"""Bulk report ingest CLI (spec §6.4).

Walks the report corpus and loads it into the vector store. Run it from `backend/`:

    python -m app.rag.ingest

Corpus layout (metadata is derived from the PATH, not the document text):

    data/reports/<era>/<FY folder>/<AS-number>.docx
    e.g. data/reports/2022-2026/FY2025 (AS0599-AS0611)/AS0599.docx
                      └─ era        └─ fiscal_year        └─ report_id

For each .docx: derive report_id / fiscal_year / era from the path, extract text
into sections (extract.py), chunk section-aware (chunker.py), embed
(embeddings.py), and store with metadata on every chunk (store.py).

Idempotent: reports already in the store are skipped, so re-running only picks up
new files. Use --force to re-ingest everything. Progress prints per file since the
full corpus (~1 GB) takes a while.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ..config import get_settings, repo_path

# AS-number anywhere in the filename stem: "AS0599", "AS0599_final" -> AS0599.
_REPORT_ID = re.compile(r"(AS\d+)", re.IGNORECASE)
# Fiscal year inside the parent folder name: "FY2025 (AS0599-AS0611)" -> FY2025.
_FISCAL_YEAR = re.compile(r"(FY\s?\d{4})", re.IGNORECASE)


@dataclass
class ReportFile:
    path: Path
    report_id: str
    era: str
    fiscal_year: str | None  # None for older eras that have no FY subfolders


def _derive_metadata(path: Path, root: Path) -> ReportFile | None:
    """Derive report_id / era / fiscal_year from a report's path, or None if unrecognizable.

    The corpus has two layouts. Newer eras (2017-2021, 2022-2026) nest reports under
    FY subfolders: ``<era>/FY2025 (...)/AS0599.docx``. Older eras (1994-1999,
    2000-2006, 2006 and earlier) place the .docx directly inside the era folder:
    ``<era>/AS0001.docx``, with no fiscal year. We tell them apart by depth under the
    corpus root, so a file is classified by where it actually sits.
    """
    rid = _REPORT_ID.search(path.stem)
    if rid is None:
        return None
    report_id = rid.group(1).upper()

    rel_parts = path.relative_to(root).parts
    # rel_parts[0] is the era folder; the file is either directly inside it
    # (era/file.docx) or under an FY subfolder (era/FY/.../file.docx).
    era = rel_parts[0] if len(rel_parts) >= 2 else ""

    if len(rel_parts) >= 3:
        # Under an FY subfolder: parse the fiscal year from the immediate parent.
        fy = _FISCAL_YEAR.search(path.parent.name)
        fiscal_year = fy.group(1).upper().replace(" ", "") if fy else path.parent.name
    else:
        # Directly in the era folder (older eras): no fiscal year.
        fiscal_year = None

    return ReportFile(path=path, report_id=report_id, era=era, fiscal_year=fiscal_year)


def iter_report_files(root: Path) -> Iterator[ReportFile]:
    """Yield every recognizable .docx report under root, recursively and in stable order."""
    for path in sorted(root.rglob("*.docx")):
        if path.name.startswith("~$"):  # Word lock/temp files
            continue
        rf = _derive_metadata(path, root)
        if rf is None:
            print(f"  ! skipping (no AS-number in filename): {path.relative_to(root)}", flush=True)
            continue
        yield rf


def _ingest_one(report: ReportFile, embedder, store, settings) -> int:
    """Extract, chunk, embed, and store one report. Returns the number of chunks written."""
    from .chunker import chunk_document
    from .extract import extract_docx

    doc = extract_docx(report.path)
    base_metadata = {
        "report_id": report.report_id,
        # ChromaDB metadata cannot be None; older reports without an FY store "".
        "fiscal_year": report.fiscal_year or "",
        "era": report.era,
        "report_title": doc.title,
    }
    chunks = chunk_document(
        doc,
        base_metadata,
        size_tokens=settings.chunk_size_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    if not chunks:
        return 0
    # Re-ingest cleanly: drop any stale chunks for this report before writing.
    store.delete_report(report.report_id)
    embeddings = embedder.encode_documents([c.text for c in chunks])
    store.add_chunks(chunks, embeddings)
    return len(chunks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest the ITAC report corpus into ChromaDB.")
    parser.add_argument("--reports", help="Override the reports root (default: config reports_path).")
    parser.add_argument("--force", action="store_true", help="Re-ingest reports already in the store.")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N new reports (0 = all).")
    args = parser.parse_args(argv)

    settings = get_settings()
    root = repo_path(args.reports) if args.reports else repo_path(settings.reports_path)

    if not root.exists():
        print(f"Reports directory not found: {root}")
        print("Nothing to ingest. Create it and add <era>/<FY>/<AS-number>.docx files.")
        return 0

    files = list(iter_report_files(root))
    if not files:
        print(f"No .docx reports found under {root}. Nothing to ingest.")
        return 0

    print(f"Found {len(files)} report(s) under {root}")

    # Heavy imports (torch + chromadb) are deferred until we know there is work.
    print("Loading embedding model and vector store...", flush=True)
    from .embeddings import Embedder
    from .store import ReportStore

    store = ReportStore()
    embedder = Embedder()
    already = set() if args.force else store.ingested_report_ids()
    if already:
        print(f"{len(already)} report(s) already ingested; they will be skipped.")

    started = time.time()
    processed = skipped = total_chunks = 0

    for i, report in enumerate(files, start=1):
        location = report.era + (f"/{report.fiscal_year}" if report.fiscal_year else "")
        label = f"[{i}/{len(files)}] {report.report_id} ({location})"
        if report.report_id in already:
            skipped += 1
            continue

        t0 = time.time()
        try:
            n = _ingest_one(report, embedder, store, settings)
        except Exception as e:  # noqa: BLE001 - keep going; one bad file shouldn't abort the run
            print(f"{label}  ERROR: {e}", flush=True)
            continue

        processed += 1
        total_chunks += n
        already.add(report.report_id)
        print(f"{label}  {n} chunks in {time.time() - t0:.1f}s  (total {total_chunks} chunks)", flush=True)

        if args.limit and processed >= args.limit:
            print(f"Reached --limit {args.limit}; stopping.")
            break

    elapsed = time.time() - started
    print(
        f"\nDone: {processed} ingested, {skipped} skipped, {total_chunks} chunks written "
        f"in {elapsed:.1f}s. Store now holds {store.count()} chunks."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

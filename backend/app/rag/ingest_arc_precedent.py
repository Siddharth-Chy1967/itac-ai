"""Mine real (AR title -> ARC code) precedent pairs from past reports.

Assessors cite the ARC code inline on an AR heading, e.g.::

    AR-1 Replace Smash My Trash Service with an In-House Stationary Trash
    Compactor (ARC 3.6193)

Those headings are real, assessor-authored phrasings of what an ARC code means in
practice, which match a user's own description ("replace our trash haul with an
in-house compactor") far better than the terse official book text ("INSTALL
EQUIPMENT (E.G. COMPACTOR) TO REDUCE DISPOSAL COSTS"). We embed the AR titles into
a separate `arc_precedent` collection that ARC lookup queries alongside the book.

Why parse the docx XML directly instead of reusing the ingested chunks: the report
extractor walks only top-level body paragraphs and requires an ``AR #N`` marker, so
it silently drops titles that live in a Word text box or use the ``AR-1`` form
(hyphen) -- which is exactly how the 3.6193 example is written. Reading every
``<w:p>`` in ``document.xml`` catches text-box paragraphs too, and a permissive
``AR[ #-.:]*N`` lead handles the punctuation variants.

Only citations to a real leaf code (present in the ARC book) are kept, and each
precedent carries that code's official description/hierarchy so display is
identical no matter whether a book or precedent entry won the match.

Run from ``backend/``::

    python -m app.rag.ingest_arc_precedent            # mine + embed + store
    python -m app.rag.ingest_arc_precedent --dry-run  # mine + print only
"""
from __future__ import annotations

import argparse
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

from ..config import get_settings, repo_path
from .ingest import iter_report_files
from .ingest_arc import ArcEntry, parse_arc_pdf

_DEFAULT_ARC_PDF = "data/reference/ARC List - V21.1.pdf"

# One <w:p> paragraph (they do not nest), then every <w:t> text run inside it.
_PARAGRAPH = re.compile(r"<w:p[ >](?:(?!</w:p>).)*?</w:p>", re.DOTALL)
_TEXT_RUN = re.compile(r"<w:t[^>]*>([^<]*)</w:t>")
# Inline ARC citation; the base leaf code, dropping any ".N" instance suffix.
_CITE = re.compile(r"ARC\s*#?\s*(\d\.\d{3,4})(?:\.\d+)?", re.IGNORECASE)
# An AR-heading lead in any punctuation form: "AR-1", "AR #1", "AR 1", "AR No. 1".
_AR_LEAD = re.compile(r"^\s*AR[\s#\-.:]*(?:No\.?\s*)?\d+\b", re.IGNORECASE)


@dataclass
class PrecedentPair:
    """One mined (AR title, ARC code) pair with the code's official book context."""

    code: str
    ar_title: str
    report_id: str
    fiscal_year: str
    book: ArcEntry  # official leaf this code resolves to

    @property
    def entry_id(self) -> str:
        return f"{self.report_id}::{self.code}::{abs(hash(self.ar_title)) % 10_000_000}"

    @property
    def metadata(self) -> dict:
        # Carry the official description/hierarchy so display is identical whether a
        # book or precedent entry wins the match; keep provenance for the UI.
        return {
            "code": self.code,
            "description": self.book.description,
            "chapter": self.book.chapter,
            "subsection": self.book.subsection,
            "report_id": self.report_id,
            "fiscal_year": self.fiscal_year,
            "ar_title": self.ar_title,
            "source": "precedent",
        }


def _paragraph_texts(xml: str) -> list[str]:
    """Reconstruct each paragraph's text from its runs (text boxes included)."""
    out = []
    for pm in _PARAGRAPH.finditer(xml):
        text = "".join(_TEXT_RUN.findall(pm.group(0)))
        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        if text.strip():
            out.append(text)
    return out


def _clean_title(raw: str) -> str:
    """Strip the AR-number lead, any section number, and the trailing ARC citation."""
    title = _CITE.split(raw)[0]
    title = re.sub(r"^\s*AR[\s#\-.:]*(?:No\.?\s*)?\d+\s*[:.\-)]?\s*", "", title, flags=re.IGNORECASE)
    title = re.sub(r"^\s*\d+(?:\.\d+)*\s*[:.\-)]?\s*", "", title)  # "4.5" section number
    title = re.sub(r"\s+", " ", title).strip()
    return title.strip("(").strip().strip(",").strip()


def mine_precedents(reports_root: Path, book: dict[str, ArcEntry]) -> list[PrecedentPair]:
    """Extract distinct (AR title, leaf code) precedent pairs from every report."""
    pairs: list[PrecedentPair] = []
    seen: set[tuple[str, str]] = set()  # (code, title.lower()) dedup across the corpus

    for rf in iter_report_files(reports_root):
        try:
            with zipfile.ZipFile(rf.path) as z:
                xml = z.read("word/document.xml").decode("utf-8", "ignore")
        except Exception:  # noqa: BLE001 - a corrupt/odd docx should not abort the run
            continue

        for para in _paragraph_texts(xml):
            cite = _CITE.search(para)
            if not cite or not _AR_LEAD.search(para):
                continue
            code = cite.group(1)
            entry = book.get(code)
            if entry is None:  # citation to a non-leaf / unknown code -- skip
                continue
            title = _clean_title(para)
            if not (4 <= len(title) <= 120):
                continue
            key = (code, title.lower())
            if key in seen:
                continue
            seen.add(key)
            pairs.append(
                PrecedentPair(
                    code=code,
                    ar_title=title,
                    report_id=rf.report_id,
                    fiscal_year=rf.fiscal_year or "",
                    book=entry,
                )
            )
    return pairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mine ARC precedent pairs from the report corpus.")
    parser.add_argument("--reports", help="Override the reports root (default: config reports_path).")
    parser.add_argument("--pdf", help=f"Path to the ARC PDF (default: {_DEFAULT_ARC_PDF}).")
    parser.add_argument("--dry-run", action="store_true", help="Mine and print; do not embed or store.")
    parser.add_argument("--limit", type=int, default=0, help="Print only the first N pairs in a dry run.")
    args = parser.parse_args(argv)

    settings = get_settings()
    root = repo_path(args.reports) if args.reports else repo_path(settings.reports_path)
    pdf_path = repo_path(args.pdf) if args.pdf else repo_path(_DEFAULT_ARC_PDF)
    if not root.exists():
        print(f"Reports directory not found: {root}")
        return 1
    if not pdf_path.exists():
        print(f"ARC PDF not found: {pdf_path}")
        return 1

    book = {e.code: e for e in parse_arc_pdf(pdf_path)}
    print(f"Mining precedent pairs from reports under {root} ...", flush=True)
    pairs = mine_precedents(root, book)
    codes = {p.code for p in pairs}
    print(f"Mined {len(pairs)} precedent pairs across {len(codes)} distinct ARC codes.")

    if args.dry_run:
        shown = pairs[: args.limit] if args.limit else pairs
        for p in shown:
            print(f"  {p.code:<8} [{p.report_id}] {p.ar_title}")
        return 0

    print("Loading embedding model and precedent store...", flush=True)
    from .embeddings import Embedder
    from .store import ARC_PRECEDENT_COLLECTION_NAME, ArcStore

    store = ArcStore(ARC_PRECEDENT_COLLECTION_NAME)
    embedder = Embedder()

    store.reset()
    embeddings = embedder.encode_documents([p.ar_title for p in pairs])
    store.add_entries(
        ids=[p.entry_id for p in pairs],
        texts=[p.ar_title for p in pairs],
        metadatas=[p.metadata for p in pairs],
        embeddings=embeddings,
    )
    print(f"Done: {store.count()} precedent entries stored in the arc_precedent collection.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

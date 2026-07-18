"""One-time ingest of the official IAC ARC leaf codes into the vector store.

The ARC (Assessment Recommendation Code) list is a hierarchical catalog: a
chapter (``2. Energy Management``), a section (``2.4 Combustion Systems``), a
category (``2.42 ...``), a subsection (``2.423 ...``), and finally the LEAF
recommendations we want (``2.4236 ELIMINATE LEAKS IN INERT GAS AND COMPRESSED
AIR LINES/ VALVES``). Only the leaves are real, codable recommendations.

How leaves are told apart from the scaffolding (verified against ARC v21.1):

- A leaf code has 3 or 4 digits after the dot (``X.YYY`` or ``X.YYYY``); shorter
  codes (``2.4``, ``2.42``) are always parent headers.
- A leaf's description is ALL CAPS ("INSTALL OCCUPANCY SENSORS"); a same-length
  subsection header is Title Case ("Operations", "Flue Gas - Recuperation").
  This is what separates the leaf ``4.111 SOMETHING IN CAPS`` from the
  subsection ``2.111 Operations``.

The Table of Contents (which repeats every code with dot-leaders and a page
number) is skipped by rejecting any line containing a run of leader dots.

Run from ``backend/``::

    python -m app.rag.ingest_arc                 # parse + embed + store
    python -m app.rag.ingest_arc --dry-run       # parse + print, no embedding

Rebuilds the ``arc_reference`` collection from scratch each run (the corpus is
~600 entries derived entirely from the PDF, so there is nothing to preserve).
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from ..config import get_settings, repo_path

# Default location of the ARC PDF (real filename has spaces and dots).
_DEFAULT_ARC_PDF = "data/reference/ARC List - V21.1.pdf"

# A code line: leading code then its text. The code is a single digit, a dot, and
# one-or-more digits. Depth (digits after the dot) decides what it is.
_CODE_LINE = re.compile(r"^(\d\.\d+)\s+(\S.*)$")
# A chapter header: "2. Energy Management" (single digit, dot, space, title).
_CHAPTER_LINE = re.compile(r"^(\d)\.\s+(\S.*)$")
# Runs of leader dots only appear in the Table of Contents.
_TOC_LEADER = re.compile(r"\.{4,}")
# Page header/footer repeated on every page.
_RUNNING_HEADER = re.compile(r"^IAC Assessment Recommendation Code System", re.IGNORECASE)


@dataclass
class ArcEntry:
    """One leaf ARC recommendation with its display hierarchy."""

    code: str          # e.g. "3.6193"
    description: str    # ALL-CAPS recommendation text
    chapter: str        # X.Y level, e.g. "3.6 Waste Disposal"
    subsection: str     # X.YYY level, e.g. "3.619 Miscellaneous" ("" if the leaf IS X.YYY)

    @property
    def embed_text(self) -> str:
        """Text embedded for retrieval: code + description (per spec)."""
        return f"{self.code} {self.description}"

    @property
    def metadata(self) -> dict:
        return {
            "code": self.code,
            "description": self.description,
            "chapter": self.chapter,
            "subsection": self.subsection,
        }


def _is_all_caps(text: str) -> bool:
    """True for a leaf description: has letters and none of them are lowercase.

    Digits, punctuation, ``&``, ``/`` and parentheses are allowed (e.g.
    "INSTALL EQUIPMENT (E.G. COMPACTOR) TO REDUCE DISPOSAL COSTS"). A single
    lowercase letter means it is a Title-Case subsection header, not a leaf.
    """
    return any(c.isupper() for c in text) and not any(c.islower() for c in text)


def parse_arc_pdf(pdf_path: Path) -> list[ArcEntry]:
    """Extract every leaf ARC entry from the PDF, with running parent hierarchy."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))

    entries: list[ArcEntry] = []
    seen_codes: set[str] = set()
    # Most recent header seen at each level, carried down onto the leaves.
    section = ""      # X.Y   e.g. "3.6 Waste Disposal"
    subsection = ""   # X.YYY e.g. "3.619 Miscellaneous"

    for page in reader.pages:
        for raw in page.extract_text().splitlines():
            line = raw.strip()
            if not line or _TOC_LEADER.search(line) or _RUNNING_HEADER.match(line):
                continue
            if _CHAPTER_LINE.match(line):
                # New chapter resets the section/subsection context.
                section = subsection = ""
                continue

            m = _CODE_LINE.match(line)
            if not m:
                continue
            code, text = m.group(1), m.group(2).strip()
            after_dot = len(code.split(".", 1)[1])

            if after_dot == 1:            # X.Y  section header
                section = f"{code} {text}"
                subsection = ""
            elif after_dot == 2:          # X.YY category header (not tracked for display)
                subsection = ""
            elif after_dot in (3, 4):
                if _is_all_caps(text):    # LEAF recommendation
                    if code in seen_codes:
                        continue
                    seen_codes.add(code)
                    entries.append(
                        ArcEntry(code=code, description=text, chapter=section, subsection=subsection)
                    )
                elif after_dot == 3:      # X.YYY subsection header
                    subsection = f"{code} {text}"
            # after_dot >= 5 is not part of the ARC leaf scheme; ignore.

    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest the IAC ARC leaf codes into ChromaDB.")
    parser.add_argument("--pdf", help=f"Path to the ARC PDF (default: {_DEFAULT_ARC_PDF}).")
    parser.add_argument("--dry-run", action="store_true", help="Parse and print; do not embed or store.")
    parser.add_argument("--limit", type=int, default=0, help="Print only the first N entries in a dry run.")
    args = parser.parse_args(argv)

    pdf_path = repo_path(args.pdf) if args.pdf else repo_path(_DEFAULT_ARC_PDF)
    if not pdf_path.exists():
        print(f"ARC PDF not found: {pdf_path}")
        return 1

    print(f"Parsing {pdf_path.name} ...", flush=True)
    entries = parse_arc_pdf(pdf_path)
    print(f"Parsed {len(entries)} leaf ARC codes.")

    if args.dry_run:
        shown = entries[: args.limit] if args.limit else entries
        for e in shown:
            print(f"  {e.code:<8} {e.description}")
            print(f"           chapter={e.chapter!r} subsection={e.subsection!r}")
        return 0

    print("Loading embedding model and ARC store...", flush=True)
    from .embeddings import Embedder
    from .store import ArcStore

    store = ArcStore()
    embedder = Embedder()

    store.reset()  # rebuild from scratch
    embeddings = embedder.encode_documents([e.embed_text for e in entries])
    store.add_entries(
        ids=[e.code for e in entries],
        texts=[e.embed_text for e in entries],
        metadatas=[e.metadata for e in entries],
        embeddings=embeddings,
    )
    print(f"Done: {store.count()} ARC codes stored in the arc_reference collection.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

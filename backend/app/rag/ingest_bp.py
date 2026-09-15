"""One-time ingest of the Best Practice list into its own vector store (Section B).

Fully standalone: this touches only the `bp_reference` collection and shares no
code, collection, or reference with the ARC feature.

The list (data/reference/Best_Practice_List.docx) is a flat Word outline:
  - Heading 1  -> category            ("Renewable Energy Practices")
  - Heading 2  -> best-practice item  ("Solar Panels:")
  - body text  -> a template paragraph describing the practice, with bracketed
                  fill-in instructions ("(Insert location of solar panels)") that
                  tell the assessor what to record on-site.

The fill-in instructions are stripped from the embedded text (they are noise for
similarity) but kept in metadata so the checklist can tell the student what to
actually note. Each item embeds as "<item name>. <template without fill-ins>".

Run from backend/::

    python -m app.rag.ingest_bp --dry-run      # parse + counts, no embedding
    python -m app.rag.ingest_bp                 # parse + embed + store
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ..config import get_settings, repo_path

_DEFAULT_BP_DOCX = "data/reference/Best_Practice_List.docx"

# A parenthetical fill-in instruction directed at the report writer, e.g.
# "(Insert location ...)", "(Include what they want ...)", "(Write a description ...)",
# or a placeholder blank like "(Number of PSI)". Anchored on an authoring verb /
# placeholder phrase at the START of the parenthetical so acronym expansions and
# clarifiers ("(MCCs)", "(a byproduct of wastewater treatment)", "(SCADA)") — which
# are real descriptive content — are left in the embedded text.
_FILL_IN = re.compile(
    r"\(\s*(?:insert|include|only include|write|enter|specify|add|note|indicate|"
    r"list|state|provide|number of|type of|amount of|estimate|quantity of|"
    r"name of|location of)\b[^)]*\)",
    re.IGNORECASE,
)


@dataclass
class BPItem:
    """One best-practice entry with its category and on-site fill-in prompts."""

    category: str
    name: str
    template: str = ""          # body text, fill-ins stripped
    fill_ins: list[str] = field(default_factory=list)

    @property
    def embed_text(self) -> str:
        return f"{self.name}. {self.template}".strip().rstrip(".") if self.template else self.name

    @property
    def metadata(self) -> dict:
        return {
            "category": self.category,
            "name": self.name,
            "template": self.template,
            # Chroma metadata cannot hold a list; join for storage.
            "fill_ins": " | ".join(self.fill_ins),
        }


def _clean_name(text: str) -> str:
    return text.strip().rstrip(":").strip()


def parse_bp_docx(path: Path) -> list[BPItem]:
    """Parse the Best Practice list into category-grouped items (H1/H2 + body)."""
    import docx

    document = docx.Document(str(path))

    items: list[BPItem] = []
    category = ""
    current: BPItem | None = None

    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = (para.style.name if para.style else "") or ""

        if style == "Heading 1":
            category = _clean_name(text)
            current = None
        elif style == "Heading 2":
            current = BPItem(category=category, name=_clean_name(text))
            items.append(current)
        elif current is not None and category:
            # Body/normal paragraph under the current item: accumulate as template,
            # pulling out the "(Insert ...)" fill-in prompts.
            current.fill_ins.extend(m.group(0) for m in _FILL_IN.finditer(text))
            cleaned = _FILL_IN.sub("", text)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            current.template = (current.template + " " + cleaned).strip() if current.template else cleaned

    return items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest the Best Practice list into ChromaDB (bp_reference).")
    parser.add_argument("--docx", help=f"Path to the Best Practice docx (default: {_DEFAULT_BP_DOCX}).")
    parser.add_argument("--dry-run", action="store_true", help="Parse and print counts; do not embed or store.")
    parser.add_argument("--limit", type=int, default=0, help="Print only the first N items in a dry run.")
    args = parser.parse_args(argv)

    path = repo_path(args.docx) if args.docx else repo_path(_DEFAULT_BP_DOCX)
    if not path.exists():
        print(f"Best Practice docx not found: {path}")
        return 1

    print(f"Parsing {path.name} ...", flush=True)
    items = parse_bp_docx(path)
    categories = list(dict.fromkeys(i.category for i in items))
    print(f"Parsed {len(items)} best-practice items across {len(categories)} categories.")

    if args.dry_run:
        shown = items[: args.limit] if args.limit else items
        for i in shown:
            fi = f"  fill-ins={i.fill_ins}" if i.fill_ins else ""
            print(f"  [{i.category}] {i.name}{fi}")
        return 0

    print("Loading embedding model and BP store...", flush=True)
    from .embeddings import Embedder
    from .store import BPStore

    store = BPStore()
    embedder = Embedder()

    store.reset()
    embeddings = embedder.encode_documents([i.embed_text for i in items])
    store.add_entries(
        ids=[f"bp::{n}" for n in range(len(items))],
        texts=[i.embed_text for i in items],
        metadatas=[i.metadata for i in items],
        embeddings=embeddings,
    )
    print(f"Done: {store.count()} best-practice items stored in the bp_reference collection.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""DOCX extraction into ordered sections (spec §6.1).

ITAC/IAC report templates do NOT use Word heading styles, so we cannot rely on
paragraph.style to find section boundaries. Instead we pattern-match the two
structural markers the template actually uses:

- **Bold numbered titles** ("1. Executive Summary", "2.3 Compressed Air"): a
  leading `N`/`N.N` number on a bold paragraph.
- **AR markers** ("AR #1", "AR No. 2", "AR3 ..."): the Assessment Recommendation
  tags that head each recommendation block.

Everything between one marker and the next accumulates as that section's body.
Paragraphs and tables are walked in true document order (python-docx does not
expose them interleaved by default, hence `iter_block_items`). Tables serialize
to pipe-joined rows so their content survives into the retrievable text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import docx
from docx.document import Document as _Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

# A leading number like "1", "2.3", "4.1.2" followed by title text. The bold check
# (see `_is_bold`) is what keeps this from matching numbered list items or "1. See
# appendix" body lines.
_SECTION_NUM = re.compile(r"^\s*(\d+(?:\.\d+)*)[.)]?\s+\S")

# AR tag at the start of a line: "AR #1", "AR No. 2", "AR3". Specific enough to
# trust on pattern alone, no bold requirement.
_AR_MARKER = re.compile(r"^\s*AR\s*#?\s*(?:No\.?\s*)?(\d+)\b", re.IGNORECASE)

_PREAMBLE = "(preamble)"


@dataclass
class Section:
    """One report section: a heading plus the body paragraphs beneath it."""

    heading: str
    paragraphs: list[str] = field(default_factory=list)
    ar_number: Optional[str] = None
    level: int = 1

    @property
    def text(self) -> str:
        return "\n".join(self.paragraphs).strip()


@dataclass
class ExtractedDoc:
    title: str
    sections: list[Section]


def iter_block_items(parent: _Document) -> Iterator[object]:
    """Yield Paragraphs and Tables in document order (standard python-docx recipe)."""
    body = parent.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, parent)
        elif child.tag == qn("w:tbl"):
            yield Table(child, parent)


def _is_bold(para: Paragraph) -> bool:
    """True if the paragraph reads as bold (any text-bearing run bold, or a bold style)."""
    runs = [r for r in para.runs if r.text.strip()]
    if runs and any(bool(r.bold) for r in runs):
        return True
    try:
        style = para.style
        return bool(style and style.font is not None and style.font.bold)
    except Exception:  # noqa: BLE001 - style resolution is best-effort
        return False


def _classify(text: str, para: Paragraph) -> Optional[tuple[str, Optional[str], int]]:
    """Return (heading, ar_number, level) if this paragraph starts a section, else None."""
    stripped = text.strip()
    ar = _AR_MARKER.match(stripped)
    if ar:
        return stripped, ar.group(1), 2
    m = _SECTION_NUM.match(stripped)
    if m and _is_bold(para):
        level = m.group(1).count(".") + 1
        return stripped, None, level
    return None


def _table_rows(table: Table) -> Iterator[str]:
    """Serialize a table to one pipe-joined line per row (blank rows dropped)."""
    for row in table.rows:
        cells = [c.text.strip().replace("\n", " ") for c in row.cells]
        line = " | ".join(cells).strip()
        if line.strip(" |"):
            yield line


def extract_docx(path: str | Path) -> ExtractedDoc:
    """Extract a .docx into a title and an ordered list of sections."""
    document = docx.Document(str(path))

    sections: list[Section] = []
    current = Section(heading=_PREAMBLE, level=0)
    title = ""

    for block in iter_block_items(document):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue
            if not title:
                title = text  # first non-empty line, a fallback (ingest sets report_id)
            cls = _classify(text, block)
            if cls is not None:
                heading, ar_number, level = cls
                if current.paragraphs or current.heading != _PREAMBLE:
                    sections.append(current)
                current = Section(heading=heading, ar_number=ar_number, level=level)
            else:
                current.paragraphs.append(text)
        elif isinstance(block, Table):
            current.paragraphs.extend(_table_rows(block))

    sections.append(current)
    # Drop an empty leading preamble (content before the first real heading).
    sections = [s for s in sections if s.text or s.heading != _PREAMBLE]
    return ExtractedDoc(title=title, sections=sections)

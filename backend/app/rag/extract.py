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

# AR tag at the start of a line, in any punctuation form the corpus uses:
# "AR #1", "AR No. 2", "AR3", and crucially "AR-1" (the hyphen form newer reports
# use, which the old `AR\s*#?` pattern missed entirely). The `[\s#\-.:]*` separator
# class stays permissive but the trailing `\d` requirement keeps "ARC 2.4236" and
# words like "Area" from matching (no digit follows the separators there).
_AR_MARKER = re.compile(r"^\s*AR[\s#\-.:]*(?:No\.?\s*)?(\d+)\b", re.IGNORECASE)

# A facility-profile section heading (the plant/process description used by Pass B
# of the pre-assessment checklist). Anchored at line start, optional leading section
# number, so "The process description is in Appendix B" (mid-sentence) does not
# match. Present in every era: newer reports use unnumbered "Plant Description";
# older ones use "2.1 Facility Description" / "2.0 General Background".
_PROFILE_HEADING = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*[.)]?\s*)?"
    r"(?:(?:plant|facility|process)\s+description|(?:general|company)\s+background)\b",
    re.IGNORECASE,
)

FACILITY_PROFILE = "facility_profile"
_PREAMBLE = "(preamble)"

# The "Principal Products: <text>" header field, captured up to the next known
# field label (so we get "Prefabricated wood buildings", not the whole header
# blob). This is the reliable cross-era facility signal: newer reports (2022-2026)
# have no "Plant Description" heading but do carry this line, so it becomes the
# synthetic facility-profile chunk that Pass B (process similarity) needs.
_PRINCIPAL_PRODUCTS = re.compile(
    r"Principal\s+Products?\s*[:|]\s*(.+?)"
    r"(?=\s*(?:Assessment\s+Date|Audit\s+Date|Report\s+(?:Date|No)|Location|"
    r"Number\s+of|SIC\b|NAICS|S\.?\s*I\.?\s*C\.?|\||\n|$))",
    re.IGNORECASE,
)


def _is_profile_heading(text: str) -> bool:
    """True for a plant/process-description heading (short enough to be a heading)."""
    return len(text) <= 60 and bool(_PROFILE_HEADING.match(text))


def _principal_products(title: str, sections: list["Section"]) -> str:
    """The facility's principal-products text from the report header, or "" if absent."""
    m = _PRINCIPAL_PRODUCTS.search(title or "")
    if not m:
        for s in sections:
            m = _PRINCIPAL_PRODUCTS.search(f"{s.heading} {s.text}")
            if m:
                break
    if not m:
        return ""
    # Collapse whitespace and trim trailing separators/labels the lookahead left.
    return re.sub(r"\s+", " ", m.group(1)).strip().strip(":|").strip()


@dataclass
class Section:
    """One report section: a heading plus the body paragraphs beneath it."""

    heading: str
    paragraphs: list[str] = field(default_factory=list)
    ar_number: Optional[str] = None
    level: int = 1
    section_type: str = ""  # "facility_profile" for plant/process-description sections

    @property
    def text(self) -> str:
        return "\n".join(self.paragraphs).strip()


@dataclass
class ExtractedDoc:
    title: str
    sections: list[Section]


def _iter_textbox_paragraphs(element, parent: _Document) -> Iterator[Paragraph]:
    """Yield paragraphs nested in text boxes anchored under `element`, deduped.

    python-docx's block iteration only walks top-level body children, so paragraphs
    inside a Word text box (`w:txbxContent`, reached via a drawing anchor) are never
    surfaced. In this corpus text boxes hold diagram/floor-plan labels and equation
    nomenclature rather than structural headings, but we surface them so nothing is
    silently dropped and any future report that does put real content there is
    captured. Text boxes wrapped in `mc:AlternateContent` store their text twice (a
    modern DrawingML copy and a VML fallback); each distinct paragraph is yielded
    once.
    """
    seen: set[str] = set()
    for txbx in element.iter(qn("w:txbxContent")):
        for p in txbx.iter(qn("w:p")):
            para = Paragraph(p, parent)
            key = para.text.strip()
            if key and key not in seen:
                seen.add(key)
                yield para


def iter_block_items(parent: _Document) -> Iterator[object]:
    """Yield Paragraphs and Tables in document order (standard python-docx recipe).

    Text-box paragraphs anchored in a block are yielded right after it, so their
    content lands in the section it visually sits in.
    """
    body = parent.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, parent)
            yield from _iter_textbox_paragraphs(child, parent)
        elif child.tag == qn("w:tbl"):
            yield Table(child, parent)
            yield from _iter_textbox_paragraphs(child, parent)


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
    # An unnumbered profile heading ("Plant Description") starts a section too, so
    # newer reports (which do not number it) still get a facility-profile section.
    if _is_profile_heading(stripped):
        return stripped, None, 2
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
                section_type = FACILITY_PROFILE if _is_profile_heading(heading) else ""
                current = Section(
                    heading=heading, ar_number=ar_number, level=level, section_type=section_type
                )
            else:
                current.paragraphs.append(text)
        elif isinstance(block, Table):
            current.paragraphs.extend(_table_rows(block))

    sections.append(current)
    # Drop an empty leading preamble (content before the first real heading).
    sections = [s for s in sections if s.text or s.heading != _PREAMBLE]

    # Facility-profile fallback: newer reports (2022-2026) have no "Plant
    # Description" heading but carry a "Principal Products:" header field. Surface
    # it as its own short, high-signal facility-profile section so Pass B (process
    # similarity) has something to match on for every era, not just the ones with a
    # description heading. Rich description sections, when present, are already
    # tagged above; this adds to them rather than replacing them.
    products = _principal_products(title, sections)
    if products:
        sections.insert(
            0,
            Section(
                heading="Principal Products",
                paragraphs=[products],
                section_type=FACILITY_PROFILE,
            ),
        )

    return ExtractedDoc(title=title, sections=sections)

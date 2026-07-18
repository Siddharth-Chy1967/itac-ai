"""Tests for the RAG ingest pipeline's pure logic (spec §6.1, §6.2, §6.4).

Covers the parts that do not need the ML stack: path-derived metadata, section/AR
boundary detection by pattern (not Word styles), and section-aware chunking with
overlap. Embedding and vector-store writes are exercised by running the CLI, not
unit-tested here.
"""
from __future__ import annotations

from pathlib import Path

from docx import Document

from app.rag.chunker import chunk_document
from app.rag.extract import ExtractedDoc, Section, extract_docx
from app.rag.ingest import ReportFile, _derive_metadata, _ingest_one, iter_report_files


# --- path-derived metadata (spec §6.4) ------------------------------------

def test_derive_metadata_from_path():
    root = Path("data/reports")
    p = root / "2022-2026" / "FY2025 (AS0599-AS0611)" / "AS0599.docx"
    rf = _derive_metadata(p, root)
    assert rf is not None
    assert rf.report_id == "AS0599"
    assert rf.fiscal_year == "FY2025"
    assert rf.era == "2022-2026"


def test_derive_metadata_uppercases_and_strips_fy_spacing():
    root = Path("reports")
    p = root / "2018-2022" / "FY 2019" / "as0123_final.docx"
    rf = _derive_metadata(p, root)
    assert rf is not None
    assert rf.report_id == "AS0123"
    assert rf.fiscal_year == "FY2019"


def test_derive_metadata_rejects_non_as_filename():
    root = Path("reports")
    assert _derive_metadata(root / "2022-2026" / "FY2025" / "notes.docx", root) is None


def test_derive_metadata_era_folder_without_fy():
    # Older eras keep .docx directly in the era folder, no FY subfolder.
    root = Path("data/reports")
    rf = _derive_metadata(root / "1994-1999" / "AS0001.docx", root)
    assert rf is not None
    assert rf.report_id == "AS0001"
    assert rf.era == "1994-1999"
    assert rf.fiscal_year is None


def test_iter_report_files_handles_mixed_era_layouts(tmp_path):
    root = tmp_path / "reports"
    # Old era: file directly inside the era folder (no FY).
    old = root / "2000-2006"
    old.mkdir(parents=True)
    (old / "AS0042.docx").write_bytes(b"")
    # New era: file nested under an FY subfolder.
    new = root / "2022-2026" / "FY2025 (AS0599-AS0611)"
    new.mkdir(parents=True)
    (new / "AS0599.docx").write_bytes(b"")

    by_id = {rf.report_id: rf for rf in iter_report_files(root)}
    assert by_id["AS0042"].era == "2000-2006"
    assert by_id["AS0042"].fiscal_year is None
    assert by_id["AS0599"].era == "2022-2026"
    assert by_id["AS0599"].fiscal_year == "FY2025"


def test_ingest_one_stores_empty_fiscal_year_for_old_era(tmp_path):
    # None fiscal_year must reach ChromaDB as "" (metadata cannot be None).
    path = tmp_path / "AS0042.docx"
    _build_report(path)
    report = ReportFile(path=path, report_id="AS0042", era="2000-2006", fiscal_year=None)

    captured: dict = {}

    class FakeStore:
        def delete_report(self, report_id):
            pass

        def add_chunks(self, chunks, embeddings):
            captured["chunks"] = chunks

    class FakeEmbedder:
        def encode_documents(self, texts):
            return [[0.0] for _ in texts]

    class FakeSettings:
        chunk_size_tokens = 700
        chunk_overlap_tokens = 100

    n = _ingest_one(report, FakeEmbedder(), FakeStore(), FakeSettings())
    assert n > 0
    assert all(c.metadata["fiscal_year"] == "" for c in captured["chunks"])
    assert all(c.metadata["era"] == "2000-2006" for c in captured["chunks"])


def test_iter_report_files_walks_tree_and_skips_junk(tmp_path):
    root = tmp_path / "reports"
    good = root / "2022-2026" / "FY2025 (AS0599-AS0611)"
    good.mkdir(parents=True)
    (good / "AS0599.docx").write_bytes(b"")
    (good / "AS0600.docx").write_bytes(b"")
    (good / "~$AS0599.docx").write_bytes(b"")  # Word lock file, must be skipped
    (good / "cover_letter.docx").write_bytes(b"")  # no AS-number, must be skipped

    found = list(iter_report_files(root))
    ids = sorted(rf.report_id for rf in found)
    assert ids == ["AS0599", "AS0600"]
    assert all(rf.era == "2022-2026" and rf.fiscal_year == "FY2025" for rf in found)


# --- section / AR detection by pattern (spec §6.1) ------------------------

def _build_report(path: Path) -> None:
    doc = Document()
    doc.add_paragraph("Prepared for Acme Manufacturing.")  # preamble, before any heading

    h1 = doc.add_paragraph()
    h1.add_run("1. Executive Summary").bold = True
    doc.add_paragraph("We assessed the facility in June.")

    h2 = doc.add_paragraph()
    h2.add_run("2.1 Compressed Air").bold = True
    doc.add_paragraph("Leaks were found on three headers.")

    ar = doc.add_paragraph()
    ar.add_run("AR #1: Install VFDs on supply fans").bold = True
    doc.add_paragraph("Anticipated savings: $10,000/yr.")

    # Numbered but NOT bold -> body text, not a heading.
    doc.add_paragraph("3. see appendix for the full data log")

    doc.save(str(path))


def test_extract_detects_bold_numbered_and_ar_headings(tmp_path):
    path = tmp_path / "AS0599.docx"
    _build_report(path)

    doc = extract_docx(path)
    headings = [s.heading for s in doc.sections]

    assert "1. Executive Summary" in headings
    assert "2.1 Compressed Air" in headings
    assert "AR #1: Install VFDs on supply fans" in headings

    # The non-bold numbered line did not start a new section.
    assert not any(h.startswith("3. see appendix") for h in headings)

    # AR section carries its parsed number and its body.
    ar_section = next(s for s in doc.sections if s.ar_number)
    assert ar_section.ar_number == "1"
    assert "10,000" in ar_section.text

    # Sub-section level reflects the dotted number depth.
    ca = next(s for s in doc.sections if s.heading.startswith("2.1"))
    assert ca.level == 2


def test_extract_serializes_tables_into_section_text(tmp_path):
    path = tmp_path / "AS0601.docx"
    doc = Document()
    h = doc.add_paragraph()
    h.add_run("1. Utility Summary").bold = True
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Month"
    table.cell(0, 1).text = "kWh"
    table.cell(1, 0).text = "Jun-25"
    table.cell(1, 1).text = "880"
    doc.save(str(path))

    extracted = extract_docx(path)
    body = extracted.sections[0].text
    assert "Month | kWh" in body
    assert "Jun-25 | 880" in body


# --- section-aware chunking (spec §6.2) -----------------------------------

def test_chunk_document_one_chunk_per_small_section():
    doc = ExtractedDoc(
        title="AS0599",
        sections=[
            Section(heading="1. Executive Summary", paragraphs=["Short summary."]),
            Section(heading="AR #1", paragraphs=["Install VFDs."], ar_number="1"),
        ],
    )
    chunks = chunk_document(doc, {"report_id": "AS0599"}, size_tokens=700, overlap_tokens=100)

    assert len(chunks) == 2
    assert [c.metadata["chunk_index"] for c in chunks] == [0, 1]
    assert chunks[0].text.startswith("1. Executive Summary")
    assert chunks[1].metadata["ar_number"] == "1"
    assert chunks[1].metadata["report_id"] == "AS0599"
    # Empty ar_number is stored as "" (ChromaDB forbids None metadata values).
    assert chunks[0].metadata["ar_number"] == ""


def test_chunk_document_splits_long_section_with_overlap():
    # 10 paragraphs of ~40 tokens each (~160 chars) -> ~400 tokens; force splitting.
    paras = [f"Paragraph {i} " + ("word " * 35) for i in range(10)]
    doc = ExtractedDoc(title="AS0599", sections=[Section(heading="2. Long", paragraphs=paras)])

    chunks = chunk_document(doc, {"report_id": "AS0599"}, size_tokens=120, overlap_tokens=40)

    assert len(chunks) > 1
    assert all(c.metadata["section"] == "2. Long" for c in chunks)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    # Overlap: the last paragraph of one chunk reappears at the start of the next.
    first_paras = [p.strip() for p in chunks[0].text.split("\n")]
    second_paras = [p.strip() for p in chunks[1].text.split("\n")]
    assert first_paras[-1] in second_paras

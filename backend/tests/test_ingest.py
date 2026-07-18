"""Tests for the RAG ingest pipeline's pure logic (spec §6.1, §6.2, §6.4).

Covers the parts that do not need the ML stack: path-derived metadata, section/AR
boundary detection by pattern (not Word styles), and section-aware chunking with
overlap. Embedding and vector-store writes are exercised by running the CLI, not
unit-tested here.
"""
from __future__ import annotations

from pathlib import Path

from docx import Document

from docx import Document as _Doc

from app.rag.chunker import chunk_document
from app.rag.extract import FACILITY_PROFILE, ExtractedDoc, Section, extract_docx
from app.rag.ingest import ReportFile, _derive_metadata, _ingest_one, extract_sic, iter_report_files


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


# --- Feature 2 extraction: AR-N headings, SIC, facility profile --------------

def test_extract_detects_hyphen_ar_heading(tmp_path):
    # Newer reports write "AR-1" (hyphen); the broadened marker must tag it.
    path = tmp_path / "AS0615.docx"
    d = _Doc()
    h = d.add_paragraph()
    h.add_run("AR-1 Replace Smash My Trash Service with a Compactor (ARC 3.6193)").bold = True
    d.add_paragraph("Anticipated savings: $5,000/yr.")
    d.save(str(path))

    doc = extract_docx(path)
    ar = next((s for s in doc.sections if s.ar_number), None)
    assert ar is not None and ar.ar_number == "1"
    assert "Compactor" in ar.heading


def test_arc_prefix_not_mistaken_for_ar_heading(tmp_path):
    # "ARC 2.4236" must NOT be read as an AR heading (no digit right after "AR").
    path = tmp_path / "AS0400.docx"
    d = _Doc()
    p = d.add_paragraph()
    p.add_run("See ARC 2.4236 for the leak-repair code.").bold = True
    d.save(str(path))
    doc = extract_docx(path)
    assert all(s.ar_number is None for s in doc.sections)


def test_extract_tags_profile_heading(tmp_path):
    path = tmp_path / "AS0300.docx"
    d = _Doc()
    h = d.add_paragraph()
    h.add_run("Process Description").bold = True  # unnumbered, newer-report style
    d.add_paragraph("The facility manufactures wood cabinets from particleboard.")
    d.save(str(path))

    doc = extract_docx(path)
    prof = [s for s in doc.sections if s.section_type == FACILITY_PROFILE]
    assert any(s.heading == "Process Description" for s in prof)
    assert any("particleboard" in s.text for s in prof)


def test_principal_products_fallback_creates_profile_section(tmp_path):
    # No description heading, only a "Principal Products" field -> synthetic profile.
    path = tmp_path / "AS0515.docx"
    d = _Doc()
    d.add_paragraph("Principal Products: Residential Framing")
    h = d.add_paragraph()
    h.add_run("AR #1 Fix compressed air leaks").bold = True
    d.add_paragraph("body text")
    d.save(str(path))

    doc = extract_docx(path)
    prof = [s for s in doc.sections if s.section_type == FACILITY_PROFILE]
    assert len(prof) == 1
    assert prof[0].heading == "Principal Products"
    assert prof[0].text == "Residential Framing"


def test_iter_textbox_paragraphs_captures_and_dedups():
    # A text box wrapped in AlternateContent stores its text twice; capture once.
    from docx import Document
    from docx.oxml import parse_xml

    from app.rag.extract import _iter_textbox_paragraphs

    w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    xml = (
        f'<w:p xmlns:w="{w}">'
        f"<w:r><w:t>anchor</w:t></w:r>"
        f"<w:txbxContent><w:p><w:r><w:t>Assembly Area</w:t></w:r></w:p></w:txbxContent>"
        f"<w:txbxContent><w:p><w:r><w:t>Assembly Area</w:t></w:r></w:p></w:txbxContent>"
        f"</w:p>"
    )
    paras = list(_iter_textbox_paragraphs(parse_xml(xml), Document()))
    assert [p.text for p in paras] == ["Assembly Area"]  # deduped 2 -> 1


def test_extract_sic_handles_all_era_formats():
    def d(text):
        return ExtractedDoc(title="", sections=[Section(heading="h", paragraphs=[text])])

    assert extract_sic(d("SIC: 2452")) == "2452"
    assert extract_sic(d("SIC/NAICS:3559/334418")) == "3559"  # SIC first, NAICS is 6-digit
    assert extract_sic(d("principal products:Aluminum Windowss.i.c. code(s):3231")) == "3231"
    assert extract_sic(d("no classification here, just $2,500 saved")) == ""


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


def test_chunk_stamps_section_type_and_passes_through_sic():
    doc = ExtractedDoc(
        title="AS0515",
        sections=[
            Section(heading="Principal Products", paragraphs=["Residential Framing"],
                    section_type=FACILITY_PROFILE),
            Section(heading="AR #1", paragraphs=["Fix leaks."], ar_number="1"),
        ],
    )
    base = {"report_id": "AS0515", "sic4": "2452", "sic3": "245", "sic2": "24"}
    chunks = chunk_document(doc, base, size_tokens=700, overlap_tokens=100)

    prof, ar = chunks
    assert prof.metadata["section_type"] == "facility_profile"
    assert ar.metadata["section_type"] == ""  # AR section is not a profile
    # SIC prefixes from base_metadata reach every chunk.
    assert all(c.metadata["sic4"] == "2452" and c.metadata["sic2"] == "24" for c in chunks)


def test_ingest_one_stamps_sic_on_every_chunk(tmp_path):
    path = tmp_path / "AS0515.docx"
    d = _Doc()
    d.add_paragraph("SIC: 2452")
    h = d.add_paragraph()
    h.add_run("AR #1 Fix leaks").bold = True
    d.add_paragraph("Anticipated savings: $5,000/yr.")
    d.save(str(path))
    report = ReportFile(path=path, report_id="AS0515", era="2022-2026", fiscal_year="FY2025")

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
    assert all(c.metadata["sic4"] == "2452" for c in captured["chunks"])
    assert all(c.metadata["sic3"] == "245" and c.metadata["sic2"] == "24" for c in captured["chunks"])

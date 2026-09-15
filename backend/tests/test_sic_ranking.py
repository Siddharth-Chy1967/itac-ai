"""Tests for "most common ARs for SIC X" and partial-SIC parsing, fully mocked.

Reuses the FakeStore/FakeEmbedder from the checklist suite: no ChromaDB, no torch.
"""
from __future__ import annotations

import pytest

from app.chat import orchestrator
from app.chat.orchestrator import _classify_mode, _parse_checklist_query
from app.chat.prompts import render_sic_ranking_answer
from app.rag.waterfall import ThemeGroup, clean_ar_title, rank_ars_for_sic
from app.skills.router import RouterDecision

from test_orchestrator import RecordingProvider, _collect, _patch
from test_waterfall import FakeEmbedder, FakeStore, _ar

RANK_DEFAULTS = dict(min_reports=3, group_threshold=0.82, top_n=10)


# --- partial SIC parsing ----------------------------------------------------
# The engine always tolerated a 2- or 3-digit stem; the message regex did not, so
# "my SIC is 25xx" silently parsed to None and the industry pass never ran.

@pytest.mark.parametrize(
    "message,expected",
    [
        ("my SIC is 2599, a cabinet shop", "2599"),
        ("my SIC is 25xx, a cabinet shop", "25xx"),
        ("pre-assessment for SIC 302x", "302x"),
        ("SIC code: 259", "259"),
        ("sic 25XX plastics", "25XX"),
        ("sic 30*", "30*"),
    ],
)
def test_partial_sic_parses_out_of_the_message(message, expected):
    assert _parse_checklist_query(message)[0] == expected


@pytest.mark.parametrize(
    "message",
    [
        "what should I look for at a bakery",   # no SIC at all
        "checklist for a plant built in 1998",  # a year is not a SIC
        "pre-assessment for SIC 30281",         # 5 digits is not a SIC
    ],
)
def test_non_sic_numbers_are_not_parsed_as_sic(message):
    assert _parse_checklist_query(message)[0] is None


def test_wildcard_sic_still_classifies_as_checklist():
    # Previously the intent regex also demanded four digits, so a bare wildcard SIC
    # fell through to an ungrounded answer.
    assert _classify_mode("my SIC is 25xx") == "checklist"


# --- waterfall widening drives the ranking ----------------------------------

def _rank_store():
    """3028 has nothing, 302 has two reports, 30 has four."""
    r = [("R1", "2022-2026"), ("R2", "2000-2006"), ("R3", "1994-1999"), ("R4", "2017-2021")]
    refs = {rid: era for rid, era in r}
    from app.rag.waterfall import ReportRef

    return FakeStore(
        sic_map={
            ("sic4", "3028"): [],
            ("sic3", "302"): [ReportRef("R1", refs["R1"]), ReportRef("R2", refs["R2"])],
            ("sic2", "30"): [ReportRef(rid, era) for rid, era in r],
        },
        ar_map={
            "R1": [_ar("R1", "1", "AR#1 Repair Compressed Air Leaks", era="2022-2026")],
            "R2": [_ar("R2", "1", "AR#1 Fix air leaks", era="2000-2006")],
            "R3": [_ar("R3", "1", "AR#1 Compressed air leak repair", era="1994-1999")],
            "R4": [_ar("R4", "2", "AR#2 Install LED lighting", era="2017-2021")],
        },
    )


def test_ranking_widens_to_the_family_when_the_exact_code_is_absent():
    res = rank_ars_for_sic(sic="3028", store=_rank_store(), embedder=FakeEmbedder(), **RANK_DEFAULTS)
    assert res.sic_level == "sic2"
    assert res.sic_pattern == "30xx"
    assert res.requested_pattern == "3028"
    assert res.widened is True
    assert res.total_reports == 4


def test_ranking_counts_every_era_not_just_modern():
    # The air-leak theme spans 2022, 2000 and 1994 reports; excluding the old ones
    # is what made these lists come back nearly empty.
    res = rank_ars_for_sic(sic="3028", store=_rank_store(), embedder=FakeEmbedder(), **RANK_DEFAULTS)
    air = next(t for t in res.themes if "air" in t.representative.lower())
    assert air.report_count == 3
    assert air.modern_count == 1
    assert air.dated is False


def test_ranking_is_descending_by_report_count():
    res = rank_ars_for_sic(sic="3028", store=_rank_store(), embedder=FakeEmbedder(), **RANK_DEFAULTS)
    counts = [t.report_count for t in res.themes]
    assert counts == sorted(counts, reverse=True)


def test_ranking_not_widened_when_the_exact_code_has_enough():
    from app.rag.waterfall import ReportRef

    store = FakeStore(
        sic_map={("sic4", "3089"): [ReportRef(f"R{i}", "2022-2026") for i in range(3)]},
        ar_map={f"R{i}": [_ar(f"R{i}", "1", "AR#1 Repair Compressed Air Leaks")] for i in range(3)},
    )
    res = rank_ars_for_sic(sic="3089", store=store, embedder=FakeEmbedder(), **RANK_DEFAULTS)
    assert res.sic_level == "sic4" and res.widened is False and res.thin is False


def test_wildcard_input_starts_at_the_level_its_digits_support():
    res = rank_ars_for_sic(sic="25xx", store=_rank_store(), embedder=FakeEmbedder(), **RANK_DEFAULTS)
    # "25xx" carries two digits, so sic4/sic3 are skipped entirely.
    assert res.sic_level == "sic2"
    assert res.widened is False  # 25xx IS the pattern asked for


# --- era confidence ---------------------------------------------------------

def test_dated_flag_is_withheld_when_the_modern_sample_is_too_small():
    from app.rag.waterfall import ReportRef

    store = FakeStore(
        sic_map={("sic4", "2599"): [ReportRef("O1", "2000-2006"), ReportRef("O2", "2000-2006"),
                                    ReportRef("M1", "2022-2026")]},
        ar_map={"O1": [_ar("O1", "1", "AR#1 Install T-8 lighting", era="2000-2006")],
                "O2": [_ar("O2", "1", "AR#1 Install T-8 lighting", era="2000-2006")],
                "M1": [_ar("M1", "1", "AR#1 Repair air leaks")]},
    )
    res = rank_ars_for_sic(sic="2599", store=store, embedder=FakeEmbedder(), **RANK_DEFAULTS)
    assert res.modern_reports == 1
    assert res.era_confident is False
    text = render_sic_ranking_answer(res)
    # With one modern report, "0 since 2017" is sample size, not obsolescence.
    assert "[dated]" not in text
    assert "too small" in text


def test_dated_flag_shows_when_there_is_enough_modern_evidence():
    from app.rag.waterfall import ReportRef

    store = FakeStore(
        sic_map={("sic4", "2599"): [ReportRef("O1", "2000-2006")]
                 + [ReportRef(f"M{i}", "2022-2026") for i in range(3)]},
        ar_map={"O1": [_ar("O1", "1", "AR#1 Install T-8 lighting", era="2000-2006")],
                **{f"M{i}": [_ar(f"M{i}", "1", "AR#1 Repair air leaks")] for i in range(3)}},
    )
    res = rank_ars_for_sic(sic="2599", store=store, embedder=FakeEmbedder(), **RANK_DEFAULTS)
    assert res.era_confident is True
    text = render_sic_ranking_answer(res)
    assert "[dated]" in text


# --- title cleaning ---------------------------------------------------------

def test_toc_page_number_is_stripped_from_the_title():
    # A tab before the number marks it as a table-of-contents page reference.
    assert clean_ar_title("AR#4 Insulate the Thermoformers\t43") == "Insulate the Thermoformers"


def test_a_number_that_is_part_of_the_title_survives():
    # Only a TAB marks a page number; a space before the digits is real content.
    assert clean_ar_title("AR #1 Install T-8 Lamps in Plant 2") == "Install T-8 Lamps in Plant 2"
    assert clean_ar_title("AR#6 Add Economizers, Building 3") == "Add Economizers, Building 3"


# --- rendering --------------------------------------------------------------

def _ranking(**kw):
    from app.rag.waterfall import SicARRanking

    base = dict(
        themes=[ThemeGroup("Repair Compressed Air Leaks", ["Repair Compressed Air Leaks"],
                           ["R1", "R2", "R3"], ["R1"])],
        total_reports=4, modern_reports=3, total_ars=7, sic_level="sic2",
        sic_pattern="30xx", requested_pattern="3028", widened=True, thin=False,
        era_confident=True,
    )
    base.update(kw)
    return SicARRanking(**base)


def test_render_shows_counts_and_era_split():
    text = render_sic_ranking_answer(_ranking())
    assert "Most common ARs for SIC 30xx" in text
    assert "4 matching reports, 7 ARs" in text
    assert "Repair Compressed Air Leaks" in text
    assert "3  (1 since 2017)" in text
    assert "covers the wider 30xx family" in text


def test_render_never_leaks_similarity_scores():
    text = render_sic_ranking_answer(_ranking())
    assert "0." not in text
    assert "similarity" not in text.lower()


def test_render_reports_an_empty_corpus_honestly():
    text = render_sic_ranking_answer(_ranking(themes=[], total_reports=0, total_ars=0))
    assert "No reports" in text and "3- and 2-digit families" in text


# --- orchestrator wiring ----------------------------------------------------

@pytest.mark.asyncio
async def test_run_chat_sic_ranking_bypasses_the_llm(monkeypatch):
    provider = RecordingProvider(["should not be used"])
    _patch(monkeypatch, RouterDecision(skills=[], needs_rag=True), provider)

    async def fake_rank(sic):
        assert sic == "3028"
        return _ranking()

    monkeypatch.setattr(orchestrator, "_rank_sic_ars", fake_rank)

    events = await _collect(
        orchestrator.run_chat("most common ARs for SIC 3028", [])
    )
    text = "".join(e.data["t"] for e in events if e.event == "token")
    assert "Most common ARs for SIC 30xx" in text
    # Counted results are rendered, never phrased by the model.
    assert provider.system is None
    assert events[-1].event == "done"
    sources = [e for e in events if e.event == "sources"]
    assert sources and sources[0].data[0]["report_id"] == "Repair Compressed Air Leaks"


@pytest.mark.asyncio
async def test_run_chat_ranking_falls_back_when_store_is_down(monkeypatch):
    provider = RecordingProvider(["normal answer"])
    _patch(monkeypatch, RouterDecision(skills=[], needs_rag=True), provider)

    async def broken(sic):
        return None

    monkeypatch.setattr(orchestrator, "_rank_sic_ars", broken)

    events = await _collect(orchestrator.run_chat("most common ARs for SIC 3028", []))
    assert not any(e.event == "sources" for e in events)
    assert provider.system is not None  # fell through to a normal LLM answer


@pytest.mark.asyncio
async def test_checklist_question_does_not_route_to_ranking(monkeypatch):
    provider = RecordingProvider(["x"])
    _patch(monkeypatch, RouterDecision(skills=[], needs_rag=True), provider)

    async def fail(sic):  # pragma: no cover - must not be called
        raise AssertionError("ranking ran for a checklist question")

    monkeypatch.setattr(orchestrator, "_rank_sic_ars", fail)
    monkeypatch.setattr(orchestrator, "_build_checklist", lambda *a, **k: _none())

    async def _none():
        return None

    await _collect(orchestrator.run_chat("pre-assessment for SIC 3028", []))


# --- grouping precision (Step 3 fixes) --------------------------------------

class SameVectorEmbedder:
    """Every title embeds identically, so cosine alone would merge everything."""

    def encode_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def encode_query(self, text):
        return [1.0, 0.0]


def test_titles_differing_in_their_object_are_not_merged():
    from app.rag import waterfall as wf

    sections = [
        _ar("R1", "1", "Install High Efficiency Motors", era="1994-1999"),
        _ar("R2", "1", "Install High Efficiency Motors", era="1994-1999"),
        _ar("R3", "1", "Install High Efficiency Air Conditioners", era="1994-1999"),
        _ar("R4", "1", "Install High Efficiency Lighting", era="2000-2006"),
        _ar("R5", "1", "Reduce Compressor Air Pressure"),
        _ar("R6", "1", "Reduce Inlet Compressor Temperature"),
        _ar("R7", "1", "Recycle Wooden Pallets"),
        _ar("R8", "1", "Recycle Cardboard"),
    ]
    groups = wf.group_ar_titles(sections, SameVectorEmbedder(), threshold=0.82, top_n=10)
    by_rep = {g.representative: g for g in groups}
    assert by_rep["Install High Efficiency Motors"].report_count == 2
    assert "Install High Efficiency Air Conditioners" in by_rep
    assert "Install High Efficiency Lighting" in by_rep
    assert "Reduce Inlet Compressor Temperature" in by_rep
    assert "Recycle Cardboard" in by_rep and "Recycle Wooden Pallets" in by_rep


def test_paraphrases_sharing_their_object_still_merge():
    from app.rag import waterfall as wf

    sections = [
        _ar("R1", "1", "Install Variable Frequency Drive on Cooling Tower Pump Motors"),
        _ar("R2", "1", "Install Variable Frequency Drive on Dust Collector Motor"),
        _ar("R3", "1", "Install Occupancy Sensors"),
        _ar("R4", "1", "Program Occupancy Sensors"),
    ]
    groups = wf.group_ar_titles(sections, SameVectorEmbedder(), threshold=0.82, top_n=10)
    assert sorted(g.report_count for g in groups) == [2, 2]


def test_label_is_the_most_common_title_in_the_group():
    from app.rag import waterfall as wf

    sections = [
        _ar("R1", "1", "Replace Exit Lighting with LEDs"),
        _ar("R2", "1", "Replace Lighting with LEDs"),
        _ar("R3", "1", "Replace Lighting with LEDs"),
    ]
    (group,) = wf.group_ar_titles(sections, SameVectorEmbedder(), threshold=0.82, top_n=10)
    assert group.representative == "Replace Lighting with LEDs"
    assert group.report_count == 3


def test_equal_counts_order_modern_first_then_alphabetical_and_ignore_store_order():
    from app.rag import waterfall as wf

    sections = [
        _ar("O1", "1", "Install Skylights", era="2000-2006"),
        _ar("M1", "1", "Clean Solar Panels", era="2022-2026"),
        _ar("M2", "1", "Compost Sludge", era="2022-2026"),
    ]
    first = wf.group_ar_titles(sections, SameVectorEmbedder(), threshold=0.82, top_n=10)
    again = wf.group_ar_titles(list(reversed(sections)), SameVectorEmbedder(), threshold=0.82, top_n=10)
    order = [g.representative for g in first]
    assert order == ["Clean Solar Panels", "Compost Sludge", "Install Skylights"]
    assert [g.representative for g in again] == order


def test_trailing_comma_is_stripped_from_titles():
    assert clean_ar_title("AR#6 Repair Compressed Air Leaks,") == "Repair Compressed Air Leaks"


def test_ranking_top_n_is_capped_at_ten_even_if_configured_higher():
    from app.config import MAX_LISTED_RESULTS

    themes = [ThemeGroup(f"Theme {i}", [f"Theme {i}"], [f"R{i}"], []) for i in range(25)]
    text = render_sic_ranking_answer(_ranking(themes=themes, total_reports=25, total_ars=25))
    import re

    rows = [line for line in text.splitlines() if re.match(r"^\s*\d+\.\s", line)]
    assert len(rows) == MAX_LISTED_RESULTS


def test_render_notes_single_report_themes():
    themes = [
        ThemeGroup("Repair Compressed Air Leaks", ["Repair Compressed Air Leaks"], ["R1", "R2"], ["R1"]),
        ThemeGroup("Compost Sludge", ["Compost Sludge"], ["R3"], ["R3"]),
    ]
    text = render_sic_ranking_answer(_ranking(themes=themes))
    assert "1 of these themes appears in a single report only" in text

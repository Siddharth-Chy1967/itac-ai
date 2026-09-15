"""Tests for the pre-assessment checklist (Feature 2), fully mocked.

No ChromaDB, no torch: a FakeStore serves canned metadata/semantic results and a
keyword-based FakeEmbedder makes clustering deterministic, so the waterfall,
merge, recency preference, AR collection, grouping, and checklist formatting are
all exercised without the real stack.
"""
from __future__ import annotations

import pytest

from app.chat import orchestrator
from app.chat.orchestrator import _classify_mode, _parse_checklist_query
from app.chat.prompts import CHECKLIST_MODE_INSTRUCTION, format_checklist_material
from app.rag import waterfall as wf
from app.rag.waterfall import (
    ARSection,
    ChecklistResult,
    MatchedReport,
    ReportRef,
    ThemeGroup,
    build_checklist,
    collect_ar_titles,
    merge_passes,
    process_similarity,
    sic_waterfall,
)
from app.skills.router import RouterDecision

from test_orchestrator import RecordingProvider, _collect, _patch


# --- fakes ------------------------------------------------------------------

class FakeStore:
    def __init__(self, sic_map=None, profile_hits=None, ar_map=None):
        self.sic_map = sic_map or {}          # (level, value) -> [ReportRef]
        self.profile_hits = profile_hits or []  # [ReportRef] with scores
        self.ar_map = ar_map or {}            # report_id -> [ARSection]
        self.profile_called_with = None
        self.ar_called_with = None

    def reports_for_sic(self, level, value):
        return list(self.sic_map.get((level, value), []))

    def profile_search(self, embedding, k):
        self.profile_called_with = k
        return list(self.profile_hits)[:k]

    def ar_sections(self, report_ids):
        self.ar_called_with = set(report_ids)
        out = []
        for rid in report_ids:
            out += self.ar_map.get(rid, [])
        return out


class FakeEmbedder:
    """Maps text to a one-hot theme vector so same-theme titles cluster exactly."""

    THEMES = ["air", "light", "solar", "motor", "waste"]

    def _vec(self, text):
        low = text.lower()
        v = [1.0 if k in low else 0.0 for k in self.THEMES]
        if not any(v):
            v[-1] = 0.5  # unknown -> a weak, shared-but-distinct bucket
        return v

    def encode_query(self, text):
        return self._vec(text)

    def encode_documents(self, texts):
        return [self._vec(t) for t in texts]


def _ar(rid, num, title, era="2022-2026"):
    return ARSection(report_id=rid, ar_number=num, title=title, era=era)


DEFAULTS = dict(min_reports=3, process_k=15, process_threshold=0.6, group_threshold=0.82, top_themes=10)


# --- Pass A: SIC waterfall widening -----------------------------------------

def test_sic_waterfall_uses_sic4_when_enough():
    store = FakeStore(sic_map={("sic4", "2599"): [ReportRef("R1", "2022-2026"),
                                                  ReportRef("R2", "2022-2026"),
                                                  ReportRef("R3", "2022-2026")]})
    reports, level = sic_waterfall(store, "2599", min_reports=3)
    assert level == "sic4" and len(reports) == 3


def test_sic_waterfall_widens_to_sic3():
    store = FakeStore(sic_map={
        ("sic4", "2599"): [ReportRef("R1", "2022-2026")],
        ("sic3", "259"): [ReportRef("R1", "2022-2026"), ReportRef("R2", "2017-2021"),
                          ReportRef("R3", "2022-2026")],
    })
    reports, level = sic_waterfall(store, "2599", min_reports=3)
    assert level == "sic3" and {r.report_id for r in reports} == {"R1", "R2", "R3"}


def test_sic_waterfall_falls_back_to_widest_sic2():
    store = FakeStore(sic_map={
        ("sic4", "2599"): [ReportRef("R1", "2022-2026")],
        ("sic3", "259"): [ReportRef("R1", "2022-2026"), ReportRef("R2", "2022-2026")],
        ("sic2", "25"): [ReportRef("R1", "2022-2026"), ReportRef("R2", "2022-2026")],
    })
    reports, level = sic_waterfall(store, "2599", min_reports=3)
    assert level == "sic2" and len(reports) == 2  # never reached 3, widest kept


def test_sic_waterfall_empty_sic():
    assert sic_waterfall(FakeStore(), "", min_reports=3) == ([], None)


# --- Pass B: process similarity threshold -----------------------------------

def test_process_similarity_filters_by_threshold_and_dedups():
    store = FakeStore(profile_hits=[
        ReportRef("R1", "2022-2026", 0.70),
        ReportRef("R2", "2022-2026", 0.50),   # below 0.6 -> dropped
        ReportRef("R1", "2022-2026", 0.82),   # same report, better score
    ])
    got = process_similarity(store, FakeEmbedder(), "wood cabinets", k=15, threshold=0.6)
    assert [r.report_id for r in got] == ["R1"]
    assert got[0].score == 0.82  # best chunk kept
    assert store.profile_called_with == 15


# --- merge + recency preference ---------------------------------------------

def test_merge_dedupes_and_tracks_source():
    merged = merge_passes(
        [ReportRef("R1", "2022-2026"), ReportRef("R2", "2022-2026")],
        [ReportRef("R2", "2022-2026"), ReportRef("R3", "2017-2021")],
    )
    by = {m.report_id: m for m in merged}
    assert by["R1"].from_industry and not by["R1"].from_process
    assert by["R2"].from_industry and by["R2"].from_process
    assert by["R3"].from_process and not by["R3"].from_industry


def test_recency_keeps_modern_when_enough():
    merged = [MatchedReport("R1", "2022-2026"), MatchedReport("R2", "2017-2021"),
              MatchedReport("R3", "2022-2026"), MatchedReport("O1", "2000-2006")]
    kept, fallback = wf.apply_recency_preference(merged, min_reports=3)
    assert {m.report_id for m in kept} == {"R1", "R2", "R3"}  # old-era dropped
    assert fallback is False


def test_recency_falls_back_to_old_era_when_modern_thin():
    merged = [MatchedReport("R1", "2022-2026"), MatchedReport("O1", "2000-2006"),
              MatchedReport("O2", "1994-1999")]
    kept, fallback = wf.apply_recency_preference(merged, min_reports=3)
    assert {m.report_id for m in kept} == {"R1", "O1", "O2"}  # padded with old-era
    assert fallback is True


# --- AR collection: exhaustive + dedup --------------------------------------

def test_collect_ar_titles_dedups_and_drops_bare_labels():
    raw = [
        _ar("R1", "1", "AR-1 Repair Compressed Air Leaks (ARC 2.4236)"),
        _ar("R1", "1", "AR-1"),                       # bare summary-table label
        _ar("R1", "1", "AR-1 Repair Compressed Air Leaks"),  # same AR, split chunk
        _ar("R1", "2", "AR-2 Install LED Lighting (ARC 2.7142)"),
    ]
    got = {(a.report_id, a.ar_number): a.title for a in collect_ar_titles(raw)}
    assert got[("R1", "1")] == "Repair Compressed Air Leaks"  # richest kept, cleaned
    assert got[("R1", "2")] == "Install LED Lighting"
    assert len(got) == 2  # bare "AR-1" dropped, not a third entry


# --- grouping ---------------------------------------------------------------

def test_group_ar_titles_clusters_counts_and_top_n():
    sections = [
        _ar("R1", "1", "Repair compressed air leaks"),
        _ar("R2", "1", "Fix air leaks"),
        _ar("R3", "1", "Eliminate compressed air leaks"),
        _ar("R4", "1", "Install LED lighting"),
        _ar("R5", "2", "Upgrade to efficient lighting"),
        _ar("R6", "1", "Install solar panels"),
    ]
    groups = wf.group_ar_titles(sections, FakeEmbedder(), threshold=0.82, top_n=2)
    assert len(groups) == 2  # top 2 of the 3 themes
    assert groups[0].report_count == 3 and "air" in groups[0].representative.lower()
    assert groups[1].report_count == 2 and "lighting" in groups[1].representative.lower()


def test_group_counts_distinct_reports_not_ars():
    # One report writing the same theme twice counts once.
    sections = [_ar("R1", "1", "fix air leaks"), _ar("R1", "2", "repair air leaks"),
                _ar("R2", "1", "air leak repair")]
    groups = wf.group_ar_titles(sections, FakeEmbedder(), threshold=0.82, top_n=10)
    assert len(groups) == 1
    assert groups[0].report_count == 2  # R1 counted once


# --- build_checklist: single vs hybrid passes -------------------------------

def _hybrid_store():
    return FakeStore(
        sic_map={("sic4", "2599"): [ReportRef("R1", "2022-2026"), ReportRef("R2", "2022-2026"),
                                    ReportRef("R3", "2022-2026")]},
        profile_hits=[ReportRef("R3", "2022-2026", 0.8), ReportRef("R4", "2017-2021", 0.75)],
        ar_map={
            "R1": [_ar("R1", "1", "Fix air leaks")],
            "R2": [_ar("R2", "1", "Repair air leaks")],
            "R3": [_ar("R3", "1", "Install LED lighting")],
            "R4": [_ar("R4", "1", "Compressed air leak repair")],
        },
    )


def test_build_checklist_sic_only_runs_pass_a_alone():
    store = _hybrid_store()
    res = build_checklist(sic="2599", description=None, store=store, embedder=FakeEmbedder(), **DEFAULTS)
    assert store.profile_called_with is None  # Pass B never ran
    assert res.n_industry == 3 and res.n_process == 0
    assert res.sic_level == "sic4" and res.sic_pattern == "2599"


def test_build_checklist_description_only_runs_pass_b_alone():
    store = _hybrid_store()
    res = build_checklist(sic=None, description="wood products", store=store, embedder=FakeEmbedder(), **DEFAULTS)
    assert res.sic_level is None
    assert res.n_industry == 0 and res.n_process == 2  # R3, R4


def test_build_checklist_hybrid_runs_both_and_reports_breakdown():
    store = _hybrid_store()
    res = build_checklist(sic="2599", description="wood", store=store, embedder=FakeEmbedder(), **DEFAULTS)
    assert store.profile_called_with == 15
    assert res.total_reports == 4          # R1,R2,R3 (SIC) + R4 (process)
    assert res.n_industry == 3 and res.n_process == 2  # R3 counted in both
    ids = {m.report_id for m in res.matched}
    assert ids == {"R1", "R2", "R3", "R4"}


def test_build_checklist_marks_thin_when_few_reports():
    store = FakeStore(sic_map={("sic4", "2599"): [ReportRef("R1", "2022-2026")],
                               ("sic3", "259"): [ReportRef("R1", "2022-2026")],
                               ("sic2", "25"): [ReportRef("R1", "2022-2026")]},
                      ar_map={"R1": [_ar("R1", "1", "Fix air leaks")]})
    res = build_checklist(sic="2599", description=None, store=store, embedder=FakeEmbedder(), **DEFAULTS)
    assert res.thin is True


def test_old_era_reports_excluded_from_ar_grouping_when_modern_suffices():
    # Enough modern reports matched, so the era preference holds and the old-era
    # report -- which is not even in the merged set -- contributes no themes.
    modern = [ReportRef("R1", "2022-2026"), ReportRef("R2", "2022-2026"),
              ReportRef("R3", "2017-2021")]
    store = FakeStore(
        sic_map={("sic4", "2599"): modern + [ReportRef("O1", "2000-2006")]},
        ar_map={"R1": [_ar("R1", "1", "Fix air leaks")],
                "R2": [_ar("R2", "1", "Repair air leaks")],
                "R3": [_ar("R3", "1", "Air leak repair")],
                "O1": [_ar("O1", "1", "Old solar recommendation", era="2000-2006")]},
    )
    res = build_checklist(sic="2599", description=None, store=store, embedder=FakeEmbedder(), **DEFAULTS)
    assert store.ar_called_with == {"R1", "R2", "R3"}
    all_titles = [t for g in res.themes for t in g.examples]
    assert not any("solar" in t.lower() for t in all_titles)


def test_old_era_reports_join_ar_grouping_when_modern_is_thin():
    # Only one modern report matched. Grouping themes from that alone is what made
    # thin-SIC checklists come back near-empty, so the recency fallback that already
    # widened `matched` must widen the theme basis too.
    store = FakeStore(
        sic_map={("sic4", "2599"): [ReportRef("R1", "2022-2026")],
                 ("sic3", "259"): [ReportRef("R1", "2022-2026"), ReportRef("O1", "2000-2006")],
                 ("sic2", "25"): [ReportRef("R1", "2022-2026"), ReportRef("O1", "2000-2006")]},
        ar_map={"R1": [_ar("R1", "1", "Fix air leaks")],
                "O1": [_ar("O1", "1", "Old solar recommendation", era="2000-2006")]},
    )
    res = build_checklist(sic="2599", description=None, store=store, embedder=FakeEmbedder(), **DEFAULTS)
    assert store.ar_called_with == {"R1", "O1"}
    all_titles = [t for g in res.themes for t in g.examples]
    assert any("solar" in t.lower() for t in all_titles)
    # The old-era theme is still marked as carrying no modern support.
    solar = next(g for g in res.themes if "solar" in g.representative.lower())
    assert solar.modern_count == 0 and solar.dated is True


# --- intent detection + query parsing ---------------------------------------

@pytest.mark.parametrize("msg", [
    "my SIC is 2599, what should I look for",
    "pre-assessment for a wood cabinet shop",
    "SIC 3559 checklist",
    "what should I look for at a metal stamping facility",
    "assigned SIC 2452, prepare for the site visit",
])
def test_classify_checklist_positive(msg):
    assert _classify_mode(msg) == "checklist"


@pytest.mark.parametrize("msg", [
    "what ARC code fits occupancy sensors",       # arc
    "which reports mention solar cleaning",        # enumerate
    "what did AS0494 recommend",                    # answer
])
def test_classify_checklist_negative(msg):
    assert _classify_mode(msg) != "checklist"


def test_parse_checklist_query_hybrid():
    sic, desc = _parse_checklist_query(
        "my SIC is 2599, wood cabinet manufacturing, OSB, particleboard, dust collection")
    assert sic == "2599"
    assert "cabinet" in desc and "dust collection" in desc
    assert "sic" not in desc.lower()


def test_parse_checklist_query_sic_only():
    sic, desc = _parse_checklist_query("my sic is 2599")
    assert sic == "2599" and desc is None


def test_parse_checklist_query_description_only():
    sic, desc = _parse_checklist_query("what should I look for at a wood cabinet shop with dust collection")
    assert sic is None
    assert "cabinet" in desc


# --- checklist formatting ---------------------------------------------------

def _result(themes, total=14, n_ind=6, n_proc=8, pattern="259x", thin=False):
    return ChecklistResult(themes=themes, total_reports=total, n_industry=n_ind,
                           n_process=n_proc, sic_level="sic3", sic_pattern=pattern, thin=thin)


def test_format_checklist_material_lists_themes_with_counts_no_scores():
    themes = [ThemeGroup("Repair compressed air leaks", ["Fix air leaks", "Repair air leaks"],
                         ["R1", "R2", "R3"]),
              ThemeGroup("Install LED lighting", ["Upgrade lighting"], ["R4", "R5"])]
    block = format_checklist_material(_result(themes))
    assert block.startswith("<reference_material>")
    assert "14 past reports" in block
    assert "6 matched by SIC 259x" in block and "8 matched by process description" in block
    assert "found in 3 reports" in block and "found in 2 reports" in block
    assert "0." not in block  # no similarity scores leak in


def test_format_checklist_material_marks_thin():
    themes = [ThemeGroup("Fix air leaks", ["Fix air leaks"], ["R1"])]
    block = format_checklist_material(_result(themes, total=1, n_ind=1, n_proc=0, thin=True))
    assert "thin" in block.lower()


def test_format_checklist_material_no_themes():
    block = format_checklist_material(_result([], total=4))
    assert "no recurring ar themes" in block.lower()


def test_checklist_instruction_is_terse_field_actions():
    t = CHECKLIST_MODE_INSTRUCTION
    assert "OUTPUT MODE: PRE-ASSESSMENT CHECKLIST" in t
    assert "field" in t.lower() and "action" in t.lower()
    assert "(found in N past reports)" in t
    assert "no scores" in t.lower()


# --- orchestrator wiring ----------------------------------------------------

def _canned_result():
    return ChecklistResult(
        themes=[ThemeGroup("Repair compressed air leaks", ["Fix air leaks"], ["R1", "R2", "R3"])],
        total_reports=4, n_industry=3, n_process=2, sic_level="sic4", sic_pattern="2599",
        thin=False, matched=[MatchedReport("R1", "2022-2026", True, False),
                             MatchedReport("R3", "2022-2026", True, True)],
    )


@pytest.mark.asyncio
async def test_run_chat_checklist_emits_sources_and_swaps_instruction(monkeypatch):
    provider = RecordingProvider(["Based on 4 past reports..."])
    _patch(monkeypatch, RouterDecision(skills=[], needs_rag=True), provider)

    async def fake_build(sic, description):
        assert sic == "2599"
        return _canned_result()

    monkeypatch.setattr(orchestrator, "_build_checklist", fake_build)

    events = await _collect(orchestrator.run_chat(
        "my SIC is 2599, wood cabinet manufacturing with dust collection", []))

    src = [e for e in events if e.event == "sources"]
    assert len(src) == 1
    assert src[0].data[0] == {"report_id": "R1", "fiscal_year": "2022-2026",
                              "section": "SIC match", "score": None}
    assert src[0].data[1]["section"] == "SIC + process match"
    assert provider.system is not None
    assert "OUTPUT MODE: PRE-ASSESSMENT CHECKLIST" in provider.system
    assert "Repair compressed air leaks" in provider.system
    assert events[-1].event == "done"


@pytest.mark.asyncio
async def test_run_chat_checklist_falls_back_when_store_down(monkeypatch):
    provider = RecordingProvider(["x"])
    _patch(monkeypatch, RouterDecision(skills=[], needs_rag=True), provider)

    async def broken(sic, description):
        return None

    monkeypatch.setattr(orchestrator, "_build_checklist", broken)

    events = await _collect(orchestrator.run_chat("pre-assessment for SIC 2599", []))
    assert not any(e.event == "sources" for e in events)
    assert "OUTPUT MODE: PRE-ASSESSMENT CHECKLIST" not in provider.system


@pytest.mark.asyncio
async def test_run_chat_non_checklist_unaffected(monkeypatch):
    provider = RecordingProvider(["x"])
    _patch(monkeypatch, RouterDecision(skills=[], needs_rag=True), provider)

    async def fail(sic, description):  # pragma: no cover - must not be called
        raise AssertionError("checklist ran for a non-checklist question")

    monkeypatch.setattr(orchestrator, "_build_checklist", fail)

    await _collect(orchestrator.run_chat("What did AS0494 recommend?", []))
    assert "OUTPUT MODE: PRE-ASSESSMENT CHECKLIST" not in provider.system

"""Tests for enumerate mode: intent detection, dedup-by-report, threshold
direction, the terse output contract, and orchestrator wiring.

Enumerate mode answers "which reports exist about X" with a deduplicated list
(one line per report) instead of a writeup. Scores stay in the sources payload
for the UI cards and never reach the chat text.
"""
from __future__ import annotations

import pytest

from app.chat import orchestrator
from app.chat.orchestrator import (
    _apply_enumerate_floor,
    _classify_mode,
    _enumerate_topic,
)
from app.chat.prompts import (
    ENUMERATE_MODE_INSTRUCTION,
    format_enumerate_material,
    render_enumerate_answer,
)
from app.rag.retrieve import Retrieval
from app.skills.router import RouterDecision

from test_orchestrator import RecordingProvider, _collect, _patch


def _hit(report_id, fy, section, score, text="chunk text"):
    return (text, {"report_id": report_id, "fiscal_year": fy, "section": section}, score)


# --- _classify_mode intent detection ---------------------------------------

@pytest.mark.parametrize(
    "message",
    [
        "Which reports mention solar panel cleaning?",
        "What reports do we have on compressed air leaks?",
        "List all reports that cover roof coating",
        "Are there any reports about lighting retrofits?",
        "How many reports mention boiler insulation?",
        "Do we have any reports on chiller replacement?",
        "which past assessments covered HVAC scheduling",
        "reports that mention solar panel cleaning",
        # Bare "reports <preposition> X" shapes: a plural-reports listing question
        # phrased without a leading which/what/list. These must route to the grounded
        # enumerate mode, not loose answer mode (regression: "Reports about double
        # paned windows" fell through to answer mode and the model fabricated a report).
        "Reports about double paned windows",
        "reports on compressed air leaks",
        "reports regarding roof coating",
        "reports concerning lighting retrofits",
        "assessments involving boiler insulation",
        "reports related to chiller replacement",
        "reports dealing with HVAC scheduling",
        # "reports for/with X": a "find me reports for double paned windows" listing
        # request. Regression: "for"/"with" were missing from the bare-preposition
        # shape, so these fell through to answer mode and got a prose single-report
        # reply instead of the enumerated list.
        "Find me reports for double paned windows",
        "Find me reports with double paned windows",
        "reports for compressed air leaks",
        "reports with roof coating",
    ],
)
def test_classify_enumerate_positive(message):
    assert _classify_mode(message) == "enumerate"


@pytest.mark.parametrize(
    "message",
    [
        "What did AS0494 recommend for solar panel cleaning?",
        "Summarize the compressed air AR in AS0552",
        "How do I format the recommendations section?",
        "What savings did the report claim for boiler insulation?",
        "Run me through the methodology for the AR",
        "Build me a utility bill breakdown table",
        "hi",
    ],
)
def test_classify_answer_negative(message):
    assert _classify_mode(message) == "answer"


# --- topic extraction: strip the enumerate boilerplate before embedding -----

@pytest.mark.parametrize(
    "message, topic",
    [
        ("Which reports mention solar panel cleaning?", "solar panel cleaning"),
        ("What reports do we have on compressed air leaks?", "compressed air leaks"),
        ("List all reports that cover roof coating", "roof coating"),
        ("Are there any reports about lighting retrofits?", "lighting retrofits"),
        ("How many reports mention boiler insulation?", "boiler insulation"),
        ("Reports about double paned windows", "double paned windows"),
        ("reports related to chiller replacement", "chiller replacement"),
        # The lead-in must not eat into a topic word that merely starts with a
        # connector: "for" in "forklifts", "contain" is fully matched then stops.
        ("Find me reports containing forklifts", "forklifts"),
        ("reports on forklifts", "forklifts"),
        # A hyphenated topic starting with a connective word survives intact.
        ("reports about on-site solar generation", "on-site solar generation"),
    ],
)
def test_enumerate_topic_strips_boilerplate(message, topic):
    assert _enumerate_topic(message) == topic


def test_enumerate_topic_none_when_no_substantive_topic():
    # Nothing but boilerplate: no topic survives, so the caller keeps the raw message.
    assert _enumerate_topic("list all reports") is None
    assert _enumerate_topic("do we have any reports") is None


# --- adaptive enumerate floor ----------------------------------------------

def test_apply_enumerate_floor_anchors_to_top_hit_on_a_genuine_match():
    # A genuine topic tops ~0.70; the floor drops to top - band so the cluster
    # survives, well below the raw 0.72 that used to bury it.
    r = Retrieval(
        query="double paned windows",
        hits=[
            _hit("AS0326", "", "Process Description", 0.695),
            _hit("AS0273", "", "Process Description", 0.626),
            _hit("AS0152", "", "Energy Conservation", 0.616),
            _hit("AS0500", "", "Off Topic", 0.55),
        ],
        min_relevance=0.60,
    )
    _apply_enumerate_floor(r, sanity_floor=0.60, band=0.10)
    assert r.min_relevance == pytest.approx(0.60)  # max(0.60, 0.695 - 0.10)
    assert r.has_match is True
    assert [e["report_id"] for e in r.enumerated()] == ["AS0326", "AS0273", "AS0152"]


def test_apply_enumerate_floor_prunes_weak_tail_on_a_strong_query():
    # A strong query (top ~0.81) lifts the floor to top - band, pruning a weak tail.
    r = Retrieval(
        query="compressed air leaks",
        hits=[
            _hit("AS0600", "FY2024", "Compressed Air", 0.807),
            _hit("AS0601", "FY2024", "Compressed Air", 0.720),
            _hit("AS0602", "FY2023", "Loosely Related", 0.650),
        ],
        min_relevance=0.60,
    )
    _apply_enumerate_floor(r, sanity_floor=0.60, band=0.10)
    assert r.min_relevance == pytest.approx(0.707)  # 0.807 - 0.10
    assert [e["report_id"] for e in r.enumerated()] == ["AS0600", "AS0601"]


def test_apply_enumerate_floor_rejects_nonsense_below_sanity_floor():
    # Off-topic query: even its best hit is below the sanity floor, so the floor stays
    # there (above every hit) and nothing matches.
    r = Retrieval(
        query="quantum teleportation of bananas",
        hits=[_hit("AS0506", "FY2021", "Occupancy Sensors", 0.531)],
        min_relevance=0.60,
    )
    _apply_enumerate_floor(r, sanity_floor=0.60, band=0.10)
    assert r.min_relevance == pytest.approx(0.60)
    assert r.has_match is False
    assert r.enumerated() == []


# --- threshold filter: corrected comparison direction ----------------------

def test_matches_keeps_scores_at_or_above_floor():
    # Similarity convention: higher = more relevant, so >= min_relevance matches.
    r = Retrieval(
        query="q",
        hits=[
            _hit("AS0494", "FY2024", "AR#3. Clean Solar Panels", 0.77),
            _hit("AS0552", "FY2025", "AR#4. Clean Solar Panels", 0.72),
            _hit("AS0557", "FY2025", "AR#3. Install Solar Photovoltaic Panels", 0.68),
        ],
        min_relevance=0.72,
    )
    assert r.has_match is True
    assert [h[1]["report_id"] for h in r.matches()] == ["AS0494", "AS0552"]


def test_no_match_when_all_below_floor():
    r = Retrieval(query="q", hits=[_hit("AS0100", "FY2018", "Background", 0.5)], min_relevance=0.72)
    assert r.has_match is False
    assert r.enumerated() == []


# --- enumerated(): dedup by report keeping best score + all sections --------

def test_enumerated_dedups_by_report_keeping_best_score_and_sections():
    r = Retrieval(
        query="solar panel cleaning",
        hits=[
            _hit("AS0494", "FY2024", "AR#3. Clean Solar Panels", 0.77),
            _hit("AS0552", "FY2025", "AR#4. Clean Solar Panels", 0.76),
            _hit("AS0552", "FY2025", "AR#4 Clean Solar Panels (ARC 2.6125)", 0.75),
            _hit("AS0494", "FY2024", "AR#3 Clean Solar Panels", 0.74),
            _hit("AS0552", "FY2025", "AR#4. Clean Solar Panels", 0.73),  # dup section
        ],
        min_relevance=0.72,
    )
    entries = r.enumerated()

    # Newest-first: AS0552 (FY2025) leads AS0494 (FY2024) despite its lower score.
    assert [e["report_id"] for e in entries] == ["AS0552", "AS0494"]
    as0552, as0494 = entries
    assert as0494["score"] == 0.77  # best chunk score kept
    assert as0494["sections"] == ["AR#3. Clean Solar Panels", "AR#3 Clean Solar Panels"]
    assert as0552["score"] == 0.76
    # All distinct matching sections, best first, exact duplicates collapsed.
    assert as0552["sections"] == [
        "AR#4. Clean Solar Panels",
        "AR#4 Clean Solar Panels (ARC 2.6125)",
    ]


def test_enumerated_orders_newest_first_with_score_tiebreak():
    # Reports are listed latest fiscal year first, regardless of relevance score.
    # Same-year reports fall back to score; older-era reports with no fiscal year
    # ("") sort to the very bottom.
    r = Retrieval(
        query="compressed air leaks",
        hits=[
            _hit("AS0100", "FY2018", "Compressed Air", 0.80),  # oldest FY, top score
            _hit("AS0600", "FY2025", "Compressed Air", 0.70),
            _hit("AS0601", "FY2025", "Compressed Air", 0.74),  # same FY as AS0600
            _hit("AS0010", "", "Compressed Air", 0.79),         # old era, no FY -> last
        ],
        min_relevance=0.60,
    )
    # FY2025 before FY2018 before the no-FY report; within FY2025, higher score wins.
    assert [e["report_id"] for e in r.enumerated()] == [
        "AS0601",
        "AS0600",
        "AS0100",
        "AS0010",
    ]


def test_enumerated_excludes_below_threshold_reports():
    r = Retrieval(
        query="q",
        hits=[
            _hit("AS0494", "FY2024", "AR#3. Clean Solar Panels", 0.77),
            _hit("AS0557", "FY2025", "AR#3. Install Solar Photovoltaic Panels", 0.68),
        ],
        min_relevance=0.72,
    )
    assert [e["report_id"] for e in r.enumerated()] == ["AS0494"]


def test_enumerated_sources_shape_carries_score_for_ui():
    r = Retrieval(
        query="q",
        hits=[
            _hit("AS0494", "FY2024", "AR#3. Clean Solar Panels", 0.7654),
            _hit("AS0494", "FY2024", "AR#3 Clean Solar Panels", 0.7421),
        ],
        min_relevance=0.72,
    )
    assert r.enumerated_sources() == [
        {
            "report_id": "AS0494",
            "fiscal_year": "FY2024",
            "section": "AR#3. Clean Solar Panels; AR#3 Clean Solar Panels",
            "score": 0.765,
        }
    ]


# --- the terse output contract ---------------------------------------------

def test_enumerate_instruction_is_terse_list_only():
    text = ENUMERATE_MODE_INSTRUCTION
    assert "OUTPUT MODE: ENUMERATE" in text
    assert "One line per matching report" in text
    # Scores/confidence and ranking commentary are explicitly banned from chat text.
    assert "Never state relevance scores" in text
    assert "closest match" in text
    assert "most relevant" in text
    assert "No introduction, no closing paragraph" in text
    # The no-match case is a single plain line.
    assert "No matching reports found in the knowledge base" in text


def test_format_enumerate_material_lists_reports_without_scores():
    r = Retrieval(
        query="q",
        hits=[
            _hit("AS0494", "FY2024", "AR#3. Clean Solar Panels", 0.7654),
            _hit("AS0552", "FY2025", "AR#4. Clean Solar Panels", 0.7638),
        ],
        min_relevance=0.72,
    )
    block = format_enumerate_material(r)

    assert block.startswith("<reference_material>")
    assert block.endswith("</reference_material>")
    assert "- AS0494 (FY2024) - AR#3. Clean Solar Panels" in block
    assert "- AS0552 (FY2025) - AR#4. Clean Solar Panels" in block
    # No score data leaks into the prompt text the model might echo.
    assert "0.76" not in block
    assert "score" not in block.lower()


def test_format_enumerate_material_empty():
    r = Retrieval(query="q", hits=[], min_relevance=0.72)
    block = format_enumerate_material(r)
    assert "No report in the knowledge base matched" in block
    assert "do not invent report IDs" in block


# --- deterministic enumerate rendering (bypasses the LLM) -------------------

def test_render_enumerate_answer_lists_reports_newest_first_without_scores():
    r = Retrieval(
        query="compressed air leaks",
        hits=[
            _hit("AS0494", "FY2024", "AR#3. Clean Solar Panels", 0.77),
            _hit("AS0552", "FY2025", "AR#4. Clean Solar Panels", 0.76),
            _hit("AS0010", "", "Repair Compressed Air Leaks", 0.75),  # old era, no FY
        ],
        min_relevance=0.60,
    )
    text = render_enumerate_answer(r)

    assert text.splitlines()[0] == "3 reports in the knowledge base match, newest first:"
    lines = [ln for ln in text.splitlines() if ln.startswith("- ")]
    assert lines == [
        "- AS0552 (FY2025): AR#4. Clean Solar Panels",
        "- AS0494 (FY2024): AR#3. Clean Solar Panels",
        "- AS0010 (older era): Repair Compressed Air Leaks",
    ]
    # No scores or confidence leak into the chat text.
    assert "0.7" not in text
    assert "score" not in text.lower()


def test_render_enumerate_answer_no_match_is_one_plain_line():
    r = Retrieval(
        query="q",
        hits=[_hit("AS0100", "FY2018", "Background", 0.5)],
        min_relevance=0.72,
    )
    assert (
        render_enumerate_answer(r)
        == "No matching reports found in the knowledge base."
    )


# --- orchestrator wiring ----------------------------------------------------

def _enumerate_retrieval() -> Retrieval:
    return Retrieval(
        query="solar panel cleaning",
        hits=[
            _hit("AS0494", "FY2024", "AR#3. Clean Solar Panels", 0.7654),
            _hit("AS0552", "FY2025", "AR#4. Clean Solar Panels", 0.7638),
            _hit("AS0494", "FY2024", "AR#3 Clean Solar Panels", 0.7421),
            _hit("AS0557", "FY2025", "AR#3. Install Solar Photovoltaic Panels", 0.6829),
        ],
        min_relevance=0.72,
    )


@pytest.mark.asyncio
async def test_run_chat_enumerate_dedups_sources_and_swaps_instruction(monkeypatch):
    provider = RecordingProvider(["AS0494"])
    _patch(monkeypatch, RouterDecision(skills=[], needs_rag=True), provider)

    calls = {"enumerate": 0, "answer": 0}

    async def fake_enumerate(message):
        calls["enumerate"] += 1
        return _enumerate_retrieval()

    async def fake_answer(message):
        calls["answer"] += 1
        return None

    monkeypatch.setattr(orchestrator, "_retrieve_enumerate", fake_enumerate)
    monkeypatch.setattr(orchestrator, "_retrieve_reference", fake_answer)

    events = await _collect(
        orchestrator.run_chat("Which reports mention solar panel cleaning?", [])
    )

    # The enumerate seam ran instead of the answer-mode one.
    assert calls == {"enumerate": 1, "answer": 0}

    # Sources are deduplicated by report (AS0494 once, not per chunk) with the
    # score kept for the UI cards; the sub-threshold install AR is excluded.
    source_events = [e for e in events if e.event == "sources"]
    assert len(source_events) == 1
    # Newest-first: AS0552 (FY2025) precedes AS0494 (FY2024) even though AS0494
    # scores marginally higher; recency, not score, drives the list order.
    assert source_events[0].data == [
        {
            "report_id": "AS0552",
            "fiscal_year": "FY2025",
            "section": "AR#4. Clean Solar Panels",
            "score": 0.764,
        },
        {
            "report_id": "AS0494",
            "fiscal_year": "FY2024",
            "section": "AR#3. Clean Solar Panels; AR#3 Clean Solar Panels",
            "score": 0.765,
        },
    ]

    # The list is rendered deterministically and the LLM is bypassed entirely, so a
    # weak model cannot drop real reports or invent fake ones: the provider is never
    # streamed (its system prompt stays unset).
    assert provider.system is None

    answer = "".join(e.data["t"] for e in events if e.event == "token")
    # Newest-first list, one line per report; the sub-threshold install AR is absent.
    assert "- AS0552 (FY2025)" in answer
    assert "- AS0494 (FY2024)" in answer
    assert answer.index("AS0552") < answer.index("AS0494")
    assert "AS0557" not in answer
    assert events[-1].event == "done"


@pytest.mark.asyncio
async def test_run_chat_answer_mode_untouched_by_enumerate(monkeypatch):
    # A non-enumerate question never touches the enumerate seam or instruction.
    provider = RecordingProvider(["x"])
    _patch(monkeypatch, RouterDecision(skills=[], needs_rag=True), provider)

    async def fail_enumerate(message):  # pragma: no cover - must not be called
        raise AssertionError("enumerate retrieval ran for an answer-mode question")

    monkeypatch.setattr(orchestrator, "_retrieve_enumerate", fail_enumerate)

    await _collect(orchestrator.run_chat("What did AS0494 recommend?", []))

    assert provider.system is not None
    assert "OUTPUT MODE: ENUMERATE" not in provider.system


@pytest.mark.asyncio
async def test_run_chat_enumerate_falls_back_to_answer_when_store_down(monkeypatch):
    # Retrieval unavailable: the model must not be told to say "no matching
    # reports" about a search that never ran, so no enumerate instruction.
    provider = RecordingProvider(["x"])
    _patch(monkeypatch, RouterDecision(skills=[], needs_rag=True), provider)

    async def broken(message):
        return None

    monkeypatch.setattr(orchestrator, "_retrieve_enumerate", broken)

    events = await _collect(
        orchestrator.run_chat("Which reports mention solar panel cleaning?", [])
    )

    assert not any(e.event == "sources" for e in events)
    assert provider.system is not None
    assert "OUTPUT MODE: ENUMERATE" not in provider.system

"""Tests for enumerate mode: intent detection, dedup-by-report, threshold
direction, the terse output contract, and orchestrator wiring.

Enumerate mode answers "which reports exist about X" with a deduplicated list
(one line per report) instead of a writeup. Scores stay in the sources payload
for the UI cards and never reach the chat text.
"""
from __future__ import annotations

import pytest

from app.chat import orchestrator
from app.chat.orchestrator import _classify_mode
from app.chat.prompts import ENUMERATE_MODE_INSTRUCTION, format_enumerate_material
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

    assert [e["report_id"] for e in entries] == ["AS0494", "AS0552"]
    as0494, as0552 = entries
    assert as0494["score"] == 0.77  # best chunk score kept
    assert as0494["sections"] == ["AR#3. Clean Solar Panels", "AR#3 Clean Solar Panels"]
    assert as0552["score"] == 0.76
    # All distinct matching sections, best first, exact duplicates collapsed.
    assert as0552["sections"] == [
        "AR#4. Clean Solar Panels",
        "AR#4 Clean Solar Panels (ARC 2.6125)",
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
    assert source_events[0].data == [
        {
            "report_id": "AS0494",
            "fiscal_year": "FY2024",
            "section": "AR#3. Clean Solar Panels; AR#3 Clean Solar Panels",
            "score": 0.765,
        },
        {
            "report_id": "AS0552",
            "fiscal_year": "FY2025",
            "section": "AR#4. Clean Solar Panels",
            "score": 0.764,
        },
    ]

    # The terse list contract and the deduplicated report list reached the model.
    assert provider.system is not None
    assert "OUTPUT MODE: ENUMERATE" in provider.system
    assert "- AS0494 (FY2024)" in provider.system
    assert "- AS0552 (FY2025)" in provider.system
    assert "AS0557" not in provider.system
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

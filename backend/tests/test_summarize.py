"""Tests for summarize mode: a specific report/AR is fetched by identifier and
summarized, instead of a semantic top-k that lands on the wrong report's AR.

Covers intent + target resolution, the store's deterministic fetch, the output
contract, and orchestrator wiring, including the reliability guards that keep it
from ever summarizing the wrong (or no) report and from disturbing the existing
enumerate/answer/arc search paths.
"""
from __future__ import annotations

import pytest

from app.chat import orchestrator
from app.chat.orchestrator import _narrow_ar_from_context, _summarize_target
from app.chat.prompts import SUMMARIZE_MODE_INSTRUCTION
from app.models import Message
from app.skills.router import RouterDecision

from test_orchestrator import RecordingProvider, _collect, _patch


def _msg(role, content):
    return Message(role=role, content=content)


# --- target resolution ------------------------------------------------------

def test_target_from_explicit_report_id_and_ar():
    assert _summarize_target("summarize AR-5 of AS0612", []) == ("AS0612", "5")


def test_target_lowercase_id_and_ar_hash_form():
    assert _summarize_target("give me the vitals for as0612 ar#5", []) == ("AS0612", "5")


def test_target_report_only_when_no_ar_named():
    assert _summarize_target("summarize AS0612", []) == ("AS0612", None)


def test_target_resolves_report_from_history_on_followup():
    # The classic workflow: search surfaced one report, then "summarize the vitals".
    history = [
        _msg("user", "what does AS0612 say about conveyor motors"),
        _msg("assistant", "AS0612 (FY2024) recommends replacing conveyor motors with MDRs."),
    ]
    assert _summarize_target("summarize the vitals", history) == ("AS0612", None)


def test_no_target_without_summary_phrasing():
    # A report id alone is not a summarize request; it stays a normal answer.
    assert _summarize_target("what did AS0612 recommend", []) is None


def test_no_target_when_history_report_is_ambiguous():
    # An enumerate list names several reports; "summarize the vitals" cannot pick one,
    # so summarize defers rather than guess.
    history = [
        _msg("user", "which reports mention conveyor motors"),
        _msg("assistant", "- AS0593 (FY2024)\n- AS0600 (FY2024)\n- AS0612 (FY2024)"),
    ]
    assert _summarize_target("summarize the vitals", history) is None


def test_no_target_when_no_report_anywhere():
    assert _summarize_target("summarize the compressed air AR", []) is None


def test_message_id_wins_over_ambiguous_history():
    # Even with a multi-report list in history, naming the id in the message resolves it.
    history = [_msg("assistant", "- AS0593 (FY2024)\n- AS0600 (FY2024)")]
    assert _summarize_target("summarize AS0612's vitals", history) == ("AS0612", None)


def test_arc_code_is_not_read_as_an_ar_number():
    # "ARC 2.4111" must not resolve ar_number=2; only a real "AR-5" form does.
    assert _summarize_target("summarize AS0612 (ARC 2.4111)", []) == ("AS0612", None)


# --- narrowing a vague follow-up to one AR from context ---------------------

def _ar_rows(*titles):
    # titles: (ar_number, section) pairs
    return [("body", {"report_id": "AS0440", "ar_number": ar, "section": sec}) for ar, sec in titles]


def test_narrow_picks_the_ar_the_conversation_named():
    rows = _ar_rows(
        ("1", "AR #1. Install High Efficiency Lighting"),
        ("5", "AR #5. Turn off Grinders"),
    )
    history = [_msg("assistant", "AS0440 recommends turning off grinders when idle.")]
    assert _narrow_ar_from_context(rows, "summarize the vitals", history) == "5"


def test_narrow_returns_none_when_no_ar_topic_in_context():
    # Nothing in the conversation points at one AR: stay whole-report (per-AR breakdown).
    rows = _ar_rows(
        ("1", "AR #1. Install High Efficiency Lighting"),
        ("5", "AR #5. Turn off Grinders"),
    )
    assert _narrow_ar_from_context(rows, "summarize the vitals", []) is None


def test_narrow_returns_none_when_context_is_ambiguous_between_ars():
    rows = _ar_rows(
        ("1", "AR #1. Install High Efficiency Lighting"),
        ("5", "AR #5. Turn off Grinders"),
    )
    history = [_msg("user", "tell me about the lighting and the grinders")]
    assert _narrow_ar_from_context(rows, "summarize the vitals", history) is None


def test_narrow_single_ar_report_is_that_ar():
    rows = _ar_rows(("3", "AR #3. Repair Compressed Air Leaks"))
    assert _narrow_ar_from_context(rows, "summarize the vitals", []) == "3"


# --- store fetch (deterministic, ordered) -----------------------------------

class FakeCollection:
    def __init__(self, rows):
        # rows: list of (document, metadata)
        self._rows = rows

    def get(self, where, include):
        def _match(meta):
            if "$and" in where:
                return all(_match_clause(meta, c) for c in where["$and"])
            return _match_clause(meta, where)

        def _match_clause(meta, clause):
            (k, v), = clause.items()
            return meta.get(k) == v

        docs, metas = [], []
        for doc, meta in self._rows:
            if _match(meta):
                docs.append(doc)
                metas.append(meta)
        return {"documents": docs, "metadatas": metas}


def _fake_store(rows):
    from app.rag.store import ReportStore

    store = ReportStore.__new__(ReportStore)  # skip __init__ (no chromadb)
    store.collection = FakeCollection(rows)
    return store


def test_chunks_for_returns_only_that_report_in_order():
    rows = [
        ("AR-5 body", {"report_id": "AS0612", "ar_number": "5", "chunk_index": 11}),
        ("header", {"report_id": "AS0612", "ar_number": "", "chunk_index": 2}),
        ("other report", {"report_id": "AS0600", "ar_number": "4", "chunk_index": 3}),
    ]
    got = _fake_store(rows).chunks_for("AS0612")
    assert [m["chunk_index"] for _t, m in got] == [2, 11]  # sorted, AS0600 excluded


def test_chunks_for_narrows_to_one_ar():
    rows = [
        ("AR-5 body", {"report_id": "AS0612", "ar_number": "5", "chunk_index": 11}),
        ("AR-6 body", {"report_id": "AS0612", "ar_number": "6", "chunk_index": 13}),
    ]
    got = _fake_store(rows).chunks_for("AS0612", "5")
    assert [t for t, _m in got] == ["AR-5 body"]


def test_chunks_for_empty_when_report_absent():
    assert _fake_store([]).chunks_for("AS9999") == []


def test_prioritize_keeps_each_ars_richest_chunk_despite_cap():
    from app.chat.orchestrator import _SUMMARIZE_MAX_CHUNKS, _prioritize_report_chunks

    # An AR whose detailed chunk sits at a high index, behind a wall of other chunks:
    # it must survive the cap so its vitals are not reported as "not stated".
    rows = [
        ("filler " * 3, {"ar_number": str(i), "chunk_index": i, "section": f"AR-{i}"})
        for i in range(1, _SUMMARIZE_MAX_CHUNKS + 5)
    ]
    rows.append(
        ("RICH grinders body with all the savings numbers " * 20,
         {"ar_number": "99", "chunk_index": 999, "section": "AR-99 Turn off Grinders"})
    )
    kept = _prioritize_report_chunks(rows)
    assert len(kept) <= _SUMMARIZE_MAX_CHUNKS
    assert any(m.get("ar_number") == "99" for _t, m in kept)  # the high-index rich AR survived


# --- output contract --------------------------------------------------------

def test_summarize_instruction_is_grounded_and_handles_thin_reports():
    text = SUMMARIZE_MODE_INSTRUCTION
    normalized = " ".join(text.split())  # collapse line-wrapping
    assert "OUTPUT MODE: AR / REPORT SUMMARY" in text
    assert "VERBATIM" in text
    assert "not stated in this report" in normalized
    # The thin-report case must be handled honestly, never by borrowing/inventing.
    assert "limited detail" in normalized
    assert "any other report" in normalized


# --- orchestrator wiring ----------------------------------------------------

def _rows():
    return [
        (
            "AR-5 Replace Conveyor Motors with MDRs (2.4111)\nEstimated Cost Savings = $1,245/yr",
            {"report_id": "AS0612", "fiscal_year": "FY2024", "section": "AR-5 Replace Conveyor Motors with MDRs", "ar_number": "5", "chunk_index": 11},
        ),
    ]


@pytest.mark.asyncio
async def test_run_chat_summarize_fetches_by_id_and_summarizes(monkeypatch):
    provider = RecordingProvider(["summary"])
    _patch(monkeypatch, RouterDecision(skills=[], needs_rag=True), provider)

    captured = {}

    async def fake_fetch(report_id, ar_number, message, history):
        captured["target"] = (report_id, ar_number)
        return _rows()

    monkeypatch.setattr(orchestrator, "_fetch_report_chunks", fake_fetch)

    events = await _collect(orchestrator.run_chat("summarize AR-5 of AS0612", []))

    assert captured["target"] == ("AS0612", "5")
    # The full AR text and the summary contract reached the model, grounded on AS0612.
    assert provider.system is not None
    assert "OUTPUT MODE: AR / REPORT SUMMARY" in provider.system
    assert "AS0612" in provider.system
    assert "$1,245/yr" in provider.system
    # Sources name the fetched section; the model still streams the summary.
    source_events = [e for e in events if e.event == "sources"]
    assert source_events and source_events[0].data[0]["report_id"] == "AS0612"
    assert events[-1].event == "done"


@pytest.mark.asyncio
async def test_run_chat_summarize_defers_when_no_report_resolves(monkeypatch):
    # "summarize the vitals" with nothing in context: summarize must NOT engage, and
    # the normal answer path runs instead (reliability: never summarize a guess).
    provider = RecordingProvider(["x"])
    _patch(monkeypatch, RouterDecision(skills=[], needs_rag=True), provider)

    async def fail_fetch(report_id, ar_number, message, history):  # pragma: no cover - must not run
        raise AssertionError("summarize fetch ran without a resolvable report")

    monkeypatch.setattr(orchestrator, "_fetch_report_chunks", fail_fetch)

    await _collect(orchestrator.run_chat("summarize the vitals", []))

    assert provider.system is not None
    assert "OUTPUT MODE: AR / REPORT SUMMARY" not in provider.system


@pytest.mark.asyncio
async def test_run_chat_summarize_falls_back_when_report_has_no_chunks(monkeypatch):
    # The id resolved but the store returned nothing (unknown/typo id): fall through
    # to the normal answer path rather than assert a summary about an empty fetch.
    provider = RecordingProvider(["x"])
    _patch(monkeypatch, RouterDecision(skills=[], needs_rag=True), provider)

    async def empty_fetch(report_id, ar_number, message, history):
        return []

    monkeypatch.setattr(orchestrator, "_fetch_report_chunks", empty_fetch)

    events = await _collect(orchestrator.run_chat("summarize AS9999", []))

    assert not any(e.event == "sources" for e in events)
    assert provider.system is not None
    assert "OUTPUT MODE: AR / REPORT SUMMARY" not in provider.system


@pytest.mark.asyncio
async def test_run_chat_plain_report_question_does_not_trigger_summarize(monkeypatch):
    # A report id without summary phrasing stays a normal answer (guards against
    # hijacking the existing search path).
    provider = RecordingProvider(["x"])
    _patch(monkeypatch, RouterDecision(skills=[], needs_rag=True), provider)

    async def fail_fetch(report_id, ar_number, message, history):  # pragma: no cover - must not run
        raise AssertionError("summarize fetch ran for a plain answer question")

    monkeypatch.setattr(orchestrator, "_fetch_report_chunks", fail_fetch)

    async def no_ref(message):
        return None

    monkeypatch.setattr(orchestrator, "_retrieve_reference", no_ref)

    await _collect(orchestrator.run_chat("what did AS0612 recommend?", []))

    assert provider.system is not None
    assert "OUTPUT MODE: AR / REPORT SUMMARY" not in provider.system

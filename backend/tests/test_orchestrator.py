"""Tests for orchestrator skill wiring (spec §5).

Verifies that the router's chosen skills are (a) emitted in the early `skills`
event and (b) have their bodies injected into the system prompt handed to the
provider. Uses monkeypatched router/registry/provider so no real LLM is called.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest

from app.chat import orchestrator
from app.chat.prompts import BASE_SYSTEM_PROMPT, assemble_system_prompt
from app.models import Message
from app.skills.loader import Skill
from app.skills.router import RouterDecision

FAKE_REGISTRY = {
    "pnm-bill-breakdown": Skill(
        name="pnm-bill-breakdown",
        description="PNM breakdown skill.",
        body="PNM_BODY_MARKER: build the Table 2-3 breakdown.",
        path="pnm.md",
    ),
    "no-em-dash": Skill(
        name="no-em-dash",
        description="No em dashes.",
        body="NO_EM_DASH_BODY",
        path="nod.md",
    ),
}


class RecordingProvider:
    """Captures the system prompt passed to stream() and yields canned tokens."""

    def __init__(self, tokens: list[str]):
        self._tokens = tokens
        self.system: str | None = None

    async def stream(self, messages, system, max_tokens) -> AsyncIterator[str]:
        self.system = system
        for t in self._tokens:
            yield t

    async def complete(self, messages, system, max_tokens, json_mode=False) -> str:
        return "{}"


async def _collect(agen) -> list[orchestrator.Event]:
    return [ev async for ev in agen]


def _patch(monkeypatch, decision: RouterDecision, provider: RecordingProvider):
    monkeypatch.setattr(orchestrator, "get_registry", lambda: FAKE_REGISTRY)
    monkeypatch.setattr(orchestrator, "get_provider", lambda: provider)
    monkeypatch.setattr(orchestrator, "get_router_provider", lambda: provider)

    async def fake_route(message, history, registry, prov):
        return decision

    monkeypatch.setattr(orchestrator, "route_skills", fake_route)

    # Default: no retrieval, so tests not about RAG are unaffected and never touch
    # torch/chromadb. RAG tests override this seam with their own Retrieval.
    async def _no_retrieval(message):
        return None

    monkeypatch.setattr(orchestrator, "_retrieve_reference", _no_retrieval)


# --- assemble_system_prompt unit tests ------------------------------------

def test_assemble_no_skills_is_base_only():
    assert assemble_system_prompt([]) == BASE_SYSTEM_PROMPT


def test_assemble_includes_skill_body():
    prompt = assemble_system_prompt([FAKE_REGISTRY["pnm-bill-breakdown"]])
    assert BASE_SYSTEM_PROMPT in prompt
    assert "PNM_BODY_MARKER" in prompt
    assert '<skill name="pnm-bill-breakdown">' in prompt


def test_assemble_reference_material_appended():
    prompt = assemble_system_prompt([], reference_material="<reference_material>X</reference_material>")
    assert prompt.endswith("<reference_material>X</reference_material>")


# --- orchestrator wiring tests --------------------------------------------

@pytest.mark.asyncio
async def test_run_chat_emits_skills_event_then_tokens(monkeypatch):
    provider = RecordingProvider(["Hello", " world"])
    _patch(monkeypatch, RouterDecision(skills=["pnm-bill-breakdown"]), provider)

    events = await _collect(orchestrator.run_chat("build the breakdown", []))

    assert events[0].event == "skills"
    # Skill injection is disabled: the event always carries an empty list, so no
    # badges render, even when the router still names a skill.
    assert events[0].data == {"skills": []}
    tokens = [e.data["t"] for e in events if e.event == "token"]
    assert tokens == ["Hello", " world"]
    assert events[-1].event == "done"


@pytest.mark.asyncio
async def test_run_chat_never_injects_skill_body_into_system(monkeypatch):
    # Skill injection is disabled: routing was misfiring on general/RAG questions,
    # so no skill body reaches the prompt even when the router selects one.
    provider = RecordingProvider(["x"])
    _patch(monkeypatch, RouterDecision(skills=["pnm-bill-breakdown"]), provider)

    await _collect(orchestrator.run_chat("build the breakdown", []))

    assert provider.system == BASE_SYSTEM_PROMPT
    assert "PNM_BODY_MARKER" not in provider.system
    assert "NO_EM_DASH_BODY" not in provider.system


@pytest.mark.asyncio
async def test_run_chat_no_skills_uses_base_prompt_only(monkeypatch):
    provider = RecordingProvider(["x"])
    _patch(monkeypatch, RouterDecision(skills=[]), provider)

    await _collect(orchestrator.run_chat("hi", []))

    assert provider.system == BASE_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_run_chat_passes_history_plus_user_message(monkeypatch):
    captured = {}

    class Prov(RecordingProvider):
        async def stream(self, messages, system, max_tokens):
            captured["messages"] = messages
            self.system = system
            for t in self._tokens:
                yield t

    provider = Prov(["x"])
    _patch(monkeypatch, RouterDecision(skills=[]), provider)

    history = [Message(role="user", content="earlier"), Message(role="assistant", content="reply")]
    await _collect(orchestrator.run_chat("now", history))

    msgs = captured["messages"]
    assert [m.content for m in msgs] == ["earlier", "reply", "now"]
    assert msgs[-1].role == "user"


# --- RAG retrieval wiring tests (spec §5 step 2, §6) ----------------------

def _match_retrieval() -> "orchestrator.Retrieval":
    from app.rag.retrieve import Retrieval

    return Retrieval(
        query="compressed air leaks",
        hits=[(
            "AR #1: Repair compressed air leaks\nAnticipated savings: 45,000 kWh/yr.",
            {"report_id": "AS0599", "fiscal_year": "FY2025", "section": "AR #1: Repair leaks"},
            0.87,
        )],
        min_relevance=0.6,
    )


def _no_match_retrieval() -> "orchestrator.Retrieval":
    from app.rag.retrieve import Retrieval

    return Retrieval(
        query="how do I format a table",
        hits=[(
            "Some loosely related paragraph.",
            {"report_id": "AS0123", "fiscal_year": "FY2019", "section": "2. Background"},
            0.41,
        )],
        min_relevance=0.6,
    )


@pytest.mark.asyncio
async def test_run_chat_rag_match_emits_sources_and_injects_citation(monkeypatch):
    provider = RecordingProvider(["ok"])
    _patch(monkeypatch, RouterDecision(skills=[], needs_rag=True), provider)

    async def fake_retrieve(message):
        return _match_retrieval()

    monkeypatch.setattr(orchestrator, "_retrieve_reference", fake_retrieve)

    events = await _collect(orchestrator.run_chat("compressed air leaks", []))

    source_events = [e for e in events if e.event == "sources"]
    assert len(source_events) == 1
    assert source_events[0].data == [
        {"report_id": "AS0599", "fiscal_year": "FY2025", "section": "AR #1: Repair leaks", "score": 0.87}
    ]
    # The retrieved chunk and a citation instruction reached the model.
    assert provider.system is not None
    assert "AS0599" in provider.system
    assert "FY2025" in provider.system
    assert "cite" in provider.system.lower()
    assert "45,000 kWh/yr" in provider.system


@pytest.mark.asyncio
async def test_run_chat_rag_no_match_instructs_plainly_without_sources(monkeypatch):
    provider = RecordingProvider(["ok"])
    _patch(monkeypatch, RouterDecision(skills=[], needs_rag=True), provider)

    async def fake_retrieve(message):
        return _no_match_retrieval()

    monkeypatch.setattr(orchestrator, "_retrieve_reference", fake_retrieve)

    events = await _collect(orchestrator.run_chat("how do I format a table", []))

    # No confident match -> no sources event.
    assert not any(e.event == "sources" for e in events)
    # The prompt tells the model to say so plainly and not present it as precedent.
    assert provider.system is not None
    assert "confident match" in provider.system.lower()
    assert "closest" in provider.system.lower()
    assert "AS0123" in provider.system  # offered as the closest, not a citation
    assert "precedent" in provider.system.lower()


@pytest.mark.asyncio
async def test_run_chat_retrieves_even_when_needs_rag_false(monkeypatch):
    # Regression: the router returned needs_rag=False on report questions the corpus
    # clearly matched (e.g. the roof coating AR), silently suppressing retrieval.
    # Retrieval now runs regardless, so a confident match always reaches the user.
    provider = RecordingProvider(["x"])
    _patch(monkeypatch, RouterDecision(skills=[], needs_rag=False), provider)

    calls = {"n": 0}

    async def spy(message):
        calls["n"] += 1
        return _match_retrieval()

    monkeypatch.setattr(orchestrator, "_retrieve_reference", spy)

    events = await _collect(orchestrator.run_chat("what ARs were written for roof coating", []))

    assert calls["n"] == 1  # retrieval ran despite needs_rag=False
    source_events = [e for e in events if e.event == "sources"]
    assert len(source_events) == 1
    assert source_events[0].data[0]["report_id"] == "AS0599"
    assert provider.system is not None
    assert "AS0599" in provider.system  # the match was injected to cite


@pytest.mark.asyncio
async def test_run_chat_retrieves_with_router_rewritten_query(monkeypatch):
    # A follow-up naming no topic ("the AR") must be embedded as the router's
    # self-contained rewrite, not raw, or it matches generic AR boilerplate.
    provider = RecordingProvider(["x"])
    _patch(
        monkeypatch,
        RouterDecision(skills=[], needs_rag=True, retrieval_query="cool roof coating AR methodology"),
        provider,
    )

    embedded = {}

    async def spy(query):
        embedded["q"] = query
        return _match_retrieval()

    monkeypatch.setattr(orchestrator, "_retrieve_reference", spy)

    captured = {}

    class Prov(RecordingProvider):
        async def stream(self, messages, system, max_tokens):
            captured["messages"] = messages
            self.system = system
            for t in self._tokens:
                yield t

    provider = Prov(["x"])
    monkeypatch.setattr(orchestrator, "get_provider", lambda: provider)

    history = [Message(role="user", content="roof coating ARs?"), Message(role="assistant", content="AS0609.")]
    await _collect(orchestrator.run_chat("Run me through the methodology for the AR", history))

    assert embedded["q"] == "cool roof coating AR methodology"
    # The model still answers the raw message, not the rewrite.
    assert captured["messages"][-1].content == "Run me through the methodology for the AR"


@pytest.mark.asyncio
async def test_run_chat_retrieves_with_raw_message_when_no_rewrite(monkeypatch):
    # First turn: the router returns no rewrite, so the raw message is embedded.
    provider = RecordingProvider(["x"])
    _patch(monkeypatch, RouterDecision(skills=[], needs_rag=True, retrieval_query=None), provider)

    embedded = {}

    async def spy(query):
        embedded["q"] = query
        return _match_retrieval()

    monkeypatch.setattr(orchestrator, "_retrieve_reference", spy)

    await _collect(orchestrator.run_chat("What ARs were written for roof top coating?", []))

    assert embedded["q"] == "What ARs were written for roof top coating?"


@pytest.mark.asyncio
async def test_run_chat_no_match_disclaimer_suppressed_when_needs_rag_false(monkeypatch):
    # A miss on a formatting/general turn: retrieval still ran, but the "nothing on
    # file" disclaimer would be noise, so nothing is injected.
    provider = RecordingProvider(["x"])
    _patch(monkeypatch, RouterDecision(skills=[], needs_rag=False), provider)

    async def fake_retrieve(message):
        return _no_match_retrieval()

    monkeypatch.setattr(orchestrator, "_retrieve_reference", fake_retrieve)

    events = await _collect(orchestrator.run_chat("what font do report tables use", []))

    assert not any(e.event == "sources" for e in events)
    assert provider.system == BASE_SYSTEM_PROMPT  # no reference material injected


# --- table-mode wiring tests (spec §5.3, §7) ------------------------------

_MINIMAL_SPEC = (
    '{"sheet_name": "Meter 1", "title": "T",'
    ' "groups": [{"label": "Date", "vmerge": true,'
    ' "cols": [{"key": "date", "header": "D", "type": "date"}]}],'
    ' "rows": [{"date": "Jun-25"}], "totals": {"date": "TOTALS"}}'
)


@pytest.mark.asyncio
async def test_run_chat_table_mode_renders_and_emits_file_event(monkeypatch, tmp_path):
    tokens = ["Here is your table.\n\n```tablespec\n", _MINIMAL_SPEC, "\n```\n"]
    provider = RecordingProvider(tokens)
    _patch(monkeypatch, RouterDecision(skills=[], wants_table=True), provider)
    monkeypatch.setattr(orchestrator, "get_settings", lambda: SimpleNamespace(export_path=str(tmp_path)))

    events = await _collect(orchestrator.run_chat("build the analysis table", []))

    file_events = [e for e in events if e.event == "file"]
    assert len(file_events) == 1
    assert file_events[0].data["filename"] == "Meter_1.xlsx"
    assert file_events[0].data["url"].startswith("/api/export/download/")
    assert events[-1].event == "done"
    # the workbook was actually written to the export dir
    assert list(tmp_path.glob("*.xlsx"))


@pytest.mark.asyncio
async def test_run_chat_table_mode_invalid_spec_emits_error(monkeypatch, tmp_path):
    bad_spec = '{"sheet_name": "S", "title": "T", "groups": [], "rows": [], "totals": {}}'
    tokens = ["prose\n\n```tablespec\n", bad_spec, "\n```"]
    provider = RecordingProvider(tokens)
    _patch(monkeypatch, RouterDecision(skills=[], wants_table=True), provider)
    monkeypatch.setattr(orchestrator, "get_settings", lambda: SimpleNamespace(export_path=str(tmp_path)))

    events = await _collect(orchestrator.run_chat("build a broken table", []))

    assert events[-1].event == "error"
    assert "validation" in events[-1].data["message"].lower()
    # nothing was rendered, and no dangling `done` after the terminal error
    assert not list(tmp_path.glob("*.xlsx"))
    assert not any(e.event == "done" for e in events)


@pytest.mark.asyncio
async def test_run_chat_table_mode_no_block_falls_back_to_prose(monkeypatch, tmp_path):
    # wants_table was set but the model answered in prose; no file, just done.
    provider = RecordingProvider(["Sorry, I need the bill data first."])
    _patch(monkeypatch, RouterDecision(skills=[], wants_table=True), provider)
    monkeypatch.setattr(orchestrator, "get_settings", lambda: SimpleNamespace(export_path=str(tmp_path)))

    events = await _collect(orchestrator.run_chat("make a table", []))

    assert not any(e.event == "file" for e in events)
    assert events[-1].event == "done"


@pytest.mark.asyncio
async def test_run_chat_table_instruction_added_to_system_prompt(monkeypatch):
    provider = RecordingProvider(["x"])
    _patch(monkeypatch, RouterDecision(skills=[], wants_table=True), provider)

    await _collect(orchestrator.run_chat("make a table", []))

    assert provider.system is not None
    assert "OUTPUT MODE: TABLE" in provider.system
    assert "```tablespec" in provider.system


# --- table-mode self-correction retry (spec §9) ---------------------------

# Invalid: passes JSON parsing but fails schema validation (no groups).
_BAD_SPEC = '{"sheet_name": "S", "title": "T", "groups": [], "rows": [], "totals": {}}'


class RetryProvider(RecordingProvider):
    """Streams a (bad) response, then returns a canned reply from the retry `complete`."""

    def __init__(self, tokens: list[str], complete_reply: str):
        super().__init__(tokens)
        self._complete_reply = complete_reply
        self.complete_calls = 0
        self.complete_messages = None

    async def complete(self, messages, system, max_tokens, json_mode=False) -> str:
        self.complete_calls += 1
        self.complete_messages = messages
        return self._complete_reply


@pytest.mark.asyncio
async def test_run_chat_table_retry_fixes_invalid_spec(monkeypatch, tmp_path):
    # First attempt (streamed) is invalid; the retry returns a valid spec.
    good_reply = "```tablespec\n" + _MINIMAL_SPEC + "\n```"
    provider = RetryProvider(["```tablespec\n", _BAD_SPEC, "\n```"], good_reply)
    _patch(monkeypatch, RouterDecision(skills=[], wants_table=True), provider)
    monkeypatch.setattr(orchestrator, "get_settings", lambda: SimpleNamespace(export_path=str(tmp_path)))

    events = await _collect(orchestrator.run_chat("build the analysis table", []))

    assert provider.complete_calls == 1  # exactly one self-correction attempt
    file_events = [e for e in events if e.event == "file"]
    assert len(file_events) == 1
    assert file_events[0].data["filename"] == "Meter_1.xlsx"
    assert events[-1].event == "done"
    assert list(tmp_path.glob("*.xlsx"))
    # the retry prompt fed the validation error back to the model
    assert "at least one group" in provider.complete_messages[-1].content


@pytest.mark.asyncio
async def test_run_chat_table_retry_both_attempts_fail(monkeypatch, tmp_path):
    # Both the streamed response and the retry are invalid: terminal error.
    bad_reply = "```tablespec\n" + _BAD_SPEC + "\n```"
    provider = RetryProvider(["```tablespec\n", _BAD_SPEC, "\n```"], bad_reply)
    _patch(monkeypatch, RouterDecision(skills=[], wants_table=True), provider)
    monkeypatch.setattr(orchestrator, "get_settings", lambda: SimpleNamespace(export_path=str(tmp_path)))

    events = await _collect(orchestrator.run_chat("build a broken table", []))

    assert provider.complete_calls == 1  # retried once, then gave up
    assert events[-1].event == "error"
    assert "validation" in events[-1].data["message"].lower()
    assert not list(tmp_path.glob("*.xlsx"))
    assert not any(e.event == "done" for e in events)


@pytest.mark.asyncio
async def test_run_chat_surfaces_llm_error(monkeypatch):
    from app.llm.base import LLMError

    class FailingProvider(RecordingProvider):
        async def stream(self, messages, system, max_tokens):
            raise LLMError("boom", retryable=False)
            yield  # pragma: no cover - make this an async generator

    provider = FailingProvider([])
    _patch(monkeypatch, RouterDecision(skills=[]), provider)

    events = await _collect(orchestrator.run_chat("hi", []))
    assert events[0].event == "skills"
    assert events[-1].event == "error"
    assert "boom" in events[-1].data["message"]

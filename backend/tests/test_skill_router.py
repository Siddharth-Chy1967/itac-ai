"""Tests for the skills loader and LLM router (spec §5.1, §8).

Uses a mock LLMProvider so no real model is called. Skill fixtures are
written to a temp directory so tests do not depend on the (currently empty)
real skill files.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from app.llm.base import LLMError, LLMProvider
from app.models import Message
from app.skills.loader import Skill, load_skills
from app.skills.router import RouterDecision, parse_decision, route_skills

# --- fixtures -------------------------------------------------------------

SKILL_FILES = {
    "utility-analysis-formatting.md": (
        "---\n"
        "name: utility-analysis-formatting\n"
        "description: Use when formatting a utility bill analysis or rate breakdown.\n"
        "---\n"
        "# Utility analysis\nBody text here.\n"
    ),
    "pnm-bill-breakdown.md": (
        "---\n"
        "name: pnm-bill-breakdown\n"
        "description: >\n"
        "  Use when breaking down a PNM electric bill into demand and energy charges.\n"
        "---\n"
        "PNM body.\n"
    ),
    "no-em-dash.md": (
        "---\n"
        "name: no-em-dash\n"
        "description: Never use em dashes. Baked into base prompt, not routed.\n"
        "---\n"
        "No em dash body.\n"
    ),
    "empty-skill.md": "",  # should be skipped, not crash
    "no-frontmatter.md": "Just some text, no frontmatter.\n",  # skipped
}


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    for name, content in SKILL_FILES.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    return tmp_path


@pytest.fixture
def registry(skills_dir: Path) -> dict[str, Skill]:
    return {s.name: s for s in load_skills(skills_dir)}


class MockProvider(LLMProvider):
    """Returns a canned string for `complete`. Records the call count."""

    def __init__(self, responses: list[str]):
        self._responses = responses
        self.calls = 0

    async def complete(self, messages, system, max_tokens, json_mode=False) -> str:  # noqa: D401
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        resp = self._responses[idx]
        if isinstance(resp, Exception):
            raise resp
        return resp

    async def stream(self, messages, system, max_tokens) -> AsyncIterator[str]:
        yield ""


# --- loader tests ---------------------------------------------------------

def test_loader_parses_valid_skills(registry: dict[str, Skill]):
    assert set(registry) == {
        "utility-analysis-formatting",
        "pnm-bill-breakdown",
        "no-em-dash",
    }


def test_loader_folded_description(registry: dict[str, Skill]):
    # `description: >` folds newlines into spaces.
    desc = registry["pnm-bill-breakdown"].description
    assert "\n" not in desc
    assert "PNM electric bill" in desc


def test_loader_skips_empty_and_frontmatterless(skills_dir: Path):
    names = {s.name for s in load_skills(skills_dir)}
    assert "empty-skill" not in names
    assert "no-frontmatter" not in names


def test_loader_rejects_duplicate_names(tmp_path: Path):
    dup = (
        "---\nname: dup\ndescription: first\n---\nbody\n"
    )
    (tmp_path / "a.md").write_text(dup, encoding="utf-8")
    (tmp_path / "b.md").write_text(dup, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate skill name"):
        load_skills(tmp_path)


def test_loader_missing_directory(tmp_path: Path):
    assert load_skills(tmp_path / "does-not-exist") == []


# --- parse_decision tests -------------------------------------------------

def test_parse_filters_unknown_and_never_routed(registry):
    raw = '{"skills": ["utility-analysis-formatting", "no-em-dash", "bogus"], ' \
          '"needs_rag": false, "wants_table": true}'
    decision = parse_decision(raw, registry)
    assert decision.skills == ["utility-analysis-formatting"]  # no-em-dash + bogus dropped
    assert decision.needs_rag is False
    assert decision.wants_table is True


def test_parse_strips_markdown_fences(registry):
    raw = '```json\n{"skills": ["pnm-bill-breakdown"], "needs_rag": true, "wants_table": false}\n```'
    decision = parse_decision(raw, registry)
    assert decision.skills == ["pnm-bill-breakdown"]


def test_parse_caps_at_three(tmp_path: Path):
    for i in range(5):
        (tmp_path / f"s{i}.md").write_text(
            f"---\nname: s{i}\ndescription: skill {i}\n---\nbody\n", encoding="utf-8"
        )
    reg = {s.name: s for s in load_skills(tmp_path)}
    raw = '{"skills": ["s0","s1","s2","s3","s4"]}'
    decision = parse_decision(raw, reg)
    assert decision.skills == ["s0", "s1", "s2"]


def test_parse_invalid_json_raises(registry):
    with pytest.raises(Exception):
        parse_decision("not json at all", registry)


# --- retrieval_query parsing ----------------------------------------------

def test_parse_extracts_retrieval_query(registry):
    raw = '{"skills": [], "needs_rag": true, "retrieval_query": "cool roof coating AR methodology"}'
    assert parse_decision(raw, registry).retrieval_query == "cool roof coating AR methodology"


def test_parse_retrieval_query_absent_is_none(registry):
    # First-turn shape: the field is never requested, so callers keep the raw message.
    raw = '{"skills": [], "needs_rag": true, "wants_table": false}'
    assert parse_decision(raw, registry).retrieval_query is None


@pytest.mark.parametrize("value", ['""', '"   "', "null", "123", '["a"]'])
def test_parse_unusable_retrieval_query_is_none(registry, value):
    raw = f'{{"skills": [], "retrieval_query": {value}}}'
    assert parse_decision(raw, registry).retrieval_query is None


def test_parse_retrieval_query_clamped(registry):
    from app.skills.router import MAX_QUERY_CHARS

    raw = '{"skills": [], "retrieval_query": "%s"}' % ("roof " * 200)
    assert len(parse_decision(raw, registry).retrieval_query) == MAX_QUERY_CHARS


# --- router prompt shape --------------------------------------------------

def test_prompt_asks_for_retrieval_query_only_with_history(registry):
    from app.skills.router import build_router_prompt

    first = build_router_prompt("What ARs cover roof coating?", [], registry)
    assert "retrieval_query" not in first

    history = [
        Message(role="user", content="What ARs were written for roof top coating?"),
        Message(role="assistant", content="AS0609 and AS0607 cover it."),
    ]
    followup = build_router_prompt("Run me through the methodology for the AR", history, registry)
    assert "retrieval_query" in followup
    # The history it must resolve "the AR" against is in the prompt.
    assert "roof top coating" in followup


# --- route_skills tests ---------------------------------------------------

@pytest.mark.asyncio
async def test_route_happy_path(registry):
    provider = MockProvider(
        ['{"skills": ["utility-analysis-formatting"], "needs_rag": true, "wants_table": false}']
    )
    decision = await route_skills("format this utility bill", [], registry, provider)
    assert decision.skills == ["utility-analysis-formatting"]
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_route_retries_once_then_succeeds(registry):
    provider = MockProvider(["garbage", '{"skills": ["pnm-bill-breakdown"]}'])
    decision = await route_skills("break down my PNM bill", [], registry, provider)
    assert decision.skills == ["pnm-bill-breakdown"]
    assert provider.calls == 2


@pytest.mark.asyncio
async def test_route_falls_back_after_two_failures(registry):
    provider = MockProvider(["garbage", "still garbage"])
    decision = await route_skills("hello", [], registry, provider)
    assert decision == RouterDecision(skills=[], needs_rag=True, wants_table=False)
    assert provider.calls == 2


@pytest.mark.asyncio
async def test_route_falls_back_on_llm_error(registry):
    provider = MockProvider([LLMError("boom", retryable=False)])
    decision = await route_skills("hello", [], registry, provider)
    assert decision.skills == []
    assert decision.needs_rag is True


@pytest.mark.asyncio
async def test_route_empty_registry_skips_llm():
    provider = MockProvider(["should not be called"])
    decision = await route_skills("anything", [], {}, provider)
    assert decision.skills == []
    assert provider.calls == 0


@pytest.mark.asyncio
async def test_route_history_included_in_prompt(registry):
    # The router should incorporate recent history; we assert it does not crash
    # and returns a valid decision when history is present.
    provider = MockProvider(['{"skills": [], "needs_rag": true, "wants_table": false}'])
    history = [
        Message(role="user", content="earlier question"),
        Message(role="assistant", content="earlier answer"),
    ]
    decision = await route_skills("follow up", history, registry, provider)
    assert isinstance(decision, RouterDecision)
    assert provider.calls == 1

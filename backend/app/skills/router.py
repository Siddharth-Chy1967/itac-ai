"""LLM-based skill router (spec §5.1, §5.2).

A single cheap LLM call decides which skill instruction files apply to a user
request, plus two flags: `needs_rag` (whether the report corpus is relevant at all)
and `wants_table` (switch output to TableSpec JSON). On follow-up turns it also
returns `retrieval_query`, a self-contained rewrite of the request for embedding
search, since a reference like "the AR" retrieves nothing on its own. All of it
comes from the one call, which already receives history.

Contract (spec §5.1):
- `no-em-dash` is never routed; it is baked into the base system prompt.
- At most 3 skills; empty list if none apply.
- On JSON parse failure: retry once, then fall back to
  `{"skills": [], "needs_rag": True, "wants_table": False}`.
- Every routing decision is logged so misfires are debuggable.
"""
from __future__ import annotations

import json
import logging

from pydantic import BaseModel, Field

from ..llm.base import LLMError, LLMProvider
from ..models import Message
from .loader import Skill

logger = logging.getLogger(__name__)

# Baked into the base system prompt; never offered to the router (spec §5.1).
NEVER_ROUTED = frozenset({"no-em-dash"})

MAX_SKILLS = 3
# Router replies are tiny JSON, but leave generous headroom so a provider that
# ignores thinkingBudget=0 (or a future reasoning model) cannot truncate output.
ROUTER_MAX_TOKENS = 512
# How many prior conversation turns to give the router for context (spec §5 step 1).
HISTORY_TURNS = 2
# A rewritten retrieval query is a keyword phrase, not prose; anything longer is the
# model ignoring the instruction, so clamp rather than embed a paragraph.
MAX_QUERY_CHARS = 300

FALLBACK = {"skills": [], "needs_rag": True, "wants_table": False}


class RouterDecision(BaseModel):
    """Parsed router output (spec §5.1)."""

    skills: list[str] = Field(default_factory=list)
    needs_rag: bool = True
    wants_table: bool = False
    # Self-contained rewrite of the user's message for embedding-based retrieval,
    # with pronouns and vague references ("the AR") resolved against recent history.
    # None on the first turn (the raw message is already self-contained) and whenever
    # the router omits it; callers fall back to the raw message.
    retrieval_query: str | None = None


def _routable(registry: dict[str, Skill]) -> list[Skill]:
    return [s for name, s in registry.items() if name not in NEVER_ROUTED]


def build_router_prompt(message: str, history: list[Message], registry: dict[str, Skill]) -> str:
    """Assemble the router prompt (spec §5.1 shape).

    On a follow-up turn the router is additionally asked for `retrieval_query`: a
    self-contained rewrite of the request for embedding search. It is only asked for
    when there is history to resolve references against.
    """
    recent = history[-HISTORY_TURNS * 2 :]

    decides = (
        "which skill instruction files apply, whether the past-report knowledge base "
        "should be searched (needs_rag), and whether the answer should be rendered as "
        "a table (wants_table)"
    )
    if recent:
        decides += ", plus a self-contained search query for the knowledge base "\
                   "(retrieval_query)"

    lines = [
        "You are a routing classifier for the ITAC assistant. For each user request "
        f"you decide these independent things: {decides}.",
        "",
        "needs_rag: true whenever answering could draw on past ITAC/IAC assessment "
        "reports. That includes any question about what was recommended, written, "
        "measured, or done for a facility, client, or Assessment Recommendation "
        "(AR), and any request for precedent or prior examples. Set it false ONLY "
        "for pure formatting or how-to questions about conventions and templates, "
        "and for general knowledge that involves no report content. When unsure, "
        "true. Choosing a skill does NOT make needs_rag false: a request can need "
        "both a formatting skill and the report corpus, and a request that matches "
        "no skill can still need the corpus. Decide needs_rag on its own merits.",
        "",
        "wants_table: true only when the user wants a table, workbook, spreadsheet, "
        "or xlsx built or rendered. False when prose answers the request.",
        "",
        "Skills available:",
    ]
    for skill in _routable(registry):
        # Collapse any newlines in the description to keep the listing compact.
        desc = " ".join(skill.description.split())
        lines.append(f"- {skill.name}: {desc}")

    if recent:
        lines.append("")
        lines.append("Recent conversation:")
        for turn in recent:
            lines.append(f"{turn.role}: {turn.content}")

    lines.append("")
    lines.append(f"User request: {message}")

    if recent:
        lines.append(
            'Respond ONLY with JSON: {"skills": [...], "needs_rag": true|false, '
            '"wants_table": true|false, "retrieval_query": "..."}'
        )
        lines.append(
            "retrieval_query: the knowledge base is searched by embedding similarity, "
            "so a follow-up that names no topic (\"run me through the methodology for "
            "the AR\") matches nothing useful on its own. Rewrite the user request "
            "into a short search query that names the actual subject from the recent "
            "conversation above, resolving every pronoun and vague reference (\"the "
            "AR\", \"it\", \"that report\") into the thing it refers to. Keywords, not "
            "a sentence: under 20 words. Example: after a discussion of roof coating, "
            '"run me through the methodology for the AR" becomes "cool roof coating '
            'AR methodology". If the request already names its own topic, repeat it '
            "unchanged."
        )
    else:
        lines.append(
            'Respond ONLY with JSON: '
            '{"skills": [...], "needs_rag": true|false, "wants_table": true|false}'
        )

    lines.append(f"Select at most {MAX_SKILLS} skills. Empty list if none apply.")
    return "\n".join(lines)


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` markdown fences if present."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()


def parse_decision(raw: str, registry: dict[str, Skill]) -> RouterDecision:
    """Parse raw model output into a validated `RouterDecision`.

    Raises `ValueError`/`json.JSONDecodeError` on unparseable output so the
    caller can retry.
    """
    data = json.loads(_strip_fences(raw))
    if not isinstance(data, dict):
        raise ValueError("Router output is not a JSON object")

    valid = {name for name in registry if name not in NEVER_ROUTED}
    requested = data.get("skills") or []
    if not isinstance(requested, list):
        requested = []

    # Keep only known, routable skills; preserve order; drop dups; cap at MAX_SKILLS.
    chosen: list[str] = []
    for name in requested:
        if name in valid and name not in chosen:
            chosen.append(name)
        if len(chosen) >= MAX_SKILLS:
            break

    # Only a non-empty string is usable; anything else (missing, null, a number, the
    # model echoing back "...") leaves it None so the caller keeps the raw message.
    raw_query = data.get("retrieval_query")
    retrieval_query = None
    if isinstance(raw_query, str) and raw_query.strip():
        retrieval_query = raw_query.strip()[:MAX_QUERY_CHARS]

    return RouterDecision(
        skills=chosen,
        needs_rag=bool(data.get("needs_rag", True)),
        wants_table=bool(data.get("wants_table", False)),
        retrieval_query=retrieval_query,
    )


async def route_skills(
    message: str,
    history: list[Message],
    registry: dict[str, Skill],
    provider: LLMProvider,
) -> RouterDecision:
    """Run skill routing for one request (spec §5.1).

    A single LLM call in JSON mode. On parse failure, retry once; on a second
    failure or an LLM error, fall back to the safe default.
    """
    if not _routable(registry):
        # Nothing to route to; skip the LLM call entirely.
        decision = RouterDecision(**FALLBACK)
        logger.info("Skill routing (empty registry) for %r -> %s", message[:80], decision.skills)
        return decision

    prompt = build_router_prompt(message, history, registry)
    router_messages = [Message(role="user", content=prompt)]

    async def _call() -> str:
        return await provider.complete(
            router_messages,
            system="You are a routing classifier. Output only JSON.",
            max_tokens=ROUTER_MAX_TOKENS,
            json_mode=True,
        )

    for attempt in (1, 2):
        try:
            raw = await _call()
            decision = parse_decision(raw, registry)
            logger.info(
                "Skill routing for %r -> skills=%s needs_rag=%s wants_table=%s "
                "retrieval_query=%r",
                message[:80],
                decision.skills,
                decision.needs_rag,
                decision.wants_table,
                decision.retrieval_query,
            )
            return decision
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Router JSON parse failed (attempt %d): %s", attempt, e)
        except LLMError as e:
            logger.warning("Router LLM error (attempt %d): %s", attempt, e)

    decision = RouterDecision(**FALLBACK)
    logger.warning("Skill routing fell back to default for %r", message[:80])
    return decision

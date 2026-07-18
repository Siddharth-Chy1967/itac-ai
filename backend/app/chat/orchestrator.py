"""Chat orchestrator (spec §5).

Runs the per-request pipeline and yields SSE-shaped events for the /api/chat
endpoint to serialize. This is the skill-wiring slice of the pipeline:

    1. SKILL ROUTING      -> route_skills(); skill injection is disabled, so the
                             decision is kept only for its needs_rag/wants_table flags
    2. RAG RETRIEVAL      -> retrieve on every turn; emit `sources`, feed the
                             reference-material slot (implemented)
    3. PROMPT ASSEMBLY    -> base prompt + reference material
    4. MAIN COMPLETION    -> streaming
    5. POST-PROCESS       -> TableSpec render (xlsx engine)

Each yielded `Event` maps directly onto the spec §5.4 SSE protocol
(`skills`, `sources`, `token`, `file`, `done`, `error`).
"""
from __future__ import annotations

import asyncio
import logging
import re
import secrets
from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pydantic import ValidationError

from ..config import get_settings
from ..llm.base import LLMError
from ..llm.factory import get_provider, get_router_provider
from ..models import Message
from ..rag.retrieve import Retrieval, retrieve
from ..skills.loader import Skill, get_registry
from ..skills.router import route_skills
from ..xlsx.renderer import render_utility_table
from ..xlsx.schema import TableSpec, extract_tablespec_block
from .prompts import (
    assemble_system_prompt,
    format_arc_material,
    format_enumerate_material,
    format_reference_material,
)

logger = logging.getLogger(__name__)

MAX_TOKENS = 2048


# --- retrieval-mode intent detection ----------------------------------------
# Three special modes sit in front of the default "answer" mode:
#   arc       -> "which ARC code fits this recommendation" (arc_reference lookup)
#   enumerate -> "which reports exist about X" (deduped report list)
#   answer    -> everything else (the best match answers the question)
# A cheap deterministic heuristic decides. ARC is checked first: it keys on the
# literal "ARC" code system or an explicit "code this AR" request, so it cannot be
# confused with enumerate's plural-"reports" shapes.
_ARC_INTENT = re.compile(
    r"""
    \b(?:
        arc\s+code                              # "arc code", "arc code for"
      | (?:which|what|what's|whats)\s+arc\b      # "which arc", "what arc"
      | code\s+for\s+(?:this|my|the)\s+(?:ar|recommendation)\b
      | code\s+this\s+ar\b                       # "code this AR"
      | my\s+ar\s+(?:name|description|title)\s+is\b
      | arc\s+for\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
_ENUMERATE_INTENT = re.compile(
    r"""
    \b(?:
        (?:which|what|how\s+many)\s+(?:\w+\s+){0,3}?(?:reports|assessments)\b
      | list\s+(?:\w+\s+){0,3}?(?:reports|assessments)\b
      | (?:all|any)\s+(?:reports?|assessments?)\s+
            (?:that|which|with|about|on|for|mention\w*|cover\w*|involv\w*|
               includ\w*|discuss\w*|address\w*|relat\w*)\b
      | (?:reports|assessments)\s+(?:that\s+|which\s+)?
            (?:mention\w*|cover\w*|involv\w*|includ\w*|contain\w*|discuss\w*|
               address\w*|deal\w*|exist\w*)\b
      | (?:do|does)\s+we\s+have\s+(?:\w+\s+){0,2}?(?:reports?|assessments?)\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _classify_mode(message: str) -> str:
    """`"arc"` / `"enumerate"` / `"answer"` by intent (ARC takes precedence)."""
    if _ARC_INTENT.search(message):
        return "arc"
    if _ENUMERATE_INTENT.search(message):
        return "enumerate"
    return "answer"


# --- RAG retrieval seam -----------------------------------------------------
# The embedder loads a model and the store opens ChromaDB, so build each once and
# reuse it across requests. Retrieval is funneled through `_retrieve_reference` so
# tests can patch that one seam without pulling in torch/chromadb.


@lru_cache(maxsize=1)
def _get_embedder():
    from ..rag.embeddings import Embedder

    return Embedder()


@lru_cache(maxsize=1)
def _get_store():
    from ..rag.store import ReportStore

    return ReportStore()


@lru_cache(maxsize=1)
def _get_arc_store():
    from ..rag.store import ArcLookupStore

    return ArcLookupStore()


async def _retrieve(message: str, k: int, min_relevance: float) -> Retrieval | None:
    """Retrieve for `message`, or None if retrieval is unavailable or errors.

    The CPU-bound embed + query runs off the event loop. Any failure (e.g. the
    store not built yet) is logged and swallowed so chat still answers, just
    without reference material.
    """

    def _run() -> Retrieval:
        return retrieve(
            message,
            embedder=_get_embedder(),
            store=_get_store(),
            k=k,
            min_relevance=min_relevance,
        )

    try:
        return await asyncio.to_thread(_run)
    except Exception:  # noqa: BLE001 - retrieval must never break the chat stream
        logger.exception("RAG retrieval failed; continuing without reference material")
        return None


async def _retrieve_reference(message: str) -> Retrieval | None:
    """Answer-mode retrieval: few chunks, the standard relevance floor."""
    settings = get_settings()
    return await _retrieve(message, settings.max_rag_chunks, settings.rag_min_relevance)


async def _retrieve_enumerate(message: str) -> Retrieval | None:
    """Enumerate-mode retrieval: a wide net with its own calibrated floor.

    k=20 so every report on a topic surfaces (dedup collapses them afterwards),
    filtered at `rag_enumerate_min_relevance` rather than the answer-mode floor.
    """
    settings = get_settings()
    return await _retrieve(
        message, settings.rag_enumerate_k, settings.rag_enumerate_min_relevance
    )


# Leading action verbs an AR description tends to open with. Used only to break
# near-ties in ARC ranking (see `_apply_verb_alignment`): a description that opens
# with the same action as the user's request is a better intent match than one that
# merely shares a surface noun.
_ACTION_VERBS = frozenset(
    {
        "replace", "install", "reduce", "eliminate", "repair", "upgrade", "insulate",
        "optimize", "recover", "convert", "add", "adjust", "reuse", "recycle",
        "implement", "utilize", "use", "improve", "increase", "lower", "minimize",
        "remove", "switch", "electrify", "apply", "maintain",
    }
)
# A leading ARC code like "3.6193 " on a book entry's text; stripped before reading
# the verb so "3.6193 INSTALL ..." reads as "install".
_LEADING_CODE = re.compile(r"^\s*\d\.\d{3,4}\s+")


def _leading_verb(text: str) -> str | None:
    """The first word of `text` (past any ARC code prefix) if it is an action verb."""
    stripped = _LEADING_CODE.sub("", text or "").lstrip()
    first = re.split(r"[^A-Za-z]", stripped, maxsplit=1)[0].lower()
    return first if first in _ACTION_VERBS else None


def _apply_verb_alignment(message: str, retrieval: Retrieval, boost: float) -> None:
    """Re-rank ARC hits in place, boosting candidates whose action verb matches.

    Only reorders; the stored similarity scores are left untouched so the weakness
    floor and displayed numbers stay honest. A no-op unless the message opens with
    an action verb, so ordinary noun-phrase queries ("occupancy sensors") are
    unaffected.
    """
    query_verb = _leading_verb(message)
    if not query_verb or not retrieval.hits:
        return

    def _key(hit):
        text, meta, score = hit
        candidate_verb = _leading_verb(meta.get("ar_title") or text)
        return score + (boost if candidate_verb == query_verb else 0.0)

    retrieval.hits = sorted(retrieval.hits, key=_key, reverse=True)


async def _retrieve_arc(message: str) -> Retrieval | None:
    """ARC-lookup retrieval against the composite ARC store (book + precedents).

    Reuses `retrieve()` (same embed + query + threshold path as report search),
    just pointed at the ARC store with its own k and weakness floor, then applies a
    small verb-alignment tie-break. Returns None on any failure so ARC mode can
    fall back to a normal answer.
    """
    settings = get_settings()

    def _run() -> Retrieval:
        r = retrieve(
            message,
            embedder=_get_embedder(),
            store=_get_arc_store(),
            k=settings.rag_arc_k,
            min_relevance=settings.rag_arc_min_relevance,
        )
        _apply_verb_alignment(message, r, settings.rag_arc_verb_boost)
        return r

    try:
        return await asyncio.to_thread(_run)
    except Exception:  # noqa: BLE001 - retrieval must never break the chat stream
        logger.exception("ARC retrieval failed; continuing without ARC lookup")
        return None


def _arc_sources(retrieval: Retrieval, top_n: int = 3) -> list[dict]:
    """`sources` payload for ARC lookup: one entry per candidate code.

    Mapped onto the shared source-card shape (report_id/fiscal_year/section/score)
    so the existing UI renders it: the code stands in for report_id, the chapter
    for fiscal_year context, the official description for section. Scores stay here
    (for the card) but never reach the chat text.
    """
    out = []
    for _text, meta, score in retrieval.hits[:top_n]:
        out.append(
            {
                "report_id": meta.get("code"),
                "fiscal_year": meta.get("chapter", ""),
                "section": meta.get("description", ""),
                "score": round(score, 3),
            }
        )
    return out


# Sanitize a sheet name into a safe download filename stem.
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _export_filename(spec: TableSpec) -> str:
    """Human-friendly download name shown to the user (the on-disk name is the token)."""
    stem = _UNSAFE_NAME.sub("_", spec.sheet_name).strip("_") or "table"
    return f"{stem}.xlsx"


def _validate_spec(block: str) -> tuple[TableSpec | None, str | None]:
    """Parse+validate a tablespec JSON string. Returns (spec, None) or (None, error)."""
    try:
        return TableSpec.model_validate_json(block), None
    except ValidationError as e:
        return None, str(e)


async def _retry_tablespec(
    provider,
    messages: list[Message],
    system: str,
    prior_response: str,
    error: str,
) -> str | None:
    """Re-prompt the model once to fix its own invalid tablespec JSON (spec §9).

    Feeds back the previous response and the validation error so the model can
    self-correct. Returns the corrected tablespec JSON string, or None if the
    reply contained nothing usable. A provider `LLMError` propagates to the caller.
    """
    correction = Message(
        role="user",
        content=(
            "The tablespec JSON in your previous response failed schema validation "
            "and could not be rendered:\n\n"
            f"{error}\n\n"
            "Return a corrected response with exactly one ```tablespec fenced block "
            "whose JSON satisfies the schema. Do not change any values that were "
            "already correct; only fix what the error describes."
        ),
    )
    convo = [*messages, Message(role="assistant", content=prior_response), correction]
    reply = await provider.complete(convo, system=system, max_tokens=MAX_TOKENS)
    return extract_tablespec_block(reply) or reply.strip() or None


async def _render_table(
    text: str,
    provider,
    messages: list[Message],
    system: str,
) -> Event | None:
    """Extract a TableSpec from a table-mode response, render it, return a `file` Event.

    Returns None when the response contains no ```tablespec block (the model gave a
    plain prose answer, so there is nothing to render). On a validation failure the
    model gets one self-correction retry (spec §9); only if that also fails do we
    return an `error` Event, so a broken workbook is never handed to the user
    (spec §7.1). The rendered file is saved under `export_path` keyed by an opaque
    token that `GET /api/export/download/{token}` serves (spec §7.3).
    """
    block = extract_tablespec_block(text)
    if block is None:
        return None

    spec, error = _validate_spec(block)
    if spec is None:
        logger.info("TableSpec invalid; re-prompting model to self-correct: %s", error)
        retry_block = await _retry_tablespec(provider, messages, system, text, error or "")
        spec = None
        if retry_block is not None:
            spec, error = _validate_spec(retry_block)
        if spec is None:
            logger.warning("TableSpec still invalid after retry: %s", error)
            return Event(
                "error",
                {"message": "The generated table failed validation and was not rendered."},
            )

    export_dir = Path(get_settings().export_path)
    export_dir.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(16)
    render_utility_table(spec, str(export_dir / f"{token}.xlsx"))
    logger.info("Rendered table %r -> token %s", spec.sheet_name, token)

    return Event(
        "file",
        {"filename": _export_filename(spec), "url": f"/api/export/download/{token}"},
    )


@dataclass
class Event:
    """One SSE event: `event: <event>\\ndata: <json data>` (spec §5.4)."""

    event: str
    data: object


async def run_chat(
    message: str,
    history: list[Message],
    mode: str = "chat",
) -> AsyncIterator[Event]:
    """Execute the chat pipeline for one request, yielding events in order.

    The `skills` event is emitted early (spec §5.4) so the frontend can render
    skill badges before tokens stream in.
    """
    registry = get_registry()

    # 1. SKILL ROUTING (cheap LLM call; has its own retry + safe fallback).
    decision = await route_skills(message, history, registry, get_router_provider())
    yield Event("skills", {"skills": []})

    selected: list[Skill] = []  # skill injection disabled

    # 2. RAG RETRIEVAL. Runs on every turn rather than only when the router sets
    #    needs_rag: the flag misfired False on report questions the corpus clearly
    #    matched, and the relevance threshold is a deterministic judge of whether
    #    precedent exists. A confident match emits `sources` and instructs the model
    #    to cite report_id/fiscal_year (spec §5.3, §6).
    #
    #    needs_rag now only decides whether a *miss* is worth reporting: on a
    #    genuine report question, "nothing on file" is the answer and the model is
    #    told to say so plainly; on a formatting or general-knowledge turn that
    #    disclaimer is noise, so it is left out. A stray False therefore costs at
    #    most a disclaimer, never a real citation.
    #    Retrieval embeds the router's `retrieval_query` when it supplied one: a
    #    follow-up like "run me through the methodology for the AR" names no topic,
    #    so embedding it raw matches generic AR boilerplate instead of the report
    #    under discussion. The rewrite resolves it against history. The raw message
    #    is what the model still answers.
    #    Enumerate mode ("which reports mention X") is detected on the raw message:
    #    it retrieves wide (k=20, its own floor), dedups hits by report, and swaps
    #    in the terse list-only output contract. ARC mode ("which ARC code fits
    #    this recommendation") queries the separate arc_reference collection and
    #    emits a terse code list. Answer mode is untouched.
    reference_material: str | None = None
    enumerate_mode = False
    arc_mode = False
    mode = _classify_mode(message)
    if mode == "arc":
        retrieval = await _retrieve_arc(decision.retrieval_query or message)
        # Store unavailable / empty: leave arc_mode False so this falls through to
        # a normal answer rather than asserting "no ARC code" about a lookup that
        # never ran.
        if retrieval is not None and retrieval.hits:
            arc_mode = True
            yield Event("sources", _arc_sources(retrieval))
            reference_material = format_arc_material(retrieval)
    elif mode == "enumerate":
        retrieval = await _retrieve_enumerate(decision.retrieval_query or message)
        # Store unavailable: leave enumerate_mode False (fall back to answer mode)
        # rather than have the model claim "no matching reports" about a search
        # that never ran.
        if retrieval is not None:
            enumerate_mode = True
            if retrieval.has_match:
                yield Event("sources", retrieval.enumerated_sources())
            reference_material = format_enumerate_material(retrieval)
    else:
        retrieval = await _retrieve_reference(decision.retrieval_query or message)
        if retrieval is not None:
            if retrieval.has_match:
                yield Event("sources", retrieval.sources())
                reference_material = format_reference_material(retrieval)
            elif decision.needs_rag:
                reference_material = format_reference_material(retrieval)

    # 3. PROMPT ASSEMBLY.
    system = assemble_system_prompt(
        selected,
        reference_material=reference_material,
        wants_table=decision.wants_table,
        enumerate_mode=enumerate_mode,
        arc_mode=arc_mode,
    )

    # 4. MAIN COMPLETION (streaming). Buffer tokens so table mode can reparse the
    #    full response after streaming; prose mode ignores the buffer.
    messages = [*history, Message(role="user", content=message)]
    provider = get_provider()
    chunks: list[str] = []
    try:
        async for token in provider.stream(messages, system=system, max_tokens=MAX_TOKENS):
            yield Event("token", {"t": token})
            chunks.append(token)

        # 5. POST-PROCESS. In table mode the model emits a ```tablespec block
        #    (spec §5.3); validate it and render the .xlsx, emitting a `file`
        #    event before `done`. A validation failure is surfaced as an `error`
        #    instead of handing back a malformed workbook.
        if decision.wants_table and not enumerate_mode and not arc_mode:
            table_event = await _render_table("".join(chunks), provider, messages, system)
            if table_event is not None:
                yield table_event
                if table_event.event == "error":
                    return  # table failed; the error is terminal, no `done`
        yield Event("done", {})
    except LLMError as e:
        yield Event("error", {"message": str(e)})
    except Exception as e:  # noqa: BLE001 - surface anything else as an SSE error
        logger.exception("Unexpected orchestrator error")
        yield Event("error", {"message": f"Unexpected error: {e}"})

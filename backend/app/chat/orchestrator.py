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
from .prompts import assemble_system_prompt, format_reference_material

logger = logging.getLogger(__name__)

MAX_TOKENS = 2048


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


async def _retrieve_reference(message: str) -> Retrieval | None:
    """Retrieve for `message`, or None if retrieval is unavailable or errors.

    The CPU-bound embed + query runs off the event loop. Any failure (e.g. the
    store not built yet) is logged and swallowed so chat still answers, just
    without reference material.
    """
    settings = get_settings()

    def _run() -> Retrieval:
        return retrieve(
            message,
            embedder=_get_embedder(),
            store=_get_store(),
            k=settings.max_rag_chunks,
            min_relevance=settings.rag_min_relevance,
        )

    try:
        return await asyncio.to_thread(_run)
    except Exception:  # noqa: BLE001 - retrieval must never break the chat stream
        logger.exception("RAG retrieval failed; continuing without reference material")
        return None


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
    reference_material: str | None = None
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
        if decision.wants_table:
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

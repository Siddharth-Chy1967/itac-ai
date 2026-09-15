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

from ..config import MAX_LISTED_RESULTS, get_settings
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
    format_bp_material,
    format_checklist_material,
    format_reference_material,
    render_arc_answer,
    render_bp_writeup_answer,
    render_enumerate_answer,
    render_field_guide_answer,
    render_sic_ranking_answer,
)

logger = logging.getLogger(__name__)

MAX_TOKENS = 2048


# --- retrieval-mode intent detection ----------------------------------------
# Special modes sit in front of the default "answer" mode:
#   arc       -> "which ARC code fits this recommendation" (arc_reference lookup)
#   checklist -> "pre-assessment: what should I look for" (SIC/process waterfall)
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
# A literal ARC code in the message ("what is the name for ARC code 3.4113"): one
# digit, a dot, then 3-4 digits, not embedded in a longer number. Its presence
# alone routes to ARC mode and triggers an exact (non-semantic) lookup.
_ARC_CODE_PATTERN = re.compile(r"(?<!\d)(\d\.\d{3,4})(?!\d)")
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
      | (?:reports|assessments)\s+
            (?:about|on|for|with|regarding|concerning|involving|mentioning|
               covering|discussing|featuring|describing|related\s+to|
               dealing\s+with|pertaining\s+to)\b
      | (?:do|does)\s+we\s+have\s+(?:\w+\s+){0,2}?(?:reports?|assessments?)\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
# Boilerplate stripped from an enumerate message to isolate the topic actually
# embedded. "Find me reports containing double paned windows" must embed "double
# paned windows", not the whole sentence: the lead-in ("find me reports
# containing") is semantic noise that drags the query away from the topic (top hit
# 0.61 with it, 0.70 without) and, worse, the enumerate relevance floor was
# calibrated on the BARE topic phrase, so embedding the full sentence measured a
# systematically lower score against a floor set for the higher one. Everything up
# to and including "reports"/"assessments" plus any run of connective/question
# filler ("that", "about", "on", "do we have", ...) is dropped; the remainder is the
# topic. Each filler token is boundaried with `(?=\s)` so a topic word that merely
# starts with one ("for" in "forklifts") is never eaten.
_ENUMERATE_STRIP = (
    r"that|which|with|about|on|for|regarding|concerning|involving|mentioning|"
    r"covering|discussing|featuring|describing|related|dealing|pertaining|to|"
    r"mention\w*|cover\w*|involv\w*|includ\w*|contain\w*|discuss\w*|address\w*|"
    r"deal\w*|exist\w*|"
    r"do|does|did|we|have|has|had|you|any|some|there|is|are|got|us|me|the|of"
)
_ENUMERATE_TOPIC = re.compile(
    rf"^.*?\b(?:reports?|assessments?)\b(?:\s+(?:{_ENUMERATE_STRIP})(?=\s|$))*"
    r"\s*(?P<topic>.+)$",
    re.IGNORECASE | re.DOTALL,
)


def _enumerate_topic(message: str) -> str | None:
    """The bare search topic in an enumerate message, or None if none is isolable.

    "Which reports mention solar panel cleaning?" -> "solar panel cleaning". None
    when nothing substantive survives the strip (a bare "list all reports"), so the
    caller falls back to the raw message.
    """
    m = _ENUMERATE_TOPIC.search(message)
    if not m:
        return None
    topic = m.group("topic").strip(" ?.!,:;\"'")
    return topic if re.search(r"[A-Za-z]{2,}", topic) else None


# A SIC code as a user types it: a full 4-digit code ("2599") or a truncated one with
# the unknown digits written as wildcards ("259x", "25xx"). Students routinely ask by
# family rather than exact code, and the code they name may not exist in the corpus at
# all, so both forms must parse; `sic_waterfall` strips the wildcards and starts at the
# widest level the digits support. Two wildcards is the limit on purpose -- "2xxx" is a
# whole division, too broad to draw a facility comparison from.
# `(?!\d)` rather than a trailing \b: it rejects a 5-digit run (not a SIC) the same way,
# but a \b after a "*" wildcard would fail and silently give back the bare digits.
_SIC_DIGITS = r"\d{2,4}[xX*]{0,2}(?!\d)"
# Pre-assessment checklist intent: a student prepping for a site visit, keyed on a
# SIC code, "pre-assessment", or an explicit "what should I look for" request.
_CHECKLIST_INTENT = re.compile(
    rf"""
    \b(?:
        (?:my|assigned)\s+sic\b
      | sic\s*(?:code)?\s*[:#]?\s*{_SIC_DIGITS}
      | pre[-\s]?assessment\b
      | what\s+(?:should\s+i|to)\s+look\s+for\b
      | checklist\b
      | prepar\w*\s+for\s+(?:the\s+|my\s+|an?\s+)?(?:assessment|site\s*visit|visit)\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
# Boilerplate stripped from a checklist message to isolate the facility description
# fed to Pass B (process similarity).
_CHECKLIST_BOILER = re.compile(
    r"""
    \b(?:
        my\s+sic\s+is | assigned\s+sic | sic\s+code | sic
      | pre[-\s]?assessment(?:\s+for)? | preassessment
      | what\s+should\s+i\s+look\s+for(?:\s+(?:at|in|for))? | what\s+to\s+look\s+for
      | checklist(?:\s+for)? | prepar\w*\s+for\s+(?:the\s+|my\s+|an?\s+)?(?:assessment|site\s*visit|visit)
      | assessment\s+for
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)
# The SIC near a "sic" keyword, tolerating filler ("is", "code", ":"). Accepts the
# wildcard forms too (see `_SIC_DIGITS`); previously this demanded exactly four digits,
# so "my SIC is 25xx" parsed to None and the industry pass silently never ran.
_SIC_IN_MESSAGE = re.compile(rf"\bsic\b[^0-9]{{0,10}}?({_SIC_DIGITS})", re.IGNORECASE)


# "What ARs does this industry usually get?" -- a frequency question about a SIC, not a
# request for a site-visit checklist. It must be tested BEFORE `_CHECKLIST_INTENT`,
# which any message naming a SIC also matches, and it engages only when a SIC is
# actually present: "most common ARs" with no industry attached is an answer-mode
# question about the corpus at large, which this ranking cannot speak to.
_SIC_RANK_INTENT = re.compile(
    r"""
    \b(?:most\s+|top\s+)?
    (?:common(?:ly)?|frequent(?:ly)?|recurring|recurrent|repeated|typical|popular|usual|top)
    \s+(?:\w+\s+){0,2}?
    (?:a\.?r\.?s?|recommendations?)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


# "What do I need to measure on site for <AR>?" -- the field data-collection list. Keys
# on an explicit data-gathering verb tied to measurement/collection vocabulary, so an
# ordinary question about an AR ("how do LED savings get calculated") does not land
# here. The topic is whatever follows the trigger phrase (see `_field_topic`).
_FIELD_INTENT = re.compile(
    r"""
    \b(?:
        what\s+(?:data|measurements?|numbers|parameters|information|info)\s+
            (?:do\s+i|should\s+i|to|does\s+\w+)\s*
            (?:need|collect|gather|measure|record|take|bring)?
      | what\s+(?:do\s+i|should\s+i)\s+(?:need\s+to\s+)?
            (?:measure|collect|gather|record|look\s+at)
      | (?:data|measurements?|parameters|readings)\s+(?:to\s+|i\s+)?
            (?:collect|gather|take|record|measure|need)
      | (?:collect|gather|measure|record)\s+(?:on[-\s]?site|in\s+the\s+field|during\s+the\s+visit)
      | on[-\s]?site\s+(?:data|measurements?|readings)
      | field\s+(?:data|measurements?|checklist)
      | vitals\s+for\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
# Everything the trigger phrase can be followed by before the topic proper begins. The
# collection verbs appear here as well as in the trigger because the trigger's own verb
# group is optional, so a phrasing like "what data do I need TO COLLECT for X" matches
# on "what data do I need" and leaves the verb for this pass to strip.
_FIELD_STRIP = (
    r"for|on|about|regarding|when|to|do|doing|writing|an?|the|my|this|that|ar|"
    r"recommendation|write[-\s]?up|assessment|site\s*visit|visit|in|order|"
    r"collect|gather|measure|record|take|bring|need|site|field|data|checklist"
)
_FIELD_TOPIC = re.compile(
    rf"(?:{_FIELD_INTENT.pattern})(?:\s+(?:{_FIELD_STRIP})(?=\s|$))*\s*(?P<topic>.+)$",
    re.IGNORECASE | re.VERBOSE,
)
# A trailing "... AR" / "... recommendation" adds nothing to the embedding.
_FIELD_TOPIC_TAIL = re.compile(
    r"\s+(?:ar|a\.r\.|recommendation|write[-\s]?up)s?\s*$", re.IGNORECASE
)


def _field_topic(message: str) -> str | None:
    """The AR topic a field-data request is about ("...measure for compressed air leaks").

    Same reasoning as `_enumerate_topic`: the relevance floor is calibrated on the bare
    topic, and embedding the whole question drags the score down against a floor set for
    the phrase alone.
    """
    m = _FIELD_TOPIC.search(message)
    if not m:
        return None
    topic = _FIELD_TOPIC_TAIL.sub("", m.group("topic")).strip(" ?.!,:;\"'")
    return topic if re.search(r"[A-Za-z]{3}", topic) else None


def _classify_mode(message: str) -> str:
    """`"arc"` / `"field"` / `"sic_rank"` / `"checklist"` / `"enumerate"` / `"answer"`."""
    if _ARC_INTENT.search(message) or _ARC_CODE_PATTERN.search(message):
        return "arc"
    if _FIELD_INTENT.search(message):
        return "field"
    if _SIC_RANK_INTENT.search(message) and _SIC_IN_MESSAGE.search(message):
        return "sic_rank"
    if _CHECKLIST_INTENT.search(message):
        return "checklist"
    if _ENUMERATE_INTENT.search(message):
        return "enumerate"
    return "answer"


# --- summarize-mode intent + target resolution ------------------------------
# "Summarize this AR / the vitals / the payback for AS0612" -> fetch that specific
# report (or AR) by identifier and summarize it, instead of semantic top-k which
# lands on the globally-nearest AR from the wrong report. Deliberately NARROW: it
# keys on explicit summary phrasing only (not "run me through" / "methodology",
# which stay on the semantic-rewrite answer path they are tested to use), and it
# engages ONLY when a single report_id resolves (from the message or the report
# already in context). No resolvable report -> it defers to the normal answer path
# rather than guess, so it can never summarize the wrong report.
_SUMMARIZE_INTENT = re.compile(
    r"""
    \b(?:
        summar(?:ize|ise|y)                       # summarize / summarise / summary
      | (?:the\s+)?vitals\b                        # "the vitals"
      | recap | rundown | tl;?dr
      | give\s+me\s+(?:the\s+)?(?:numbers|details|specifics|breakdown)
      | (?:payback|pay\s*back)\s+period
      | implementation\s+cost
      | cost\s+savings
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)
# A report identifier as the corpus uses it ("AS0612"). Case-insensitive so "as0612"
# in a typed follow-up still resolves; the 4-digit run is what makes it a real id and
# not the word "as".
_REPORT_ID = re.compile(r"\bAS\d{4}\b", re.IGNORECASE)
# An AR reference in free text ("AR-5", "AR #5", "AR5", "AR No. 2"), unanchored. The
# trailing digit requirement keeps "ARC 2.4111" and "Area" from matching.
_AR_REF = re.compile(r"\bAR[\s#\-.:]*(?:No\.?\s*)?(\d+)\b", re.IGNORECASE)
# When summarizing a whole report (no AR named), cap how many chunks are fed so a
# large report does not blow the context window; AR and summary sections are kept
# preferentially (see `_prioritize_report_chunks`).
_SUMMARIZE_MAX_CHUNKS = 20


def _report_ids(text: str) -> list[str]:
    """Distinct report ids in `text`, uppercased, first-seen order."""
    return list(dict.fromkeys(m.group(0).upper() for m in _REPORT_ID.finditer(text or "")))


def _report_id_from_history(history: list[Message]) -> str | None:
    """The report currently in context: the single id in the most recent turn to name one.

    Scans newest-first and returns on the first turn that mentions any report id. If
    that turn names exactly one, it is the target; if it names several (e.g. an
    enumerate list the user has not yet narrowed), the reference is ambiguous, so
    return None and let summarize defer rather than pick arbitrarily.
    """
    for msg in reversed(history):
        ids = _report_ids(msg.content or "")
        if ids:
            return ids[0] if len(ids) == 1 else None
    return None


def _summarize_target(
    message: str, history: list[Message]
) -> tuple[str, str | None] | None:
    """(report_id, ar_number|None) for a summarize request, or None to not engage.

    Engages only on explicit summary phrasing AND a single resolvable report_id: an
    id typed in the message wins; otherwise the report already in context (history).
    `ar_number` is taken from the message only (an explicit "AR-5"); absent, the whole
    report is summarized. Returns None whenever no single report resolves, so the
    caller falls through to the normal answer path instead of guessing.
    """
    if not _SUMMARIZE_INTENT.search(message):
        return None
    ids = _report_ids(message)
    report_id = ids[0] if len(ids) == 1 else (None if ids else _report_id_from_history(history))
    if report_id is None:
        return None
    ar_m = _AR_REF.search(message)
    ar_number = ar_m.group(1) if ar_m else None
    return report_id, ar_number


def _parse_checklist_query(message: str) -> tuple[str | None, str | None]:
    """Split a checklist request into (sic, facility_description); either may be None.

    A SIC (full or wildcarded, see `_SIC_DIGITS`) is taken only when tied to the word
    "sic", so a stray year is not mistaken for one. The description is the message with
    the SIC mention and the checklist boilerplate removed, kept only when substantive
    (>= 3 words).
    """
    sic_m = _SIC_IN_MESSAGE.search(message)
    sic = sic_m.group(1) if sic_m else None

    desc = message
    if sic_m:
        desc = desc[: sic_m.start()] + " " + desc[sic_m.end() :]
    desc = _CHECKLIST_BOILER.sub(" ", desc)
    desc = re.sub(r"\s+", " ", desc).strip(" ,.:;-")
    description = desc if len(re.findall(r"[A-Za-z]{2,}", desc)) >= 3 else None
    return sic, description


# Best-practice write-up intent: an explicit request to DRAFT one or more named
# practices. Two shapes: "write me a best practice for <topics>" and "write up the
# <topic> best practice". Keyed on a drafting verb so a plain "what are the best
# practices for X" (no verb) stays a checklist/answer question, not a write-up.
_BP_WRITEUP_VERB = r"(?:write|draft|compose|generate|create|give\s+me|make|prepare)"
_BP_WRITEUP_FOR = re.compile(
    rf"\b{_BP_WRITEUP_VERB}\b(?:\s+(?:me|us|out|up))?\s+(?:an?\s+|the\s+)?"
    r"best[-\s]?practices?(?:\s+\w+)?\s+(?:for|on|about|regarding)\s+(?P<topic>.+)",
    re.IGNORECASE,
)
_BP_WRITEUP_THE = re.compile(
    rf"\b{_BP_WRITEUP_VERB}\b(?:\s+(?:me|us|out|up))?\s+(?:the\s+)?"
    r"(?P<topic>.+?)\s+best[-\s]?practice\b",
    re.IGNORECASE,
)
# A topic phrase that is purely a back-reference ("... for these", "for all of them"):
# it names no practice itself, it points at whatever the user described earlier in the
# same message, so the topics are pulled from the text before the drafting clause.
_BP_BACKREF = re.compile(
    r"^(?:these|those|them|it|that|this|they|all(?:\s+of\s+(?:these|those|them))?"
    r"|the\s+above)"
    r"(?:\s+(?:ones?|items?|things?|systems?|practices?|measures?|equipment|above))?$",
    re.IGNORECASE,
)
# Lead-in on a described item ("my facility already has <topic>") stripped so the bare
# practice is what gets embedded against the best-practice list.
_BP_TOPIC_LEAD = re.compile(
    r"^(?:(?:my|our|the)\s+(?:facility|plant|site|shop|building)\s+)?"
    r"(?:already\s+)?(?:has|had|have|got|uses?|used|includes?|installed|runs?|"
    r"operates?|does|did|with)\s+",
    re.IGNORECASE,
)
# Split a topic phrase into individual practice topics on list separators.
_BP_TOPIC_SPLIT = re.compile(r"\s*(?:,|;|/|&|\band\b)\s*", re.IGNORECASE)


def _split_bp_topics(text: str) -> list[str]:
    """Split a phrase into distinct practice topics, cleaned and deduped.

    "LEDs, VFDs and sludge reusing" -> ["LEDs", "VFDs", "sludge reusing"]. Each
    fragment has any "my facility has ..." lead-in stripped; empty or too-short
    fragments are dropped. Order is preserved, case-insensitive duplicates removed.
    """
    seen: set[str] = set()
    out: list[str] = []
    for part in _BP_TOPIC_SPLIT.split(text):
        frag = _BP_TOPIC_LEAD.sub("", part).strip(" .?!,:;\"'")
        if len(frag) < 3 or not re.search(r"[A-Za-z]{2,}", frag):
            continue
        key = frag.lower()
        if key not in seen:
            seen.add(key)
            out.append(frag)
    return out


def _bp_writeup_topics(message: str) -> list[str]:
    """The practice topics when `message` is a write-up request, else [].

    "write me a best practice for LED lighting" -> ["LED lighting"]; a multi-topic
    "write me best practices for LEDs, VFDs and sludge" -> all three. A back-reference
    ("... for these") is resolved to the items the user described earlier in the same
    message. A "checklist" is never a write-up (guards the "write me a checklist of
    best practices for ..." overlap).
    """
    if re.search(r"\bchecklist\b", message, re.IGNORECASE):
        return []
    m = _BP_WRITEUP_FOR.search(message) or _BP_WRITEUP_THE.search(message)
    if not m:
        return []
    topic = m.group("topic").strip(" .?!,:;\"'")
    if len(topic) < 3:
        return []
    # "... for these": the topic points back at what was described before the drafting
    # clause, so mine the topics from that earlier text instead of the pronoun itself.
    if _BP_BACKREF.match(topic):
        return _split_bp_topics(message[: m.start()])
    return _split_bp_topics(topic)


def _bp_writeup_topic(message: str) -> str | None:
    """The first write-up topic, or None. Thin wrapper over `_bp_writeup_topics`."""
    topics = _bp_writeup_topics(message)
    return topics[0] if topics else None


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


@lru_cache(maxsize=1)
def _get_bp_store():
    from ..rag.store import BPStore

    return BPStore()


async def warmup() -> None:
    """Pre-build the heavy singletons so the first chat turn doesn't pay for them.

    Every retrieval dependency here is `@lru_cache(maxsize=1)`, so it is built once
    and reused for the process lifetime. Left to lazy loading, that one-time cost
    (import torch + load the embed model onto the GPU, open the three ChromaDB
    collections, and cold-load the Ollama chat model into VRAM) all lands on the
    user's first message. Running it here — off the event loop, kicked off at
    startup — moves the wait to boot instead.

    Each step is best-effort: a warmup failure (GPU busy, store not ingested yet,
    Ollama not running) is logged and swallowed, so the app still starts and the
    real request path retries and surfaces any genuine error itself.
    """
    def _build_retrieval() -> None:
        _get_embedder()  # torch import + embed model onto the GPU
        _get_store()
        _get_arc_store()
        _get_bp_store()

    try:
        await asyncio.to_thread(_build_retrieval)
        logger.info("Warmup: embedder + stores ready")
    except Exception:  # noqa: BLE001 - warmup must never break startup
        logger.exception("Warmup: retrieval pre-build failed; will lazy-load on first use")

    # Only Ollama benefits from a VRAM warmup; a throwaway Gemini call would just
    # spend API quota. A 1-token completion is enough to make Ollama resident-load
    # the chat model.
    if get_settings().llm_provider.lower() == "ollama":
        try:
            await get_provider().complete(
                [Message(role="user", content="ok")], system="", max_tokens=1
            )
            logger.info("Warmup: Ollama chat model loaded")
        except Exception:  # noqa: BLE001 - warmup must never break startup
            logger.exception("Warmup: Ollama model load failed; will load on first use")


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


# Generic AR-title words that carry no facility-specific signal, so they never help
# decide which AR a vague follow-up ("summarize the vitals") refers to. Everything
# else in a title (grinders, lighting, compressor, chiller, ...) is the distinctive
# topic that a context match keys on. The action verbs (`_ACTION_VERBS`, defined
# lower in this file) are folded in at call time by `_stopwords()`.
_AR_TITLE_FILLER = frozenset(
    {
        "the", "and", "for", "with", "after", "off", "on", "your", "our", "system",
        "systems", "unit", "units", "new", "existing", "high", "low", "ar", "arc",
        "efficiency", "efficient", "energy", "cost", "savings", "usage", "annual",
    }
)


def _stopwords() -> frozenset[str]:
    """Filler + action verbs; built at call time since `_ACTION_VERBS` is defined below."""
    return _AR_TITLE_FILLER | _ACTION_VERBS


def _ar_title_keywords(title: str) -> set[str]:
    """The distinctive (non-generic) topic words in an AR section title."""
    stripped = _AR_REF.sub(" ", _LEADING_CODE.sub("", title or ""))
    words = {w for w in re.findall(r"[a-z]{3,}", stripped.lower())}
    return words - _stopwords()


def _narrow_ar_from_context(
    rows: list[tuple[str, dict]], message: str, history: list[Message]
) -> str | None:
    """Which single AR a vague summarize refers to, from the conversation, or None.

    A whole-report fetch spans every AR; asking a small model to "summarize the
    vitals" over all of them at once invites it to blend one AR's numbers into
    another. When the report has one AR, that is the target; when it has several,
    narrow to the one whose distinctive title words the recent conversation actually
    mentions (the search that surfaced the report, e.g. "grinders"). If the report
    has several ARs and none (or more than one) matches, return None so the caller
    keeps the whole report but under the strict per-AR contract.
    """
    titles: dict[str, str] = {}
    for _text, meta in rows:
        ar = meta.get("ar_number")
        if ar and ar not in titles:
            titles[ar] = meta.get("section") or ""
    if len(titles) <= 1:
        return next(iter(titles), None)

    context = " ".join([message, *(h.content or "" for h in history[-4:])]).lower()
    ctx_words = set(re.findall(r"[a-z]{3,}", context)) - _stopwords()

    best: list[str] = []
    best_score = 0
    for ar, title in titles.items():
        score = len(_ar_title_keywords(title) & ctx_words)
        if score > best_score:
            best, best_score = [ar], score
        elif score == best_score and score > 0:
            best.append(ar)
    return best[0] if best_score >= 1 and len(best) == 1 else None


def _prioritize_report_chunks(rows: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    """Trim a whole-report fetch to the highest-signal chunks, in document order.

    A large report can carry more chunks than the summary context should hold. Keep
    the AR-section chunks (`ar_number` set) and any summary/recommendation section,
    which is where the vitals and methodology live, then cap. Order (chunk_index) is
    already applied by `chunks_for`, so slicing keeps the report's top-to-bottom flow.
    """
    def _is_signal(meta: dict) -> bool:
        if meta.get("ar_number"):
            return True
        section = (meta.get("section") or "").lower()
        return "summary" in section or "recommendation" in section

    signal = [r for r in rows if _is_signal(r[1])] or rows

    # Guarantee every AR is represented by its richest (longest) chunk, which is where
    # its vitals live, BEFORE the cap: otherwise an AR whose detail sits at a high
    # chunk_index gets sliced off and the summary wrongly reports "not stated" for it.
    by_ar: dict[str, tuple[str, dict]] = {}
    for row in signal:
        ar = row[1].get("ar_number")
        if ar and (ar not in by_ar or len(row[0]) > len(by_ar[ar][0])):
            by_ar[ar] = row

    kept: dict[int, tuple[str, dict]] = {}
    # Per-AR richest picks first (richest ARs win if there are more ARs than budget),
    # then fill the remaining budget with the rest in document order.
    for row in sorted(by_ar.values(), key=lambda r: len(r[0]), reverse=True):
        if len(kept) >= _SUMMARIZE_MAX_CHUNKS:
            break
        kept[row[1].get("chunk_index", 0)] = row
    for row in signal:
        if len(kept) >= _SUMMARIZE_MAX_CHUNKS:
            break
        kept.setdefault(row[1].get("chunk_index", 0), row)

    return [row for _idx, row in sorted(kept.items())]


async def _fetch_report_chunks(
    report_id: str, ar_number: str | None, message: str, history: list[Message]
) -> list[tuple[str, dict]]:
    """Fetch the chunks to summarize, off the event loop, or [] on failure.

    A deterministic metadata fetch (not semantic search). With an explicit AR, that
    AR's chunks are returned as-is. Without one, the whole report is fetched, then
    narrowed to the single AR the conversation is about when that can be told
    (`_narrow_ar_from_context`); this keeps a vague follow-up like "summarize the
    vitals" from blending several ARs' numbers. If it cannot be narrowed, the whole
    report is trimmed to its highest-signal chunks and summarized per-AR. Any failure
    is logged and swallowed so summarize falls back to a normal answer.
    """
    def _run() -> list[tuple[str, dict]]:
        rows = _get_store().chunks_for(report_id, ar_number)
        if ar_number is None and rows:
            narrowed = _narrow_ar_from_context(rows, message, history)
            if narrowed is not None:
                return [r for r in rows if r[1].get("ar_number") == narrowed]
            if len(rows) > _SUMMARIZE_MAX_CHUNKS:
                rows = _prioritize_report_chunks(rows)
        return rows

    try:
        return await asyncio.to_thread(_run)
    except Exception:  # noqa: BLE001 - retrieval must never break the chat stream
        logger.exception("Summarize fetch failed; continuing without it")
        return []


def _summarize_retrieval(query: str, rows: list[tuple[str, dict]]) -> Retrieval:
    """Wrap fetched chunks as a confident `Retrieval` so `format_reference_material`
    renders them with its grounding block (each labeled, cite report_id/fiscal_year).

    The chunks were fetched by identifier, so every one is a real match: score 1.0 and
    a 0.0 floor make `has_match`/`matches()` include them all.
    """
    hits = [(text, meta, 1.0) for text, meta in rows]
    return Retrieval(query=query, hits=hits, min_relevance=0.0)


def _summarize_sources(rows: list[tuple[str, dict]]) -> list[dict]:
    """`sources` payload for summarize: the report sections fed to the model, deduped."""
    out: list[dict] = []
    seen: set[tuple] = set()
    for _text, meta in rows:
        key = (meta.get("report_id"), meta.get("section"))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "report_id": meta.get("report_id"),
                "fiscal_year": meta.get("fiscal_year"),
                "section": meta.get("section"),
                "score": None,
            }
        )
    return out


def _apply_enumerate_floor(retrieval: Retrieval, sanity_floor: float, band: float) -> None:
    """Raise the enumerate relevance floor to a value adapted to THIS query.

    nomic cosine scores are compressed and query-dependent: a specific topic that is
    genuinely in the corpus tops out around 0.70 while an off-topic query still tops
    ~0.59, so no single absolute floor both surfaces the real matches and rejects the
    nonsense. Anchor the floor to the query's own best hit instead: keep the cluster
    within `band` of the top score, but never drop below the absolute `sanity_floor`
    that a query with no real match at all fails to clear. On a genuine hit this keeps
    the top cluster; on nonsense (`top_score < sanity_floor`) the effective floor
    stays at `sanity_floor` above every hit, so nothing matches and enumerate reports
    "none found" honestly.
    """
    retrieval.min_relevance = max(sanity_floor, retrieval.top_score - band)


async def _retrieve_enumerate(message: str) -> Retrieval | None:
    """Enumerate-mode retrieval: a wide net, a query-adaptive floor, a recency cap.

    `rag_enumerate_k` chunks so every report on a topic surfaces (dedup collapses them
    afterwards). The floor is not the fixed answer-mode value but adapted per query
    from the top hit (see `_apply_enumerate_floor`), since a single absolute floor
    cannot separate a genuine topic (tops ~0.70) from nonsense (tops ~0.59) across
    nomic's compressed, query-dependent score scale.

    The wide net and the floor together still leave a long tail the floor cannot rank:
    most of the corpus predates fiscal-year labels, so beyond the newest few reports
    "newest first" has nothing left to sort on. `max_reports` trims to that trustworthy
    window (see `Retrieval.enumerated`).
    """
    settings = get_settings()
    retrieval = await _retrieve(
        message, settings.rag_enumerate_k, settings.rag_enumerate_min_relevance
    )
    if retrieval is not None:
        _apply_enumerate_floor(
            retrieval, settings.rag_enumerate_min_relevance, settings.rag_enumerate_band
        )
        retrieval.max_reports = settings.rag_enumerate_max_reports
    return retrieval


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


async def _direct_arc_lookup(message: str) -> Retrieval | None:
    """Exact ARC-code lookup when the message contains a literal code, else None.

    Reverse direction ("what is the name for ARC 3.4113"): resolve the code straight
    from arc_reference/arc_precedent metadata rather than semantic search, which does
    not match a bare code against its prose description. Returned as a single
    confident hit (score 1.0) so the display renders it as one definitive answer.
    """
    m = _ARC_CODE_PATTERN.search(message)
    if not m:
        return None
    code = m.group(1)

    def _run() -> Retrieval | None:
        hit = _get_arc_store().get_by_code(code)
        if hit is None:
            return None
        text, meta = hit
        return Retrieval(query=message, hits=[(text, meta, 1.0)], min_relevance=0.65)

    try:
        return await asyncio.to_thread(_run)
    except Exception:  # noqa: BLE001 - retrieval must never break the chat stream
        logger.exception("Direct ARC code lookup failed")
        return None


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


async def _build_checklist(sic: str | None, description: str | None):
    """Run the pre-assessment waterfall off the event loop, or None on failure."""
    settings = get_settings()

    def _run():
        from ..rag.waterfall import build_checklist

        return build_checklist(
            sic=sic,
            description=description,
            store=_get_store(),
            embedder=_get_embedder(),
            min_reports=settings.checklist_min_reports,
            process_k=settings.rag_process_k,
            process_threshold=settings.rag_process_min_relevance,
            group_threshold=settings.rag_ar_group_threshold,
            top_themes=min(settings.checklist_top_themes, MAX_LISTED_RESULTS),
        )

    try:
        return await asyncio.to_thread(_run)
    except Exception:  # noqa: BLE001 - retrieval must never break the chat stream
        logger.exception("Checklist retrieval failed; continuing without it")
        return None


async def _build_field_guide(topic: str):
    """Run the on-site data lookup off the event loop, or None on failure."""
    settings = get_settings()

    def _run():
        from ..rag.fielddata import TitleGate, build_field_guide

        return build_field_guide(
            topic=topic,
            store=_get_store(),
            embedder=_get_embedder(),
            k=settings.rag_field_k,
            threshold=settings.rag_field_min_relevance,
            band=settings.rag_field_band,
            max_instances=settings.field_max_instances,
            min_support=settings.field_min_support,
            top_n=min(settings.field_top_parameters, MAX_LISTED_RESULTS),
            gate=TitleGate(
                exact=settings.rag_field_title_exact,
                min_coverage=settings.rag_field_title_min_coverage,
                min_combined=settings.rag_field_title_min_combined,
                coverage_weight=settings.rag_field_title_coverage_weight,
            ),
        )

    try:
        return await asyncio.to_thread(_run)
    except Exception:  # noqa: BLE001 - retrieval must never break the chat stream
        logger.exception("Field-data lookup failed; continuing without it")
        return None


def _field_sources(guide) -> list[dict]:
    """`sources` payload for the field guide: the past write-ups it counted."""
    return [
        {
            "report_id": inst.report_id,
            "fiscal_year": inst.era,
            "section": inst.title,
            "score": round(inst.score, 3),
        }
        for inst in guide.instances
    ]


async def _rank_sic_ars(sic: str):
    """Rank the AR themes most written for a SIC, off the event loop, or None on failure."""
    settings = get_settings()

    def _run():
        from ..rag.waterfall import rank_ars_for_sic

        return rank_ars_for_sic(
            sic=sic,
            store=_get_store(),
            embedder=_get_embedder(),
            min_reports=settings.checklist_min_reports,
            group_threshold=settings.rag_ar_group_threshold,
            top_n=min(settings.checklist_top_themes, MAX_LISTED_RESULTS),
        )

    try:
        return await asyncio.to_thread(_run)
    except Exception:  # noqa: BLE001 - retrieval must never break the chat stream
        logger.exception("SIC AR ranking failed; continuing without it")
        return None


def _sic_rank_sources(result) -> list[dict]:
    """`sources` payload for the SIC ranking: one entry per ranked theme.

    `fiscal_year` carries the era split rather than a year, because a theme spans many
    reports across eras and the card has no single one to show.
    """
    return [
        {
            "report_id": theme.representative,
            "fiscal_year": f"{theme.modern_count} of {theme.report_count} since 2017",
            "section": "; ".join(theme.examples[:4]),
            "score": None,
        }
        for theme in result.themes
    ]


def _checklist_sources(result) -> list[dict]:
    """`sources` payload for the checklist: the matched reports and how each matched."""
    out = []
    for m in result.matched:
        if m.from_industry and m.from_process:
            how = "SIC + process match"
        elif m.from_industry:
            how = "SIC match"
        else:
            how = "process match"
        out.append(
            {"report_id": m.report_id, "fiscal_year": m.era, "section": how, "score": None}
        )
    return out


async def _build_bp_candidates(sic: str | None, description: str | None):
    """Run the B3 best-practice retrieval off the event loop, or None on failure."""
    settings = get_settings()

    def _run():
        from ..rag.waterfall import build_bp_candidates

        return build_bp_candidates(
            sic=sic,
            description=description,
            store=_get_store(),
            bp_store=_get_bp_store(),
            embedder=_get_embedder(),
            min_reports=settings.checklist_min_reports,
            precedent_k=settings.bp_precedent_k,
            precedent_threshold=settings.bp_precedent_min_relevance,
            semantic_k=settings.bp_semantic_k,
            semantic_threshold=settings.bp_semantic_min_relevance,
            per_category=settings.bp_per_category,
        )

    try:
        return await asyncio.to_thread(_run)
    except Exception:  # noqa: BLE001 - retrieval must never break the chat stream
        logger.exception("BP checklist retrieval failed; continuing without it")
        return None


def _bp_sources(result) -> list[dict]:
    """`sources` payload for the BP checklist: the comparable precedent facilities."""
    return [
        {
            "report_id": rid,
            "fiscal_year": "",
            "section": f"comparable facility (SIC {result.sic_pattern})",
            "score": None,
        }
        for rid in result.precedent_report_ids
    ]


async def _lookup_bp_writeup(topic: str):
    """Look up the single official BP template for a write-up topic, or None."""
    settings = get_settings()

    def _run():
        from ..rag.waterfall import lookup_bp_item

        return lookup_bp_item(
            _get_bp_store(),
            _get_embedder(),
            topic,
            threshold=settings.bp_writeup_min_relevance,
        )

    try:
        return await asyncio.to_thread(_run)
    except Exception:  # noqa: BLE001 - retrieval must never break the chat stream
        logger.exception("BP write-up lookup failed; continuing without it")
        return None


async def _lookup_bp_writeups(topics: list[str]) -> list:
    """Resolve each write-up topic to its official BP template, deduped by name.

    One `lookup_bp_item` per topic; topics that match nothing over the write-up floor
    are dropped, so the caller drafts only the practices that are actually on the ITAC
    best-practice list. Returns [] when none of the topics match.
    """
    items = []
    seen: set[str] = set()
    for topic in topics:
        item = await _lookup_bp_writeup(topic)
        if item is not None and item.name not in seen:
            seen.add(item.name)
            items.append(item)
    return items


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


def _sources_event(payload: list[dict]) -> Event:
    """A `sources` event holding at most MAX_LISTED_RESULTS cards.

    Every mode emits its source cards through here, so the cap holds for all of them
    (a checklist over a 2-digit SIC family matches 50+ reports, a summarize fetch lists
    every section) without each builder having to remember it. Payloads arrive best-
    or newest-first, so truncation keeps the strongest entries.
    """
    return Event("sources", payload[:MAX_LISTED_RESULTS])


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
    #    in the terse list-only output contract. ARC mode ("which ARC code fits this
    #    recommendation") queries the separate arc_reference/arc_precedent collections
    #    and renders the answer deterministically (verbatim code + title), bypassing
    #    the LLM so a code cannot be fabricated. Answer mode is untouched.
    reference_material: str | None = None
    arc_answer: str | None = None  # ARC lookup renders deterministically, no LLM
    enumerate_answer: str | None = None  # enumerate lists render deterministically too
    sic_rank_answer: str | None = None  # SIC AR frequency table, likewise deterministic
    field_answer: str | None = None  # on-site data list, likewise deterministic
    bp_writeup_answer: str | None = None  # BP write-up renders deterministically too
    enumerate_mode = False
    checklist_mode = False
    bp_checklist_mode = False
    bp_writeup_mode = False
    summarize_mode = False

    # An explicit drafting request ("write me the best practices for LEDs and skylights")
    # pre-empts every other mode, INCLUDING the checklist: the write verb is the strong
    # signal, so it wins even when the router also flagged the facility description as a
    # checklist. It resolves each named practice to its official bp_reference item and
    # emits that literature verbatim, deterministically (no LLM), so the reply is exactly
    # the per-heading text with no assumptions or commentary. A back-reference ("... for
    # these") is resolved to the items described earlier in the same message. Topics that
    # match no BP item are dropped; if none resolve, it falls through to the checklist /
    # keyword modes, so a non-drafting request (a bare facility question) is unaffected.
    topics = _bp_writeup_topics(message)
    if topics:
        items = await _lookup_bp_writeups(topics)
        if items:
            bp_writeup_mode = True
            bp_writeup_answer = render_bp_writeup_answer(items)

    # Best-practice pre-assessment is router-gated (wants_bp_checklist), not keyword-
    # classified: it fires when the router judges the message to be a facility
    # description paired with a best-practices ask. It pre-empts the keyword modes
    # below; if it yields no candidates (or the store is down) it falls through to
    # them, so a misfire costs at most a normal answer.
    if not bp_writeup_mode and decision.wants_bp_checklist:
        sic, description = _parse_checklist_query(message)
        result = await _build_bp_candidates(sic, description)
        if result is not None and result.candidates:
            bp_checklist_mode = True
            if result.precedent_report_ids:
                yield _sources_event(_bp_sources(result))
            reference_material = format_bp_material(result)

    # Summarize a specific report/AR ("summarize the vitals for AS0612", or a
    # follow-up "summarize this AR" on the report already in context). Fetches that
    # report/AR by identifier (deterministic, not semantic) and feeds the full text so
    # the grounded model summarizes the RIGHT report. Engages only when a single
    # report_id resolves and the fetch actually returns chunks; otherwise it leaves
    # summarize_mode False and falls through to the keyword modes below, so a misfire
    # (unknown id, empty report) costs at most a normal answer. Pre-empts the keyword
    # modes but sits behind the router-gated BP paths, matching their precedence.
    if not (bp_writeup_mode or bp_checklist_mode):
        target = _summarize_target(message, history)
        if target is not None:
            rows = await _fetch_report_chunks(*target, message, history)
            if rows:
                summarize_mode = True
                yield _sources_event(_summarize_sources(rows))
                reference_material = format_reference_material(
                    _summarize_retrieval(message, rows)
                )

    mode = (
        None
        if (bp_writeup_mode or bp_checklist_mode or summarize_mode)
        else _classify_mode(message)
    )
    if bp_writeup_mode or bp_checklist_mode or summarize_mode:
        pass  # reference material already assembled above
    elif mode == "arc":
        # A literal ARC code in the message resolves by exact lookup first; only a
        # descriptive request falls through to semantic search.
        retrieval = await _direct_arc_lookup(message)
        if retrieval is None:
            retrieval = await _retrieve_arc(decision.retrieval_query or message)
        # Store unavailable / empty: leave arc_answer None so this falls through to a
        # normal answer rather than asserting "no ARC code" about a lookup that never
        # ran. The answer is rendered deterministically from the retrieved records
        # (verbatim code + title), never by the LLM, so a fabricated code is
        # impossible; the model is bypassed entirely for ARC.
        if retrieval is not None and retrieval.hits:
            yield _sources_event(_arc_sources(retrieval))
            arc_answer = render_arc_answer(
                retrieval, gap_threshold=get_settings().rag_arc_gap_threshold
            )
    elif mode == "field":
        # "What do I measure on site for compressed air leaks": count the input
        # parameters past write-ups of that AR recorded. Rendered deterministically --
        # the whole point is that no measurement appears unless a report used it, and
        # letting the model phrase the list is how an unrecorded one gets in. Falls
        # through to a normal answer when no topic parses or the store is down.
        field_topic = _field_topic(message) or decision.retrieval_query
        if field_topic:
            guide = await _build_field_guide(field_topic)
            if guide is not None:
                if guide.instances:
                    yield _sources_event(_field_sources(guide))
                field_answer = render_field_guide_answer(guide)
    elif mode == "sic_rank":
        # "Most common ARs for SIC 3089": a frequency table over the corpus. Rendered
        # deterministically (like ARC and enumerate) because the counts ARE the answer
        # and a small model rephrasing them would quietly corrupt them. A SIC always
        # parses here (`_classify_mode` required one), but the store may be down, and
        # the waterfall may find nothing at all -- either way, fall through.
        sic, _description = _parse_checklist_query(message)
        if sic:
            result = await _rank_sic_ars(sic)
            if result is not None:
                if result.themes:
                    yield _sources_event(_sic_rank_sources(result))
                sic_rank_answer = render_sic_ranking_answer(result)
    elif mode == "checklist":
        # Pre-assessment: split into SIC and/or facility description and run the
        # two-pass waterfall. If neither parses (a bare "checklist" with nothing to
        # search on) or the store is unavailable, fall through to a normal answer.
        sic, description = _parse_checklist_query(message)
        if sic or description:
            result = await _build_checklist(sic, description)
            if result is not None:
                checklist_mode = True
                if result.matched:
                    yield _sources_event(_checklist_sources(result))
                reference_material = format_checklist_material(result)
    elif mode == "enumerate":
        # Embed the bare topic, not the whole sentence. On a follow-up the router's
        # rewrite already resolves references, so prefer it; on a first turn strip the
        # "find me reports containing ..." boilerplate down to the topic; failing both,
        # fall back to the raw message.
        enumerate_query = (
            decision.retrieval_query or _enumerate_topic(message) or message
        )
        retrieval = await _retrieve_enumerate(enumerate_query)
        # Store unavailable: leave enumerate_mode False (fall back to answer mode)
        # rather than have the model claim "no matching reports" about a search
        # that never ran. When the store answers, the report list is rendered
        # deterministically (like ARC) and the LLM is bypassed: the answer is a pure
        # lookup, so handing it to a small local model only invites fabricated report
        # IDs and invented savings figures.
        if retrieval is not None:
            enumerate_mode = True
            if retrieval.has_match:
                yield _sources_event(retrieval.enumerated_sources())
            enumerate_answer = render_enumerate_answer(retrieval)
    else:
        retrieval = await _retrieve_reference(decision.retrieval_query or message)
        if retrieval is not None:
            if retrieval.has_match:
                yield _sources_event(retrieval.sources())
                reference_material = format_reference_material(retrieval)
            elif decision.needs_rag:
                reference_material = format_reference_material(retrieval)

    # ARC lookup is deterministic: emit the rendered answer verbatim and finish,
    # bypassing the LLM entirely so no code or title can be paraphrased or invented.
    if arc_answer is not None:
        yield Event("token", {"t": arc_answer})
        yield Event("done", {})
        return

    # Enumerate is likewise a pure lookup: emit the deterministic report list and
    # finish, so the model can neither drop real reports nor invent fake ones.
    if enumerate_answer is not None:
        yield Event("token", {"t": enumerate_answer})
        yield Event("done", {})
        return

    # The SIC frequency table is a counted result: emit it verbatim so no count,
    # AR title, or era split can be paraphrased into something untrue.
    if sic_rank_answer is not None:
        yield Event("token", {"t": sic_rank_answer})
        yield Event("done", {})
        return

    # The field-data list is likewise counted evidence: emit it verbatim so no
    # measurement the reports never recorded can be added to it.
    if field_answer is not None:
        yield Event("token", {"t": field_answer})
        yield Event("done", {})
        return

    # BP write-up is a pure lookup too: emit each matched practice's official literature
    # verbatim and finish, so the reply is exactly the clean per-heading text with no
    # model preamble, paraphrase, or invented commentary.
    if bp_writeup_answer is not None:
        yield Event("token", {"t": bp_writeup_answer})
        yield Event("done", {})
        return

    # 3. PROMPT ASSEMBLY.
    system = assemble_system_prompt(
        selected,
        reference_material=reference_material,
        wants_table=decision.wants_table,
        enumerate_mode=enumerate_mode,
        checklist_mode=checklist_mode,
        bp_checklist_mode=bp_checklist_mode,
        bp_writeup_mode=bp_writeup_mode,
        summarize_mode=summarize_mode,
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
        if (
            decision.wants_table
            and not enumerate_mode
            and not checklist_mode
            and not bp_checklist_mode
            and not bp_writeup_mode
            and not summarize_mode
        ):
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

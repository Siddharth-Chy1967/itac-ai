"""Application configuration (spec §3).

Single source of truth for settings. Everything reads config through
`get_settings()`; nothing touches `os.environ` directly.

The `.env` lives at the repository root (itac-ai/.env). We resolve its
absolute path from this file's location so the app finds it regardless of
the current working directory (e.g. running uvicorn from backend/).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# config.py -> app -> backend -> itac-ai (repo root)
_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Try the repo-root .env first, then a .env in the current dir.
        env_file=(str(_REPO_ROOT / ".env"), ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # LLM provider selection
    llm_provider: str = "gemini"  # gemini | ollama | anthropic

    # Provider credentials / endpoints
    gemini_api_key: str = ""
    anthropic_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    # Models
    ollama_model: str = "llama3.3"
    gemini_model: str = "gemini-2.0-flash"
    anthropic_model: str = "claude-haiku-latest"
    router_model_override: str = ""  # optional cheaper model just for skill routing

    # RAG (unused in Session A, defined for completeness)
    # nomic-embed-text-v1.5 covers up to 8192 tokens, so full ~700-token chunks
    # are embedded without truncation (all-MiniLM-L6-v2 capped at ~256).
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    chroma_path: str = "./data/chroma"
    upload_path: str = "./data/uploads"
    # Where rendered .xlsx files from chat land, keyed by download token (spec §7.3).
    export_path: str = "./data/exports"
    # Corpus of ITAC/IAC reports for bulk ingest (spec §6.4, rag.ingest CLI).
    reports_path: str = "./data/reports"
    max_rag_chunks: int = 6
    chunk_size_tokens: int = 700
    chunk_overlap_tokens: int = 100
    # Cosine-similarity floor for treating the top retrieval hit as a real match.
    # Below this, retrieval reports "no confident match" so the model does not
    # present a weak result as an established precedent. Tune once the real corpus
    # is ingested (nomic+cosine: a strong match scores ~0.85+, a weak one ~0.5).
    rag_min_relevance: float = 0.6
    # ARC code lookup: query the arc_reference collection top-5, return top-3.
    rag_arc_k: int = 5
    # Verb-alignment tie-break for ARC lookup: when the user's leading action verb
    # (replace/install/reduce/...) matches a candidate's leading verb, add this to
    # its ranking score. Small on purpose: it only breaks near-ties where a literal
    # token match (e.g. "baler") would otherwise outrank the better-intent code.
    rag_arc_verb_boost: float = 0.03
    # ARC display: show the top candidate as a single confident answer only when it
    # leads #2 by at least this raw-similarity margin; otherwise present the top 3 as
    # equal options. Calibrated on real cases: a clear winner (compactor -> 3.6193,
    # gap ~0.30) headlines as one answer; a genuine cluster (solar panel cleaning,
    # gap ~0.08) stays 3 tied options.
    rag_arc_gap_threshold: float = 0.15
    # Weakness floor for "no confident ARC match". Calibrated 2026-07-17 on the
    # 579-code arc_reference collection: the four intended lookups scored
    # 0.662-0.819 on their best hit, while nonsense queries ("color of the sky",
    # "schedule a meeting") topped out at 0.632. 0.65 sits in that gap, so real
    # lookups clear it and nonsense falls below and is flagged as a guess.
    rag_arc_min_relevance: float = 0.65
    # Pre-assessment checklist (Feature 2). Pass B (process similarity) queries the
    # facility_profile chunks; the threshold is a placeholder pending calibration
    # against the Clayton Homes <-> woodworking pair on real data.
    rag_process_k: int = 15
    rag_process_min_relevance: float = 0.6
    # AR-title near-duplicate grouping: greedy cosine merge threshold. Placeholder
    # pending calibration against a real sample of AR titles (the "show me the
    # sample groupings" gate) once the corpus is re-ingested.
    rag_ar_group_threshold: float = 0.82
    # One threshold reused three ways per the era policy: SIC waterfall widens
    # sic4->sic3->sic2 when fewer than this many distinct reports match; the recency
    # preference falls back to <=2006 reports when fewer than this many modern
    # reports match; and a merged set below this is marked "thin".
    checklist_min_reports: int = 3
    checklist_top_themes: int = 10
    # Best Practice pre-assessment (Section B). The precedent path maps each AR title
    # from comparable (same-SIC, modern) reports onto its nearest Best Practice item;
    # the semantic fallback embeds the facility description directly against the
    # 115-item bp_reference when precedent is thin (fewer than checklist_min_reports
    # modern reports matched). All four are PLACEHOLDERS pending real-data calibration
    # against a few worked facility descriptions.
    bp_precedent_k: int = 2  # BP items considered per precedent AR title
    bp_precedent_min_relevance: float = 0.55
    bp_semantic_k: int = 40  # BP items pulled for the semantic fallback
    bp_semantic_min_relevance: float = 0.45
    bp_per_category: int = 5  # cap on semantic candidates kept per category
    # BP write-up ("write me the LED lighting best practice"): the single best
    # bp_reference item must clear this to be drafted from its official template;
    # below it, the request falls through to a normal answer. Calibrated 2026-07-19
    # on the 115-item collection: three real topics (LED lighting, compressed air
    # leaks, solar panels) scored 0.74-0.79 on their best hit while a nonsense topic
    # ("teleportation devices") topped out at 0.56. 0.65 sits in that gap. Revisit
    # with more worked examples.
    bp_writeup_min_relevance: float = 0.65
    # Enumerate mode ("which reports mention X"): cast a wide net, then dedup by
    # report and filter with a query-adaptive floor. This is CHUNKS retrieved, not
    # reports -- a report contributes several chunks, and the floor (not k) is what
    # bounds relevance, so k only needs to be large enough that it never truncates
    # before the floor cuts in. A common AR title (e.g. "Repair Compressed Air
    # Leaks") recurs in 100+ reports all scoring within ~0.01 of each other; at the
    # old k=20 Chroma returned an arbitrary 20 of those near-ties and the recent
    # reports fell outside the cut, so enumerate looked like it only had old ones.
    # k=400 clears that: the most common topic surfaces ~140 distinct reports, all
    # the genuinely-matching recent ones included, and off-topic chunks past the
    # floor are still discarded. Cheap for HNSW over 27k chunks.
    rag_enumerate_k: int = 400
    # Absolute sanity floor: a query whose BEST hit is below this has no real match
    # and enumerate reports "none found". The per-query floor is then anchored to the
    # top hit (see _apply_enumerate_floor), because nomic cosine scores are compressed
    # and query-dependent, so one fixed floor cannot both surface a genuine topic and
    # reject nonsense. Recalibrated 2026-07-23 on the ingested corpus (27k chunks)
    # over the BARE topic phrase (the runtime query now strips the enumerate
    # boilerplate before embedding): genuine topics top 0.65-0.81 ("double paned
    # windows" 0.695, forklifts 0.761, solar panel cleaning 0.765, compressed air
    # 0.807), while off-topic queries top below ("color of the sky" 0.587, "quantum
    # teleportation of bananas" 0.531). 0.60 sits in that gap. Earlier this was a
    # fixed 0.72 floor calibrated on one bare query, which buried every genuine topic
    # scoring below it (double paned windows returned nothing). Revisit with more
    # worked topics; err toward recall since a stray weak report is cheaper here than
    # a missed one.
    rag_enumerate_min_relevance: float = 0.60
    # Width of the adaptive band kept below the top hit: on a strong query (top ~0.80)
    # this prunes a weak 0.62 tail; on a borderline one it keeps the whole cluster.
    rag_enumerate_band: float = 0.10
    # How many reports the enumerate list keeps after the newest-first sort. The
    # relevance floor bounds relevance but not trustworthiness: only 172 of the 474
    # ingested reports carry a fiscal year (the 1994-2006 eras have none), so past the
    # newest handful the list is near-ties on score in arbitrary order, which is what
    # made rows beyond the top few read as junk. 5 keeps the window where "newest
    # first" is a real ordering. Accuracy over quantity -- raise only if users start
    # asking for reports the list demonstrably has but is cutting.
    rag_enumerate_max_reports: int = 5
    # On-site data collection ("what do I measure for compressed air leaks"). Searches
    # AR-section chunks for past write-ups of the same recommendation, then counts the
    # input parameters they recorded. See rag/fielddata.py.
    #
    # k is CHUNKS scanned before dedup to distinct write-ups; a common AR contributes
    # several chunks each, so it must be well above `field_max_instances`.
    rag_field_k: int = 250
    # Absolute floor a write-up must clear to count as the same recommendation.
    # Calibrated 2026-08-15 on the ingested corpus: genuine same-AR write-ups score
    # 0.71-0.85 ("repair compressed air leaks" tops 0.854, "insulate bare equipment"
    # 0.797, "install plug load timers" 0.744), and the first off-topic neighbour
    # ("Install Skylights" under an insulation query) sits at 0.635.
    rag_field_min_relevance: float = 0.62
    # Adaptive band below the top match, as in enumerate. Load-bearing: at a fixed
    # floor an insulation query swept in skylight and scrap-metal ARs whose tables
    # then contributed parameters from the wrong recommendation.
    rag_field_band: float = 0.10
    # How many past write-ups are read for parameters. 12 is plenty of evidence for a
    # recurring AR and bounds the per-request Chroma fetches.
    field_max_instances: int = 12
    # A parameter must appear in at least this many distinct write-ups to be listed, so
    # one assessor's idiosyncratic table row never becomes an instruction. Relaxes to 1
    # (flagged low-support) when too few clear the bar.
    field_min_support: int = 2
    field_top_parameters: int = 12
    # Title gate: whether a candidate write-up's AR title is the same recommendation as
    # the topic (see fielddata.title_matches). Calibrated 2026-09-14 on 45 topics against
    # the ingested corpus, comparing the topic and each title on the query side of nomic:
    # an identical or near-identical title scores >= 0.90 and always matches. Below that,
    # genuine paraphrases ("Install Strip Doors" for strip curtains, 0.83) and wrong
    # neighbours ("Install High Efficiency Motors" for transformers, 0.82; "Repair Steam
    # Leaks" for steam traps, 0.85) overlap completely on similarity, so the title must
    # also cover the topic's distinctive words (IDF-weighted over all AR titles) and
    # clear a combined score. These values accept every paraphrase in the calibration
    # set that shares the topic's vocabulary and reject motors-for-transformers,
    # power-strip-timers-for-exhaust-fans and boiler-coils-for-economizers outright.
    rag_field_title_exact: float = 0.90
    rag_field_title_min_coverage: float = 0.60
    rag_field_title_min_combined: float = 0.93
    rag_field_title_coverage_weight: float = 0.25

    # --- Utility-bill extraction (Session F) --------------------------------
    # Vision model that reads bill pages. Qwen2.5-VL is the calibrated default;
    # it must be a model Ollama reports with vision capability.
    bills_vlm_model: str = "qwen2.5vl:latest"
    # Text-only model for page checks that need no image. Blank reuses the VLM so
    # only one model occupies VRAM.
    bills_text_model: str = ""
    # Context sizing. Raised per request up to the max for long bills.
    bills_num_ctx: int = 16384
    bills_max_num_ctx: int = 32768
    bills_timeout_seconds: float = 900.0
    bills_keep_alive: str = "15m"
    # OCR (RapidOCR on CPU) feeds exact characters alongside the page image.
    bills_ocr: bool = True
    bills_ocr_dpi: int = 300
    bills_ocr_min_score: float = 0.5
    bills_image_max_side: int = 1344
    bills_image_token_budget: int = 6000
    # Attempts per request when the reply is not valid JSON for the schema.
    bills_max_attempts: int = 2
    # Working root for bill jobs: uploads, page/extraction caches and workbooks.
    # Inside Docker this resolves under the mounted ./data volume, never the image.
    bills_path: str = "./data/bills"
    # Reconciliation tolerance in dollars. Real bills round to the cent, so an exact
    # equality test flags healthy bills for review; see validation_checks.
    bills_amount_tolerance: float = 0.02

    # Pre-build the chat/retrieval singletons at startup (see chat.orchestrator.warmup).
    # That puts torch, the embedding model and three Chroma collections in memory
    # before anything asks for them, which a bill-only run never uses and which
    # competes with the VLM for the same GPU. Turn it off on a constrained machine:
    # the first chat turn then pays the load cost instead.
    warmup_on_startup: bool = True

    @field_validator(
        "llm_provider",
        "gemini_model",
        "ollama_model",
        "anthropic_model",
        "router_model_override",
        mode="before",
    )
    @classmethod
    def _strip_inline_comment(cls, v: object) -> object:
        """Defensively drop stray inline `#` comments / whitespace from model names.

        python-dotenv does not strip an inline comment when the value before it
        is empty (e.g. `ROUTER_MODEL_OVERRIDE=   # note`), which would otherwise
        smuggle the comment text in as a model name and 404 the API call.
        """
        if isinstance(v, str):
            v = v.split("#", 1)[0].strip()
        return v

    @property
    def active_model(self) -> str:
        """The model name for the currently selected provider."""
        return {
            "gemini": self.gemini_model,
            "ollama": self.ollama_model,
            "anthropic": self.anthropic_model,
        }.get(self.llm_provider.lower(), "")


# Hard ceiling on how many results any mode LISTS back to the user: report lists,
# source cards, AR themes, ARC candidates, field parameters, best-practice items. A
# presentation-layer cap only -- retrieval depth and relevance scoring are unchanged,
# so a mode can still read more evidence than it shows (e.g. enumerate scans 400
# chunks). Deliberately a constant, not a Setting: no .env value may lift it, since a
# 100+ row reply was the bug it exists to prevent.
MAX_LISTED_RESULTS = 10


@lru_cache
def get_settings() -> Settings:
    return Settings()


def repo_path(rel_or_abs: str) -> Path:
    """Resolve a config path to an absolute one, anchoring relatives at the repo root.

    The gitignored `data/` tree lives at the repo root, but tools run from various
    working directories (uvicorn from backend/, `python -m rag.ingest` likewise).
    Anchoring relative paths here keeps the chroma DB and report corpus pointing at
    the same place regardless of cwd.
    """
    p = Path(rel_or_abs)
    return p if p.is_absolute() else (_REPO_ROOT / p)

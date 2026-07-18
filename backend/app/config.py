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
    # Weakness floor for "no confident ARC match". Calibrated 2026-07-17 on the
    # 579-code arc_reference collection: the four intended lookups scored
    # 0.662-0.819 on their best hit, while nonsense queries ("color of the sky",
    # "schedule a meeting") topped out at 0.632. 0.65 sits in that gap, so real
    # lookups clear it and nonsense falls below and is flagged as a guess.
    rag_arc_min_relevance: float = 0.65
    # Enumerate mode ("which reports mention X"): cast a wider net than answer
    # mode, then dedup by report and filter with its own similarity floor.
    rag_enumerate_k: int = 20
    # Calibrated 2026-07-17 on the ingested corpus (26k chunks) with the query
    # "solar panel cleaning": genuine Clean Solar Panels chunks scored
    # 0.738-0.765 across 6 reports (incl. ground truth AS0494/AS0552); the first
    # off-topic hit (a solar *install* AR) scored 0.683. 0.72 sits in that gap.
    rag_enumerate_min_relevance: float = 0.72

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

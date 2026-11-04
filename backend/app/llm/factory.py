"""Provider factory (spec §4.2).

`get_provider()` returns a singleton for the configured LLM_PROVIDER.
`get_router_provider()` returns the ROUTER_MODEL_OVERRIDE variant (same
provider, cheaper model) if set, else the main provider.
"""
from __future__ import annotations

from functools import lru_cache

from ..config import get_settings
from .base import LLMProvider, LLMError
from .gemini import GeminiProvider
from .ollama import OllamaProvider
from .openai import OpenAIProvider


def _build(provider_name: str, model_override: str | None = None) -> LLMProvider:
    name = provider_name.lower()
    s = get_settings()
    if name == "openai":
        return OpenAIProvider(api_key=s.resolved_openai_api_key,
                              model=model_override or s.resolved_openai_model,
                              reasoning_effort=s.openai_reasoning_effort)
    if name == "gemini":
        return GeminiProvider(api_key=s.gemini_api_key, model=model_override or s.gemini_model)
    if name == "ollama":
        return OllamaProvider(base_url=s.ollama_base_url, model=model_override or s.ollama_model)
    if name == "anthropic":
        raise LLMError(
            "Anthropic provider is not implemented yet (Session A ships gemini + ollama).",
            retryable=False,
        )
    raise LLMError(f"Unknown LLM_PROVIDER: {provider_name!r}", retryable=False)


@lru_cache
def get_provider() -> LLMProvider:
    return _build(get_settings().llm_provider)


@lru_cache
def get_router_provider() -> LLMProvider:
    s = get_settings()
    if s.router_model_override:
        return _build(s.llm_provider, model_override=s.router_model_override)
    return get_provider()

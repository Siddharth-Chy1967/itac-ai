"""Which model reads a bill, behind one small interface.

    provider.extract(images, ocr_text, schema, instructions=...) -> dict

`images` are the page renders (PNG bytes, in page order), `ocr_text` is the
deterministic page text, `schema` is the Pydantic model the answer must match.
The provider only transports: it never interprets, repairs or fills in a value.
Validation against the schema and against the page happens in the caller.

Three implementations, chosen by BILLS_LLM_PROVIDER in .env:

    ollama             local Ollama (qwen2.5vl); the existing LocalInferenceClient
    openai             OpenAI Responses API with strict JSON-schema output
    openai_compatible  any /chat/completions endpoint (OpenRouter, vLLM, LM Studio)
                       at BILLS_LLM_BASE_URL, with a json_schema response_format

Results are cached per `cache_label`, which names both the provider and the model,
so a switch never reads back another model's answers.
"""
from __future__ import annotations

import base64
import json
import logging
import math
import re
import struct
import time
from dataclasses import dataclass
from typing import Sequence, TypeVar

from pydantic import BaseModel, ValidationError

from .bill_schema import strict_json_schema
from .local_inference import (
    LOCAL_INPUT_RULES, LocalInferenceClient, LocalInferenceConfig, PageMaterial,
    load_pdf_pages,
)

logger = logging.getLogger(__name__)
Model = TypeVar("Model", bound=BaseModel)

OLLAMA, OPENAI, OPENAI_COMPATIBLE = "ollama", "openai", "openai_compatible"
PROVIDERS = (OLLAMA, OPENAI, OPENAI_COMPATIBLE)
DEFAULT_MODELS = {OPENAI: "gpt-6-luna"}
DEFAULT_BASE_URLS = {OPENAI: "https://api.openai.com/v1"}
REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")
# Longest side of a page render sent to a hosted model. OpenAI "high" detail fits
# an image inside 2048x2048, so rendering larger only costs upload time.
HOSTED_IMAGE_SIDE = 2048


class ProviderError(RuntimeError):
    """The model could not be reached or returned no usable answer."""


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-._")
    if not slug:
        raise ValueError(f"Invalid model name: {text!r}")
    return slug


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model: str
    api_key: str = ""
    base_url: str = ""
    reasoning_effort: str = "medium"
    timeout_seconds: float = 900.0
    max_attempts: int = 3
    image_detail: str = "high"

    @classmethod
    def from_settings(cls, settings) -> "ProviderConfig":
        provider = (settings.bills_llm_provider or OLLAMA).strip().lower().replace("-", "_")
        if provider not in PROVIDERS:
            raise ValueError(f"BILLS_LLM_PROVIDER must be one of {', '.join(PROVIDERS)}; "
                             f"got {settings.bills_llm_provider!r}")
        model = settings.bills_llm_model.strip()
        if not model:
            model = settings.bills_vlm_model if provider == OLLAMA else DEFAULT_MODELS.get(provider, "")
        if not model:
            raise ValueError(f"BILLS_LLM_MODEL is required for provider {provider!r}")
        base_url = settings.bills_llm_base_url.strip() or DEFAULT_BASE_URLS.get(provider, "")
        if provider == OPENAI_COMPATIBLE and not base_url:
            raise ValueError("BILLS_LLM_BASE_URL is required for provider 'openai_compatible'")
        effort = (settings.bills_llm_reasoning_effort or "").strip().lower()
        if effort and effort not in REASONING_EFFORTS:
            raise ValueError(f"BILLS_LLM_REASONING_EFFORT must be one of {', '.join(REASONING_EFFORTS)}")
        api_key = settings.bills_llm_api_key.strip()
        if not api_key and provider == OPENAI:
            import os

            api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        return cls(
            provider=provider, model=model, api_key=api_key,
            base_url=base_url.rstrip("/"), reasoning_effort=effort,
            timeout_seconds=settings.bills_timeout_seconds,
            max_attempts=max(1, settings.bills_max_attempts + 1),
        )

    @property
    def cache_label(self) -> str:
        """Folder name for this provider+model's cached answers."""
        if self.provider == OLLAMA:
            return f"ollama-{_slug(self.model)}"
        if self.provider == OPENAI and self.base_url == DEFAULT_BASE_URLS[OPENAI]:
            return f"openai-{_slug(self.model)}"
        host = re.sub(r"^https?://", "", self.base_url).split("/")[0]
        return f"{self.provider}-{_slug(host)}-{_slug(self.model)}"

    def describe(self) -> str:
        where = f" at {self.base_url}" if self.base_url else ""
        effort = f", reasoning {self.reasoning_effort}" if self.reasoning_effort else ""
        return f"{self.model} via {self.provider}{where}{effort}"


# --- the interface -----------------------------------------------------------


class Provider:
    """Base class. Subclasses implement `extract`."""

    config: ProviderConfig
    image_side: int = HOSTED_IMAGE_SIDE

    @property
    def cache_label(self) -> str:
        return self.config.cache_label

    def describe(self) -> str:
        return self.config.describe()

    def ensure_ready(self) -> None:
        """Fail fast, with the fix, when the model cannot be used."""

    def extract(self, images: Sequence[bytes], ocr_text: str, schema: type[BaseModel], *,
                instructions: str, max_output_tokens: int = 8192) -> dict:
        raise NotImplementedError


def image_size(png: bytes) -> tuple[int, int]:
    """(width, height) from a PNG header."""
    if png[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG image")
    return struct.unpack(">II", png[16:24])


def _data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


# --- Ollama ------------------------------------------------------------------


class OllamaProvider(Provider):
    """The local model, through the existing LocalInferenceClient transport."""

    def __init__(self, config: ProviderConfig, inference: LocalInferenceConfig):
        from dataclasses import replace

        self.config = config
        self.inference = replace(inference, model=config.model)
        self.client = LocalInferenceClient(self.inference)
        self.image_side = self.inference.image_max_side

    def ensure_ready(self) -> None:
        self.client.ensure_ready(self.config.model, vision=True)

    def extract(self, images, ocr_text, schema, *, instructions, max_output_tokens=8192) -> dict:
        tokens = 0
        for png in images:
            width, height = image_size(png)
            tokens += math.ceil(width / 28) * math.ceil(height / 28)
        result = self.client._chat(
            self.config.model, schema, instructions, ocr_text,
            [base64.b64encode(png).decode("ascii") for png in images],
            visual_tokens=tokens, max_output_tokens=max_output_tokens,
        )
        return result.model_dump(mode="json")


# --- hosted, over HTTP -------------------------------------------------------


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    calls: int = 0

    def add(self, input_tokens: int, output_tokens: int, reasoning_tokens: int = 0) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.reasoning_tokens += reasoning_tokens
        self.calls += 1


class _HttpProvider(Provider):
    path = ""

    def __init__(self, config: ProviderConfig, http=None):
        self.config = config
        self.usage = Usage()
        self._http = http

    def _client(self):
        if self._http is None:
            import httpx

            self._http = httpx.Client(timeout=self.config.timeout_seconds)
        return self._http

    def ensure_ready(self) -> None:
        if not self.config.api_key and self.config.provider == OPENAI:
            raise ProviderError(
                "BILLS_LLM_API_KEY is empty. Add your OpenAI API key to .env, or set "
                "BILLS_LLM_PROVIDER=ollama to use the local model."
            )

    def _post(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        url = self.config.base_url + self.path
        last = "no attempt made"
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                response = self._client().post(url, headers=headers, json=payload)
            except Exception as error:  # timeouts and connection resets are transient
                last = f"{type(error).__name__}: {error}"
            else:
                if response.status_code == 200:
                    return response.json()
                last = f"HTTP {response.status_code}: {response.text[:500]}"
                if response.status_code not in (408, 409, 429) and response.status_code < 500:
                    break  # a bad request or a bad key will not improve on retry
            if attempt < self.config.max_attempts:
                time.sleep(min(30, 2 ** attempt))
        raise ProviderError(f"{self.config.describe()} failed: {last}")


class OpenAIResponsesProvider(_HttpProvider):
    """OpenAI's Responses API: images + text in, strict JSON-schema text out."""

    path = "/responses"

    def payload(self, images, ocr_text, schema, instructions, max_output_tokens) -> dict:
        content = [{"type": "input_text", "text": ocr_text}]
        content += [{"type": "input_image", "image_url": _data_url(png),
                     "detail": self.config.image_detail} for png in images]
        payload = {
            "model": self.config.model,
            "instructions": instructions,
            "input": [{"role": "user", "content": content}],
            "text": {"format": {
                "type": "json_schema",
                "name": schema.__name__,
                "schema": strict_json_schema(schema),
                "strict": True,
            }},
            "max_output_tokens": max_output_tokens,
            # Client bills: nothing needs to be kept on the provider's side.
            "store": False,
        }
        if self.config.reasoning_effort:
            payload["reasoning"] = {"effort": self.config.reasoning_effort}
        if self.config.reasoning_effort == "none":
            payload["temperature"] = 0  # only accepted without reasoning
        return payload

    def extract(self, images, ocr_text, schema, *, instructions, max_output_tokens=8192) -> dict:
        # Reasoning tokens count against max_output_tokens; leave room for them.
        budget = max_output_tokens * (1 if self.config.reasoning_effort in ("", "none") else 4)
        body = self._post(self.payload(images, ocr_text, schema, instructions, budget))
        usage = body.get("usage") or {}
        self.usage.add(usage.get("input_tokens", 0), usage.get("output_tokens", 0),
                       (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0))
        if body.get("status") == "incomplete":
            reason = (body.get("incomplete_details") or {}).get("reason", "unknown")
            raise ProviderError(f"{self.config.model} stopped before finishing ({reason}).")
        texts = []
        for item in body.get("output") or []:
            if item.get("type") != "message":
                continue
            for part in item.get("content") or []:
                if part.get("type") == "refusal":
                    raise ProviderError(f"{self.config.model} refused: {part.get('refusal')}")
                if part.get("type") == "output_text":
                    texts.append(part.get("text", ""))
        if not texts:
            raise ProviderError(f"{self.config.model} returned no output text.")
        return json.loads("".join(texts))


class OpenAICompatibleProvider(_HttpProvider):
    """Any Chat Completions endpoint that accepts a json_schema response_format."""

    path = "/chat/completions"

    def payload(self, images, ocr_text, schema, instructions, max_output_tokens) -> dict:
        content = [{"type": "text", "text": ocr_text}]
        content += [{"type": "image_url", "image_url": {"url": _data_url(png),
                                                        "detail": self.config.image_detail}}
                    for png in images]
        payload = {
            "model": self.config.model,
            "messages": [{"role": "system", "content": instructions},
                         {"role": "user", "content": content}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": schema.__name__, "schema": strict_json_schema(schema), "strict": True,
            }},
            "max_tokens": max_output_tokens,
        }
        if self.config.reasoning_effort:
            payload["reasoning_effort"] = self.config.reasoning_effort
        if self.config.reasoning_effort in ("", "none"):
            payload["temperature"] = 0
        return payload

    def extract(self, images, ocr_text, schema, *, instructions, max_output_tokens=8192) -> dict:
        body = self._post(self.payload(images, ocr_text, schema, instructions,
                                       max_output_tokens * 4))
        usage = body.get("usage") or {}
        self.usage.add(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
        choices = body.get("choices") or []
        if not choices:
            raise ProviderError(f"{self.config.model} returned no choices.")
        message = choices[0].get("message") or {}
        if message.get("refusal"):
            raise ProviderError(f"{self.config.model} refused: {message['refusal']}")
        if choices[0].get("finish_reason") == "length":
            raise ProviderError(f"{self.config.model} stopped at its output limit.")
        text = message.get("content") or ""
        fenced = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", text, re.DOTALL)
        return json.loads(fenced.group(1) if fenced else text)


def make_provider(settings, *, http=None) -> Provider:
    config = ProviderConfig.from_settings(settings)
    if config.provider == OLLAMA:
        return OllamaProvider(config, LocalInferenceConfig.from_settings(settings))
    if config.provider == OPENAI:
        return OpenAIResponsesProvider(config, http=http)
    return OpenAICompatibleProvider(config, http=http)


# --- the parse_pdf / parse_text surface the grouping stage uses ----------------


class ProviderClient:
    """`parse_pdf` / `parse_text` for any provider, as LocalInferenceClient offers.

    Page grouping (bill_extraction) was written against LocalInferenceClient; this
    gives it the same surface over a hosted model. With Ollama the original client
    is used directly (see `grouping_client`).
    """

    def __init__(self, provider: Provider, inference: LocalInferenceConfig):
        self.provider = provider
        from dataclasses import replace

        # Hosted models see page images at their own resolution, not the VRAM budget.
        self.inference = replace(inference, image_max_side=provider.image_side,
                                 image_token_budget=10**9)

    @property
    def cache_label(self) -> str:
        return self.provider.cache_label

    def describe(self) -> str:
        return self.provider.describe()

    def ensure_ready(self, *args, **kwargs) -> None:
        self.provider.ensure_ready()

    def close(self) -> None:
        pass

    def parse_pdf(self, output_model: type[Model], instructions: str, prompt: str,
                  pdf_paths, *, labels=None, max_output_tokens: int = 2048) -> Model:
        pages = load_pdf_pages(pdf_paths, self.inference, labels)
        return self._parse(output_model, instructions, page_prompt(prompt, pages),
                           [page.image_png for page in pages], max_output_tokens)

    def parse_text(self, output_model: type[Model], instructions: str, text: str, *,
                   max_output_tokens: int = 2048) -> Model:
        return self._parse(output_model, instructions, text, [], max_output_tokens)

    def _parse(self, output_model, instructions, text, images, max_output_tokens):
        system = instructions.strip() + "\n\n" + LOCAL_INPUT_RULES.strip()
        answer = self.provider.extract(images, text, output_model, instructions=system,
                                       max_output_tokens=max_output_tokens)
        try:
            return output_model.model_validate(answer)
        except ValidationError as error:
            raise ProviderError(f"{self.provider.describe()} answer does not match "
                                f"{output_model.__name__}: {error}") from error


def page_prompt(prompt: str, pages: Sequence[PageMaterial]) -> str:
    sections = [prompt.strip()]
    for number, page in enumerate(pages, start=1):
        transcript = page.transcript.strip() or "(no text could be read from this page)"
        sections.append(f"=== {page.label}: image {number} of {len(pages)}; "
                        f"transcript from {page.transcript_source} ===\n{transcript}")
    return "\n\n".join(sections)


def grouping_client(provider: Provider, inference: LocalInferenceConfig):
    """The client page grouping talks to, for whichever provider is configured."""
    if isinstance(provider, OllamaProvider):
        return provider.client
    return ProviderClient(provider, inference)

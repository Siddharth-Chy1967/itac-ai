"""OpenAI provider (Responses API).

`POST https://api.openai.com/v1/responses`. Streaming is SSE whose
`response.output_text.delta` events carry the text. JSON mode via
`text.format = {"type": "json_object"}`.

By default it shares the bill reader's key and model (see Settings.resolved_openai_*).
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

import httpx

from ..models import Message
from .base import RETRY_BACKOFF_SECONDS, RETRYABLE_STATUS, LLMProvider, LLMError

logger = logging.getLogger(__name__)

_URL = "https://api.openai.com/v1/responses"
_TIMEOUT = httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=10.0)
# Reasoning tokens count against max_output_tokens. Without headroom a small
# budget (skill routing asks for a few hundred) is spent thinking and the reply
# comes back empty.
_REASONING_HEADROOM = 4000

# Running token totals for this process, for spend reporting.
USAGE = {"calls": 0, "input_tokens": 0, "cached_tokens": 0, "output_tokens": 0}


def _record(usage: dict | None) -> None:
    if not usage:
        return
    USAGE["calls"] += 1
    USAGE["input_tokens"] += usage.get("input_tokens", 0)
    USAGE["cached_tokens"] += (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
    USAGE["output_tokens"] += usage.get("output_tokens", 0)
    logger.info("openai usage: %s in, %s out", usage.get("input_tokens"), usage.get("output_tokens"))


def _output_text(body: dict) -> str:
    texts = []
    for item in body.get("output") or []:
        if item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if part.get("type") == "refusal":
                raise LLMError(f"OpenAI refused: {part.get('refusal')}", retryable=False)
            if part.get("type") == "output_text":
                texts.append(part.get("text", ""))
    return "".join(texts)


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, reasoning_effort: str = "low"):
        self._api_key = api_key
        self._model = model
        self._effort = (reasoning_effort or "").strip().lower()

    def _require_key(self) -> None:
        if not self._api_key:
            raise LLMError(
                "No OpenAI key. Set OPENAI_API_KEY (or BILLS_LLM_API_KEY) in .env, "
                "or switch LLM_PROVIDER=ollama to use the local model.",
                retryable=False,
            )

    @property
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    def _payload(self, messages: list[Message], system: str, max_tokens: int,
                 json_mode: bool, stream: bool) -> dict:
        thinking = self._effort not in ("", "none")
        payload: dict = {
            "model": self._model,
            "input": [{"role": m.role, "content": m.content} for m in messages],
            "max_output_tokens": max_tokens + (_REASONING_HEADROOM if thinking else 0),
            "store": False,
            "stream": stream,
        }
        if system:
            payload["instructions"] = system
        if self._effort:
            payload["reasoning"] = {"effort": self._effort}
        if json_mode:
            payload["text"] = {"format": {"type": "json_object"}}
            # json_object requires the word "JSON" in the input messages themselves
            # (the instructions field does not count).
            if "json" not in "".join(m.content for m in messages).lower():
                payload["input"].append({"role": "developer", "content": "Reply with JSON only."})
        return payload

    def _error(self, status: int, body: str) -> LLMError:
        return LLMError(f"OpenAI API error {status}: {body[:500]}",
                        retryable=status in RETRYABLE_STATUS, status=status)

    # ── complete ────────────────────────────────────────────────
    async def complete(
        self,
        messages: list[Message],
        system: str,
        max_tokens: int,
        json_mode: bool = False,
    ) -> str:
        self._require_key()
        payload = self._payload(messages, system, max_tokens, json_mode, stream=False)

        async def _once() -> str:
            try:
                async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                    resp = await client.post(_URL, headers=self._headers, json=payload)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                raise LLMError(f"OpenAI connection error: {e}", retryable=True) from e
            if resp.status_code >= 400:
                raise self._error(resp.status_code, resp.text)
            body = resp.json()
            _record(body.get("usage"))
            return _output_text(body)

        try:
            return await _once()
        except LLMError as e:
            if not e.retryable:
                raise
            await asyncio.sleep(RETRY_BACKOFF_SECONDS)
            return await _once()

    # ── stream ──────────────────────────────────────────────────
    async def stream(
        self,
        messages: list[Message],
        system: str,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        self._require_key()
        payload = self._payload(messages, system, max_tokens, json_mode=False, stream=True)

        # Retry only if the connection fails before any token is emitted.
        for attempt in range(2):
            emitted = 0
            try:
                async for tok in self._stream_once(payload):
                    emitted += 1
                    yield tok
                return
            except LLMError as e:
                if e.retryable and attempt == 0 and emitted == 0:
                    await asyncio.sleep(RETRY_BACKOFF_SECONDS)
                    continue
                raise

    async def _stream_once(self, payload: dict) -> AsyncIterator[str]:
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                async with client.stream("POST", _URL, headers=self._headers, json=payload) as resp:
                    if resp.status_code >= 400:
                        body = (await resp.aread()).decode("utf-8", "replace")
                        raise self._error(resp.status_code, body)
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        chunk = line[len("data:"):].strip()
                        if not chunk or chunk == "[DONE]":
                            continue
                        try:
                            event = json.loads(chunk)
                        except json.JSONDecodeError:
                            continue
                        kind = event.get("type", "")
                        if kind == "response.output_text.delta":
                            if event.get("delta"):
                                yield event["delta"]
                        elif kind == "response.completed":
                            _record((event.get("response") or {}).get("usage"))
                            return
                        elif kind in ("response.failed", "error"):
                            detail = event.get("error") or (event.get("response") or {}).get("error")
                            raise LLMError(f"OpenAI stream error: {detail}", retryable=False)
        except (httpx.TimeoutException, httpx.TransportError) as e:
            raise LLMError(f"OpenAI connection error: {e}", retryable=True) from e

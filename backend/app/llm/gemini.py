"""Gemini provider (spec §4.2, default).

REST calls to generativelanguage.googleapis.com. Streaming via SSE
(`:streamGenerateContent?alt=sse`). Native JSON mode via
`responseMimeType: application/json`.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import httpx

from ..models import Message
from .base import RETRY_BACKOFF_SECONDS, RETRYABLE_STATUS, LLMProvider, LLMError

_API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)


def _to_contents(messages: list[Message]) -> list[dict]:
    """Map our Message list to Gemini `contents` (assistant -> model)."""
    contents: list[dict] = []
    for m in messages:
        role = "model" if m.role == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": m.content}]})
    return contents


def _extract_text(candidate: dict) -> str:
    parts = candidate.get("content", {}).get("parts", []) or []
    return "".join(p.get("text", "") for p in parts)


def _check_block(payload: dict) -> None:
    """Raise if the prompt was blocked (no candidates returned)."""
    feedback = payload.get("promptFeedback", {})
    reason = feedback.get("blockReason")
    if reason:
        raise LLMError(f"Gemini blocked the request: {reason}", retryable=False)


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        self._api_key = api_key
        self._model = model

    def _require_key(self) -> None:
        if not self._api_key:
            raise LLMError(
                "GEMINI_API_KEY is not set. Add it to .env or switch LLM_PROVIDER=ollama.",
                retryable=False,
            )

    def _payload(self, messages: list[Message], system: str, max_tokens: int, json_mode: bool) -> dict:
        gen_config: dict = {"maxOutputTokens": max_tokens}
        if json_mode:
            gen_config["responseMimeType"] = "application/json"
            # Gemini 2.5 models "think" before answering, consuming output-token
            # budget. For structured JSON calls (skill routing, TableSpec) that
            # thinking is unnecessary and, under a tight budget, truncates the
            # actual JSON (finishReason=MAX_TOKENS with empty text). Disable it.
            gen_config["thinkingConfig"] = {"thinkingBudget": 0}
        payload: dict = {
            "contents": _to_contents(messages),
            "generationConfig": gen_config,
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        return payload

    def _url(self, method: str, sse: bool = False) -> str:
        suffix = f"?alt=sse&key={self._api_key}" if sse else f"?key={self._api_key}"
        return f"{_API_ROOT}/{self._model}:{method}{suffix}"

    # ── complete ────────────────────────────────────────────────
    async def complete(
        self,
        messages: list[Message],
        system: str,
        max_tokens: int,
        json_mode: bool = False,
    ) -> str:
        self._require_key()
        payload = self._payload(messages, system, max_tokens, json_mode)

        async def _once() -> str:
            try:
                async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                    resp = await client.post(self._url("generateContent"), json=payload)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                raise LLMError(f"Gemini connection error: {e}", retryable=True) from e
            if resp.status_code >= 400:
                raise LLMError(
                    f"Gemini API error {resp.status_code}: {resp.text}",
                    retryable=resp.status_code in RETRYABLE_STATUS,
                    status=resp.status_code,
                )
            data = resp.json()
            candidates = data.get("candidates")
            if not candidates:
                _check_block(data)
                return ""
            return _extract_text(candidates[0])

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
        payload = self._payload(messages, system, max_tokens, json_mode=False)

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
                async with client.stream(
                    "POST", self._url("streamGenerateContent", sse=True), json=payload
                ) as resp:
                    if resp.status_code >= 400:
                        body = (await resp.aread()).decode("utf-8", "replace")
                        raise LLMError(
                            f"Gemini API error {resp.status_code}: {body}",
                            retryable=resp.status_code in RETRYABLE_STATUS,
                            status=resp.status_code,
                        )
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        chunk = line[len("data:"):].strip()
                        if not chunk or chunk == "[DONE]":
                            continue
                        try:
                            data = json.loads(chunk)
                        except json.JSONDecodeError:
                            continue
                        candidates = data.get("candidates")
                        if not candidates:
                            _check_block(data)
                            continue
                        text = _extract_text(candidates[0])
                        if text:
                            yield text
        except (httpx.TimeoutException, httpx.TransportError) as e:
            raise LLMError(f"Gemini connection error: {e}", retryable=True) from e

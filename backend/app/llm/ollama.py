"""Ollama provider (spec §4.2, fully local/free).

`POST {base_url}/api/chat`. Streaming is newline-delimited JSON. JSON mode
via `format: json`.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import httpx

from ..models import Message
from .base import RETRY_BACKOFF_SECONDS, RETRYABLE_STATUS, LLMProvider, LLMError

_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)


def _to_messages(messages: list[Message], system: str) -> list[dict]:
    out: list[dict] = []
    if system:
        out.append({"role": "system", "content": system})
    out.extend({"role": m.role, "content": m.content} for m in messages)
    return out


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str, model: str):
        self._base_url = base_url.rstrip("/")
        self._model = model

    def _payload(self, messages: list[Message], system: str, max_tokens: int,
                 json_mode: bool, stream: bool) -> dict:
        payload: dict = {
            "model": self._model,
            "messages": _to_messages(messages, system),
            "stream": stream,
            "options": {"num_predict": max_tokens},
        }
        if json_mode:
            payload["format"] = "json"
        return payload

    @property
    def _url(self) -> str:
        return f"{self._base_url}/api/chat"

    # ── complete ────────────────────────────────────────────────
    async def complete(
        self,
        messages: list[Message],
        system: str,
        max_tokens: int,
        json_mode: bool = False,
    ) -> str:
        payload = self._payload(messages, system, max_tokens, json_mode, stream=False)

        async def _once() -> str:
            try:
                async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                    resp = await client.post(self._url, json=payload)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                raise LLMError(
                    f"Ollama connection error (is `ollama serve` running at {self._base_url}?): {e}",
                    retryable=True,
                ) from e
            if resp.status_code >= 400:
                raise LLMError(
                    f"Ollama API error {resp.status_code}: {resp.text}",
                    retryable=resp.status_code in RETRYABLE_STATUS,
                    status=resp.status_code,
                )
            return resp.json().get("message", {}).get("content", "")

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
        payload = self._payload(messages, system, max_tokens, json_mode=False, stream=True)

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
                async with client.stream("POST", self._url, json=payload) as resp:
                    if resp.status_code >= 400:
                        body = (await resp.aread()).decode("utf-8", "replace")
                        raise LLMError(
                            f"Ollama API error {resp.status_code}: {body}",
                            retryable=resp.status_code in RETRYABLE_STATUS,
                            status=resp.status_code,
                        )
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if data.get("error"):
                            raise LLMError(f"Ollama error: {data['error']}", retryable=False)
                        text = data.get("message", {}).get("content", "")
                        if text:
                            yield text
                        if data.get("done"):
                            return
        except (httpx.TimeoutException, httpx.TransportError) as e:
            raise LLMError(
                f"Ollama connection error (is `ollama serve` running at {self._base_url}?): {e}",
                retryable=True,
            ) from e

"""Abstract LLM provider interface (spec §4.1).

One abstract interface, multiple implementations. All providers normalize
errors to a common `LLMError` carrying a `retryable` flag; callers retry once
on transient failures with a short backoff.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TypeVar

from ..models import Message

RETRY_BACKOFF_SECONDS = 0.5

# HTTP status codes that indicate a transient, retryable failure.
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})

T = TypeVar("T")


class LLMError(Exception):
    """Normalized error raised by every provider."""

    def __init__(self, message: str, *, retryable: bool = False, status: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


async def with_retry(factory: Callable[[], Awaitable[T]]) -> T:
    """Await `factory()`; on a retryable LLMError, retry once after a short backoff."""
    try:
        return await factory()
    except LLMError as e:
        if not e.retryable:
            raise
        await asyncio.sleep(RETRY_BACKOFF_SECONDS)
        return await factory()


class LLMProvider(ABC):
    """Provider-agnostic LLM interface. Signatures match spec §4.1 exactly."""

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        system: str,
        max_tokens: int,
        json_mode: bool = False,
    ) -> str:
        """Return a single completion string.

        `json_mode=True` instructs/configures the model to emit only JSON.
        """
        ...

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        system: str,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        """Yield completion text deltas as they arrive."""
        ...

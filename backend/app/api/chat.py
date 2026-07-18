"""POST /api/chat with SSE streaming (spec §5.4).

Thin HTTP layer: it delegates the whole pipeline (skill routing, prompt
assembly, streaming completion) to `chat.orchestrator.run_chat` and serializes
each yielded event as SSE. RAG `sources` and xlsx `file` events will flow
through automatically once the orchestrator emits them.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..chat.orchestrator import run_chat
from ..models import ChatRequest

router = APIRouter()


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    async def event_stream() -> AsyncIterator[str]:
        async for ev in run_chat(req.message, req.history, req.mode):
            yield _sse(ev.event, ev.data)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable proxy buffering (e.g. nginx)
        },
    )

"""GET /api/health (spec §2)."""
from __future__ import annotations

from fastapi import APIRouter

from ..config import get_settings

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "provider": s.llm_provider,
        "model": s.active_model,
    }

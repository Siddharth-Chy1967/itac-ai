"""FastAPI app factory: CORS + router mounting (spec §2).

Session A mounts /api/health and /api/chat. Session B adds /api/skills.
Session C adds /api/export (xlsx download). Later sessions add the ingest router.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import chat, export, health, skills

# Vite/React dev origins.
DEV_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
]


def create_app() -> FastAPI:
    app = FastAPI(title="ITAC AI Assistant", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=DEV_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/api")
    app.include_router(chat.router, prefix="/api")
    app.include_router(skills.router, prefix="/api")
    app.include_router(export.router, prefix="/api")
    return app


app = create_app()

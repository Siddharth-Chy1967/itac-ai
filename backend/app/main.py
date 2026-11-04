"""FastAPI app factory: CORS + router mounting (spec §2).

Session A mounts /api/health and /api/chat. Session B adds /api/skills.
Session C adds /api/export (xlsx download). Later sessions add the ingest router.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import bills, chat, export, health, skills
from .chat.orchestrator import warmup
from .config import get_settings

logger = logging.getLogger(__name__)

# Vite/React dev origins.
DEV_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
]


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Kick off model/store warmup in the background at startup.

    The task is fire-and-forget so it does not block the server from accepting
    requests: a message sent in the first few seconds still lazy-loads on the
    request path, but by the time the user has typed anything the singletons are
    usually already built, so the first real turn is fast. A reference is held on
    `app.state` so the task is not garbage-collected mid-run.
    """
    if get_settings().warmup_on_startup:
        app.state.warmup_task = asyncio.create_task(warmup())
    else:
        logger.info("Warmup disabled; chat singletons load on first use.")
    # The bill worker must be created on the running loop, and stopped with it, so
    # a batch can never outlive the app that is reporting its progress.
    store = bills.get_store()
    store.start()
    try:
        yield
    finally:
        await store.stop()


def create_app() -> FastAPI:
    app = FastAPI(title="ARchie (ITAC AI)", version="0.1.0", lifespan=_lifespan)

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
    app.include_router(bills.router, prefix="/api")
    return app


app = create_app()

"""Shared request/response models (spec §4.1, §5.4)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["user", "assistant"]


class Message(BaseModel):
    """A single conversation turn. `Message = {role, content}` per spec §4.1."""

    role: Role
    content: str


class ChatRequest(BaseModel):
    """Body of POST /api/chat (spec §5.4)."""

    message: str
    history: list[Message] = Field(default_factory=list)
    mode: Literal["chat", "notes"] = "chat"

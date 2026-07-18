"""Skills API (spec §2, §9).

- GET /api/skills          -> [{name, description}] for the Skills browser.
- GET /api/skills/{name}   -> {name, description, body} full markdown viewer.

The registry is loaded once at startup by `skills.loader.get_registry()`.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..skills.loader import get_registry

router = APIRouter()


class SkillSummary(BaseModel):
    name: str
    description: str


class SkillDetail(BaseModel):
    name: str
    description: str
    body: str


@router.get("/skills")
async def list_skills() -> list[SkillSummary]:
    registry = get_registry()
    return [
        SkillSummary(name=s.name, description=s.description)
        for s in registry.values()
    ]


@router.get("/skills/{name}")
async def get_skill(name: str) -> SkillDetail:
    skill = get_registry().get(name)
    if skill is None:
        raise HTTPException(status_code=404, detail=f"Unknown skill: {name}")
    return SkillDetail(name=skill.name, description=skill.description, body=skill.body)

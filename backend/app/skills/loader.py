"""Skill registry loader (spec §8).

Parses the markdown skill files in `itac-ai/skills/` into a registry of
`{name, description, body, path}`. Each file mirrors the Anthropic skill
format: a YAML frontmatter block delimited by `---` lines, followed by the
markdown body.

The loader is deliberately tolerant: a file with no frontmatter, empty
content, or a missing `name`/`description` is skipped with a warning rather
than crashing startup (the four new skill files ship empty until Siddharth
writes them, per spec §12). Duplicate `name` values are a hard error, since
they would make routing ambiguous.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# loader.py -> skills -> app -> backend -> itac-ai (repo root)
_REPO_ROOT = Path(__file__).resolve().parents[3]
SKILLS_DIR = _REPO_ROOT / "skills"

_FRONTMATTER_DELIM = "---"


class Skill(BaseModel):
    """A loaded skill file (spec §8)."""

    name: str
    description: str
    body: str
    path: str


def _lenient_frontmatter(raw_fm: str) -> dict:
    """Best-effort extraction of `name` and `description` without strict YAML.

    Skill descriptions are prose and routinely contain `: `, quotes, and other
    characters that break unquoted YAML scalars. When `yaml.safe_load` fails
    (or drops these keys), we recover them by hand: read `name`/`description`
    line by line, supporting both a single-line value and a folded/literal
    block (`description: >` or `|` followed by indented lines).
    """
    result: dict[str, str] = {}
    lines = raw_fm.splitlines()
    i = 0
    key_re = re.compile(r"^(?P<key>[A-Za-z0-9_-]+):(?P<rest>.*)$")

    while i < len(lines):
        m = key_re.match(lines[i])
        if not m:
            i += 1
            continue
        key = m.group("key").strip().lower()
        rest = m.group("rest").strip()
        i += 1

        if rest in (">", "|", ">-", "|-", ">+", "|+"):
            # Folded/literal block: gather following indented lines.
            block: list[str] = []
            while i < len(lines) and (lines[i].strip() == "" or lines[i][:1] in (" ", "\t")):
                block.append(lines[i].strip())
                i += 1
            value = " ".join(part for part in block if part)
        else:
            value = rest.strip().strip("'\"")

        if key in ("name", "description") and value:
            result[key] = value

    return result


def _split_frontmatter(text: str) -> tuple[dict, str] | None:
    """Split `---`-delimited frontmatter from the markdown body.

    Returns `(frontmatter_dict, body)` or `None` if the file has no
    frontmatter block. Tries strict YAML first, then falls back to a lenient
    line parser so prose descriptions with `: `/quotes still load.
    """
    stripped = text.lstrip("﻿")  # tolerate a UTF-8 BOM
    if not stripped.lstrip().startswith(_FRONTMATTER_DELIM):
        return None

    # Drop everything before the opening delimiter, then split on the closing one.
    after_open = stripped.split(_FRONTMATTER_DELIM, 1)[1]
    if _FRONTMATTER_DELIM not in after_open:
        return None
    raw_fm, body = after_open.split(_FRONTMATTER_DELIM, 1)
    body = body.lstrip("\n")

    data: dict = {}
    try:
        parsed = yaml.safe_load(raw_fm)
        if isinstance(parsed, dict):
            data = parsed
    except yaml.YAMLError:
        data = {}

    # Recover missing name/description via the lenient parser.
    if not data.get("name") or not data.get("description"):
        lenient = _lenient_frontmatter(raw_fm)
        for key in ("name", "description"):
            if not data.get(key) and lenient.get(key):
                data[key] = lenient[key]

    return data, body


def _parse_skill_file(path: Path) -> Skill | None:
    """Parse one skill file into a `Skill`, or `None` if it is not usable."""
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        logger.warning("Skipping empty skill file: %s", path.name)
        return None

    parsed = _split_frontmatter(text)
    if parsed is None:
        logger.warning("Skipping skill file with no valid frontmatter: %s", path.name)
        return None

    fm, body = parsed
    name = fm.get("name")
    description = fm.get("description")
    if not name or not description:
        logger.warning(
            "Skipping skill file missing name/description in frontmatter: %s", path.name
        )
        return None

    return Skill(
        name=str(name).strip(),
        description=str(description).strip(),
        body=body.strip(),
        path=str(path),
    )


def load_skills(skills_dir: Path | None = None) -> list[Skill]:
    """Load and parse every `*.md` skill file in `skills_dir`.

    Raises `ValueError` on a duplicate skill `name`. Files that cannot be
    parsed are skipped with a warning.
    """
    directory = skills_dir or SKILLS_DIR
    skills: list[Skill] = []
    seen: dict[str, str] = {}

    if not directory.is_dir():
        logger.warning("Skills directory does not exist: %s", directory)
        return skills

    for path in sorted(directory.glob("*.md")):
        skill = _parse_skill_file(path)
        if skill is None:
            continue
        if skill.name in seen:
            raise ValueError(
                f"Duplicate skill name {skill.name!r}: {path.name} and {seen[skill.name]}"
            )
        seen[skill.name] = path.name
        skills.append(skill)

    logger.info("Loaded %d skill(s) from %s", len(skills), directory)
    return skills


@lru_cache
def get_registry() -> dict[str, Skill]:
    """Startup-cached registry keyed by skill name."""
    return {skill.name: skill for skill in load_skills()}

"""Offline SIC / NAICS names, so ARchie never guesses what an industry code means.

The tables under app/data are built by scripts/build_industry_codes.py from the 1987
SIC manual (OSHA list) and the Census NAICS concordances. Looking a code up is a dict
read: no network, no model, no cost per question.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_DATA = Path(__file__).resolve().parents[1] / "data"


@dataclass(frozen=True)
class SicEntry:
    sic4: str
    sic4_name: str
    sic3: str
    sic3_name: str
    sic2: str
    sic2_name: str
    division: str
    division_name: str


@lru_cache
def _sic_table() -> dict[str, SicEntry]:
    with open(_DATA / "sic_codes.csv", newline="", encoding="utf-8") as f:
        return {row["sic4"]: SicEntry(**row) for row in csv.DictReader(f)}


@lru_cache
def _naics_table() -> dict[str, tuple[str, list[str]]]:
    with open(_DATA / "naics_to_sic.csv", newline="", encoding="utf-8") as f:
        return {row["naics"]: (row["naics_title"], row["sic4"].split())
                for row in csv.DictReader(f)}


def _digits(code: str) -> str:
    return re.sub(r"\D", "", code or "")


def sic_entry(code: str) -> SicEntry | None:
    """The exact 4-digit SIC, or None."""
    digits = _digits(code)
    return _sic_table().get(digits) if len(digits) == 4 else None


def _group_names(digits: str) -> tuple[str, str, str]:
    """(industry group, major group, division) names for a 2-4 digit prefix."""
    for entry in _sic_table().values():
        if entry.sic4.startswith(digits[:2]):
            group = ""
            if len(digits) >= 3:
                match = next((e for e in _sic_table().values() if e.sic3 == digits[:3]), None)
                group = match.sic3_name if match else ""
            return group, entry.sic2_name, entry.division_name
    return "", "", ""


def describe_sic(code: str) -> str | None:
    """A one-line plain description of a SIC (4-digit, 3/2-digit or "25xx" form).

    "2431" -> "SIC 2431 Millwork (group 243: Millwork, Veneer, Plywood, And Structural
    Wood; major group 24: Lumber And Wood Products, Except Furniture; Manufacturing)"
    """
    digits = _digits(code)
    if len(digits) < 2:
        return None
    if len(digits) >= 4:
        entry = _sic_table().get(digits[:4])
        if entry is None:
            group, major, division = _group_names(digits[:3])
            if not major:
                return None
            label = f"SIC {digits[:4]} (not a code in the 1987 SIC manual)"
            parts = [f"group {digits[:3]}: {group}"] if group else []
            parts += [f"major group {digits[:2]}: {major}", division]
            return f"{label}; nearest family: " + "; ".join(parts)
        return (f"SIC {entry.sic4} {entry.sic4_name} (group {entry.sic3}: {entry.sic3_name}; "
                f"major group {entry.sic2}: {entry.sic2_name}; {entry.division_name})")
    group, major, division = _group_names(digits)
    if not major:
        return None
    if len(digits) == 3 and group:
        return f"SIC group {digits}: {group} (major group {digits[:2]}: {major}; {division})"
    return f"SIC major group {digits[:2]}: {major} ({division})"


def naics_to_sic(code: str) -> tuple[str, list[str]] | None:
    """(NAICS title, [SIC codes]) for a 6-digit NAICS of any 1997-2022 vintage.

    A shorter prefix (e.g. 3219) returns every SIC its 6-digit children map to, most
    common first, titled by the first child.
    """
    digits = _digits(code)
    table = _naics_table()
    if len(digits) == 6 and digits in table:
        return table[digits]
    if 2 <= len(digits) < 6:
        counts: dict[str, int] = {}
        title = ""
        for naics, (name, sics) in table.items():
            if naics.startswith(digits):
                title = title or name
                for sic in sics:
                    counts[sic] = counts.get(sic, 0) + 1
        if counts:
            return title, sorted(counts, key=lambda s: -counts[s])
    return None

"""Meter readings rebuilt from where OCR found each box on the page.

The vision model decides well *what* a bill's readings are, and badly *which
label* each number belongs to once they sit in a table. SRP prints its readings
as a label-value list under the headings "Meter #" and "Energy":

    Meter #                                        Energy
    On Pk kWh                                      61,535
    On Pk kW            7/21/2021      15:30        528.2
    16 Month Peak Demand                            689.5

and qwen2.5vl returned every reading as label "Energy" with the row name in
`meter_number` -- or tagged one with the corner heading, or collapsed a whole
row into one string. OCR has none of those problems: it reads each box's text
exactly and says where the box is. A label and its value share a printed line,
so the pairing is geometry, not judgement.

So the page is the authority on which label goes with which number:

1. `printed_rows` rebuilds label-value rows from the OCR boxes of each page.
2. `reconcile` gives every reading the model returned the label of the printed
   row carrying the same number, and adds readings the model dropped or mangled
   when their printed row names a unit or lines up with rows that do.

Nothing here reads a number the page does not print, and no model is involved:
the numbers come from OCR, the pairing from position, and the meaning of each
label is still decided later by `semantics`.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .extraction_models import Measurement
from .local_inference import LocalInferenceConfig, OcrItem, ocr_lines, ocr_page_items

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PrintedRow:
    """One printed line holding a label and exactly one reading."""

    label: str
    value_text: str
    unit: str | None
    page: int
    # Where the value box ends and how tall it is, in page pixels: readings in one
    # printed block share a right edge (numbers are right-aligned).
    right: float = 0.0
    height: float = 0.0


# --- reading the page -------------------------------------------------------

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
# A value box: one number, optionally with a percent sign or a unit after it.
_VALUE = re.compile(r"^\(?-?\d[\d,]*(?:\.\d+)?\)?\s*(?:%|[A-Za-z]{1,6})?$")
_DATE_OR_TIME = re.compile(r"\d{1,2}/\d{1,2}(?:/\d{2,4})?|^\d{1,2}:\d{2}")
# Printed units, longest spelling first so "kWh" is never read as "kW".
_UNITS = {
    "kvarh": "kVARh", "kvah": "kVAh", "kvar": "kVAR", "kva": "kVA", "kwh": "kWh",
    "mwh": "MWh", "kw": "kW", "mw": "MW", "therms": "Therms", "therm": "Therms",
    "ccf": "CCF", "mcf": "MCF", "mmbtu": "MMBtu", "dth": "Dth", "tgals": "TGals",
    "tgal": "TGals", "gallons": "Gallons", "gallon": "Gallons", "gal": "Gallons",
}
_UNIT = re.compile(
    r"(?<![A-Za-z])(" + "|".join(sorted(_UNITS, key=len, reverse=True)) + r")(?![A-Za-z])",
    re.IGNORECASE,
)


def _is_value(text: str) -> bool:
    return "$" not in text and not _DATE_OR_TIME.search(text) and bool(_VALUE.match(text))


def _is_label(text: str) -> bool:
    return len(re.findall(r"[A-Za-z]", text)) >= 2 and not _is_value(text)


def unit_in(text: str | None) -> str | None:
    if not text:
        return None
    if "%" in text:
        return "%"
    match = _UNIT.search(text)
    return _UNITS[match.group(1).casefold()] if match else None


def rows_from_items(items: list[OcrItem], page: int) -> list[PrintedRow]:
    """Label-value rows on one page.

    A row is a printed line with exactly one reading and a label to its left;
    anything between them that is a date or a time (when a demand peak was set)
    is ignored. A line with two or more readings is a real multi-column table
    whose columns cannot be told apart from one line, so it is left to the model
    rather than guessed at.
    """
    rows = []
    for line in ocr_lines(items):
        values = [item for item in line if _is_value(item[4])]
        if len(values) != 1:
            continue
        value = values[0]
        label = " ".join(item[4] for item in line if item[3] <= value[2] and _is_label(item[4]))
        if not label:
            continue
        rows.append(PrintedRow(label=label, value_text=value[4],
                               unit=unit_in(value[4]) or unit_in(label), page=page,
                               right=value[3], height=value[1] - value[0]))
    return rows


def printed_rows(pdf_path: Path, config: LocalInferenceConfig) -> list[PrintedRow]:
    """Every label-value row in a bill PDF, pages in order."""
    import pymupdf

    rows: list[PrintedRow] = []
    with pymupdf.open(pdf_path) as document:
        for number, page in enumerate(document, start=1):
            rows.extend(rows_from_items(ocr_page_items(page, config), number))
    return rows


# --- reconciling the model's readings with the page --------------------------


def _number(text: str | None) -> Decimal | None:
    """The one number a value prints, or None if it prints none or several."""
    found = _NUMBER.findall(text or "")
    if len(found) != 1:
        return None
    try:
        return Decimal(found[0].replace(",", ""))
    except InvalidOperation:
        return None


def _norm(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").casefold()).strip()


def reconcile(bill, rows: list[PrintedRow]):
    """The bill with its readings labelled as the page prints them.

    Returns a copy; the extraction itself (and its cache) is never changed.
    """
    if not rows:
        return bill
    used: set[int] = set()
    readings = []
    notes = []

    for measurement in bill.measurements:
        text = measurement.value_text or ""
        if "$" in text:
            readings.append(measurement)  # a price; shape leaves it out
            continue
        number = _number(text)
        if number is None:
            if len(_NUMBER.findall(text)) > 1:
                # A row collapsed into one string ("On Pk kWh: 273.7, Shoulder...").
                # Its figures are recovered from the page below, not parsed here.
                notes.append(f"Grid: dropped collapsed reading {measurement.label!r}: {text!r}")
                continue
            readings.append(measurement)
            continue
        candidates = [i for i, row in enumerate(rows)
                      if i not in used and _number(row.value_text) == number]
        if not candidates:
            readings.append(measurement)
            continue
        # Several printed rows can share a number (a month's on-peak demand that
        # set the 16-month peak). Prefer the one the model already named, in
        # either field; otherwise take them in page order, as the model does.
        given = {_norm(measurement.label), _norm(measurement.meter_number)}
        index = next((i for i in candidates if _norm(rows[i].label) in given), candidates[0])
        used.add(index)
        row = rows[index]
        if _norm(row.label) == _norm(measurement.label):
            if measurement.unit is None and row.unit is not None:
                measurement = measurement.model_copy(update={"unit": row.unit})
            readings.append(measurement)
            continue
        readings.append(measurement.model_copy(update={
            "label": row.label,
            # Same number, as printed: the model's text can carry the borrowed unit
            # too ("689.5 kVAH"), which no kW column could parse.
            "value_text": row.value_text,
            # The model's unit is suspect exactly when its label was: in the real
            # bills it borrowed the neighbouring row's (689.5 read as kVAh).
            "unit": row.unit,
            "meter_number": None if _names_a_row(measurement.meter_number, rows)
                            else measurement.meter_number,
            "bill_page_number": row.page,
        }))
        notes.append(f"Grid: {row.value_text} relabelled {measurement.label!r} -> {row.label!r}")

    # Readings the model dropped or collapsed: any printed row whose number no
    # reading carried, if it names a unit or sits in a block of rows that do.
    added = set()
    for index, row in enumerate(rows):
        key = (_norm(row.label), _number(row.value_text))
        if index in used or key in added:
            continue
        if row.unit is None and not _in_reading_block(row, rows):
            continue
        added.add(key)
        readings.append(_measurement_like(bill, row))
        notes.append(f"Grid: added {row.label!r} = {row.value_text}")

    return bill.model_copy(update={
        "measurements": readings,
        "extraction_notes": [*bill.extraction_notes, *notes],
    })


def _in_reading_block(row: PrintedRow, rows: list[PrintedRow]) -> bool:
    """True when a unitless row's value lines up with three or more readings.

    SRP prints "16 Month Peak Demand 689.5" with no unit, in the same
    right-aligned column as "On Pk kW 528.2". Alignment with rows that do carry
    a unit is what makes it a reading rather than some other figure on the page.
    """
    tolerance = max(row.height, 1.0) * 0.5
    aligned = [other for other in rows
               if other.unit is not None and other.page == row.page
               and abs(other.right - row.right) <= tolerance]
    return len(aligned) >= 3


def _names_a_row(text: str | None, rows: list[PrintedRow]) -> bool:
    """True when a "meter number" is really a row label or the grid's corner heading.

    "Meter #" and "Meter #1" are the heading and a position; "Meter 10953811" is a
    real serial and is kept.
    """
    key = _norm(text)
    return bool(key) and (key in {_norm(r.label) for r in rows}
                          or re.fullmatch(r"meter( \d{1,3})?", key) is not None)


def _measurement_like(bill, row: PrintedRow) -> Measurement:
    return Measurement(
        label=row.label, value_text=row.value_text, unit=row.unit, section=None,
        meter_number=None, measurement_date=None, measurement_time=None,
        service_period_text=None, bill_page_number=row.page,
    )

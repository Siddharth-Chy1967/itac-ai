"""The text printed on each page, with where it sits. Deterministic; no model.

Two sources, one shape. A digital PDF carries its own text layer, which is exact;
a scan has none, so RapidOCR reads the rendered page. Either way a page becomes a
list of boxes, `(top, bottom, left, right, text)`, and everything downstream --
the transcript the model is shown and the check that each extracted number is
really printed -- works from those boxes.

Chart axis labels are removed here, before anything else sees them. A bill's
usage history chart prints its y-axis as a column of evenly stepped numbers
(0, 5000, 10000 ... 45000). They are printed, but they are not bill figures, and
a label beside them ("Electricity (kWh)") once turned the tick 25000 into a
month's usage. An axis is recognised by shape alone: three or more bare numbers
sharing an x-position whose values form an even arithmetic sequence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Sequence

from .local_inference import (
    TEXT_LAYER_MIN_CHARS, LocalInferenceConfig, OcrItem, ocr_page_items, text_from_items,
)

# A box that is one number and nothing else: "25000", "5,000", "$1,000", "40%".
_BARE_NUMBER = re.compile(r"^\$?\s*(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s*%?$")


@dataclass(frozen=True)
class PageText:
    """One page's printed text, minus chart axis labels."""

    number: int                 # 1-based position in the PDF
    items: tuple[OcrItem, ...]  # boxes that remain after axis labels are removed
    axis_labels: tuple[OcrItem, ...]
    source: str                 # "PDF text layer" | "OCR" | "none"

    @property
    def text(self) -> str:
        return text_from_items(self.items)


def bare_number(text: str) -> Decimal | None:
    match = _BARE_NUMBER.match(text.strip())
    if not match:
        return None
    try:
        return Decimal(match.group(1).replace(",", "") + (match.group(2) or ""))
    except InvalidOperation:
        return None


def axis_label_indices(items: Sequence[OcrItem]) -> set[int]:
    """Indices of boxes that are chart axis tick labels.

    Three or more bare numbers that share an x-position (left edge, right edge or
    centre -- axes are printed all three ways) and, read top to bottom, step by a
    constant non-zero amount at a constant vertical spacing. A column of charges
    or readings practically never does both; an axis always does. Value alone is
    not enough: an SRP bill's $753.06, $532.76 and $312.46 step by exactly
    -$220.30, but sit on lines 55px and then 311px apart.
    """
    numbers = [(index, value) for index, item in enumerate(items)
               if (value := bare_number(item[4])) is not None]
    if len(numbers) < 3:
        return set()
    heights = sorted(item[1] - item[0] for item in items)
    tolerance = max(heights[len(heights) // 2], 1.0) * 0.6

    edges = (
        lambda item: item[3],                    # right-aligned ticks
        lambda item: item[2],                    # left-aligned
        lambda item: (item[2] + item[3]) / 2,    # centred
    )
    found: set[int] = set()
    for edge in edges:
        ordered = sorted(numbers, key=lambda pair: edge(items[pair[0]]))
        group = [ordered[0]]
        for pair in ordered[1:]:
            if edge(items[pair[0]]) - edge(items[group[-1][0]]) <= tolerance:
                group.append(pair)
            else:
                found |= _arithmetic_runs(group, items, tolerance)
                group = [pair]
        found |= _arithmetic_runs(group, items, tolerance)
    return found


def _arithmetic_runs(group, items, tolerance: float) -> set[int]:
    """Indices in runs of 3+ vertically consecutive values with a constant step,
    evenly spaced on the page (within `tolerance` of the run's first gap)."""
    if len(group) < 3:
        return set()

    def middle(pair) -> float:
        return (items[pair[0]][0] + items[pair[0]][1]) / 2

    column = sorted(group, key=middle)
    found: set[int] = set()
    start = 0
    while start < len(column) - 2:
        step = column[start + 1][1] - column[start][1]
        gap = middle(column[start + 1]) - middle(column[start])
        end = start + 1
        while (end + 1 < len(column)
               and column[end + 1][1] - column[end][1] == step
               and abs(middle(column[end + 1]) - middle(column[end]) - gap) <= tolerance):
            end += 1
        if step != 0 and end - start >= 2:
            found |= {index for index, _ in column[start:end + 1]}
            start = end
        else:
            start += 1
    return found


def split_axis_labels(items: Sequence[OcrItem]) -> tuple[list[OcrItem], list[OcrItem]]:
    """(kept boxes, axis-label boxes)."""
    axis = axis_label_indices(items)
    kept = [item for index, item in enumerate(items) if index not in axis]
    dropped = [item for index, item in enumerate(items) if index in axis]
    return kept, dropped


def text_layer_items(page) -> list[OcrItem]:
    """The PDF's own text as boxes, one per word, in the same shape OCR produces."""
    return [(y0, y1, x0, x1, word)
            for x0, y0, x1, y1, word, *_ in page.get_text("words") if word.strip()]


def read_page(page, number: int, config: LocalInferenceConfig) -> PageText:
    """One PyMuPDF page: text layer when it has one, else OCR; axis labels removed."""
    if len(page.get_text("text").strip()) >= TEXT_LAYER_MIN_CHARS:
        items, source = text_layer_items(page), "PDF text layer"
    elif config.ocr_enabled:
        items, source = ocr_page_items(page, config), "OCR"
    else:
        items, source = [], "none"
    kept, axis = split_axis_labels(items)
    return PageText(number=number, items=tuple(kept), axis_labels=tuple(axis), source=source)


def read_pdf(pdf_path: Path, config: LocalInferenceConfig) -> list[PageText]:
    import pymupdf

    with pymupdf.open(pdf_path) as document:
        return [read_page(page, number, config)
                for number, page in enumerate(document, start=1)]

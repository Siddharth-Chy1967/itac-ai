"""Synthetic digital bills and a fake model, for tests that need no client data.

`make_bill_pdf` writes a two-page bill with a real text layer (so page_text reads
it exactly and fast, with no OCR), including a usage-history chart axis on page 1.
`FakeProvider` answers like a model would: it finds which bill it was shown from
the page text and returns that bill's figures -- or whatever the test asks for.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import pymupdf

from app.bills.bill_schema import BillFigures
from app.bills.llm_providers import Provider, ProviderConfig


@dataclass
class Month:
    statement: str                       # "01/16/2026"
    on_kwh: str = "16,720.000"
    off_kwh: str = "14,320.000"
    total_kwh: str = "31,040.000"
    demand: str = "154.400"
    charges: list[tuple[str, str, str]] = field(default_factory=lambda: [
        ("On-Peak Use and Charge", "$820.76", "energy"),
        ("Off-Peak Use and Charge", "$420.55", "energy"),
        ("Customer Charge", "$74.69", "fixed"),
        ("Billed demand and charge", "$1,931.54", "demand"),
        ("Palo Verde Regulatory Credit", "-$57.13", "credit"),
        ("Gross Receipts Tax - State", "$216.22", "tax"),
    ])

    @property
    def total(self) -> str:
        amount = sum(Decimal(a.replace("$", "").replace(",", "")) for _, a, _ in self.charges)
        return f"${amount:,.2f}"


def make_bill_pdf(folder: Path, name: str, month: Month, *, total: str | None = None) -> Path:
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "PNM   Bill Issued: " + month.statement, fontsize=10)
    page.insert_text((72, 80), "Account 0000-TEST", fontsize=10)
    # A usage-history chart: y-axis ticks 0..45000, right-aligned, and its title.
    for index, tick in enumerate(range(45000, -1, -5000)):
        text = str(tick)
        width = pymupdf.get_text_length(text, fontsize=8)
        page.insert_text((110 - width, 150 + index * 18), text, fontsize=8)
    page.insert_text((20, 240), "Electricity (kWh)", fontsize=8)
    page.insert_text((72, 400), "Current Electricity Charges " + (total or month.total), fontsize=10)

    page = doc.new_page(width=612, height=792)
    y = 60
    lines = [
        f"On-Peak Use and Charge {month.on_kwh} kWh",
        f"Off-Peak Use and Charge {month.off_kwh} kWh",
        f"Transportation Electrification R53 {month.total_kwh} kWh",
        f"Billable Demand {month.demand}",
    ]
    lines += [f"{label} {amount}" for label, amount, _ in month.charges]
    lines.append("Current Electricity Charges " + (total or month.total))
    for line in lines:
        page.insert_text((72, y), line, fontsize=10)
        y += 18
    path = Path(folder) / name
    doc.save(path)
    doc.close()
    return path


def figures_for(month: Month, **changes) -> dict:
    """What a correct model returns for `month`, as JSON."""
    answer = {
        "utility_company": "PNM", "utility_type": "electricity",
        "account_number": "0000-TEST", "meter_number": None, "rate_schedule": None,
        "statement_date": {"text": month.statement, "page": 1},
        "due_date": None, "service_start": None, "service_end": None,
        "readings": [
            {"label": "On-Peak Use and Charge", "value": month.on_kwh, "unit": "kWh",
             "kind": "energy", "period": "on_peak", "page": 2},
            {"label": "Off-Peak Use and Charge", "value": month.off_kwh, "unit": "kWh",
             "kind": "energy", "period": "off_peak", "page": 2},
            {"label": "Transportation Electrification R53", "value": month.total_kwh,
             "unit": "kWh", "kind": "energy", "period": "total", "page": 2},
            {"label": "Billable Demand", "value": month.demand, "unit": "kW",
             "kind": "billed_demand", "period": "none", "page": 2},
        ],
        "charges": [
            {"label": label, "amount": amount, "category": category, "period": "none", "page": 2}
            for label, amount, category in month.charges
        ],
        "total_current_charges": {"text": month.total, "page": 2},
        "amount_due": {"text": month.total, "page": 1},
        "notes": [],
    }
    answer.update(changes)
    BillFigures.model_validate(answer)  # a fake answer must still fit the schema
    return answer


class FakeProvider(Provider):
    """Returns the answer registered for the statement date it sees in the text."""

    def __init__(self, answers: dict[str, dict], *, fail: set[str] = frozenset(),
                 rechecks: dict[str, dict] | None = None):
        self.config = ProviderConfig(provider="openai", model="fake-model",
                                     base_url="https://api.openai.com/v1")
        self.answers = answers
        self.fail = set(fail)
        self.rechecks = rechecks or {}   # statement date -> ChargeReview answer
        self.calls = []

    def extract(self, images, ocr_text, schema, *, instructions, max_output_tokens=8192):
        self.calls.append((len(images), ocr_text))
        if schema.__name__ == "LineVerdicts":
            # A recheck quotes lines, not the whole bill; tests register one bill.
            if not self.rechecks:
                raise RuntimeError("no recheck answer")
            return next(iter(self.rechecks.values()))
        for statement, answer in self.answers.items():
            if statement in ocr_text:
                if statement in self.fail:
                    raise RuntimeError("model unavailable")
                if schema.__name__ == "ChargeReview":
                    return self.rechecks[statement]
                return answer
        raise AssertionError("FakeProvider was shown a bill it has no answer for")

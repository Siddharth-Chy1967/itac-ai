"""Regression: the PNM bill whose chart axis became a month's usage.

The PNM bill issued 01/16/2026 prints a 13-month usage chart on page 1. Its
y-axis title "Electricity (kWh)" is rotated, and OCR puts its centre on the same
line as the axis tick "25000". The old printed-row reader (grid.py) paired
"label to the left of a number" and added a reading "Electricity (kWh) = 25000"
that no model had returned; every month of the workbook then showed 25000 kWh.

The test drives the whole batch (`pipeline.run_batch`) with only two stand-ins:
the OCR engine replays the real RapidOCR boxes of that bill
(data/pnm_2026_01_ocr.json, with account, name and address boxes removed), and
the model answers correctly -- it never mentions 25000. Anything that puts
25000 in the workbook is therefore the code inventing a figure.

It is written against entry points the old and the new pipeline share, so it
fails on the old code (25000 kWh appears) and passes on the new code.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pymupdf
from openpyxl import load_workbook

from app.bills import local_inference, pipeline
from app.bills.jobs import JobState, JobStore
from app.config import get_settings

DATA = Path(__file__).resolve().parent / "data" / "pnm_2026_01_ocr.json"
OCR_PAGES = json.loads(DATA.read_text(encoding="utf-8"))["pages"]

# What a correct model reads off this bill (page 2 charges, page 1 dates).
CHARGES = [
    ("On-Peak Use and Charge", "$820.76", "energy", "on_peak"),
    ("Off-Peak Use and Charge", "$420.55", "energy", "off_peak"),
    ("Fuel Cost Adjustment - Non-Renewable", "$659.28", "energy", "none"),
    ("Fuel Cost Adjustment - Renewable", "$0.00", "energy", "none"),
    ("Renewable Energy Rider", "$95.31", "energy", "none"),
    ("Renewable Energy Rider", "$103.78", "energy", "none"),
    ("Transportation Electrification R53", "$3.03", "energy", "none"),
    ("Customer Charge", "$74.69", "fixed", "none"),
    ("Billed demand and charge", "$1,931.54", "demand", "none"),
    ("San Juan Energy Transition Charge", "$78.74", "fee", "none"),
    ("Billed Rkva and Charge", "$0.00", "demand", "none"),
    ("Cost-Effective Energy Saving Prog.", "$175.49", "fee", "none"),
    ("Palo Verde Regulatory Credit", "-$57.13", "credit", "none"),
    ("City/County Franchise Fee", "$129.18", "tax", "none"),
    ("Gross Receipts Tax - State", "$216.22", "tax", "none"),
    ("Gross Receipts Tax - County", "$52.67", "tax", "none"),
    ("Gross Receipts Tax - City", "$69.30", "tax", "none"),
]
READINGS = [
    ("On-Peak Use and Charge", "16,720.000", "kWh", "energy", "on_peak"),
    ("Off-Peak Use and Charge", "14,320.000", "kWh", "energy", "off_peak"),
    ("Actual Demand: (Read x Constant)", "154.400", "kW", "measured_demand", "none"),
    ("Billable Demand", "154.400", "kW", "billed_demand", "none"),
]

MODEL_ANSWER = {
    "utility_company": "PNM", "utility_type": "electricity", "account_number": None,
    "meter_number": "0413226", "rate_schedule": "Gen Pwr 3C (Low LF) TOU",
    "statement_date": {"text": "01/16/2026", "page": 1},
    "due_date": {"text": "02/10/2026", "page": 1},
    "service_start": None, "service_end": None,
    "readings": [{"label": l, "value": v, "unit": u, "kind": k, "period": p, "page": 2}
                 for l, v, u, k, p in READINGS],
    "charges": [{"label": l, "amount": a, "category": c, "period": p, "page": 2}
                for l, a, c, p in CHARGES],
    "total_current_charges": {"text": "$4,773.41", "page": 2},
    "amount_due": {"text": "$4,773.41", "page": 1},
    "notes": [],
}

# The same correct answer in the old extraction schema, for the old pipeline,
# which read it from its cache instead of asking a model.
OLD_MODEL_ANSWER = {
    "utility_company": "PNM", "utility_types": ["electricity"], "account_number": None,
    "invoice_number": None, "bill_date": "01/16/2026", "due_date": "02/10/2026",
    "service_start": None, "service_end": None, "billing_days_text": "31",
    "billing_period_label": None, "current_charges_text": "$4,773.41",
    "total_due_text": "$4,773.41",
    "financial_entries": [
        {"label": l, "amount_text": a, "quantity_text": None, "quantity_unit": None,
         "rate_text": None, "rate_unit": None, "section": None, "meter_number": None,
         "service_period_text": None,
         "category": {"credit": "discount", "tax": "tax"}.get(c, "charge"),
         "bill_page_number": 2}
        for l, a, c, _ in CHARGES
    ],
    "measurements": [
        {"label": {"energy": ("On-Peak kWh" if p == "on_peak" else "Off-Peak kWh")}.get(k, l),
         "value_text": v, "unit": u, "section": None, "meter_number": None,
         "measurement_date": None, "measurement_time": None, "service_period_text": None,
         "bill_page_number": 2}
        for l, v, u, k, p in READINGS
    ],
    "additional_fields": [], "extraction_notes": [],
}


class _Answers:
    """A model that answers correctly for this bill, whichever way it is asked."""

    def __init__(self):
        from app.bills.llm_providers import Provider, ProviderConfig

        class Model(Provider):
            def __init__(inner):
                inner.config = ProviderConfig(provider="openai", model="regression-model",
                                              base_url="https://api.openai.com/v1")

            def extract(inner, images, ocr_text, schema, *, instructions, max_output_tokens=8192):
                return MODEL_ANSWER

        self.provider = Model()


class _ReplayOcr:
    """Stands in for RapidOCR: returns the recorded boxes of whichever page it sees."""

    def __call__(self, image):
        # Page 1 of the stand-in PDF carries a dark mark in its top-left corner.
        page = 0 if image[40, 40].sum() < 100 else 1
        boxes, texts = [], []
        for top, bottom, left, right, text in OCR_PAGES[page]:
            boxes.append([[left, top], [right, top], [right, bottom], [left, bottom]])
            texts.append(text)
        return SimpleNamespace(boxes=boxes, txts=texts, scores=[0.99] * len(texts))


def scanned_stand_in() -> bytes:
    """Two blank letter pages with no text layer, so the pipeline must use OCR."""
    doc = pymupdf.open()
    for number in range(2):
        page = doc.new_page(width=612, height=792)
        mark = pymupdf.Rect(0, 0, 20, 20) if number == 0 else pymupdf.Rect(590, 770, 612, 792)
        page.draw_rect(mark, color=(0, 0, 0), fill=(0, 0, 0))
    data = doc.tobytes()
    doc.close()
    return data


class PnmChartAxisIsNeverAReading(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        local_inference._OCR_CACHE.clear()
        self.addCleanup(local_inference._OCR_CACHE.clear)
        self.settings = get_settings().model_copy(update={
            "bills_cache_path": str(self.tmp / "cache"), "bills_ocr": True,
            "bills_llm_provider": "ollama",
        })
        self.bill = self.tmp / "pnm_2026_01.pdf"
        self.bill.write_bytes(scanned_stand_in())

        # The old pipeline read the model's answer from its extraction cache.
        from hashlib import sha256
        old_cache = (self.tmp / "cache" / "extraction_cache"
                     / sha256(self.bill.read_bytes()).hexdigest() / "gpt-5.6-terra")
        old_cache.mkdir(parents=True)
        (old_cache / "bill_extraction_v3.json").write_text(json.dumps(OLD_MODEL_ANSWER))

        patches = [
            patch.object(local_inference, "_get_ocr_engine", return_value=_ReplayOcr()),
            patch("app.bills.pipeline.bill_extraction.process_source_pdf",
                  side_effect=lambda *a, **k: [self.bill]),
            # New pipeline: the configured provider is the correct model above.
            patch("app.bills.pipeline.make_provider", create=True,
                  side_effect=lambda settings: _Answers().provider),
            # Old pipeline: no model for label interpretation (rules only).
            patch("app.bills.pipeline.LocalInferenceClient", create=True,
                  side_effect=RuntimeError("no model in tests")),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_pipeline(self):
        store = JobStore(self.tmp / "jobs", lambda job: None)
        job = store.create([("pnm.pdf", b"%PDF-1.4 stand-in")])
        pipeline.run_batch(job, settings=self.settings, stop_for_review=False)
        self.assertEqual(job.state, JobState.DONE, job.message)
        book = load_workbook(job.workbook)
        sheet = book[book.sheetnames[0]]
        values = []
        for row in sheet.iter_rows(min_row=4, values_only=True):
            if row[0] in ("Total", "TOTALS"):
                break
            values.extend(v for v in row[1:] if isinstance(v, (int, float)))
        return values, sheet

    def test_the_chart_tick_25000_never_reaches_the_workbook(self):
        values, _ = self.run_pipeline()
        self.assertNotIn(25000, values, "a chart axis label was written as a bill figure")

    def test_the_printed_usage_does(self):
        values, _ = self.run_pipeline()
        self.assertIn(16720, values)
        self.assertIn(14320, values)


if __name__ == "__main__":
    unittest.main()

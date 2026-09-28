"""Auditability regression tests: corrections, accounting effects, detailed checks.

Rewritten for the packaged fixture set. Two changes versus the original 36-bill
suite are deliberate, not a loosening:

1. The water bills (12 of 36) were never packaged, so tests that indexed
   `BILL_FOLDERS[1]` are repointed at a bill that actually exhibits the behaviour
   under test, found by property rather than by hard-coded index.
2. The shipped `reviewed_corrections.json` is STALE against the shipped
   `extraction_cache`: every one of its six reviewed bills either has no cache
   (the three water bills) or a cache whose SHA-256 no longer matches the one the
   correction was reviewed against. `test_shipped_corrections_are_stale` pins that
   fact. The correction-application paths are therefore exercised against a
   correction document synthesised from the live cache, which tests the apply and
   guard logic honestly instead of depending on a broken artifact.
"""
from contextlib import redirect_stdout
from copy import deepcopy
from decimal import Decimal
from hashlib import sha256
from io import BytesIO, StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
import openpyxl

import app.bills.bill_value_extraction as app
from app.bills.detailed_checks import build_checks
from app.bills.extraction_models import BillExtraction, FinancialEntry
from app.bills.reviewed_values import load_corrections, apply_corrections
from .test_fixtures import BILL_FOLDERS, FIXTURES, requires_fixtures
from app.bills.validation_checks import financial_effect, reconcile_current_charges
from app.bills.value_parsing import parse_number

TESTS_DIR = Path(__file__).resolve().parent
ROOT = FIXTURES  # project folder holding extraction_cache/

ELECTRICITY, GAS = 0, 1
DEGREE = "°"


def load_sample(folder, number):
    path = TESTS_DIR / BILL_FOLDERS[folder] / f"bill_{number:03d}.pdf"
    fingerprint = sha256(path.read_bytes()).hexdigest()
    bill, cache, _ = app.load_bill(path, fingerprint, ROOT, app.EXTRACTION_MODEL)
    return bill, fingerprint, cache, path


def correction_document(bill, fingerprint, cache, entry_index, new_amount):
    """A correction document bound to the live cache, as a reviewer would produce."""
    entry = bill.financial_entries[entry_index]
    return {
        "format_version": 1,
        "model": app.EXTRACTION_MODEL,
        "cache_version": app.CACHE_VERSION,
        "reviewed_by": "test",
        "reviewed_on": "2026-09-15",
        "bills": {
            fingerprint: {
                "original_cache_sha256": sha256(cache.read_bytes()).hexdigest(),
                "changes": [{
                    "collection": "financial_entries",
                    "field": "amount_text",
                    "entry_number": entry_index + 1,
                    "label": entry.label,
                    "bill_page_number": entry.bill_page_number,
                    "old": entry.amount_text,
                    "new": new_amount,
                    "reason": "Source review: amount misread.",
                }],
            }
        },
    }


def synthetic_bill(entries):
    return BillExtraction(
        utility_company="Example Utility", utility_types=["electricity"],
        account_number=None, invoice_number=None, bill_date=None, due_date=None,
        service_start=None, service_end=None, billing_days_text=None,
        billing_period_label=None, current_charges_text="$100.00",
        total_due_text="$100.00", financial_entries=entries, measurements=[],
        additional_fields=[], extraction_notes=[],
    )


def entry(label, amount, category="charge", section=None):
    return FinancialEntry(
        label=label, amount_text=amount, quantity_text=None, quantity_unit=None,
        rate_text=None, rate_unit=None, section=section, meter_number=None,
        service_period_text=None, category=category, bill_page_number=1,
    )


class CorrectionTests(unittest.TestCase):
    @requires_fixtures
    def test_correction_applies_only_to_copy_and_preserves_original(self):
        bill, fingerprint, cache, path = load_sample(GAS, 5)
        index = next(i for i, e in enumerate(bill.financial_entries) if e.amount_text)
        document = correction_document(bill, fingerprint, cache, index, "$1.23")
        before, raw = bill.model_dump(), cache.read_bytes()

        reviewed, audit = apply_corrections(
            bill, fingerprint, cache, path, app.EXTRACTION_MODEL, app.CACHE_VERSION, document
        )
        self.assertEqual(reviewed.financial_entries[index].amount_text, "$1.23")
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit[0]["original_value"],
                         before["financial_entries"][index]["amount_text"])
        self.assertEqual(audit[0]["reviewed_value"], "$1.23")
        # Neither the in-memory bill nor the cache on disk may be touched.
        self.assertEqual(bill.model_dump(), before)
        self.assertEqual(cache.read_bytes(), raw)

    @requires_fixtures
    def test_stale_cache_model_and_value_guards(self):
        bill, fingerprint, cache, path = load_sample(ELECTRICITY, 4)
        index = next(i for i, e in enumerate(bill.financial_entries) if e.amount_text)
        document = correction_document(bill, fingerprint, cache, index, "$1.23")

        def apply(doc, model=app.EXTRACTION_MODEL):
            return apply_corrections(bill, fingerprint, cache, path, model,
                                     app.CACHE_VERSION, doc)

        # The unmodified document applies cleanly; each mutation below must be caught.
        self.assertEqual(len(apply(document)[1]), 1)

        stale = deepcopy(document)
        stale["bills"][fingerprint]["original_cache_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "Stale"):
            apply(stale)
        with self.assertRaisesRegex(ValueError, "model"):
            apply(document, "different-model")
        wrong_value = deepcopy(document)
        wrong_value["bills"][fingerprint]["changes"][0]["old"] = "not the cached value"
        with self.assertRaisesRegex(ValueError, "old value"):
            apply(wrong_value)
        wrong_row = deepcopy(document)
        wrong_row["bills"][fingerprint]["changes"][0]["label"] = "different row"
        with self.assertRaisesRegex(ValueError, "identity"):
            apply(wrong_row)

    @requires_fixtures
    def test_shipped_corrections_are_stale(self):
        """The packaged corrections no longer match the packaged caches.

        Pinned so the breakage stays visible: the guard is behaving correctly and
        re-reviewing against current extractions is the only valid fix. Never
        "repair" this by rewriting the recorded hashes.
        """
        document = load_corrections(ROOT / "reviewed_corrections.json")
        applicable = 0
        for fingerprint, review in document["bills"].items():
            cache = app.cache_path_for(ROOT, fingerprint, app.EXTRACTION_MODEL)
            if not cache.is_file():
                continue  # water bills: never packaged
            applicable += 1
            self.assertNotEqual(sha256(cache.read_bytes()).hexdigest(),
                                review["original_cache_sha256"])
        self.assertEqual(applicable, 3)

    @requires_fixtures
    def test_reviewed_run_reports_stale_bills_without_exporting_them(self):
        with patch.object(app, "LocalInferenceClient", side_effect=AssertionError("No model")), \
             redirect_stdout(StringIO()):
            paths = app.collect_pdfs([str(TESTS_DIR / folder) for folder in BILL_FOLDERS])
            tables = app.process_pdfs(paths, project_folder=ROOT, reviewed=True)
        counts = tables["Run Summary"]["status"].value_counts().to_dict()
        self.assertEqual(counts, {"cached": 21, "error": 3})
        self.assertEqual(len(tables["Bills"]), 21)
        for message in tables["Run Summary"].query("status == 'error'")["message"]:
            self.assertIn("Stale", message)

    def test_reviewed_mode_never_runs_local_inference(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "reviewed_corrections.json").write_text(
                '{"format_version": 1, "model": "gpt-5.6-terra", '
                '"cache_version": "bill_extraction_v3", "bills": {}}'
            )
            pdf = root / "bill_001.pdf"
            pdf.write_bytes(b"%PDF-1.4 uncached test bill")
            with patch.object(app, "extract_bill_pdf", side_effect=AssertionError("No model")), \
                 redirect_stdout(StringIO()):
                tables = app.process_pdfs([pdf], project_folder=root, reviewed=True)
        self.assertEqual(tables["Run Summary"]["status"].tolist(), ["missing_cache"])


class AccountingTests(unittest.TestCase):
    def test_comparison_rows_are_not_billed_charges(self):
        bill = synthetic_bill([
            entry("Energy Charge", "$60.00"),
            entry("Service Charge", "$40.00"),
            entry("Cost", "$3.10", section="Comparison of daily averages"),
        ])
        self.assertEqual(financial_effect(bill.financial_entries[2])[0], "comparison")
        self.assertEqual(
            reconcile_current_charges(bill, app.parse_amount)["reconciliation_status"],
            "matched",
        )

    def test_unresolved_row_blocks_the_check_rather_than_being_ignored(self):
        bill = synthetic_bill([
            entry("Energy Charge", "$60.00"),
            entry("Service Charge", "$40.00"),
            entry("Mystery line", "$3.10", category="other"),
        ])
        result = reconcile_current_charges(bill, app.parse_amount)
        self.assertEqual(result["reconciliation_status"], "not_checked")
        self.assertIn("unresolved", result["reconciliation_note"])

    @requires_fixtures
    def test_account_activity_excluded_from_current_charges(self):
        bill, _, _, _ = load_sample(ELECTRICITY, 6)
        activity = [e for e in bill.financial_entries
                    if financial_effect(e)[0] == "account_activity"]
        self.assertTrue(activity)
        charges = sum(
            app.parse_amount(e.amount_text)
            for e in bill.financial_entries
            if financial_effect(e)[0] == "current_charge"
        )
        self.assertEqual(
            reconcile_current_charges(bill, app.parse_amount)["line_item_sum"], charges
        )


class DetailedCheckTests(unittest.TestCase):
    @requires_fixtures
    def test_stronger_checks_detect_errors_hidden_by_matching_total(self):
        # Current charges reconcile, yet the balance check still catches a problem.
        bill, fingerprint, _, path = load_sample(ELECTRICITY, 6)
        checks = build_checks(bill, app.parse_amount, fingerprint, path)
        self.assertEqual(checks[0]["status"], "matched")
        balance = [c for c in checks if c["check_type"] == "balance_to_total_due"]
        self.assertEqual([c["status"] for c in balance], ["review"])

        bill, fingerprint, _, path = load_sample(GAS, 2)
        checks = build_checks(bill, app.parse_amount, fingerprint, path)
        self.assertTrue(any(c["check_type"] == "section_subtotal" and c["status"] == "review"
                            for c in checks))

    @requires_fixtures
    def test_missing_balance_blocks_its_check(self):
        bill, fingerprint, _, path = load_sample(ELECTRICITY, 6)
        bill = bill.model_copy(deep=True)
        for e in bill.financial_entries:
            if e.category == "balance":
                e.amount_text = None
        checks = build_checks(bill, app.parse_amount, fingerprint, path)
        self.assertEqual(
            next(c for c in checks if c["check_type"] == "balance_to_total_due")["status"],
            "not_checked",
        )

    @requires_fixtures
    def test_ambiguous_meter_split_blocks_a_section_subtotal(self):
        """Two meters inside one section must block that section's subtotal check.

        The checker groups by (page, section), so the split has to be applied to
        rows in the very section that is currently checkable -- splitting two
        arbitrary charges elsewhere proves nothing.
        """
        bill, fingerprint, _, path = load_sample(GAS, 2)
        checked = [c for c in build_checks(bill, app.parse_amount, fingerprint, path)
                   if c["check_type"] == "section_subtotal" and c["status"] != "not_checked"]
        self.assertTrue(checked, "fixture no longer has a checkable section subtotal")
        target = checked[0]
        rows = [int(s.split(":")[1]) for s in target["calculation_sources"].split(";") if s]
        self.assertGreaterEqual(len(rows), 2)

        bill = bill.model_copy(deep=True)
        for number, meter in zip(rows[:2], ("METER_A", "METER_B")):
            bill.financial_entries[number - 1].meter_number = meter

        after = build_checks(bill, app.parse_amount, fingerprint, path)
        same = next(c for c in after if c["check_type"] == "section_subtotal"
                    and c["expected_source"] == target["expected_source"])
        self.assertEqual(same["status"], "not_checked")


class ConversionTests(unittest.TestCase):
    def test_degrees_do_not_invent_temperature_scale(self):
        self.assertEqual(parse_number("91.3" + DEGREE), Decimal("91.3"))
        with self.assertRaises(ValueError):
            parse_number("91.3" + DEGREE, "kW")


class WorkbookTests(unittest.TestCase):
    @requires_fixtures
    def test_full_fixture_export_is_auditable_and_leaves_caches_untouched(self):
        cache_files = {p: sha256(p.read_bytes()).hexdigest()
                       for p in (ROOT / "extraction_cache").rglob("*.json")}
        self.assertEqual(len(cache_files), 24)
        with patch.object(app, "LocalInferenceClient", side_effect=AssertionError("No model")), \
             redirect_stdout(StringIO()):
            paths = app.collect_pdfs([str(TESTS_DIR / folder) for folder in BILL_FOLDERS])
            tables = app.process_pdfs(paths, project_folder=ROOT)

        self.assertEqual(len(tables["Bills"]), 24)
        self.assertEqual(len(tables["Financial Entries"]), 353)
        self.assertEqual(len(tables["Measurements"]), 144)
        self.assertEqual(set(tables["Run Summary"]["status"]), {"cached"})
        self.assertEqual(tables["Bills"]["reconciliation_status"].value_counts().to_dict(),
                         {"matched": 21, "review": 3})
        self.assertEqual(tables["Validation Checks"]["status"].value_counts().to_dict(),
                         {"matched": 162, "review": 9, "not_checked": 9})

        for path, original in cache_files.items():
            self.assertEqual(sha256(path.read_bytes()).hexdigest(), original)

        buffer = BytesIO()
        app.write_workbook(tables, buffer)
        book = openpyxl.load_workbook(buffer)
        ws = book["Validation Checks"]
        headers = {cell.value: cell.column for cell in ws[1]}
        self.assertEqual(ws.cell(2, headers["calculated_value"]).data_type, "f")
        self.assertIn("'Financial Entries'!", ws.cell(2, headers["calculated_value"]).value)
        # Corrections must never carry live formulas: the sheet is an audit record.
        self.assertFalse(any(c.data_type == "f" for row in book["Corrections"] for c in row))


if __name__ == "__main__":
    unittest.main()

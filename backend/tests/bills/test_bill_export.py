"""Offline tests: no model or network calls; Excel tests use memory only."""
from contextlib import redirect_stdout
from datetime import date, time, datetime
from decimal import Decimal
from hashlib import sha256
from io import BytesIO, StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, Mock
import unittest

import openpyxl
import app.bills.bill_value_extraction as app
from app.bills.extraction_models import BillExtraction
from app.bills.value_parsing import parse_number, parse_date, parse_time
from .test_fixtures import BILL_FOLDERS, FIXTURES, requires_fixtures

# The extractor resolves caches relative to the project folder it is given,
# so for fixture-backed tests that folder is fixtures/, not the tests folder.
TESTS_DIR = Path(__file__).resolve().parent
ROOT = FIXTURES
SAMPLE = TESTS_DIR / BILL_FOLDERS[0] / "bill_004.pdf"
GAS_SAMPLE = TESTS_DIR / BILL_FOLDERS[1] / "bill_008.pdf"


def sample_bill():
    fingerprint = sha256(SAMPLE.read_bytes()).hexdigest()
    bill, cache, status = app.load_bill(SAMPLE, fingerprint, ROOT, app.EXTRACTION_MODEL)
    return bill, fingerprint, cache


class ParsingTests(unittest.TestCase):
    def test_numbers_and_percent(self):
        for text, unit, expected in [
            ("88.05%", "%", "88.05"), ("5,659.800", "MMBtu", "5659.800"),
            ("(5.200)", "MMBtu", "-5.200"), ("637.3 kW", "kW", "637.3"),
            ("0", None, "0"), (".5", None, "0.5"), ("+2.50", None, "2.50"),
        ]:
            with self.subTest(text=text):
                self.assertEqual(parse_number(text, unit), Decimal(expected))
        self.assertIsNone(parse_number(None))

    def test_unsafe_numbers_rejected(self):
        for text, unit in [("1,23", None), ("5-6", None), ("1/2/2021", None),
                           ("NaN", None), ("88.05%", "kW"), ("1234567890123456", None)]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_number(text, unit)

    def test_dates_and_times(self):
        for text in ("12/3/2021", "2021-12-03", "Dec 3, 2021", "December 3, 2021"):
            self.assertEqual(parse_date(text), date(2021, 12, 3))
        self.assertEqual(parse_time("14:30"), time(14, 30))
        for text in ("Dec-20", "02/30/2021", "03/04/21", "unknown"):
            with self.assertRaises(ValueError):
                parse_date(text)


class BatchTests(unittest.TestCase):
    @requires_fixtures
    def test_real_cache_batch_without_api(self):
        # A fixed two-bill sample (one electricity, one gas) so expected counts stay
        # stable as real batch runs add caches. The original sample also carried a
        # water bill; the water set was never packaged (see test_fixtures).
        paths = [SAMPLE, GAS_SAMPLE]
        with patch.object(app, "LocalInferenceClient", side_effect=AssertionError("No model")),              redirect_stdout(StringIO()):
            tables = app.process_pdfs(paths, project_folder=ROOT)
        self.assertEqual(len(tables["Run Summary"]), len(paths))
        self.assertEqual(len(tables["Bills"]), 2)
        self.assertEqual(len(tables["Measurements"]), 11)
        self.assertEqual(len(tables["Financial Entries"]), 29)
        self.assertEqual(set(tables["Run Summary"]["status"]), {"cached"})
        # Both reconcile exactly against their printed current charges.
        self.assertEqual(tables["Bills"]["reconciliation_status"].tolist(), ["matched", "matched"])
        self.assertEqual(tables["Bills"]["current_charges_difference"].tolist(),
                         [Decimal("0.00"), Decimal("0.00")])

    @requires_fixtures
    def test_raw_cache_unchanged_and_percent_numeric(self):
        bill, fingerprint, cache = sample_bill()
        original = cache.read_bytes()
        before = bill.model_dump()
        rows = app.bill_tables(bill, SAMPLE, fingerprint, app.EXTRACTION_MODEL, cache)
        percent_rows = [r for r in rows["Measurements"] if r["value_text"].endswith("%")]
        self.assertEqual(len(percent_rows), 2)
        for row in percent_rows:
            self.assertEqual(row["numeric_unit"], "%")
            self.assertEqual(row["value"], Decimal(row["value_text"][:-1]))
        self.assertEqual(before, bill.model_dump())
        self.assertEqual(original, cache.read_bytes())

    @requires_fixtures
    def test_cache_only_missing_cache_does_not_run_model(self):
        with patch.object(app, "LocalInferenceClient", side_effect=AssertionError("No model")):
            with self.assertRaises(FileNotFoundError):
                app.load_bill(SAMPLE, "uncached", ROOT, app.EXTRACTION_MODEL, run_inference=False)

    def test_missing_cache_is_extracted_locally_and_saved_beside_old_caches(self):
        bill = BillExtraction(
            utility_company="Example", utility_types=["water"], account_number=None,
            invoice_number=None, bill_date=None, due_date=None, service_start=None,
            service_end=None, billing_days_text=None, billing_period_label=None,
            current_charges_text="$1.00", total_due_text="$1.00", financial_entries=[],
            measurements=[], additional_fields=[], extraction_notes=[],
        )
        client = Mock(cache_label="ollama-test-model")
        client.parse_pdf.return_value = bill
        with TemporaryDirectory() as folder:
            root = Path(folder)
            loaded, path, status = app.load_bill(Path("bill.pdf"), "abc", root, app.EXTRACTION_MODEL,
                                                 get_client=lambda: client, local_label="ollama-test-model")
            self.assertEqual(status, "extracted")
            self.assertEqual(path, app.cache_path_for(root, "abc", "ollama-test-model"))
            self.assertEqual(path.read_text(encoding="utf-8"), bill.model_dump_json(indent=2))
            client.ensure_ready.assert_called_once()
            self.assertIs(client.parse_pdf.call_args.args[0], BillExtraction)
            # A second run reuses the saved local cache instead of extracting again.
            client.parse_pdf.side_effect = AssertionError("No second extraction")
            _, again, status = app.load_bill(Path("bill.pdf"), "abc", root, app.EXTRACTION_MODEL,
                                             get_client=lambda: client, local_label="ollama-test-model")
            self.assertEqual((again, status), (path, "cached"))

    @requires_fixtures
    def test_corrupt_cache_not_reextracted(self):
        with patch.object(Path, "is_file", return_value=True), \
             patch.object(Path, "read_text", return_value="not json"), \
             patch.object(app, "extract_bill_pdf", side_effect=AssertionError("No model")):
            with self.assertRaises(ValueError):
                app.load_bill(SAMPLE, "fake", ROOT, app.EXTRACTION_MODEL, run_inference=True)

    @requires_fixtures
    def test_model_separation_and_local_request(self):
        self.assertNotEqual(app.cache_path_for(ROOT, "abc", "model-one"),
                            app.cache_path_for(ROOT, "abc", "model-two"))
        with self.assertRaises(ValueError):
            app.cache_path_for(ROOT, "abc", "../unsafe")
        client = Mock()
        client.parse_pdf.return_value = "fake"
        self.assertEqual(app.extract_bill_pdf(client, SAMPLE), "fake")
        args, kwargs = client.parse_pdf.call_args
        self.assertEqual(args[1], app.BILL_EXTRACTION_INSTRUCTIONS)
        self.assertEqual(args[3], [SAMPLE])
        self.assertEqual(kwargs["max_output_tokens"], app.EXTRACTION_MAX_OUTPUT_TOKENS)
        client.parse_pdf.side_effect = RuntimeError("failure")
        with self.assertRaises(RuntimeError):
            app.extract_bill_pdf(client, SAMPLE)

    @requires_fixtures
    def test_duplicates_and_failures_accounted_for(self):
        with redirect_stdout(StringIO()):
            tables = app.process_pdfs([SAMPLE, SAMPLE, ROOT / "missing.pdf"], project_folder=ROOT)
        self.assertEqual(len(tables["Bills"]), 1)
        self.assertEqual(tables["Run Summary"]["status"].tolist(), ["cached", "duplicate", "error"])

    @requires_fixtures
    def test_invalid_conversion_flags_without_mutating(self):
        bill, fingerprint, cache = sample_bill()
        bill = bill.model_copy(deep=True)
        bill.measurements[0].value_text = "not a number"
        before = bill.model_dump()
        rows = app.bill_tables(bill, SAMPLE, fingerprint, app.EXTRACTION_MODEL, cache)
        self.assertIsNone(rows["Measurements"][0]["value"])
        self.assertIn("Unrecognized", rows["Bills"][0]["conversion_notes"])
        self.assertEqual(bill.model_dump(), before)

    def test_empty_sheets_have_headers(self):
        empty = {name: [] for name in ("Bills", "Financial Entries", "Measurements", "Additional Fields")}
        buffer = BytesIO()
        app.write_workbook(app.make_tables(empty, []), buffer)
        book = openpyxl.load_workbook(buffer)
        self.assertEqual(len(book.sheetnames), 7)
        for sheet in book:
            self.assertEqual(sheet.max_row, 1)

    @requires_fixtures
    def test_excel_types_and_formula_protection(self):
        with redirect_stdout(StringIO()):
            tables = app.process_pdfs([SAMPLE], project_folder=ROOT)
        tables["Financial Entries"].loc[0, "label"] = '=HYPERLINK("https://example.invalid","bad")'
        buffer = BytesIO()
        app.write_workbook(tables, buffer)
        book = openpyxl.load_workbook(buffer)
        def cell(sheet, field, row=2):
            ws = book[sheet]
            headers = [c.value for c in ws[1]]
            return ws.cell(row, headers.index(field) + 1)
        self.assertEqual(cell("Financial Entries", "label").data_type, "s")
        self.assertEqual(cell("Financial Entries", "amount").data_type, "n")
        self.assertEqual(cell("Bills", "current_charges").data_type, "n")
        self.assertIsInstance(cell("Bills", "due_date_value").value, datetime)
        self.assertEqual(cell("Bills", "line_item_sum").data_type, "f")
        self.assertIn("SUMIFS", cell("Bills", "line_item_sum").value)
        self.assertIsInstance(cell("Measurements", "measurement_time_value", 6).value, time)
        for row in range(2, book["Measurements"].max_row + 1):
            if cell("Measurements", "numeric_unit", row).value == "%":
                self.assertEqual(cell("Measurements", "value", row).data_type, "n")
                self.assertNotIn("%", cell("Measurements", "value", row).number_format)


if __name__ == "__main__":
    unittest.main()

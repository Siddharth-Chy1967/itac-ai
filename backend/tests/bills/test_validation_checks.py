"""Run offline: python test_validation_checks.py (no API or Excel writes)."""
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
import unittest

from app.bills.extraction_models import BillExtraction
from app.bills.validation_checks import reconcile_current_charges
from app.bills.bill_value_extraction import parse_amount
from .test_fixtures import FIXTURES, requires_fixtures

ROOT = FIXTURES


class ReconciliationTests(unittest.TestCase):
    def check_bill(self, entries, expected):
        bill = SimpleNamespace(
            financial_entries=[SimpleNamespace(category=c, amount_text=a) for c, a in entries],
            current_charges_text=expected,
        )
        return reconcile_current_charges(bill, parse_amount)

    def test_excludes_repeated_totals_payments_and_balances(self):
        result = self.check_bill([
            ("charge", "$100.00"), ("tax", "10.00"), ("discount", "$(5.00)"),
            ("subtotal", "105.00"), ("total", "105.00"), ("total", "105.00"),
            ("payment", "-50.00"), ("balance", "20.00"),
        ], "105.00")
        self.assertEqual(result["reconciliation_status"], "matched")
        self.assertEqual(result["line_item_sum"], Decimal("105.00"))

    def test_missing_invalid_and_uncertain_inputs_do_not_pass(self):
        for entries, expected in [
            ([], "0.00"), ([("charge", None)], "0.00"),
            ([("charge", "not a number")], "0.00"),
            ([("charge", "1.00")], None),
            ([("charge", "1.00")], "invalid"),
            ([("charge", "1.00"), ("other", "2.00")], "1.00"),
            ([("charge", "1.00"), ("tax", None)], "1.00"),
        ]:
            with self.subTest(entries=entries, expected=expected):
                self.assertEqual(self.check_bill(entries, expected)["reconciliation_status"], "not_checked")

    def test_does_not_deduplicate_legitimate_equal_rows(self):
        self.assertEqual(self.check_bill([("charge", "5.00"), ("charge", "5.00")], "10.00")["reconciliation_status"], "matched")

    @requires_fixtures
    def test_real_saved_bills_unchanged(self):
        # The water sample of the original trio was never packaged (see test_fixtures);
        # the two that exist still pin the real-data behaviour.
        samples = [
            ("electricity", "95ddcdb780d1f2f9e0f6498727d8066cd4658e02c479512d837eeaab5f0500f4", "matched", "0.00"),
            ("gas", "030aa618737309a7ac8f95352c4cd677031db7774b57f002c5334024544c779b", "matched", "0.00"),
        ]
        for name, fingerprint, status, difference in samples:
            with self.subTest(bill=name):
                path = ROOT / "extraction_cache" / fingerprint / "gpt-5.6-terra" / "bill_extraction_v3.json"
                original_bytes = path.read_bytes()
                bill = BillExtraction.model_validate_json(original_bytes)
                before = bill.model_dump()
                result = reconcile_current_charges(bill, parse_amount)
                self.assertEqual(result["reconciliation_status"], status)
                self.assertEqual(result["current_charges_difference"], Decimal(difference))
                self.assertEqual(bill.model_dump(), before)
                self.assertEqual(path.read_bytes(), original_bytes)


if __name__ == "__main__":
    unittest.main()


class SectionNamedRowTests(unittest.TestCase):
    """Unlabelled charge rows named by their section, from the real Dec-21 SRP bill."""

    def entry(self, amount, section, category="charge", label=None):
        from app.bills.extraction_models import FinancialEntry
        return FinancialEntry(label=label, amount_text=amount, quantity_text=None,
                              quantity_unit=None, rate_text=None, rate_unit=None,
                              section=section, meter_number=None, service_period_text=None,
                              category=category, bill_page_number=1)

    def bill(self, entries):
        return BillExtraction(
            utility_company="SRP", utility_types=["electricity"], account_number=None,
            invoice_number=None, bill_date=None, due_date=None, service_start=None,
            service_end=None, billing_days_text=None,
            billing_period_label="SERVICE FROM 11/9/2021 - 12/8/2021",
            current_charges_text="$12,614.11", total_due_text="$12,614.11",
            financial_entries=entries, measurements=[], additional_fields=[],
            extraction_notes=[],
        )

    def test_every_row_named_only_by_its_section_is_recovered(self):
        from app.bills.validation_checks import financial_effect, name_rows_from_sections
        # Exactly as qwen2.5vl returned the Dec-21 bill: every label empty.
        bill = name_rows_from_sections(self.bill([
            self.entry("-$17,056.45", "Previous Balance"),
            self.entry("$753.06", "Monthly Service Charge"),
            self.entry("$532.76", "Demand Charge"),
            self.entry("$1,318.46", "On Peak Energy Charge"),
            self.entry("$1,409.62", "Shoulder Peak Energy Charge"),
            self.entry("$5,766.01", "Off Peak Energy Charge"),
            self.entry("$1,840.97", "Facilities Charge"),
            self.entry("-$48.30", "Aggregation Discount", "discount"),
            self.entry("$312.46", "Phoenix City Tax", "tax"),
            self.entry("$729.07", "County and State Tax", "tax"),
            self.entry("$12,614.11", "This Month's Charges"),
            self.entry("$12,614.11", "PLEASE PAY", "total"),
        ]))
        effect = {e.label: financial_effect(e)[0] for e in bill.financial_entries}
        self.assertEqual(effect["On Peak Energy Charge"], "current_charge")
        # Named, but still not charges: counting these would double the month.
        self.assertEqual(effect["Previous Balance"], "account_activity")
        self.assertEqual(effect["This Month's Charges"], "summary")
        charges = sum(parse_amount(e.amount_text) for e in bill.financial_entries
                      if financial_effect(e)[0] == "current_charge")
        self.assertEqual(charges, Decimal("12614.11"))

    def test_a_section_several_rows_share_is_a_heading_not_a_name(self):
        from app.bills.validation_checks import name_rows_from_sections
        bill = name_rows_from_sections(self.bill([
            self.entry("$100.00", "Energy Charges"),
            self.entry("$50.00", "Energy Charges", label="Fuel Adj"),
        ]))
        self.assertEqual([e.label for e in bill.financial_entries], [None, "Fuel Adj"])

    def test_a_printed_label_is_never_replaced(self):
        from app.bills.validation_checks import name_rows_from_sections
        bill = name_rows_from_sections(self.bill([
            self.entry("$100.00", "Charges", label="Energy Charge")]))
        self.assertEqual(bill.financial_entries[0].label, "Energy Charge")

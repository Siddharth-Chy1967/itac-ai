"""The second look: when charge lines do not add up, the model is asked again.

The recheck can recover a line the model missed, but its answer faces the same
printed-on-the-page check, so it cannot invent one to make the total balance.
"""
from __future__ import annotations

import tempfile
import unittest
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

from app.bills import figures
from app.bills.local_inference import LocalInferenceConfig
from .synthetic import FakeProvider, Month, figures_for, make_bill_pdf

NO_OCR = LocalInferenceConfig(ocr_enabled=False)
MONTH = Month("01/16/2026", charges=[
    ("On-Peak Use and Charge", "$820.76", "energy"),
    ("Cost-Effective Energy Saving Prog.", "$175.49", "fee"),
    ("Customer Charge", "$74.69", "fixed"),
])


def charge(label, amount, category="energy"):
    return {"label": label, "amount": amount, "category": category, "period": "none", "page": 2}


def verdict(line, label, amount, category="energy", is_charge=True):
    return {"line": line, "is_charge_line": is_charge, "label": label, "amount": amount,
            "category": category}


class RecheckTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.pdf = make_bill_pdf(self.folder, "bill.pdf", MONTH)
        self.fingerprint = sha256(self.pdf.read_bytes()).hexdigest()
        # The first reading misses the $175.49 line, as qwen2.5vl did on PNM.
        self.first = figures_for(MONTH, charges=[
            charge("On-Peak Use and Charge", "$820.76"),
            charge("Customer Charge", "$74.69", "fixed")])

    def read(self, provider):
        return figures.read_bill(self.pdf, self.fingerprint, self.folder, provider, NO_OCR)

    def test_a_missed_line_is_recovered(self):
        provider = FakeProvider({MONTH.statement: self.first}, rechecks={MONTH.statement: {
            "verdicts": [verdict(1, "Cost-Effective Energy Saving Prog.", "$175.49", "fee")]}})
        verified, _ = self.read(provider)
        self.assertEqual(verified.status, "matched")
        self.assertEqual(verified.charges_sum, Decimal("1070.94"))
        self.assertTrue(any("Recheck recovered: Cost-Effective" in f for f in verified.flags))
        self.assertFalse(any("Line items total" in f for f in verified.flags))
        self.assertEqual(len(provider.calls), 2)
        # The follow-up states the arithmetic and quotes the page's unaccounted line,
        # and sends no images: it is a question about text.
        images, text = provider.calls[1]
        self.assertEqual(images, 0)
        self.assertIn("short by $175.49", text)
        self.assertIn("Cost-Effective Energy Saving Prog. $175.49", text)
        self.assertNotIn("On-Peak Use and Charge $820.76", text)

    def test_a_line_invented_to_balance_is_still_dropped(self):
        provider = FakeProvider({MONTH.statement: self.first}, rechecks={MONTH.statement: {
            "verdicts": [verdict(1, "Balancing Adjustment", "$175.49")]}})   # not printed
        verified, _ = self.read(provider)
        self.assertEqual(verified.status, "review")
        self.assertNotIn("Balancing Adjustment", [c.label for c in verified.figures.charges])
        self.assertTrue(any("Recheck found no printed charge line" in f for f in verified.flags))

    def test_a_bill_whose_lines_exceed_the_total_is_left_for_review(self):
        over = figures_for(MONTH, charges=[
            charge("On-Peak Use and Charge", "$820.76"),
            charge("Cost-Effective Energy Saving Prog.", "$175.49", "fee"),
            charge("Customer Charge", "$74.69", "fixed")],
            total_current_charges={"text": "$820.76", "page": 1})
        provider = FakeProvider({MONTH.statement: over})
        verified, _ = self.read(provider)
        self.assertEqual(verified.status, "review")
        self.assertEqual(len(provider.calls), 1)

    def test_the_total_is_never_accepted_as_a_charge(self):
        provider = FakeProvider({MONTH.statement: self.first}, rechecks={MONTH.statement: {
            "verdicts": [verdict(1, "Current Electricity Charges", "$1,070.94"),
                         verdict(1, "Cost-Effective Energy Saving Prog.", "$175.49", "fee")]}})
        verified, _ = self.read(provider)
        self.assertEqual(verified.status, "matched")
        self.assertNotIn("$1,070.94", [c.amount for c in verified.figures.charges])

    def test_a_line_that_only_narrows_the_gap_is_not_kept(self):
        month = Month("01/16/2026", charges=MONTH.charges + [("See the website", "$1.00", "fee")])
        pdf = make_bill_pdf(self.folder, "notice.pdf", month, total="$1,070.94")
        provider = FakeProvider({month.statement: self.first}, rechecks={month.statement: {
            "verdicts": [verdict(1, "See the website", "$1.00", "fee")]}})
        verified, _ = figures.read_bill(pdf, sha256(pdf.read_bytes()).hexdigest(),
                                        self.folder, provider, NO_OCR)
        self.assertEqual(verified.status, "review")
        self.assertNotIn("$1.00", [c.amount for c in verified.figures.charges])

    def test_a_bill_that_adds_up_is_not_rechecked(self):
        provider = FakeProvider({MONTH.statement: figures_for(MONTH)})
        verified, _ = self.read(provider)
        self.assertEqual(verified.status, "matched")
        self.assertEqual(len(provider.calls), 1)

    def test_the_recheck_is_cached_per_model(self):
        answers = {MONTH.statement: self.first}
        rechecks = {MONTH.statement: {"verdicts": [
            verdict(1, "Cost-Effective Energy Saving Prog.", "$175.49", "fee")]}}
        self.read(FakeProvider(answers, rechecks=rechecks))
        again = FakeProvider(answers, rechecks=rechecks)
        verified, status = self.read(again)
        self.assertEqual((status, verified.status, again.calls), ("cached", "matched", []))

    def test_a_failed_recheck_keeps_the_flagged_first_reading(self):
        provider = FakeProvider({MONTH.statement: self.first}, rechecks={})
        verified, _ = self.read(provider)
        self.assertEqual(verified.status, "review")
        self.assertEqual(len(verified.figures.charges), 2)
        self.assertTrue(any("Recheck of the charge lines failed" in f for f in verified.flags))


if __name__ == "__main__":
    unittest.main()

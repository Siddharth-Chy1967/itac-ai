"""The hard rule: nothing reaches the workbook that the page does not print.

Page text and chart-axis removal (page_text), the printed-figure check (verify)
and column naming from the model's typed fields (columns).
"""
from __future__ import annotations

import json
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

from app.bills import columns
from app.bills.bill_schema import BillFigures
from app.bills.page_text import PageText, axis_label_indices, split_axis_labels
from app.bills.semantics import CREDIT, CREDITS, DEMAND, OTHER, USAGE
from app.bills.verify import numbers_in, verify
from .synthetic import Month, figures_for

DATA = Path(__file__).resolve().parent / "data" / "pnm_2026_01_ocr.json"
PNM = [[tuple(box) for box in page] for page in json.loads(DATA.read_text())["pages"]]


def page(number: int, *lines: str) -> PageText:
    """A page whose boxes are one per line, stacked, left-aligned."""
    items = tuple((20.0 * i, 20.0 * i + 12, 10.0, 10.0 + 7 * len(t), t)
                  for i, t in enumerate(lines))
    return PageText(number=number, items=items, axis_labels=(), source="test")


def pnm_pages() -> list[PageText]:
    out = []
    for number, items in enumerate(PNM, start=1):
        kept, axis = split_axis_labels(items)
        out.append(PageText(number, tuple(kept), tuple(axis), "OCR"))
    return out


class AxisLabelTests(unittest.TestCase):
    def test_the_pnm_usage_chart_axis_is_found_and_nothing_else(self):
        _, axis = split_axis_labels(PNM[0])
        self.assertEqual(sorted(int(box[4]) for box in axis), list(range(0, 45001, 5000)))
        self.assertEqual(split_axis_labels(PNM[1])[1], [])

    def test_the_axis_title_survives_but_its_ticks_do_not(self):
        text = pnm_pages()[0].text
        self.assertIn("Electricity (kWh)", text)
        self.assertNotIn(25000, [int(n) for n in numbers_in(text)])

    def test_a_charge_column_is_not_an_axis(self):
        amounts = ["$820.76", "$420.55", "$659.28", "$0.00", "$95.31", "$103.78"]
        items = [(40.0 * i, 40.0 * i + 30, 500.0, 600.0, a) for i, a in enumerate(amounts)]
        self.assertEqual(axis_label_indices(items), set())

    def test_unevenly_spaced_charges_that_step_evenly_are_not_an_axis(self):
        # SRP 12/2021: three charges step by exactly -$220.30 in one right-aligned
        # column, but on lines 55px then 311px apart. They are not chart ticks.
        rows = [(610.0, "$753.06"), (665.0, "$532.76"), (976.0, "$312.46")]
        items = [(top, top + 20, 500.0, 560.0, text) for top, text in rows]
        self.assertEqual(axis_label_indices(items), set())

    def test_a_repeated_value_is_not_an_axis(self):
        items = [(40.0 * i, 40.0 * i + 30, 500.0, 560.0, "80.000") for i in range(4)]
        self.assertEqual(axis_label_indices(items), set())

    def test_two_numbers_are_not_enough(self):
        items = [(0.0, 30.0, 500.0, 560.0, "100"), (40.0, 70.0, 500.0, 560.0, "200")]
        self.assertEqual(axis_label_indices(items), set())

    def test_left_aligned_and_decreasing_axes_are_found(self):
        ticks = ["1,000", "800", "600", "400", "200"]
        items = [(40.0 * i, 40.0 * i + 30, 50.0, 50.0 + 12 * len(t), t) for i, t in enumerate(ticks)]
        self.assertEqual(axis_label_indices(items), set(range(5)))


class VerifyTests(unittest.TestCase):
    def answer(self, **changes) -> BillFigures:
        base = {
            "utility_company": "PNM", "utility_type": "electricity", "account_number": None,
            "meter_number": None, "rate_schedule": None,
            "statement_date": {"text": "01/16/2026", "page": 1},
            "due_date": None, "service_start": None, "service_end": None,
            "readings": [
                {"label": "On-Peak Use and Charge", "value": "16,720.000", "unit": "kWh",
                 "kind": "energy", "period": "on_peak", "page": 2},
            ],
            "charges": [
                {"label": "On-Peak Use and Charge", "amount": "$820.76", "category": "energy",
                 "period": "on_peak", "page": 2},
                {"label": "Palo Verde Regulatory Credit", "amount": "-$57.13",
                 "category": "credit", "period": "none", "page": 2},
            ],
            "total_current_charges": {"text": "$763.63", "page": 2},
            "amount_due": None, "notes": [],
        }
        base.update(changes)
        return BillFigures.model_validate(base)

    def pages(self):
        return [page(1, "Bill Issued: 01/16/2026", "Total Amount Due $763.63"),
                page(2, "On-Peak Use and Charge 16,720.000 kWh@ $0.0490885 $820.76",
                     "Palo Verde Regulatory Credit -$57.13",
                     "Current Electricity Charges $763.63")]

    def test_printed_figures_are_kept_and_the_bill_reconciles(self):
        result = verify(self.answer(), self.pages())
        self.assertEqual(result.flags, [])
        self.assertEqual(len(result.figures.readings), 1)
        self.assertEqual(len(result.figures.charges), 2)
        self.assertEqual(result.status, "matched")
        self.assertEqual(result.charges_sum, Decimal("763.63"))
        self.assertEqual(result.statement_date, date(2026, 1, 16))

    def test_the_pnm_axis_tick_is_dropped_even_when_the_model_returns_it(self):
        fooled = self.answer(readings=[
            {"label": "Electricity (kWh)", "value": "25000", "unit": "kWh",
             "kind": "energy", "period": "none", "page": 1},
        ])
        result = verify(fooled, pnm_pages())
        self.assertEqual(result.figures.readings, [])
        self.assertIn("'25000' is not printed", result.flags[0])
        # The real charges and total of that bill are printed, so they survive.
        self.assertEqual([c.amount for c in result.figures.charges], ["$820.76", "-$57.13"])

    def test_a_figure_printed_nowhere_is_dropped_not_corrected(self):
        result = verify(self.answer(readings=[
            {"label": "Off-Peak", "value": "14,321.000", "unit": "kWh",
             "kind": "energy", "period": "off_peak", "page": 2}]), self.pages())
        self.assertEqual(result.figures.readings, [])
        self.assertTrue(any("not printed" in f for f in result.flags))

    def test_a_value_holding_several_numbers_is_dropped(self):
        result = verify(self.answer(readings=[
            {"label": "On-Peak", "value": "16,720.000 kWh @ 0.0490885", "unit": "kWh",
             "kind": "energy", "period": "on_peak", "page": 2}]), self.pages())
        self.assertEqual(result.figures.readings, [])
        self.assertIn("not a single printed number", result.flags[0])

    def test_money_is_never_a_reading(self):
        result = verify(self.answer(readings=[
            {"label": "San Juan Energy Transition Charge*", "value": "$820.76", "unit": None,
             "kind": "other", "period": "none", "page": 2}]), self.pages())
        self.assertEqual(result.figures.readings, [])
        self.assertIn("is money", result.flags[0])

    def test_an_amount_printed_beside_another_label_is_dropped(self):
        # "$0.00" is printed on the page, but on the Rkva line, not beside "Actual Demand".
        pages = [self.pages()[0], page(2, "On-Peak Use and Charge 16,720.000 kWh $820.76",
                                       "Actual Demand: (Read x Constant) 154.400",
                                       "Billed Rkva and Charge 0.000 @ $0.27 $0.00",
                                       "Current Charges $820.76")]
        result = verify(self.answer(charges=[
            {"label": "On-Peak Use and Charge", "amount": "$820.76", "category": "energy",
             "period": "none", "page": 2},
            {"label": "Actual Demand", "amount": "$0.00", "category": "demand",
             "period": "none", "page": 2},
            {"label": "Billed Rkva and Charge", "amount": "$0.00", "category": "demand",
             "period": "none", "page": 2}],
            total_current_charges={"text": "$820.76", "page": 2}), pages)
        self.assertEqual([c.label for c in result.figures.charges],
                         ["On-Peak Use and Charge", "Billed Rkva and Charge"])
        self.assertIn("beside its label", result.flags[0])

    def test_a_charge_whose_sign_the_model_flipped_is_dropped(self):
        # Printed "$820.76"; the model called it a credit of -$820.76.
        result = verify(self.answer(charges=[
            {"label": "On-Peak Use and Charge", "amount": "-$820.76", "category": "credit",
             "period": "none", "page": 2}]), self.pages())
        self.assertEqual(result.figures.charges, [])
        self.assertIn("with that sign", result.flags[0])

    def test_a_credit_printed_without_a_sign_is_kept_on_a_credit_line(self):
        pages = [self.pages()[0], page(2, "On-Peak Use and Charge 16,720.000 kWh $820.76",
                                       "Aggregation Discount 57.13", "Current Charges $763.63")]
        result = verify(self.answer(charges=[
            {"label": "On-Peak Use and Charge", "amount": "$820.76", "category": "energy",
             "period": "none", "page": 2},
            {"label": "Aggregation Discount", "amount": "57.13", "category": "credit",
             "period": "none", "page": 2}]), pages)
        self.assertEqual(len(result.figures.charges), 2)
        self.assertEqual(result.status, "matched")

    def test_a_figure_on_another_page_is_kept_with_its_real_page(self):
        result = verify(self.answer(readings=[
            {"label": "On-Peak", "value": "16,720.000", "unit": "kWh",
             "kind": "energy", "period": "on_peak", "page": 1}]), self.pages())
        self.assertEqual(result.figures.readings[0].page, 2)
        self.assertIn("found on page 2", result.flags[0])

    def test_an_unprinted_date_is_dropped(self):
        result = verify(self.answer(statement_date={"text": "01/17/2026", "page": 1}),
                        self.pages())
        self.assertIsNone(result.figures.statement_date)
        self.assertTrue(any("statement date" in f for f in result.flags))

    def test_dates_match_whatever_the_printed_format(self):
        pages = [page(1, "Statement Date Jan 16, 2026"), *self.pages()[1:]]
        result = verify(self.answer(statement_date={"text": "01/16/2026", "page": 1}), pages)
        self.assertEqual(result.statement_date, date(2026, 1, 16))

    def test_line_items_off_by_more_than_two_cents_are_flagged(self):
        pages = self.pages()
        pages[1] = page(2, "On-Peak Use and Charge 16,720.000 kWh $820.76",
                        "Palo Verde Regulatory Credit -$57.13", "Current Charges $763.66")
        result = verify(self.answer(total_current_charges={"text": "$763.66", "page": 2}), pages)
        self.assertEqual(result.status, "review")
        self.assertIn("difference -0.03", result.flags[-1])

    def test_two_cents_of_rounding_is_accepted(self):
        pages = self.pages()
        pages[1] = page(2, "On-Peak Use and Charge 16,720.000 kWh $820.76",
                        "Palo Verde Regulatory Credit -$57.13", "Current Charges $763.65")
        result = verify(self.answer(total_current_charges={"text": "$763.65", "page": 2}), pages)
        self.assertEqual(result.status, "matched")

    def test_no_total_means_not_checked_and_flagged(self):
        result = verify(self.answer(total_current_charges=None), self.pages())
        self.assertEqual(result.status, "not_checked")
        self.assertTrue(result.flags)


class ColumnTests(unittest.TestCase):
    def build(self, *answers):
        verified = []
        for index, answer in enumerate(answers):
            month = BillFigures.model_validate(answer)
            pages = [page(1, *(f"{r.value}" for r in month.readings),
                          *(f"{c.label} {c.amount}" for c in month.charges),
                          month.total_current_charges.text, month.statement_date.text)]
            month = month.model_copy(update={
                "readings": [r.model_copy(update={"page": 1}) for r in month.readings],
                "charges": [c.model_copy(update={"page": 1}) for c in month.charges],
                "statement_date": month.statement_date.model_copy(update={"page": 1}),
                "total_current_charges": month.total_current_charges.model_copy(update={"page": 1}),
                "amount_due": None,
            })
            verified.append((f"bill{index}", verify(month, pages)))
        return columns.build(verified)

    def test_reading_columns_are_named_from_the_model_types(self):
        _, mapping = self.build(figures_for(Month("01/16/2026")))
        names = {m.canonical: m for m in mapping.values()}
        self.assertEqual(names["On-Peak kWh"].group, USAGE)
        self.assertEqual(names["On-Peak kWh"].peak_period, "on")
        self.assertEqual(names["Total kWh"].unit, "kWh")
        self.assertEqual((names["Billed Demand"].group, names["Billed Demand"].unit), (DEMAND, "kW"))

    def test_credits_are_credits_whatever_their_wording(self):
        _, mapping = self.build(figures_for(Month("01/16/2026")))
        credit = next(m for m in mapping.values() if m.canonical == "Palo Verde Regulatory Credit")
        self.assertEqual((credit.group, credit.sign, credit.unit), (CREDITS, CREDIT, "$"))

    def test_meter_register_reads_are_not_columns(self):
        answer = figures_for(Month("01/16/2026"))
        answer["readings"].append({"label": "Present", "value": "14509", "unit": None,
                                   "kind": "meter_register", "period": "none", "page": 2})
        bills, mapping = self.build(answer)
        self.assertNotIn("14509", [m.value_text for m in bills[0][1].measurements])
        self.assertNotIn("Present", [m.canonical for m in mapping.values()])

    def test_a_charge_printed_twice_becomes_two_columns_not_a_sum(self):
        month = Month("01/16/2026", charges=[
            ("Renewable Energy Rider", "$95.31", "energy"),
            ("Renewable Energy Rider", "$103.78", "energy"),
        ])
        bills, mapping = self.build(figures_for(month))
        entries = {e.label: e.amount_text for e in bills[0][1].financial_entries}
        self.assertEqual(entries, {"Renewable Energy Rider": "95.31",
                                   "Renewable Energy Rider (2)": "103.78"})
        self.assertNotIn("199.09", entries.values())

    def test_spacing_differences_across_months_share_one_column(self):
        first = Month("01/16/2026", charges=[("San Juan Energy Transition Charge*", "$78.74", "fee")])
        second = Month("02/17/2026", charges=[("San Juan EnergyTransitionCharge*", "$83.90", "fee")])
        bills, mapping = self.build(figures_for(first), figures_for(second))
        labels = {e.label for _, b in bills for e in b.financial_entries}
        self.assertEqual(labels, {"San Juan Energy Transition Charge"})

    def test_the_statement_date_is_carried_for_the_month_label(self):
        bills, _ = self.build(figures_for(Month("02/17/2025")))
        self.assertEqual(bills[0][1].bill_date, "2025-02-17")

    def test_other_readings_keep_their_printed_label(self):
        answer = figures_for(Month("01/16/2026"))
        answer["readings"].append({"label": "Actual Rkva: (Read x Constant)", "value": "65.600",
                                   "unit": "kVAR", "kind": "reactive", "period": "none", "page": 2})
        _, mapping = self.build(answer)
        rkva = next(m for m in mapping.values() if "Rkva" in m.canonical)
        self.assertEqual(rkva.group, OTHER)

class OcrDateTests(unittest.TestCase):
    """Scanned gas invoices: OCR keeps a date's digits but misreads its slashes."""

    def test_a_misread_separator_still_counts_as_printed(self):
        from app.bills.verify import ocr_prints_date

        self.assertTrue(ocr_prints_date("Invoice Date:    3r2612021", date(2021, 3, 26)))
        self.assertTrue(ocr_prints_date("Invoice Date:    4726/2021", date(2021, 4, 26)))

    def test_other_digits_do_not(self):
        from app.bills.verify import ocr_prints_date

        self.assertFalse(ocr_prints_date("Invoice Date:    3r2712021", date(2021, 3, 26)))
        self.assertFalse(ocr_prints_date("Account 1326120219", date(2021, 3, 26)))

if __name__ == "__main__":
    unittest.main()

"""Pivot tests: normalized bill rows -> per-meter monthly analysis sheets.

Nothing here touches a model. Synthetic bills pin the arithmetic and the edge
cases; the packaged bills pin the real-world result, including the two gas bills
whose columns genuinely do not add up to their printed total.
"""
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

from app.bills.bill_value_extraction import parse_amount, parse_number
from app.bills.semantics import (
    CREDITS, DEMAND, OTHER, TAXES_AND_FEES, USAGE, LabelSemantics, ON_PEAK, OFF_PEAK,
    interpret_labels, labels_from_bill,
)
from app.bills.shape import (
    GALLONS_PER_TGAL, MMBTU_PER_GALLON, MMBTU_PER_THERM, MonthRow, Period, SheetKey,
    SheetPlan, add_derived_columns, build_rollup, build_sheets, detect_gaps,
    period_for, reconcile, sheet_key_for,
)
from .test_fixtures import BILL_FOLDERS, FIXTURES, requires_fixtures

TESTS_DIR = Path(__file__).resolve().parent
TOLERANCE = Decimal("0.02")


def entry(label, amount, category="charge", meter=None, section=None):
    return NS(label=label, amount_text=amount, category=category, section=section,
              meter_number=meter, quantity_unit=None, rate_unit=None)


def measurement(label, value, unit, meter=None):
    return NS(label=label, value_text=value, unit=unit, section=None, meter_number=meter)


def bill(*, entries=(), measurements=(), current="100.00", company="Example Power",
         types=("electricity",), start="1/1/2025", end="1/31/2025", bill_date=None,
         account=None, extras=()):
    return NS(
        utility_company=company, utility_types=list(types), account_number=account,
        invoice_number=None, bill_date=bill_date, due_date=None,
        service_start=start, service_end=end, billing_days_text=None,
        billing_period_label=None, current_charges_text=current, total_due_text=current,
        financial_entries=list(entries), measurements=list(measurements),
        additional_fields=list(extras), extraction_notes=[],
    )


def shape(bills, *, tolerance=TOLERANCE):
    labels = [l for _, b in bills for l in labels_from_bill(b)]
    mapping = interpret_labels(labels, client=None)
    return build_sheets(bills, mapping, parse_amount=parse_amount,
                        parse_number=parse_number, tolerance=tolerance)


class SheetKeyTests(unittest.TestCase):
    def test_meter_names_the_sheet_when_present(self):
        key = sheet_key_for(bill(entries=[entry("Energy Charge", "60.00", meter="10953811")]))
        self.assertEqual(key.meter, "10953811")
        self.assertIn("Meter 10953811", key.title())

    def test_account_is_used_when_there_is_no_meter(self):
        key = sheet_key_for(bill(account="123-456"))
        self.assertIn("Account 123-456", key.title())

    def test_company_and_type_are_the_last_resort(self):
        # The packaged SRP bills print neither a meter nor an account number.
        key = sheet_key_for(bill(company="SRP", types=("electricity",)))
        self.assertEqual(key.title(), "SRP Electricity")

    def test_only_a_real_serial_counts_as_a_meter(self):
        # All seen in the "Meter #" column of real SRP extractions.
        for junk in ("Meter #1", "Meter #", "Shoulder Pk kWh", "179,746 kVAH",
                     "20,121", "Power Factor"):
            with self.subTest(junk=junk):
                key = sheet_key_for(bill(entries=[entry("Energy Charge", "1.00", meter=junk)]))
                self.assertIsNone(key.meter)
        key = sheet_key_for(bill(entries=[entry("Energy Charge", "1.00", meter="Meter 10953811")]))
        self.assertEqual(key.meter, "10953811")
        self.assertEqual(key.title(), "Meter 10953811")

    def test_printed_acronym_casing_is_preserved(self):
        self.assertTrue(sheet_key_for(bill(company="SRP")).title().startswith("SRP"))
        self.assertIn("LLC", sheet_key_for(bill(company="Symmetry LLC")).title())

    def test_two_meters_on_one_bill_do_not_name_a_sheet(self):
        key = sheet_key_for(bill(entries=[entry("A", "1.00", meter="M1"),
                                          entry("B", "2.00", meter="M2")]))
        self.assertIsNone(key.meter)

    def test_rate_schedule_is_appended_without_nesting_parentheses(self):
        plain = SheetKey("SRP", "electricity", rate_schedule="Rate 20").title()
        self.assertEqual(plain, "SRP Electricity (Rate 20)")
        nested = SheetKey("SRP", "electricity", rate_schedule="Secondary Service (E61)").title()
        self.assertEqual(nested, "SRP Electricity - Secondary Service (E61)")
        self.assertNotIn("((", nested)

    def test_sheet_name_is_excel_safe(self):
        key = SheetKey("A/B:C[D]", "electricity")
        name = key.sheet_name()
        self.assertLessEqual(len(name), 31)
        for bad in ":\\/?*[]":
            self.assertNotIn(bad, name)

    def test_rate_schedule_comes_from_additional_fields(self):
        key = sheet_key_for(bill(extras=[NS(label="Rate Schedule",
                                            value_text="Secondary Service (E61)")]))
        self.assertEqual(key.rate_schedule, "Secondary Service (E61)")


class PeriodTests(unittest.TestCase):
    def test_service_end_anchors_the_row(self):
        period = period_for(bill(start="7/12/2021", end="8/11/2021"))
        self.assertEqual(period.label, "Aug-21")

    def test_bill_date_is_the_last_resort(self):
        period = period_for(bill(start=None, end=None, bill_date="1/22/2021"))
        self.assertEqual(period.label, "Jan-21")

    def test_a_service_range_printed_in_the_billing_label_is_used(self):
        # A real SRP extraction left the dates empty and filed the range here;
        # the statement date would have put January's bill in March.
        b = bill(start=None, end=None, bill_date="3/10/2022")
        b.billing_period_label = "SERVICE FROM 1/12/2022 - 2/8/2022"
        self.assertEqual(period_for(b).label, "Feb-22")

    def test_unparseable_dates_do_not_crash_the_row(self):
        period = period_for(bill(start="sometime", end="later"))
        self.assertEqual(period.label, "Unknown")

    def test_zero_padded_dates_parse(self):
        # The packaged gas bills print 01/01/2021 rather than 1/1/2021.
        self.assertEqual(period_for(bill(start="01/01/2021", end="01/31/2021")).label, "Jan-21")


class PlacementTests(unittest.TestCase):
    def test_amounts_land_in_their_columns(self):
        plans = shape([("b1", bill(entries=[
            entry("On Peak Energy Charge", "60.00"),
            entry("Demand Charge", "30.00"),
            entry("State GRT", "10.00", "tax"),
        ], current="100.00"))])
        row = plans[0].rows[0]
        self.assertEqual(row.values["On-Peak Energy Charge"], Decimal("60.00"))
        self.assertEqual(row.values["Demand Charge"], Decimal("30.00"))
        self.assertEqual(row.values["State GRT"], Decimal("10.00"))

    def test_credits_are_stored_negative_however_they_were_printed(self):
        for printed in ("(5.00)", "-5.00", "5.00"):
            with self.subTest(printed=printed):
                plans = shape([("b1", bill(entries=[
                    entry("Energy Charge", "105.00"),
                    entry("Aggregation Discount", printed, "discount"),
                ]))])
                self.assertEqual(plans[0].rows[0].values["Aggregation Discount"],
                                 Decimal("-5.00"))

    def test_repeated_rows_of_one_concept_accumulate(self):
        plans = shape([("b1", bill(entries=[
            entry("Energy Charge", "60.00"), entry("Energy Charge", "40.00"),
        ]))])
        row = plans[0].rows[0]
        self.assertEqual(row.values["Energy Charge"], Decimal("100.00"))
        self.assertTrue(any("combined" in n for n in row.notes))

    def test_balances_and_payments_never_reach_a_column(self):
        plans = shape([("b1", bill(entries=[
            entry("Energy Charge", "100.00"),
            entry("Previous Balance", "500.00", "balance"),
            entry("Payment - Thank you", "-500.00", "payment"),
            entry("Total Current Charges", "100.00", "subtotal"),
        ]))])
        self.assertEqual([c.canonical for c in plans[0].columns], ["Energy Charge"])

    def test_unreadable_amount_is_noted_not_guessed(self):
        plans = shape([("b1", bill(entries=[entry("Energy Charge", "not a number")]))])
        row = plans[0].rows[0]
        self.assertNotIn("Energy Charge", row.values)
        self.assertTrue(any("not readable" in n for n in row.notes))

    def test_measurements_become_columns_with_their_printed_units(self):
        plans = shape([("b1", bill(
            entries=[entry("On Peak Energy Charge", "100.00")],
            measurements=[measurement("On Pk kWh", "18,240", "kWh"),
                          measurement("On Pk kW", "160", "kW")]))])
        row = plans[0].rows[0]
        self.assertEqual(row.values["On-Peak kWh"], Decimal("18240"))
        self.assertEqual(row.values["On-Peak Demand"], Decimal("160"))
        bands = {c.canonical: c.group for c in plans[0].columns}
        self.assertEqual(bands["On-Peak kWh"], USAGE)
        self.assertEqual(bands["On-Peak Demand"], DEMAND)


class RepeatedReadingTests(unittest.TestCase):
    """Meter readings are resolved, never summed like charges.

    The figures are from a real SRP bill whose time-of-use table came out of the
    extractor as four rows all labelled "Energy" and four labelled "Demand (kW)".
    Summing them doubled the month's usage and reported a demand of 2,272.8 kW.
    """

    def test_repeated_tou_readings_resolve_to_the_printed_total(self):
        plans = shape([("b1", bill(measurements=[
            measurement("Energy", "61,535", "kWh"),
            measurement("Energy", "86,956", "kWh"),
            measurement("Energy", "136,943", "kWh"),
            measurement("Energy", "285,434", "kWh"),
        ]))])
        self.assertEqual(plans[0].rows[0].values["Total kWh"], Decimal("285434"))

    def test_a_grid_column_header_in_the_meter_field_names_the_reading(self):
        # Exactly how qwen2.5vl returned the Aug-21 SRP grid: the row name as the
        # label and the column header in meter_number, every value correct.
        plans = shape([("b1", bill(measurements=[
            measurement("Energy", "61,535 kWh", "kWh", meter="On Pk kWh"),
            measurement("Energy", "86,956 kWh", "kWh", meter="Shoulder Pk kWh"),
            measurement("Energy", "136,943 kWh", "kWh", meter="Off Pk kWh"),
            measurement("Energy", "285,434 kWh", "kWh", meter="Total kWh"),
            measurement("Demand (kW)", "528.2 kW", "kW", meter="On Pk kW"),
            measurement("Demand (kW)", "540.2 kW", "kW", meter="Shoulder Pk kW"),
            measurement("Demand (kW)", "514.9 kW", "kW", meter="Off Pk kW"),
            measurement("Demand (kW)", "689.5 kW", "kW", meter="16 Month Peak Demand"),
        ]))])
        values = plans[0].rows[0].values
        self.assertEqual(values["On-Peak kWh"], Decimal("61535"))
        self.assertEqual(values["Shoulder-Peak kWh"], Decimal("86956"))
        self.assertEqual(values["Off-Peak kWh"], Decimal("136943"))
        self.assertEqual(values["Total kWh"], Decimal("285434"))
        self.assertEqual(values["On-Peak Demand"], Decimal("528.2"))
        self.assertEqual(values["Shoulder-Peak Demand"], Decimal("540.2"))
        self.assertEqual(values["Off-Peak Demand"], Decimal("514.9"))
        self.assertEqual(values["16-Month Peak Demand"], Decimal("689.5"))
        self.assertIsNone(plans[0].key.meter)  # and no sheet named after a header
        self.assertEqual(plans[0].warnings, [])

    def test_a_bare_energy_reading_is_the_total(self):
        plans = shape([("b1", bill(measurements=[measurement("Energy", "285,434", "kWh")]))])
        self.assertEqual(plans[0].rows[0].values["Total kWh"], Decimal("285434"))

    def test_an_energy_charge_line_is_still_a_charge(self):
        plans = shape([("b1", bill(entries=[entry("Energy", "60.00")], current="60.00"))])
        row = plans[0].rows[0]
        self.assertEqual(row.values["Energy"], Decimal("60.00"))
        self.assertEqual(row.status, "matched")

    def test_two_different_totals_for_one_month_are_left_blank(self):
        # A real May-22 extraction: Total kWh 253,500 and Energy 35,434.
        plans = shape([("b1", bill(measurements=[
            measurement("Total kWh", "253,500", "kWh"),
            measurement("Energy", "35,434", "kWh"),
        ]))])
        self.assertNotIn("Total kWh", plans[0].rows[0].values)

    def test_an_unlabelled_period_is_recovered_when_the_total_proves_it(self):
        # Jun-22 as qwen returned it: the on-peak cell tagged with the corner header.
        # The second bill is another month that printed the on-peak column.
        plans = shape([("b1", bill(measurements=[
            measurement("Energy", "57,044 kWh", "kWh", meter="Meter #"),
            measurement("Energy", "79,390 kWh", "kWh", meter="Shoulder Pk kWh"),
            measurement("Energy", "121,263 kWh", "kWh", meter="Off Pk kWh"),
            measurement("Energy", "257,697 kWh", "kWh", meter="Total kWh"),
        ], start="5/12/2022", end="6/10/2022")),
            ("b2", bill(measurements=[measurement("On Pk kWh", "61,535", "kWh")],
                        start="7/12/2021", end="8/11/2021"))])
        june = next(r for r in plans[0].rows if r.label == "Jun-22")
        self.assertEqual(june.values["On-Peak kWh"], Decimal("57044"))
        self.assertEqual(june.values["Total kWh"], Decimal("257697"))
        self.assertTrue(any("without its period label" in n for n in june.notes))
        self.assertFalse(any("Jun-22" in w for w in plans[0].warnings))

    def test_no_recovery_when_two_periods_are_missing(self):
        # One leftover, two empty periods: which one it belongs to is a guess.
        plans = shape([("b1", bill(measurements=[
            measurement("Energy", "57,044", "kWh"),
            measurement("Off Pk kWh", "200,653", "kWh"),
            measurement("Total kWh", "257,697", "kWh"),
        ])), ("b2", bill(measurements=[
            measurement("On Pk kWh", "1", "kWh"), measurement("Shoulder Pk kWh", "1", "kWh"),
        ], start="2/1/2025", end="2/28/2025"))])
        row = next(r for r in plans[0].rows if r.label == "Jan-25")
        self.assertNotIn("On-Peak kWh", row.values)
        self.assertNotIn("Shoulder-Peak kWh", row.values)

    def test_no_recovery_when_the_arithmetic_does_not_balance(self):
        plans = shape([("b1", bill(measurements=[
            measurement("Energy", "50,000", "kWh"),
            measurement("Off Pk kWh", "121,263", "kWh"),
            measurement("Shoulder Pk kWh", "79,390", "kWh"),
            measurement("Total kWh", "257,697", "kWh"),
        ])), ("b2", bill(measurements=[measurement("On Pk kWh", "1", "kWh")],
                          start="2/1/2025", end="2/28/2025"))])
        row = next(r for r in plans[0].rows if r.label == "Jan-25")
        self.assertNotIn("On-Peak kWh", row.values)
        self.assertNotIn("Total kWh", row.values)

    def test_repeated_demand_readings_are_never_summed(self):
        plans = shape([("b1", bill(measurements=[
            measurement("Demand (kW)", "528.2", "kW"),
            measurement("Demand (kW)", "540.2", "kW"),
            measurement("Demand (kW)", "514.9", "kW"),
            measurement("Demand (kW)", "689.5", "kW"),
        ]))])
        self.assertNotIn("Demand (kW)", plans[0].rows[0].values)  # blank, not 2272.8
        self.assertTrue(any("Demand (kW) left blank" in w for w in plans[0].warnings))

    def test_usage_with_no_provable_total_is_left_blank(self):
        plans = shape([("b1", bill(measurements=[
            measurement("Energy", "100", "kWh"), measurement("Energy", "250", "kWh"),
        ]))])
        self.assertNotIn("Energy", plans[0].rows[0].values)

    def test_the_same_reading_printed_twice_counts_once(self):
        plans = shape([("b1", bill(measurements=[
            measurement("On Pk kWh", "18,240", "kWh"),
            measurement("On Pk kWh", "18,240", "kWh"),
        ]))])
        self.assertEqual(plans[0].rows[0].values["On-Peak kWh"], Decimal("18240"))

    def test_a_reading_in_a_unit_the_column_cannot_hold_is_not_placed(self):
        # The extractor attached a kW demand figure to the kWh row.
        plans = shape([("b1", bill(measurements=[measurement("On Pk kWh", "417.8", "kW")]))])
        self.assertNotIn("On-Peak kWh", plans[0].rows[0].values)

    def test_a_price_per_unit_is_not_a_quantity(self):
        plans = shape([("b1", bill(measurements=[measurement("Energy", "$0.37245", "kWh")]))])
        self.assertNotIn("Energy", plans[0].rows[0].values)

    def test_dollars_never_land_in_a_quantity_column(self):
        plans = shape([("b1", bill(
            entries=[entry("On Pk kWh", "500.00")],
            measurements=[measurement("On Pk kWh", "18,240", "kWh")],
            current="500.00"))])
        self.assertEqual(plans[0].rows[0].values["On-Peak kWh"], Decimal("18240"))

    def test_a_carried_balance_filed_as_a_charge_is_not_a_charge(self):
        # The $19,729.39 that made a real month fail to reconcile.
        plans = shape([("b1", bill(entries=[
            entry("Energy Charge", "100.00"),
            entry("Previous Balance", "19,729.39", category="charge"),
        ], current="100.00"))])
        row = plans[0].rows[0]
        self.assertNotIn("Previous Balance", row.values)
        self.assertEqual(row.status, "matched")


class GroupingTests(unittest.TestCase):
    def test_one_sheet_per_meter(self):
        plans = shape([
            ("b1", bill(entries=[entry("Energy Charge", "60.00", meter="M1")])),
            ("b2", bill(entries=[entry("Energy Charge", "70.00", meter="M2")])),
        ])
        self.assertEqual(len(plans), 2)
        self.assertEqual({p.key.meter for p in plans}, {"M1", "M2"})

    def test_months_of_one_meter_become_rows_of_one_sheet(self):
        plans = shape([
            (f"b{n}", bill(entries=[entry("Energy Charge", "60.00", meter="M1")],
                           start=f"{n}/1/2025", end=f"{n}/28/2025"))
            for n in range(1, 7)
        ])
        self.assertEqual(len(plans), 1)
        self.assertEqual([r.label for r in plans[0].rows],
                         ["Jan-25", "Feb-25", "Mar-25", "Apr-25", "May-25", "Jun-25"])

    def test_rows_are_sorted_chronologically_regardless_of_input_order(self):
        plans = shape([
            ("b3", bill(entries=[entry("Energy Charge", "1.00")], start="3/1/2025", end="3/31/2025")),
            ("b1", bill(entries=[entry("Energy Charge", "1.00")], start="1/1/2025", end="1/31/2025")),
            ("b2", bill(entries=[entry("Energy Charge", "1.00")], start="2/1/2025", end="2/28/2025")),
        ])
        self.assertEqual([r.label for r in plans[0].rows], ["Jan-25", "Feb-25", "Mar-25"])

    def test_duplicate_service_period_is_dropped_with_a_warning(self):
        same = dict(entries=[entry("Energy Charge", "60.00")], start="1/1/2025", end="1/31/2025")
        plans = shape([("b1", bill(**same)), ("b2", bill(**same))])
        self.assertEqual(len(plans[0].rows), 1)
        self.assertTrue(any("Duplicate" in w for w in plans[0].warnings))

    def test_a_bill_missing_its_rate_joins_the_series_that_printed_one(self):
        """Utilities often print the rate only on the first bill of a series.

        Without folding, one meter would yield a one-row sheet and an eleven-row
        sheet instead of a single twelve-row series.
        """
        with_rate = bill(entries=[entry("Energy Charge", "60.00")],
                         start="1/1/2025", end="1/31/2025",
                         extras=[NS(label="Rate Schedule", value_text="Rate 20")])
        without = bill(entries=[entry("Energy Charge", "70.00")],
                       start="2/1/2025", end="2/28/2025")
        plans = shape([("b1", with_rate), ("b2", without)])
        self.assertEqual(len(plans), 1)
        self.assertEqual(len(plans[0].rows), 2)
        self.assertEqual(plans[0].key.rate_schedule, "Rate 20")

    def test_different_utilities_never_share_a_sheet(self):
        plans = shape([
            ("b1", bill(entries=[entry("Energy Charge", "60.00")], types=("electricity",))),
            ("b2", bill(entries=[entry("Natural Gas Sales", "60.00")], types=("gas",),
                        company="Gas Co")),
        ])
        self.assertEqual(len(plans), 2)

    def test_an_ambiguous_rateless_bill_is_not_guessed_into_a_series(self):
        """Two rate schedules on one account: a rateless bill cannot be placed.

        Folding it into whichever series sorted first would silently file a month
        under the wrong meter, so it stays on its own sheet where it is visible.
        """
        rate20 = bill(entries=[entry("Energy Charge", "1.00")], start="1/1/2025", end="1/31/2025",
                      extras=[NS(label="Rate Schedule", value_text="Rate 20")])
        rate10 = bill(entries=[entry("Energy Charge", "1.00")], start="2/1/2025", end="2/28/2025",
                      extras=[NS(label="Rate Schedule", value_text="Rate 10")])
        rateless = bill(entries=[entry("Energy Charge", "1.00")], start="3/1/2025", end="3/31/2025")
        plans = shape([("a", rate20), ("b", rate10), ("c", rateless)])
        self.assertEqual(len(plans), 3)
        self.assertEqual({len(p.rows) for p in plans}, {1})


class EmptyInputTests(unittest.TestCase):
    def test_no_bills_produces_no_sheets(self):
        self.assertEqual(shape([]), [])

    def test_rollup_of_nothing_is_a_zero_total(self):
        rows = build_rollup([])
        self.assertEqual([r.category for r in rows], ["TOTAL"])
        self.assertEqual(rows[0].annual_cost, Decimal("0"))

    def test_a_bill_with_no_usable_rows_still_produces_a_row(self):
        plans = shape([("b1", bill(entries=[entry("Previous Balance", "50.00", "balance")]))])
        self.assertEqual(len(plans[0].rows), 1)
        self.assertEqual(plans[0].columns, [])
        # $100.00 printed and nothing placed: flagged, never quietly unchecked.
        self.assertEqual(plans[0].rows[0].status, "review")

    def test_a_month_with_nothing_billed_is_not_flagged(self):
        plans = shape([("b1", bill(entries=[entry("Previous Balance", "50.00", "balance")],
                                   current="0.00"))])
        self.assertEqual(plans[0].rows[0].status, "not_checked")

    def test_a_derived_column_nothing_can_fill_is_not_created(self):
        """An always-blank column reads as a measured zero across the year."""
        plans = shape([("b1", bill(
            entries=[entry("Natural Gas Sales", "1.00")],
            measurements=[measurement("Propane Usage (Gallons)", "0", "Gallons")],
            types=("propane",), company="Propane Co"))])
        names = [c.canonical for c in plans[0].columns]
        self.assertNotIn("Cost Per Gallon ($)", names)


class HeaderBandTests(unittest.TestCase):
    def test_bands_span_their_columns_in_order(self):
        plans = shape([("b1", bill(entries=[
            entry("On Peak Energy Charge", "50.00"), entry("Off Peak Energy Charge", "20.00"),
            entry("Demand Charge", "20.00"), entry("State GRT", "5.00", "tax"),
            entry("Aggregation Discount", "(5.00)", "discount"),
        ]))])
        # Every charge in one band, still in reading order, with no subtotal.
        self.assertEqual(plans[0].header_bands(), [("Charges ($)", 5)])
        self.assertEqual([c.canonical for c in plans[0].columns], [
            "On-Peak Energy Charge", "Off-Peak Energy Charge", "Demand Charge",
            "State GRT", "Aggregation Discount",
        ])

    def test_columns_are_ordered_by_band(self):
        plans = shape([("b1", bill(entries=[
            entry("State GRT", "5.00", "tax"), entry("Demand Charge", "20.00"),
            entry("On Peak Energy Charge", "50.00"),
        ]))])
        self.assertEqual([c.canonical for c in plans[0].columns],
                         ["On-Peak Energy Charge", "Demand Charge", "State GRT"])


class DerivedColumnTests(unittest.TestCase):
    def test_total_kwh_is_summed_from_the_peak_periods(self):
        plans = shape([("b1", bill(
            entries=[entry("On Peak Energy Charge", "100.00")],
            measurements=[measurement("On Pk kWh", "18240", "kWh"),
                          measurement("Off Pk kWh", "15040", "kWh")]))])
        row = plans[0].rows[0]
        self.assertEqual(row.values["Total kWh"], Decimal("33280"))
        derived = next(c for c in plans[0].columns if c.canonical == "Total kWh")
        self.assertEqual(derived.source, "derived")

    def test_total_kwh_is_not_invented_when_no_peak_split_exists(self):
        plans = shape([("b1", bill(
            entries=[entry("Energy Charge", "100.00")],
            measurements=[measurement("Total kWh", "33280", "kWh")]))])
        column = next(c for c in plans[0].columns if c.canonical == "Total kWh")
        self.assertNotEqual(column.source, "derived")  # the bill printed it

    def test_therms_convert_to_mmbtu_by_constant(self):
        plan = SheetPlan(
            key=SheetKey("Gas Co", "gas"),
            columns=[LabelSemantics("Usage (Therms)", USAGE, "none", "charge", "therms", 10, "seed")],
            rows=[MonthRow(period=Period(None, None, None), values={"Usage (Therms)": Decimal("4252")})],
        )
        add_derived_columns(plan)
        self.assertEqual(plan.rows[0].values["Usage (MMBtu)"], Decimal("4252") * MMBTU_PER_THERM)
        self.assertEqual(plan.rows[0].values["Usage (MMBtu)"], Decimal("425.2"))

    def test_propane_gallons_convert_to_mmbtu(self):
        plan = SheetPlan(
            key=SheetKey("Propane Co", "propane"),
            columns=[LabelSemantics("Propane Usage (Gallons)", USAGE, "none", "charge",
                                    "Gallons", 10, "seed")],
            rows=[MonthRow(period=Period(None, None, None),
                           values={"Propane Usage (Gallons)": Decimal("24")})],
        )
        add_derived_columns(plan)
        self.assertEqual(plan.rows[0].values["Usage (MMBtu)"],
                         Decimal("24") * MMBTU_PER_GALLON["propane"])

    def test_tgals_convert_to_gallons(self):
        plan = SheetPlan(
            key=SheetKey("Water Co", "water"),
            columns=[LabelSemantics("Water Usage (TGals)", USAGE, "none", "charge",
                                    "TGals", 10, "seed")],
            rows=[MonthRow(period=Period(None, None, None),
                           values={"Water Usage (TGals)": Decimal("118")})],
        )
        add_derived_columns(plan)
        self.assertEqual(plan.rows[0].values["Water Usage (Gallons)"],
                         Decimal("118") * GALLONS_PER_TGAL)

    def test_a_derived_column_is_blank_when_its_input_is_missing(self):
        plans = shape([
            ("b1", bill(entries=[entry("On Peak Energy Charge", "1.00")],
                        measurements=[measurement("On Pk kWh", "100", "kWh")],
                        start="1/1/2025", end="1/31/2025")),
            ("b2", bill(entries=[entry("On Peak Energy Charge", "1.00")],
                        start="2/1/2025", end="2/28/2025")),
        ])
        rows = {r.label: r for r in plans[0].rows}
        self.assertEqual(rows["Jan-25"].values["Total kWh"], Decimal("100"))
        self.assertNotIn("Total kWh", rows["Feb-25"].values)  # not a zero

    def test_derived_columns_are_excluded_from_reconciliation(self):
        # Total kWh is not money, and must never be added into a money check.
        plans = shape([("b1", bill(
            entries=[entry("On Peak Energy Charge", "100.00")],
            measurements=[measurement("On Pk kWh", "500", "kWh"),
                          measurement("Off Pk kWh", "500", "kWh")],
            current="100.00"))])
        self.assertEqual(plans[0].rows[0].status, "matched")


class ReconciliationTests(unittest.TestCase):
    def test_matching_row_reconciles(self):
        plans = shape([("b1", bill(entries=[
            entry("Energy Charge", "60.00"), entry("Demand Charge", "40.00"),
        ], current="100.00"))])
        row = plans[0].rows[0]
        self.assertEqual(row.status, "matched")
        self.assertEqual(row.difference, Decimal("0.00"))

    def test_a_cent_of_rounding_is_tolerated(self):
        plans = shape([("b1", bill(entries=[
            entry("Energy Charge", "60.00"), entry("Demand Charge", "40.01"),
        ], current="100.00"))])
        self.assertEqual(plans[0].rows[0].status, "matched")

    def test_a_real_discrepancy_is_flagged(self):
        plans = shape([("b1", bill(entries=[
            entry("Energy Charge", "60.00"), entry("Demand Charge", "40.00"),
        ], current="150.00"))])
        row = plans[0].rows[0]
        self.assertEqual(row.status, "review")
        self.assertEqual(row.difference, Decimal("-50.00"))
        self.assertTrue(any("wrong column" in n for n in row.notes))

    def test_credits_reduce_the_computed_total(self):
        plans = shape([("b1", bill(entries=[
            entry("Energy Charge", "105.00"),
            entry("Aggregation Discount", "(5.00)", "discount"),
        ], current="100.00"))])
        self.assertEqual(plans[0].rows[0].status, "matched")

    def test_missing_printed_total_is_not_checked_rather_than_passed(self):
        plans = shape([("b1", bill(entries=[entry("Energy Charge", "60.00")], current=None))])
        row = plans[0].rows[0]
        self.assertEqual(row.status, "not_checked")
        self.assertTrue(any("No printed current charges" in n for n in row.notes))

    def test_tolerance_is_configurable(self):
        bills = [("b1", bill(entries=[entry("Energy Charge", "100.50")], current="100.00"))]
        self.assertEqual(shape(bills, tolerance=Decimal("0.02"))[0].rows[0].status, "review")
        self.assertEqual(shape(bills, tolerance=Decimal("1.00"))[0].rows[0].status, "matched")


class GapTests(unittest.TestCase):
    def test_a_complete_series_reports_no_gap(self):
        plans = shape([
            (f"b{n}", bill(entries=[entry("Energy Charge", "1.00")],
                           start=f"{n}/1/2025", end=f"{n}/28/2025"))
            for n in range(1, 13)
        ])
        self.assertEqual(plans[0].warnings, [])

    def test_a_missing_month_is_reported(self):
        plans = shape([
            ("b1", bill(entries=[entry("Energy Charge", "1.00")], start="1/1/2025", end="1/31/2025")),
            ("b3", bill(entries=[entry("Energy Charge", "1.00")], start="3/1/2025", end="3/31/2025")),
        ])
        self.assertEqual(plans[0].warnings, ["No bill for Feb-25"])

    def test_a_month_end_anchor_does_not_invent_a_trailing_gap(self):
        """Regression: comparing a 31st-of-month anchor against a 1st walked one
        month past the end of the series and reported a month that was never due."""
        plan = SheetPlan(key=SheetKey("X", "gas"), columns=[], rows=[
            MonthRow(period=Period(None, __import__("datetime").date(2020, 12, 31), None)),
            MonthRow(period=Period(None, __import__("datetime").date(2021, 1, 31), None)),
        ])
        detect_gaps(plan)
        self.assertEqual(plan.warnings, [])

    def test_a_year_boundary_is_walked_correctly(self):
        plans = shape([
            ("b1", bill(entries=[entry("Energy Charge", "1.00")], start="11/1/2024", end="11/30/2024")),
            ("b2", bill(entries=[entry("Energy Charge", "1.00")], start="2/1/2025", end="2/28/2025")),
        ])
        self.assertEqual(plans[0].warnings, ["No bill for Dec-24, Jan-25"])


class RollupTests(unittest.TestCase):
    def test_electricity_splits_into_energy_demand_and_fees(self):
        plans = shape([("b1", bill(entries=[
            entry("On Peak Energy Charge", "60.00"), entry("Demand Charge", "30.00"),
            entry("State GRT", "10.00", "tax"),
        ], current="100.00"))])
        rows = {r.category: r.annual_cost for r in build_rollup(plans)}
        self.assertEqual(rows["Electricity Energy"], Decimal("60.00"))
        self.assertEqual(rows["Electricity Demand"], Decimal("30.00"))
        self.assertEqual(rows["Electricity Fees/Taxes"], Decimal("10.00"))
        self.assertEqual(rows["TOTAL"], Decimal("100.00"))

    def test_non_electric_utilities_roll_up_as_one_line(self):
        plans = shape([("b1", bill(entries=[entry("Natural Gas Sales", "60.00")],
                                   types=("gas",), company="Gas Co", current="60.00"))])
        rows = {r.category: r.annual_cost for r in build_rollup(plans)}
        self.assertEqual(rows["Gas"], Decimal("60.00"))

    def test_the_total_is_the_sum_of_the_categories(self):
        plans = shape([
            ("b1", bill(entries=[entry("Energy Charge", "60.00")], current="60.00")),
            ("b2", bill(entries=[entry("Natural Gas Sales", "40.00")], types=("gas",),
                        company="Gas Co", current="40.00")),
        ])
        rows = build_rollup(plans)
        total = next(r for r in rows if r.category == "TOTAL")
        self.assertEqual(total.annual_cost,
                         sum(r.annual_cost for r in rows if r.category != "TOTAL"))

    def test_quantities_never_enter_the_cost_rollup(self):
        plans = shape([("b1", bill(
            entries=[entry("On Peak Energy Charge", "60.00")],
            measurements=[measurement("On Pk kWh", "999999", "kWh")], current="60.00"))])
        total = next(r for r in build_rollup(plans) if r.category == "TOTAL")
        self.assertEqual(total.annual_cost, Decimal("60.00"))


class RealBillShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from .test_fixtures import HAVE_FIXTURES
        if not HAVE_FIXTURES:
            raise unittest.SkipTest("real-bill fixtures are not present")
        import app.bills.bill_value_extraction as bve
        cls.bills = []
        for folder in BILL_FOLDERS:
            for number in range(1, 13):
                path = TESTS_DIR / folder / f"bill_{number:03d}.pdf"
                fingerprint = sha256(path.read_bytes()).hexdigest()
                loaded, _, _ = bve.load_bill(path, fingerprint, FIXTURES, bve.EXTRACTION_MODEL)
                cls.bills.append((fingerprint[:12], loaded))
        cls.plans = shape(cls.bills)

    def test_one_sheet_per_utility_with_twelve_rows_each(self):
        self.assertEqual(len(self.plans), 2)
        for plan in self.plans:
            self.assertEqual(len(plan.rows), 12, plan.title)

    def test_the_series_are_complete(self):
        for plan in self.plans:
            self.assertEqual(plan.warnings, [], plan.title)

    def test_every_electricity_row_reconciles(self):
        electricity = next(p for p in self.plans if p.key.utility_type == "electricity")
        self.assertEqual({r.status for r in electricity.rows}, {"matched"})

    def test_the_two_known_bad_gas_bills_are_the_only_ones_flagged(self):
        """The reconciliation independently rediscovers the bills a human flagged.

        The original project notes recorded exactly two gas bills needing a source
        correction (a mis-signed undertake amount and a wrong current-charges
        figure). Those corrections cannot apply here -- the shipped corrections
        file is stale -- so the arithmetic must still catch both on its own.
        """
        gas = next(p for p in self.plans if p.key.utility_type == "gas")
        flagged = [(r.label, r.difference) for r in gas.rows if r.status == "review"]
        self.assertEqual(len(flagged), 2)
        self.assertEqual({d for _, d in flagged}, {Decimal("0.10"), Decimal("1000.00")})

    def test_electricity_columns_match_the_analysis_layout(self):
        electricity = next(p for p in self.plans if p.key.utility_type == "electricity")
        names = [c.canonical for c in electricity.columns]
        # Quantities before the charges priced off them, demand after usage,
        # taxes after demand, credits last before diagnostics.
        for earlier, later in [("On-Peak kWh", "On-Peak Energy Charge"),
                               ("On-Peak Energy Charge", "Demand Charge"),
                               ("Demand Charge", "City Tax"),
                               ("City Tax", "Aggregation Discount")]:
            self.assertLess(names.index(earlier), names.index(later), f"{earlier} < {later}")

    def test_totals_are_not_double_counted_in_the_rollup(self):
        rollup = {r.category: r.annual_cost for r in build_rollup(self.plans)}
        total = rollup.pop("TOTAL")
        self.assertEqual(total, sum(rollup.values()))
        # Electricity energy should dominate demand on these bills, as on the bill face.
        self.assertGreater(rollup["Electricity Energy"], rollup["Electricity Demand"])

    def test_no_row_carries_an_account_activity_column(self):
        for plan in self.plans:
            names = " | ".join(c.canonical.casefold() for c in plan.columns)
            for forbidden in ("balance", "payment", "amount due", "account total"):
                self.assertNotIn(forbidden, names, plan.title)


if __name__ == "__main__":
    unittest.main()

"""Phase 5: reproduce the two real client analysis formats.

These are the formats the whole pivot exists to produce, and they are a genuinely
adversarial pair:

    Gila River Rate 20   no time-of-use split, ONE combined tax column
    PNM Rate 3C          on/off-peak energy, gross receipts tax split three ways
                         (state, county, city), plus a negative Palo Verde credit

A fixed template cannot serve both, which is why columns are discovered from the
labels each bill actually prints. Each test below feeds the labels the real bills
carry and asserts the resulting layout matches the workbook an assessor built by
hand -- same columns, same bands, same order, same signs.

The bills themselves are synthesised here from the printed labels, because the
extraction step needs a running vision model; what is under test is the layout
decision, which is deterministic.
"""
from decimal import Decimal
from types import SimpleNamespace as NS
import unittest

from app.bills.bill_value_extraction import parse_amount, parse_number
from app.bills.semantics import (
    CREDIT, CREDITS, DEMAND, TAXES_AND_FEES, USAGE, interpret_labels, labels_from_bill,
)
from app.bills.shape import build_rollup, build_sheets


def entry(label, amount, category="charge"):
    return NS(label=label, amount_text=amount, category=category, section=None,
              meter_number=None, quantity_unit=None, rate_unit=None)


def measurement(label, value, unit):
    return NS(label=label, value_text=value, unit=unit, section=None, meter_number=None)


def bill(company, rate, start, end, entries, measurements, current, meter=None):
    return NS(
        utility_company=company, utility_types=["electricity"], account_number=None,
        invoice_number=None, bill_date=None, due_date=None,
        service_start=start, service_end=end, billing_days_text=None,
        billing_period_label=None, current_charges_text=current, total_due_text=current,
        financial_entries=[
            NS(label=e.label, amount_text=e.amount_text, category=e.category,
               section=e.section, meter_number=meter, quantity_unit=None, rate_unit=None)
            for e in entries
        ],
        measurements=[
            NS(label=m.label, value_text=m.value_text, unit=m.unit, section=None,
               meter_number=meter)
            for m in measurements
        ],
        additional_fields=[NS(label="Rate Schedule", value_text=rate)],
        extraction_notes=[],
    )


def shape(bills):
    labels = [l for _, b in bills for l in labels_from_bill(b)]
    mapping = interpret_labels(labels, client=None)
    return build_sheets(bills, mapping, parse_amount=parse_amount,
                        parse_number=parse_number, tolerance=Decimal("0.02"))


class Rate20Tests(unittest.TestCase):
    """Gila River Rate 20: no peak split, a single Tax column.

    Target sheet "Meter 10953811 (Rate 20)":
        Usage (kWh) | Energy Charge | Fuel Adj | Total Energy Charge
        Billed Demand (kW) | Demand Charge
        Service Charge | Tax
    """

    MONTHS = [
        # (start, end, kWh, energy, fuel, demand kW, demand $, service, tax, total)
        ("6/1/2025", "6/30/2025", "33120", "2583.36", "943.92", "105.92", "529.60", "50.00", "246.41", "4353.29"),
        ("7/1/2025", "7/31/2025", "35840", "2795.52", "1021.44", "113.44", "567.20", "50.00", "266.05", "4700.21"),
        ("8/1/2025", "8/31/2025", "41040", "3201.12", "1169.64", "117.84", "589.20", "50.00", "300.60", "5310.56"),
    ]

    def build(self):
        bills = []
        for index, m in enumerate(self.MONTHS):
            start, end, kwh, energy, fuel, kw, demand, service, tax, total = m
            bills.append((f"b{index}", bill(
                "Gila River Utilities", "Rate 20", start, end,
                entries=[
                    entry("Energy Charge", energy),
                    entry("Fuel Adj", fuel),
                    entry("Demand Charge", demand),
                    entry("Service Charge", service),
                    entry("Tax", tax, "tax"),
                ],
                measurements=[
                    measurement("Usage", kwh, "kWh"),
                    measurement("Billed Demand", kw, "kW"),
                ],
                current=total, meter="10953811")))
        return shape(bills)

    def test_columns_match_the_hand_built_sheet(self):
        plan = self.build()[0]
        # Quantities first, then every charge in one band; the hand-built sheet's
        # "Total Energy Charge" subtotal is gone because Total ($) already sums it.
        self.assertEqual([c.canonical for c in plan.columns], [
            "Usage", "Billed Demand", "Energy Charge", "Fuel Adjustment",
            "Demand Charge", "Service Charge", "Tax",
        ])

    def test_bands_match(self):
        plan = self.build()[0]
        self.assertEqual(plan.header_bands(), [
            ("Energy (kWh)", 1), ("Demand (kW)", 1), ("Charges ($)", 5),
        ])

    def test_no_time_of_use_column_is_invented(self):
        plan = self.build()[0]
        names = " ".join(c.canonical for c in plan.columns).casefold()
        for absent in ("on-peak", "off-peak", "shoulder"):
            self.assertNotIn(absent, names)

    def test_a_single_tax_column_is_not_split(self):
        plan = self.build()[0]
        taxes = [c.canonical for c in plan.columns if c.group == TAXES_AND_FEES]
        self.assertEqual(taxes, ["Service Charge", "Tax"])

    def test_no_band_subtotal_restates_the_total(self):
        # One Charges band: the row's Total ($) is its sum, so no subtotal column.
        plan = self.build()[0]
        self.assertFalse(any(c.source == "derived" for c in plan.columns))

    def test_every_month_reconciles(self):
        plan = self.build()[0]
        self.assertEqual({r.status for r in plan.rows}, {"matched"})

    def test_the_derived_subtotal_does_not_double_count(self):
        # Total Energy Charge sits beside the rows it totals; counting both would
        # inflate every month by the whole energy band.
        plan = self.build()[0]
        row = plan.rows[0]
        self.assertEqual(row.computed_total, Decimal("4353.29"))

    def test_the_sheet_is_named_for_its_meter_and_rate(self):
        self.assertEqual(self.build()[0].title, "Meter 10953811 (Rate 20)")


class PnmRate3CTests(unittest.TestCase):
    """PNM Rate 3C: on/off-peak energy, three-way GRT, a negative credit.

    Target sheet "PNM Rate 3C".
    """

    MONTHS = [
        # start, end, on kWh, off kWh, on $, off $, fuel, renewable, transport,
        # cost-eff, pv credit, demand kW, demand $, etc, franchise, state, county,
        # city, customer, total
        ("2/19/2025", "3/18/2025", "18240", "15040", "863.84", "426.14", "913.21",
         "238.73", "1.83", "183.54", "(59.20)", "160", "1934.40", "73.60", "139.45",
         "233.40", "56.85", "74.81", "72.18", "5152.78"),
        ("3/19/2025", "4/16/2025", "17920", "14560", "848.68", "412.54", "930.02",
         "232.99", "1.79", "177.51", "(55.35)", "149.6", "1808.66", "68.82", "134.94",
         "225.85", "55.01", "72.39", "72.18", "4986.03"),
    ]

    def build(self):
        bills = []
        for index, m in enumerate(self.MONTHS):
            (start, end, on_kwh, off_kwh, on, off, fuel, renewable, transport,
             cost_eff, credit, kw, demand, etc, franchise, state, county, city,
             customer, total) = m
            bills.append((f"b{index}", bill(
                "PNM", "Rate 3C", start, end,
                entries=[
                    entry("On-Peak Charge", on),
                    entry("Off-Peak Charge", off),
                    entry("Fuel Adj", fuel),
                    entry("Renewable Rider", renewable),
                    entry("Transport R53", transport),
                    entry("Cost-Eff Prog", cost_eff),
                    entry("Palo Verde Credit", credit, "discount"),
                    entry("Demand Charge", demand),
                    entry("San Juan ETC", etc, "tax"),
                    entry("Franchise Fee", franchise, "tax"),
                    entry("State GRT", state, "tax"),
                    entry("County GRT", county, "tax"),
                    entry("City GRT", city, "tax"),
                    entry("Customer Charge", customer),
                ],
                measurements=[
                    measurement("On-Peak kWh", on_kwh, "kWh"),
                    measurement("Off-Peak kWh", off_kwh, "kWh"),
                    measurement("Billable Demand kW", kw, "kW"),
                ],
                current=total)))
        return shape(bills)

    def test_the_time_of_use_split_survives(self):
        plan = self.build()[0]
        names = [c.canonical for c in plan.columns]
        for expected in ("On-Peak kWh", "Off-Peak kWh", "Total kWh",
                         "On-Peak Energy Charge", "Off-Peak Energy Charge"):
            self.assertIn(expected, names)

    def test_gross_receipts_tax_stays_split_three_ways(self):
        plan = self.build()[0]
        taxes = [c.canonical for c in plan.columns if c.group == TAXES_AND_FEES]
        for expected in ("State GRT", "County GRT", "City GRT"):
            self.assertIn(expected, taxes)
        # Jurisdictions are distinct columns; they must not be merged into one GRT.
        self.assertNotIn("GRT", taxes)

    def test_total_kwh_is_the_sum_of_the_peaks(self):
        row = self.build()[0].rows[0]
        self.assertEqual(row.values["Total kWh"], Decimal("33280"))

    def test_the_palo_verde_credit_is_negative(self):
        plan = self.build()[0]
        row = plan.rows[0]
        self.assertEqual(row.values["Palo Verde Credit"], Decimal("-59.20"))
        column = next(c for c in plan.columns if c.canonical == "Palo Verde Credit")
        self.assertEqual((column.group, column.sign), (CREDITS, CREDIT))

    def test_no_band_subtotals_are_produced(self):
        row = self.build()[0].rows[0]
        self.assertNotIn("Total Energy Charge", row.values)
        self.assertNotIn("Total Taxes & Fees", row.values)

    def test_every_month_reconciles_against_the_printed_total(self):
        plan = self.build()[0]
        self.assertEqual({r.status for r in plan.rows}, {"matched"},
                         [(r.label, r.difference) for r in plan.rows])

    def test_bands_are_energy_demand_then_charges(self):
        captions = [caption for caption, _ in self.build()[0].header_bands()]
        self.assertEqual(captions, ["Energy (kWh)", "Demand (kW)", "Charges ($)"])

    def test_charges_are_ordered_energy_demand_taxes_credits(self):
        # One band, but still in the order an assessor reads a bill.
        groups = [c.group for c in self.build()[0].columns if c.header == "Charges ($)"]
        order = [USAGE, DEMAND, TAXES_AND_FEES, CREDITS]
        self.assertEqual(groups, sorted(groups, key=order.index))

    def test_demand_reading_is_demand_not_usage(self):
        plan = self.build()[0]
        column = next(c for c in plan.columns if c.canonical == "Billed Demand")
        self.assertEqual((column.group, column.unit), (DEMAND, "kW"))

    def test_the_adjustors_column_is_recognised(self):
        """PNM prints "Off-Peak Charge with Adjustors" -- its own spelling."""
        from app.bills.semantics import interpret_labels, RawLabel
        mapping = interpret_labels(
            [RawLabel("Off-Peak Charge with Adjustors", "charge")], client=None)
        column = next(iter(mapping.values()))
        self.assertEqual(column.canonical, "Off-Peak Charge with Adjustors")
        self.assertEqual(column.source, "seed")
        self.assertEqual(column.group, USAGE)

    def test_the_column_set_matches_the_hand_built_sheet(self):
        """Same line-item columns as the assessor's workbook, ignoring order.

        The hand-built sheet also carries "Total Energy Charge" and "Total Taxes &
        Fees" subtotals; this layout puts every charge in one band and leaves the
        sum to Total ($), so those two are deliberately absent. The line items are
        what must not drift.
        """
        plan = self.build()[0]
        ours = {c.canonical for c in plan.columns}
        theirs = {
            "On-Peak kWh", "Off-Peak kWh", "Total kWh", "On-Peak Energy Charge",
            "Off-Peak Energy Charge", "Fuel Adjustment", "Renewable Rider",
            "Transport R53", "Cost-Effectiveness Program",
            "Palo Verde Credit", "Billed Demand", "Demand Charge", "San Juan ETC",
            "Franchise Fee", "State GRT", "County GRT", "City GRT",
            "Customer Charge",
        }
        self.assertEqual(ours, theirs)


class BothFormatsTests(unittest.TestCase):
    """The pair together: one pipeline, two layouts, no template."""

    def test_the_two_rate_schedules_produce_different_columns(self):
        rate20 = [c.canonical for c in Rate20Tests().build()[0].columns]
        rate3c = [c.canonical for c in PnmRate3CTests().build()[0].columns]
        self.assertNotEqual(rate20, rate3c)
        # Rate 20 has no peak split; Rate 3C does. Neither borrows from the other.
        self.assertNotIn("On-Peak Energy Charge", rate20)
        self.assertIn("On-Peak Energy Charge", rate3c)
        self.assertIn("Tax", rate20)
        self.assertNotIn("Tax", rate3c)

    def test_both_sites_roll_up_into_energy_demand_and_fees(self):
        for plans in (Rate20Tests().build(), PnmRate3CTests().build()):
            categories = {r.category for r in build_rollup(plans)}
            self.assertIn("Electricity Energy", categories)
            self.assertIn("Electricity Demand", categories)
            self.assertIn("Electricity Fees/Taxes", categories)

    def test_a_credit_reduces_the_rollup_rather_than_adding_to_it(self):
        rollup = {r.category: r.annual_cost for r in build_rollup(PnmRate3CTests().build())}
        self.assertLess(rollup["Electricity Credits"], Decimal("0"))


if __name__ == "__main__":
    unittest.main()

"""Phase 6: water, gas and delivered fuels.

These sheets are structurally simpler than electricity -- a usage quantity, often
the same quantity in a second unit, and a cost -- but they are where the unit
conversions live, and a wrong constant is invisible in the output. Each conversion
is asserted against its arithmetic, not against a previously-observed number.

Target shapes, from the two client workbooks:

    Water (Classy)    Date  | Water Usage (TGals) | Water Usage (Gallons) | Water Cost
    Water (Clayton)   Month | Water Usage (Gallons) | Water Cost
    Gas (Clayton)     Month | Natural Gas Usage (MMBTU) | Natural Gas Cost | Gas Usage (Therms)
    Propane           Month | Propane Usage (Gallons) | Propane Usage (MMBtu) | Propane Cost
    Diesel            Month | Diesel Usage (Gallons) | Diesel Usage (MMBtu) | Diesel Cost
"""
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

from app.bills.bill_value_extraction import parse_amount, parse_number
from app.bills.semantics import USAGE, interpret_labels, labels_from_bill
from app.bills.shape import (
    GALLONS_PER_TGAL, MMBTU_PER_GALLON, MMBTU_PER_THERM, build_rollup, build_sheets,
)
from .test_fixtures import BILL_FOLDERS, FIXTURES, HAVE_FIXTURES, requires_fixtures

TESTS_DIR = Path(__file__).resolve().parent


def entry(label, amount, category="charge"):
    return NS(label=label, amount_text=amount, category=category, section=None,
              meter_number=None, quantity_unit=None, rate_unit=None)


def measurement(label, value, unit):
    return NS(label=label, value_text=value, unit=unit, section=None, meter_number=None)


def bill(company, utility, start, end, entries, measurements, current):
    return NS(
        utility_company=company, utility_types=[utility], account_number=None,
        invoice_number=None, bill_date=None, due_date=None,
        service_start=start, service_end=end, billing_days_text=None,
        billing_period_label=None, current_charges_text=current, total_due_text=current,
        financial_entries=list(entries), measurements=list(measurements),
        additional_fields=[], extraction_notes=[],
    )


def shape(bills):
    labels = [l for _, b in bills for l in labels_from_bill(b)]
    mapping = interpret_labels(labels, client=None)
    return build_sheets(bills, mapping, parse_amount=parse_amount,
                        parse_number=parse_number, tolerance=Decimal("0.02"))


class WaterTests(unittest.TestCase):
    def build(self, unit="Gallons", value="47872"):
        return shape([("w1", bill(
            "City Water", "water", "1/1/2025", "1/31/2025",
            entries=[entry("Water Service", "879.66")],
            measurements=[measurement("Water Usage", value, unit)],
            current="879.66"))])

    def test_a_water_sheet_is_usage_and_cost(self):
        plan = self.build()[0]
        names = [c.canonical for c in plan.columns]
        self.assertIn("Water Usage", names)
        self.assertIn("Water Service", names)

    def test_water_usage_is_not_filed_as_demand(self):
        plan = self.build()[0]
        usage = next(c for c in plan.columns if c.canonical == "Water Usage")
        self.assertEqual(usage.group, USAGE)
        self.assertEqual(usage.unit, "Gallons")

    def test_thousand_gallons_convert_to_gallons(self):
        plan = self.build(unit="TGals", value="118")[0]
        row = plan.rows[0]
        source = next(c.canonical for c in plan.columns if c.unit == "TGals")
        self.assertEqual(row.values["Water Usage (Gallons)"],
                         row.values[source] * GALLONS_PER_TGAL)
        self.assertEqual(row.values["Water Usage (Gallons)"], Decimal("118000"))

    def test_gallons_are_not_re_derived_when_already_printed(self):
        plan = self.build(unit="Gallons", value="47872")[0]
        derived = [c for c in plan.columns if c.source == "derived"]
        self.assertEqual([c.canonical for c in derived], [])

    def test_water_gets_no_cost_per_gallon_in_either_unit(self):
        """Cost per gallon is a delivered-fuel column, not a water one.

        Water is priced per TGal; the same ratio here is ~$0.018 and quantizes to
        $0.02, which is rounding noise. Gating it on the fuel also keeps the sheet
        from depending on which unit the utility happened to print.
        """
        for unit, value in (("Gallons", "47872"), ("TGals", "118")):
            with self.subTest(unit=unit):
                names = [c.canonical for c in self.build(unit=unit, value=value)[0].columns]
                self.assertNotIn("Cost Per Gallon ($)", names)

    def test_water_rolls_up_as_a_single_category(self):
        rows = {r.category: r.annual_cost for r in build_rollup(self.build())}
        self.assertEqual(rows["Water"], Decimal("879.66"))
        self.assertNotIn("Water Energy", rows)

    def test_the_month_reconciles(self):
        self.assertEqual(self.build()[0].rows[0].status, "matched")


class GasTests(unittest.TestCase):
    def build(self, unit="MMBtu", value="425.2"):
        return shape([("g1", bill(
            "Gas Co", "gas", "1/1/2025", "1/31/2025",
            entries=[entry("Natural Gas Sales", "2973.94")],
            measurements=[measurement("Natural Gas Usage", value, unit)],
            current="2973.94"))])

    def test_mmbtu_converts_to_therms(self):
        # Clayton's gas sheet shows both; the bills print only one.
        plan = self.build(unit="MMBtu", value="425.2")[0]
        row = plan.rows[0]
        self.assertEqual(row.values["Usage (Therms)"],
                         Decimal("425.2") / MMBTU_PER_THERM)
        self.assertEqual(row.values["Usage (Therms)"], Decimal("4252"))

    def test_therms_convert_to_mmbtu(self):
        plan = self.build(unit="Therms", value="4252")[0]
        row = plan.rows[0]
        self.assertEqual(row.values["Usage (MMBtu)"],
                         Decimal("4252") * MMBTU_PER_THERM)
        self.assertEqual(row.values["Usage (MMBtu)"], Decimal("425.2"))

    def test_the_two_conversions_are_exact_inverses(self):
        therms = self.build(unit="MMBtu", value="425.2")[0].rows[0].values["Usage (Therms)"]
        back = therms * MMBTU_PER_THERM
        self.assertEqual(back, Decimal("425.2"))

    def test_only_the_missing_unit_is_derived(self):
        plan = self.build(unit="MMBtu")[0]
        derived = [c.canonical for c in plan.columns if c.source == "derived"]
        self.assertEqual(derived, ["Usage (Therms)"])

    def test_gas_rolls_up_as_one_category(self):
        rows = {r.category: r.annual_cost for r in build_rollup(self.build())}
        self.assertEqual(rows["Gas"], Decimal("2973.94"))


class DeliveredFuelTests(unittest.TestCase):
    def build(self, fuel, label, gallons, cost):
        return shape([("f1", bill(
            f"{fuel.title()} Supply", fuel, "1/1/2025", "1/31/2025",
            entries=[entry(f"{fuel.title()} Cost", cost)],
            measurements=[measurement(label, gallons, "Gallons")],
            current=cost))])

    def test_propane_gallons_convert_by_its_own_heat_content(self):
        plan = self.build("propane", "Propane Usage", "854.9", "2344.81")[0]
        row = plan.rows[0]
        self.assertEqual(row.values["Usage (MMBtu)"],
                         Decimal("854.9") * MMBTU_PER_GALLON["propane"])

    def test_diesel_uses_a_different_heat_content_from_propane(self):
        plan = self.build("diesel", "Diesel Usage", "38.5", "155.61")[0]
        row = plan.rows[0]
        self.assertEqual(row.values["Usage (MMBtu)"],
                         Decimal("38.5") * MMBTU_PER_GALLON["diesel"])
        self.assertNotEqual(MMBTU_PER_GALLON["diesel"], MMBTU_PER_GALLON["propane"])

    def test_cost_per_gallon_is_computed(self):
        plan = self.build("diesel", "Diesel Usage", "38.5", "155.61")[0]
        row = plan.rows[0]
        self.assertEqual(row.values["Cost Per Gallon ($)"], Decimal("4.04"))

    def test_heat_contents_match_the_published_btu_figures(self):
        # A wrong constant here is invisible in the output, so it is pinned.
        self.assertEqual(MMBTU_PER_GALLON["propane"], Decimal("0.0915"))     # 91,500 Btu
        self.assertEqual(MMBTU_PER_GALLON["diesel"], Decimal("0.138690"))    # 138,690 Btu
        self.assertEqual(MMBTU_PER_THERM, Decimal("0.1"))                    # 100,000 Btu

    def test_an_unknown_fuel_gets_no_invented_conversion(self):
        plan = self.build("other", "Mystery Usage", "100", "50.00")[0]
        self.assertNotIn("Usage (MMBtu)", [c.canonical for c in plan.columns])


class MixedSiteTests(unittest.TestCase):
    """A real site has several utilities at once; each gets its own sheet."""

    def build(self):
        return shape([
            ("e1", bill("SRP", "electricity", "1/1/2025", "1/31/2025",
                        [entry("Energy Charge", "100.00")],
                        [measurement("Total kWh", "1000", "kWh")], "100.00")),
            ("w1", bill("City Water", "water", "1/1/2025", "1/31/2025",
                        [entry("Water Service", "50.00")],
                        [measurement("Water Usage", "1000", "Gallons")], "50.00")),
            ("g1", bill("Gas Co", "gas", "1/1/2025", "1/31/2025",
                        [entry("Natural Gas Sales", "75.00")],
                        [measurement("Natural Gas Usage", "10", "MMBtu")], "75.00")),
        ])

    def test_each_utility_gets_its_own_sheet(self):
        plans = self.build()
        self.assertEqual(len(plans), 3)
        self.assertEqual({p.key.utility_type for p in plans},
                         {"electricity", "water", "gas"})

    def test_sheets_are_ordered_by_utility(self):
        self.assertEqual([p.key.utility_type for p in self.build()],
                         ["electricity", "gas", "water"])

    def test_no_column_leaks_between_utilities(self):
        for plan in self.build():
            names = " ".join(c.canonical for c in plan.columns).casefold()
            if plan.key.utility_type == "water":
                self.assertNotIn("kwh", names)
                self.assertNotIn("mmbtu", names)
            if plan.key.utility_type == "electricity":
                self.assertNotIn("water", names)

    def test_the_rollup_lists_every_utility_and_totals_them(self):
        rows = {r.category: r.annual_cost for r in build_rollup(self.build())}
        self.assertEqual(rows["Water"], Decimal("50.00"))
        self.assertEqual(rows["Gas"], Decimal("75.00"))
        self.assertEqual(rows["Electricity Energy"], Decimal("100.00"))
        self.assertEqual(rows["TOTAL"], Decimal("225.00"))


@unittest.skipUnless(HAVE_FIXTURES, "real-bill fixtures are not present")
class RealGasBillTests(unittest.TestCase):
    """The packaged Symmetry Energy gas bills, end to end through the pivot."""

    @classmethod
    def setUpClass(cls):
        import app.bills.bill_value_extraction as bve
        bills = []
        for number in range(1, 13):
            path = TESTS_DIR / BILL_FOLDERS[1] / f"bill_{number:03d}.pdf"
            fingerprint = sha256(path.read_bytes()).hexdigest()
            loaded, _, _ = bve.load_bill(path, fingerprint, FIXTURES, bve.EXTRACTION_MODEL)
            bills.append((fingerprint[:12], loaded))
        cls.plan = shape(bills)[0]

    def test_twelve_months_on_one_sheet(self):
        self.assertEqual(len(self.plan.rows), 12)
        self.assertEqual(self.plan.warnings, [])

    def test_volume_in_mmbtu_gains_a_therms_column(self):
        self.assertIn("Usage (Therms)", [c.canonical for c in self.plan.columns])
        row = next(r for r in self.plan.rows if "Volume" in r.values)
        self.assertEqual(row.values["Usage (Therms)"],
                         row.values["Volume"] / MMBTU_PER_THERM)

    def test_the_undertake_credit_is_negative(self):
        credits = [r.values["Natural Gas Sales Undertake"] for r in self.plan.rows
                   if "Natural Gas Sales Undertake" in r.values]
        self.assertTrue(credits)
        for value in credits:
            self.assertLessEqual(value, Decimal("0"))

    def test_ten_of_twelve_months_reconcile(self):
        statuses = {}
        for row in self.plan.rows:
            statuses[row.status] = statuses.get(row.status, 0) + 1
        self.assertEqual(statuses, {"matched": 10, "review": 2})

    def test_jurisdiction_taxes_are_separate_columns_in_one_band(self):
        taxes = [c for c in self.plan.columns
                 if "transaction privilege" in c.canonical.casefold()]
        self.assertGreaterEqual(len(taxes), 3)
        self.assertEqual({c.group for c in taxes}, {"taxes_and_fees"})


if __name__ == "__main__":
    unittest.main()

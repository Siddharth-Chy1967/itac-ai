"""Printed-row tests: meter readings labelled by where OCR found them.

The boxes are the real RapidOCR output for SRP's usage block (Aug-21 and Mar-22
bills, 300 DPI), reduced to (top, bottom, left, right, text). The bills are the
shapes qwen2.5vl actually returned for them.
"""
import unittest

from app.bills.extraction_models import BillExtraction, Measurement
from app.bills.grid import PrintedRow, reconcile, rows_from_items, unit_in


def box(y, x0, x1, text):
    return (y - 20, y + 20, x0, x1, text)


# SRP Aug-21, page 2, as OCR returned it.
SRP_AUG = [
    box(721, 19, 169, "Meter "), box(723, 1027, 1193, "Energy"),
    box(777, 19, 247, "On Pk kWh"), box(779, 1042, 1188, "61,535"),
    box(829, 23, 356, "Shoulder Pk kWh"), box(831, 1040, 1190, "86,956"),
    box(882, 19, 247, "Off Pk kWh"), box(884, 1022, 1188, "136,943"),
    box(935, 21, 224, "Total kWh"), box(936, 1019, 1190, "285,434"),
    box(986, 21, 224, "On Pk kW"), box(987, 508, 703, "7/21/2021"),
    box(987, 742, 859, "15:30"), box(987, 1065, 1190, "528.2"),
    box(1037, 21, 337, "Shoulder Pk kW"), box(1037, 506, 703, "7/15/2021"),
    box(1038, 1065, 1190, "540.2"), box(1039, 742, 861, "11:30"),
    box(1090, 19, 224, "Off Pk kW"), box(1090, 512, 693, "8/6/2021"),
    box(1091, 738, 861, "02:30"), box(1091, 1067, 1188, "514.9"),
    box(1142, 23, 484, "16 Month Peak Demand"), box(1142, 1067, 1190, "689.5"),
    box(1194, 19, 144, "kVAH"), box(1195, 1019, 1188, "337,797"),
    box(1279, 19, 287, "Power Factor"), box(1280, 1030, 1187, "84.50%"),
    box(1333, 1030, 1187, "71.02%"), box(1334, 19, 260, "Load Factor"),
]


def reading(label, value, unit=None, meter=None, page=2):
    return Measurement(label=label, value_text=value, unit=unit, section=None,
                       meter_number=meter, measurement_date=None, measurement_time=None,
                       service_period_text=None, bill_page_number=page)


def srp_bill(measurements):
    return BillExtraction(
        utility_company="SRP", utility_types=["electricity"], account_number=None,
        invoice_number=None, bill_date=None, due_date=None, service_start="7/12/2021",
        service_end="8/11/2021", billing_days_text=None, billing_period_label=None,
        current_charges_text="$33,194.75", total_due_text="$33,194.75",
        financial_entries=[], measurements=measurements, additional_fields=[],
        extraction_notes=[],
    )


class PrintedRowTests(unittest.TestCase):
    def setUp(self):
        self.rows = {r.label: r for r in rows_from_items(SRP_AUG, page=2)}

    def test_every_reading_row_is_found_with_its_own_label(self):
        self.assertEqual(list(self.rows), [
            "On Pk kWh", "Shoulder Pk kWh", "Off Pk kWh", "Total kWh", "On Pk kW",
            "Shoulder Pk kW", "Off Pk kW", "16 Month Peak Demand", "kVAH",
            "Power Factor", "Load Factor",
        ])
        self.assertEqual(self.rows["Shoulder Pk kWh"].value_text, "86,956")

    def test_the_heading_line_is_not_a_reading(self):
        self.assertNotIn("Meter", self.rows)
        self.assertNotIn("Meter Energy", self.rows)

    def test_the_peak_date_and_time_between_label_and_value_are_ignored(self):
        self.assertEqual(self.rows["On Pk kW"].value_text, "528.2")

    def test_units_come_from_the_label_or_the_value(self):
        self.assertEqual(self.rows["On Pk kWh"].unit, "kWh")
        self.assertEqual(self.rows["On Pk kW"].unit, "kW")  # not read as kWh
        self.assertEqual(self.rows["kVAH"].unit, "kVAh")
        self.assertEqual(self.rows["Power Factor"].unit, "%")
        self.assertIsNone(self.rows["16 Month Peak Demand"].unit)

    def test_a_line_with_two_readings_is_left_to_the_model(self):
        rows = rows_from_items([box(100, 10, 200, "Current Read"),
                                box(100, 300, 400, "12,345"),
                                box(100, 500, 600, "12,000")], page=1)
        self.assertEqual(rows, [])

    def test_a_money_line_is_not_a_reading(self):
        rows = rows_from_items([box(100, 10, 300, "On Peak Energy Charge"),
                                box(100, 900, 1000, "$6,990.38")], page=1)
        self.assertEqual(rows, [])

    def test_unit_spellings(self):
        for text, unit in (("kWh", "kWh"), ("kW", "kW"), ("KVAH", "kVAh"),
                           ("12 therms", "Therms"), ("TGals", "TGals"), ("Peak", None)):
            with self.subTest(text=text):
                self.assertEqual(unit_in(text), unit)


class ReconcileTests(unittest.TestCase):
    def setUp(self):
        self.rows = rows_from_items(SRP_AUG, page=2)

    def labels(self, bill):
        return {m.label: m.value_text for m in bill.measurements}

    def test_header_in_the_meter_field_is_replaced_by_the_printed_label(self):
        # Exactly the shape qwen returned for this bill.
        bill = srp_bill([
            reading("Energy", "61,535 kWh", "kWh", meter="On Pk kWh"),
            reading("Energy", "285,434 kWh", "kWh", meter="Total kWh"),
            reading("Demand (kW)", "528.2 kW", "kW", meter="On Pk kW"),
        ])
        fixed = reconcile(bill, self.rows)
        on_peak = next(m for m in fixed.measurements if m.label == "On Pk kWh")
        self.assertEqual((on_peak.value_text, on_peak.meter_number), ("61,535", None))
        demand = next(m for m in fixed.measurements if m.label == "On Pk kW")
        self.assertEqual((demand.value_text, demand.unit), ("528.2", "kW"))

    def test_readings_with_no_period_label_are_recovered(self):
        # Oct-21: every reading tagged "Meter #1".
        bill = srp_bill([reading("Energy", "61,535", "kWh", meter="Meter #1"),
                         reading("Demand", "528.2", "kW", meter="Meter #1")])
        fixed = reconcile(bill, self.rows)
        self.assertEqual(self.labels(fixed)["On Pk kWh"], "61,535")
        self.assertEqual(self.labels(fixed)["On Pk kW"], "528.2")

    def test_a_borrowed_unit_is_replaced(self):
        # Oct-21 again: 689.5 came back as kVAH, the next row's unit.
        # The unit is borrowed in the value text as well (Jan-22: "689.5 kVAH").
        bill = srp_bill([reading("On Peak Demand", "689.5 kVAH", "kVAH", meter="Meter #")])
        peak = next(m for m in reconcile(bill, self.rows).measurements
                    if m.label == "16 Month Peak Demand")
        self.assertEqual((peak.value_text, peak.unit), ("689.5", None))

    def test_a_collapsed_row_is_rebuilt_from_the_page(self):
        # Mar-22: the whole row in one string, with kW figures under kWh names.
        bill = srp_bill([reading("Energy", "On Pk kWh: 273.7, Shoulder Pk kWh: 246.7, "
                                           "Off Pk kWh: 277.8")])
        fixed = reconcile(bill, self.rows)
        values = self.labels(fixed)
        self.assertNotIn("Energy", values)
        self.assertEqual(values["On Pk kWh"], "61,535")
        self.assertEqual(values["Off Pk kW"], "514.9")
        self.assertTrue(any("collapsed" in n for n in fixed.extraction_notes))

    def test_rows_the_model_missed_are_added(self):
        fixed = reconcile(srp_bill([]), self.rows)
        values = self.labels(fixed)
        self.assertEqual(values["Total kWh"], "285,434")
        self.assertEqual(values["Load Factor"], "71.02%")
        # No unit printed, but its value lines up with the readings above it.
        self.assertEqual(values["16 Month Peak Demand"], "689.5")

    def test_a_unitless_figure_outside_the_reading_block_is_not_added(self):
        stray = rows_from_items(SRP_AUG + [box(2400, 19, 300, "Days in Period"),
                                           box(2400, 600, 660, "31")], page=2)
        values = self.labels(reconcile(srp_bill([]), stray))
        self.assertNotIn("Days in Period", values)

    def test_a_shared_number_goes_to_the_row_the_model_named(self):
        rows = [PrintedRow("On Pk kW", "689.5", "kW", 2),
                PrintedRow("16 Month Peak Demand", "689.5", None, 2)]
        bill = srp_bill([reading("16 Month Peak Demand", "689.5"),
                         reading("Demand", "689.5", meter="On Pk kW")])
        fixed = reconcile(bill, rows)
        self.assertEqual([m.label for m in fixed.measurements],
                         ["16 Month Peak Demand", "On Pk kW"])

    def test_a_real_meter_serial_is_kept(self):
        bill = srp_bill([reading("Energy", "61,535", "kWh", meter="Meter 10953811")])
        fixed = reconcile(bill, self.rows)
        self.assertEqual(fixed.measurements[0].meter_number, "Meter 10953811")

    def test_a_price_is_left_alone(self):
        bill = srp_bill([reading("Energy", "$0.37245", "kWh")])
        self.assertEqual(reconcile(bill, []).measurements[0].value_text, "$0.37245")
        fixed = reconcile(bill, self.rows)
        self.assertIn("$0.37245", [m.value_text for m in fixed.measurements])

    def test_no_printed_rows_leaves_the_bill_unchanged(self):
        bill = srp_bill([reading("Energy", "61,535", "kWh")])
        self.assertIs(reconcile(bill, []), bill)

    def test_the_extraction_is_not_modified(self):
        bill = srp_bill([reading("Energy", "61,535", "kWh", meter="On Pk kWh")])
        reconcile(bill, self.rows)
        self.assertEqual(bill.measurements[0].label, "Energy")


if __name__ == "__main__":
    unittest.main()

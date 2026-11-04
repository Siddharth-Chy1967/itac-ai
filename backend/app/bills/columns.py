"""Turn verified bills into the rows and columns the sheet builder consumes.

The model has already said what every figure is -- a kWh reading in the on-peak
period, a demand charge, a tax -- so there is nothing left to interpret. This
module only renames: a reading's column name is built from its typed fields
(kind, period, unit), a charge's from its printed label, and each becomes the
`LabelSemantics` the review gate shows and `shape` places values by.

No vocabulary of utility wording lives here. A new utility needs no code.

Two rules keep every workbook cell a printed figure:

* Readings the model calls `meter_register` (dial reads, multipliers) are not
  analysis columns and are left out.
* A charge name printed on several lines of one bill ("Renewable Energy Rider"
  for 15 days, then again for 16 days) becomes several columns, numbered, rather
  than one column holding their sum -- a sum is not printed on the bill.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Sequence

from .bill_schema import BillFigures, Charge, Reading
from .extraction_models import AdditionalField, BillExtraction, FinancialEntry, Measurement
from .semantics import (
    CHARGE, CREDIT, CREDITS, DEMAND, NO_PEAK, OTHER, PEAK_LABELS, TAXES_AND_FEES, USAGE,
    LabelSemantics, _key, normalize_label, validate_mapping,
)
from .verify import VerifiedBill, parse_any_date, signed_amount, single_number

# Model period -> semantics period.
_PERIODS = {
    "on_peak": "on", "mid_peak": "mid", "shoulder": "shoulder", "off_peak": "off",
    "super_off_peak": "super_off", "critical_peak": "critical", "total": NO_PEAK,
    "none": NO_PEAK,
}
# Printed unit -> how the column header spells it.
_UNITS = {
    "kwh": "kWh", "mwh": "MWh", "kw": "kW", "mw": "MW", "kva": "kVA", "kvah": "kVAh",
    "kvar": "kVAR", "kvarh": "kVARh", "rkva": "kVAR", "therm": "Therms", "therms": "Therms",
    "ccf": "CCF", "mcf": "MCF", "mmbtu": "MMBtu", "dth": "Dth", "gal": "Gallons",
    "gallon": "Gallons", "gallons": "Gallons", "tgal": "TGals", "tgals": "TGals",
    "kgal": "TGals", "kgals": "TGals", "%": "%", "percent": "%",
}
_CHARGE_GROUPS = {
    "energy": USAGE, "demand": DEMAND, "fixed": TAXES_AND_FEES, "tax": TAXES_AND_FEES,
    "fee": TAXES_AND_FEES, "other": TAXES_AND_FEES, "credit": CREDITS,
}
# Left-to-right order inside a band.
_CHARGE_ORDER = {"fixed": 10, "energy": 20, "demand": 20, "fee": 30, "other": 35,
                 "tax": 40, "credit": 10}
_PERIOD_ORDER = {"critical": 0, "on": 1, "mid": 2, "shoulder": 3, "off": 4,
                 "super_off": 5, NO_PEAK: 6}
_ENTRY_CATEGORY = {"tax": "tax", "credit": "discount"}


def unit_name(unit: str | None) -> str | None:
    if not unit or not unit.strip():
        return None
    return _UNITS.get(re.sub(r"[^a-z%]", "", unit.casefold()), unit.strip())


def _period_prefix(peak: str) -> str:
    return PEAK_LABELS.get(peak, "")


def reading_column(reading: Reading) -> LabelSemantics | None:
    """The column a reading belongs in, named from its typed fields."""
    if reading.kind == "meter_register":
        return None
    peak = _PERIODS[reading.period]
    unit = unit_name(reading.unit)
    prefix = _period_prefix(peak)
    order = 10 + _PERIOD_ORDER.get(peak, 6)
    if reading.kind == "energy":
        name = f"{prefix or 'Total'} {unit or 'Usage'}"
        return LabelSemantics(name, USAGE, peak, CHARGE, unit, order, "model")
    if reading.kind in ("billed_demand", "measured_demand"):
        what = "Billed Demand" if reading.kind == "billed_demand" else "Measured Demand"
        name = f"{prefix} {what}".strip()
        return LabelSemantics(name, DEMAND, peak, CHARGE, unit or "kW",
                              order + (0 if reading.kind == "billed_demand" else 10), "model")
    name = {"power_factor": "Power Factor", "load_factor": "Load Factor"}.get(reading.kind)
    name = name or clean_label(reading.label) or reading.kind.replace("_", " ").title()
    return LabelSemantics(name, OTHER, peak, CHARGE, unit, 60, "model")


# Per-bill quantities that ride along in a printed label: "(15 Days)", "16 Days",
# "83.7% of kWh", "4.191%". They change every month, so left in the name they
# would give every month its own column.
_LABEL_QUANTITIES = re.compile(
    r"\(?\s*\d+\s*days?\s*\)?|\d+(?:\.\d+)?\s*%(?:\s*of\s*kwh)?", re.IGNORECASE)


def clean_label(label: str | None) -> str:
    """A printed charge name as a column name: printing noise removed, nothing added."""
    text = _LABEL_QUANTITIES.sub(" ", normalize_label(label))
    text = re.sub(r"\s*\*+\s*", " ", text)
    text = re.sub(r"[\s:;,\-]+$", "", text)
    return re.sub(r"\s+", " ", text).strip()


def compact_key(text: str) -> str:
    """Spacing and punctuation never make two columns: "EnergyTransition" == "Energy Transition"."""
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def charge_column(charge: Charge, name: str) -> LabelSemantics:
    peak = _PERIODS[charge.period]
    return LabelSemantics(
        name, _CHARGE_GROUPS[charge.category], peak,
        CREDIT if charge.category == "credit" else CHARGE, "$",
        _CHARGE_ORDER[charge.category] + _PERIOD_ORDER.get(peak, 6), "model",
    )


def _iso(value) -> str | None:
    parsed = parse_any_date(value.text) if value is not None else None
    return parsed.isoformat() if parsed else None


def _utility_types(figures: BillFigures) -> list[str]:
    kind = figures.utility_type
    return [kind if kind in ("electricity", "gas", "water", "sewer") else "other"]


def build(bills: Sequence[tuple[str, VerifiedBill]]
          ) -> tuple[list[tuple[str, BillExtraction]], dict[str, LabelSemantics]]:
    """(bill id, rows for `shape`) per bill, and the label -> column mapping."""
    names: dict[str, str] = {}  # compact key -> first display name seen
    votes: dict[str, Counter] = defaultdict(Counter)
    candidates: dict[tuple, LabelSemantics] = {}
    converted = []

    def register(column: LabelSemantics) -> str:
        """The display name every bill uses for this column's key."""
        compact = compact_key(column.canonical)
        name = names.setdefault(compact, column.canonical)
        column = LabelSemantics(name, column.group, column.peak_period, column.sign,
                                column.unit, column.order_hint, column.source)
        signature = (column.group, column.peak_period, column.sign, column.unit)
        votes[_key(name)][signature] += 1
        candidates[(_key(name), signature)] = column
        return name

    for bill_id, verified in bills:
        figures = verified.figures
        meter = figures.meter_number
        measurements = []
        for reading in figures.readings:
            column = reading_column(reading)
            value = single_number(reading.value)
            if column is None or value is None:
                continue
            name = register(column)
            measurements.append(Measurement(
                # The meter rides on the charges, which name the sheet. A reading's
                # meter field is also read as a label hint by `measurement_label`;
                # the column name here is already final.
                label=name, value_text=format(value, "f"), unit=column.unit, section=None,
                meter_number=None, measurement_date=None, measurement_time=None,
                service_period_text=None, bill_page_number=reading.page,
            ))

        entries = []
        seen: Counter = Counter()
        for charge in figures.charges:
            amount = signed_amount(charge)
            base = clean_label(charge.label)
            if amount is None or not base:
                continue
            seen[compact_key(base)] += 1
            occurrence = seen[compact_key(base)]
            label = base if occurrence == 1 else f"{base} ({occurrence})"
            name = register(charge_column(charge, label))
            entries.append(FinancialEntry(
                label=name, amount_text=format(amount, "f"), quantity_text=None,
                quantity_unit=None, rate_text=None, rate_unit=None, section=None,
                meter_number=meter, service_period_text=None,
                category=_ENTRY_CATEGORY.get(charge.category, "charge"),
                bill_page_number=charge.page,
            ))

        total = figures.total_current_charges
        due = figures.amount_due
        extras = []
        if figures.rate_schedule:
            extras.append(AdditionalField(label="Rate Schedule", value_text=figures.rate_schedule,
                                          section=None, bill_page_number=1))
        converted.append((bill_id, BillExtraction(
            utility_company=figures.utility_company,
            utility_types=_utility_types(figures),
            account_number=figures.account_number,
            invoice_number=None,
            bill_date=_iso(figures.statement_date),
            due_date=_iso(figures.due_date),
            service_start=_iso(figures.service_start),
            service_end=_iso(figures.service_end),
            billing_days_text=None,
            billing_period_label=None,
            current_charges_text=format(single_number(total.text), "f") if total else None,
            total_due_text=format(single_number(due.text), "f") if due else None,
            financial_entries=entries,
            measurements=measurements,
            additional_fields=extras,
            extraction_notes=[*verified.flags, *figures.notes],
        )))

    mapping: dict[str, LabelSemantics] = {}
    for key, counter in votes.items():
        # Bills of one series normally agree; when they do not, the majority decides
        # and the review gate shows the result before anything is written.
        signature, _ = counter.most_common(1)[0]
        mapping[key] = candidates[(key, signature)]
    validate_mapping(mapping)
    return converted, mapping

"""Pivot extracted bills into the analysis workbook's shape.

The extractor emits normalized long format: one row per printed line item. The
analysis workbook an assessor actually uses is a matrix -- one sheet per meter,
one row per month, one column per line item:

    Date          | Usage and Cost              | Demand and Cost   | Taxes and Fees
    Meter Read    | On-Peak kWh | Energy Charge | Billed Demand kW  | State GRT | ...
    2025-03-18    |      18240  |       863.84  |            160.0  |    233.40 | ...

Which columns exist is decided by `semantics`, from the labels the bills actually
print, so a Rate 20 bill with no time-of-use split simply has no on-peak column.
This module does the rest deterministically:

* group bills into sheets, and periods into rows;
* place each amount in its column, credits negative;
* compute the columns that are NOT printed on any bill -- `Total kWh`, MMBtu
  conversions, `Cost Per Gallon` -- from constants, never from the model;
* reconcile every row against its printed total and flag the ones that disagree;
* roll the sheets up into an annual cost summary.

Nothing here calls a model. Every number traces to a printed value or to a
conversion constant defined in this file.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Iterable, Sequence

from .semantics import (
    CREDIT, CREDITS, DEMAND, GROUP_HEADERS, NO_PEAK, OFF_PEAK, ON_PEAK, OTHER,
    SHOULDER_PEAK, TAXES_AND_FEES, USAGE, LabelSemantics, labels_from_bill,
    measurement_label, normalize_label, unit_class, units_compatible, _key,
)
from .validation_checks import financial_effect
from .value_parsing import parse_date

logger = logging.getLogger(__name__)

# --- energy content, for the MMBtu columns the bills never print ------------
#
# Site energy content at standard conditions. These are the conventional ITAC
# figures; they are constants of the fuel, not anything read off a bill.
MMBTU_PER_THERM = Decimal("0.1")
MMBTU_PER_GALLON = {
    "propane": Decimal("0.0915"),      # 91,500 Btu/gal
    "diesel": Decimal("0.138690"),     # 138,690 Btu/gal
    "fuel oil": Decimal("0.138690"),
    "gasoline": Decimal("0.120429"),   # 120,429 Btu/gal
}
GALLONS_PER_TGAL = Decimal("1000")

# Units that mean "this column is money".
MONEY = "$"


class ShapeError(Exception):
    """Raised when bills cannot be arranged into a well-defined sheet."""


# --- identity ---------------------------------------------------------------


@dataclass(frozen=True)
class SheetKey:
    """What makes one analysis sheet distinct from another.

    Meter number first, because a site with three meters wants three sheets even
    on one account. Real bills often print none of these (the packaged SRP set
    has no meter or account number at all), so the key degrades to company and
    utility type rather than collapsing everything into one sheet or refusing.
    """

    company: str
    utility_type: str
    meter: str | None = None
    account: str | None = None
    rate_schedule: str | None = None

    def title(self) -> str:
        if self.meter:
            base = f"Meter {self.meter}"
        elif self.account:
            base = f"Account {self.account}"
        else:
            # Printed casing is preserved: an acronym like SRP must not become "Srp".
            base = f"{self.company} {self.utility_type.title()}".strip()
        if self.rate_schedule:
            # The rate often already carries its own parentheses, e.g.
            # "Secondary Service (E61)"; do not nest another pair around it.
            rate = self.rate_schedule.strip()
            return f"{base} - {rate}" if "(" in rate else f"{base} ({rate})"
        return base

    def sheet_name(self) -> str:
        """Excel sheet names: 31 chars, and none of : \\ / ? * [ ]"""
        cleaned = re.sub(r"[:\\/?*\[\]]", "-", self.title()).strip() or "Sheet"
        # Trim after truncating too: a 31-character cut often lands mid-word.
        return cleaned[:31].strip() or "Sheet"


def _rate_schedule(bill) -> str | None:
    for field_ in getattr(bill, "additional_fields", []) or []:
        label = (field_.label or "").casefold()
        if "rate" in label and "schedule" in label or label.strip() == "rate":
            return normalize_label(field_.value_text) or None
    return None


_METER_PREFIX = re.compile(r"^(?:meter|no\.?|number|#|:|\s)+", re.IGNORECASE)
_NOT_A_METER = re.compile(r"kwh|kw|kva|kvar|%|therm|gal|ccf|mcf|factor|demand|peak", re.IGNORECASE)
_PRINTED_QUANTITY = re.compile(r"^\d{1,3}(,\d{3})+(\.\d+)?$")


def _meter_id(value: str | None) -> str | None:
    """The identifier in a printed meter number, or None if it is not one.

    The extractor fills this field from whatever sits in a table's "Meter #"
    column: real serials ("10953811"), but also column headers ("Shoulder Pk
    kWh"), readings ("179,746 kVAH") and ordinals ("Meter #1"). Only a real
    serial may split a sheet off; anything else would give one month its own
    sheet and leave a gap in the year.
    """
    text = _METER_PREFIX.sub("", (value or "").strip())
    if not text or _NOT_A_METER.search(text) or _PRINTED_QUANTITY.match(text):
        return None
    compact = re.sub(r"[^A-Za-z0-9]", "", text)
    if not re.search(r"\d", compact):
        return None
    # A bare short number is a position on the statement ("Meter #1"), not a serial.
    if compact.isdigit() and len(compact) < 4:
        return None
    return text


def _meter(bill) -> str | None:
    meters = {_meter_id(e.meter_number) for e in bill.financial_entries}
    meters |= {_meter_id(m.meter_number) for m in bill.measurements}
    meters.discard(None)
    # Two meters on one bill cannot name one sheet; fall back to the account.
    return meters.pop() if len(meters) == 1 else None


def sheet_key_for(bill) -> SheetKey:
    types = [t for t in (bill.utility_types or []) if t and t != "unknown"]
    return SheetKey(
        company=normalize_label(bill.utility_company) or "Unknown Utility",
        utility_type=types[0] if types else "unknown",
        meter=_meter(bill),
        account=normalize_label(bill.account_number) or None,
        rate_schedule=_rate_schedule(bill),
    )


# --- period ----------------------------------------------------------------

_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _as_date(text: str | None) -> date | None:
    if not text:
        return None
    try:
        return parse_date(text)
    except (ValueError, TypeError):
        return None


@dataclass(frozen=True)
class Period:
    """One billing period. `anchor` is what the row sorts and is labelled by."""

    start: date | None
    end: date | None
    bill_date: date | None

    @property
    def anchor(self) -> date | None:
        # The service end is what an assessor calls "the month"; fall back through
        # start then bill date rather than leaving a row unplaceable.
        return self.end or self.start or self.bill_date

    @property
    def label(self) -> str:
        anchor = self.anchor
        if anchor is None:
            return "Unknown"
        return f"{_MONTHS[anchor.month - 1]}-{anchor.year % 100:02d}"

    def key(self) -> tuple:
        return (self.start, self.end, self.bill_date)


# A printed service range: "SERVICE FROM 1/12/2022 - 2/8/2022", "1/12/2022 to 2/8/2022".
_PRINTED_RANGE = re.compile(
    r"(\d{1,2}/\d{1,2}/\d{2,4})\s*(?:-|–|to|through|thru)\s*(\d{1,2}/\d{1,2}/\d{2,4})",
    re.IGNORECASE,
)


def period_for(bill) -> Period:
    start, end = _as_date(bill.service_start), _as_date(bill.service_end)
    if start is None and end is None:
        # The extractor sometimes files the printed service range under the billing
        # label and leaves the date fields empty. Without this the row falls back to
        # the statement date, which lands it a month late -- on top of the next
        # month's bill, with a gap where it belonged.
        match = _PRINTED_RANGE.search(getattr(bill, "billing_period_label", None) or "")
        if match:
            start, end = _as_date(match.group(1)), _as_date(match.group(2))
    return Period(start=start, end=end, bill_date=_as_date(bill.bill_date))


# --- rows -------------------------------------------------------------------


@dataclass
class MonthRow:
    period: Period
    values: dict[str, Decimal] = field(default_factory=dict)
    printed_total: Decimal | None = None
    computed_total: Decimal | None = None
    difference: Decimal | None = None
    status: str = "not_checked"  # matched | review | not_checked
    notes: list[str] = field(default_factory=list)
    bill_id: str | None = None

    @property
    def label(self) -> str:
        return self.period.label


@dataclass
class SheetPlan:
    key: SheetKey
    columns: list[LabelSemantics]
    rows: list[MonthRow]
    warnings: list[str] = field(default_factory=list)

    @property
    def title(self) -> str:
        return self.key.title()

    def header_bands(self) -> list[tuple[str, int]]:
        """(band caption, column span) across the columns, in order."""
        bands: list[tuple[str, int]] = []
        for column in self.columns:
            caption = column.header
            if bands and bands[-1][0] == caption:
                bands[-1] = (caption, bands[-1][1] + 1)
            else:
                bands.append((caption, 1))
        return bands

    def column_total(self, canonical: str) -> Decimal:
        return sum((row.values[canonical] for row in self.rows
                    if canonical in row.values), Decimal("0"))


# --- amount placement -------------------------------------------------------


def _amount(text: str | None, parse_amount) -> Decimal | None:
    if text is None:
        return None
    try:
        return parse_amount(text)
    except (ValueError, TypeError, InvalidOperation):
        return None


def _quantity(text: str | None, unit: str | None, parse_number) -> Decimal | None:
    if text is None:
        return None
    for attempt in (unit, None):
        try:
            return parse_number(text, attempt)
        except (ValueError, TypeError, InvalidOperation):
            continue
    return None


def _signed(value: Decimal, column: LabelSemantics) -> Decimal:
    """A credit reduces the bill, so it is stored negative however it was printed.

    Bills are inconsistent: some print "(59.20)", some "-59.20", some "59.20"
    under a Credits heading. Normalizing here is what lets a row's columns be
    summed without knowing how each utility formats a reduction.
    """
    if column.sign != CREDIT:
        return value
    return -abs(value)


def _place(row: MonthRow, column: LabelSemantics, value: Decimal) -> None:
    """Add a value into its column; repeated rows of one concept accumulate.

    A bill can print the same charge twice (two service periods on one page), and
    the analysis column is their total, not the last one seen.
    """
    canonical = column.canonical
    if canonical in row.values:
        row.values[canonical] += value
        row.notes.append(f"{canonical}: multiple printed rows combined")
    else:
        row.values[canonical] = value


# Quantities that add up across time-of-use periods. Demand, power factor and the
# like are peaks or ratios: two of them summed is not a figure anyone measured.
ADDITIVE = {"energy", "fuel", "volume"}


def resolve_readings(values: list[Decimal], *, additive: bool) -> tuple[Decimal | None, str | None]:
    """One month's figure for a column that received several readings.

    Unlike charges, meter readings are never summed. Several readings under one
    label means a time-of-use table lost its column headers on the way out of the
    extractor ("Energy" printed under On-Peak, Shoulder, Off-Peak and Total), so
    their sum counts the usage twice. The printed total can still be recovered
    when one reading is exactly the sum of the others; otherwise the cell stays
    blank, because a blank is a visible gap and a guess is a wrong number.
    """
    distinct = sorted(set(values))
    if len(distinct) == 1:
        return distinct[0], None
    if additive:
        for candidate in distinct:
            rest = list(values)
            rest.remove(candidate)
            if abs(sum(rest) - candidate) <= max(Decimal("1"), candidate * Decimal("0.001")):
                return candidate, (f"{len(values)} unlabelled readings; used {candidate}, "
                                   "the one equal to the sum of the others")
    listed = ", ".join(str(v) for v in values)
    return None, f"{len(values)} unlabelled readings ({listed}); left blank"


def _within(a: Decimal, b: Decimal) -> bool:
    return abs(a - b) <= max(Decimal("1"), abs(b) * Decimal("0.001"))


def _recover_unlabelled_period(readings: dict, mapping: dict, row: MonthRow) -> None:
    """Put a reading that lost its period label back in its period, when provable.

    A time-of-use grid sometimes comes out with one cell unlabelled: on a real
    SRP bill the on-peak kWh was tagged with the grid's corner header ("Meter #")
    instead of "On Pk kWh". A bare usage reading is taken as the total, so the
    total column then holds two figures (57,044 and 257,697) and the on-peak
    column none. If one figure equals the labelled periods plus the others --
    257,697 = 79,390 + 121,263 + 57,044 -- it is the total, and when exactly one
    period is missing from the row, the leftover is that period. Every figure
    placed is printed on the bill; only its column is inferred, and only by
    arithmetic that has to balance.
    """
    for canonical, (total_column, values) in list(readings.items()):
        kind = unit_class(total_column.unit)
        if (total_column.group != USAGE or total_column.peak_period != NO_PEAK
                or kind not in ADDITIVE or len(set(values)) < 2):
            continue
        periods = {
            c.canonical: c for c in mapping.values()
            if c.group == USAGE and c.peak_period != NO_PEAK and unit_class(c.unit) == kind
        }
        placed = {}
        for name in periods:
            period_values = set(readings.get(name, (None, []))[1])
            if len(period_values) > 1:
                return  # a period is itself ambiguous; nothing here is provable
            if period_values:
                placed[name] = period_values.pop()
        missing = [name for name in periods if name not in placed]
        for total in sorted(set(values), reverse=True):
            leftovers = list(values)
            leftovers.remove(total)
            if not _within(sum(placed.values()) + sum(leftovers), total):
                continue
            if len(leftovers) == 1 and len(missing) == 1:
                period = periods[missing[0]]
                readings[canonical] = (total_column, [total])
                readings[period.canonical] = (period, leftovers)
                row.notes.append(
                    f"{period.canonical}: {leftovers[0]} was printed without its period "
                    f"label; placed because {total_column.canonical} {total} equals the "
                    "periods it totals")
            return


# --- derived columns --------------------------------------------------------


def _fuel_of(key: SheetKey) -> str | None:
    text = f"{key.utility_type} {key.company}".casefold()
    for fuel in MMBTU_PER_GALLON:
        if fuel in text:
            return fuel
    return None


def add_derived_columns(plan: SheetPlan) -> None:
    """Columns the analysis sheet shows but no bill prints.

    Every one is arithmetic over values already placed, or a conversion constant
    from the top of this file. Nothing is inferred, and a derived column is only
    added when its inputs are actually present -- an absent input leaves the cell
    blank rather than implying a zero.
    """
    names = {c.canonical for c in plan.columns}

    def derive(canonical, group, unit, order, compute, peak="none"):
        """Fill `canonical` from `compute(row)` and add the column only if it fills.

        A column nothing can populate is worse than a missing one: it reads as a
        measured zero across the whole year. So the values come first and the
        header only follows if at least one row actually produced a number.
        """
        filled = False
        for row in plan.rows:
            value = compute(row)
            if value is not None:
                row.values.setdefault(canonical, value)
                filled = True
        if filled and canonical not in names:
            plan.columns.append(LabelSemantics(canonical, group, peak, "charge", unit,
                                               order, "derived"))
            names.add(canonical)

    peak_kwh = [name for name in ("On-Peak kWh", "Shoulder-Peak kWh", "Off-Peak kWh")
                if name in names]
    if peak_kwh and "Total kWh" not in names:
        def total_kwh(row):
            # Only a row with every period the sheet carries has a total. Summing
            # whichever periods survived would publish a partial sum as the month's
            # usage -- and a row whose printed total was ambiguous would get one.
            if any(name not in row.values for name in peak_kwh):
                return None
            return sum((row.values[name] for name in peak_kwh), Decimal("0"))
        derive("Total kWh", USAGE, "kWh", 15, total_kwh)

    # Every remaining conversion keys off the column's UNIT, never its name. A gas
    # volume can be printed as "Volume", "Natural Gas Usage" or "Consumption", and
    # matching on wording would miss all but the one spelled the expected way.
    def by_unit(*units) -> str | None:
        wanted = {u.casefold() for u in units}
        for column in plan.columns:
            if column.unit and column.unit.casefold() in wanted and column.source != "derived":
                return column.canonical
        return None

    # Gas is quoted in therms on some bills and MMBtu on others, and the analysis
    # sheet shows both, so the conversion runs in whichever direction is missing.
    therms = by_unit("therms", "therm")
    mmbtu = by_unit("mmbtu")
    if therms and not mmbtu:
        derive("Usage (MMBtu)", USAGE, "MMBtu", 16,
               lambda row: row.values[therms] * MMBTU_PER_THERM if therms in row.values else None)
    elif mmbtu and not therms:
        derive("Usage (Therms)", USAGE, "Therms", 17,
               lambda row: row.values[mmbtu] / MMBTU_PER_THERM if mmbtu in row.values else None)

    # Gallons of a delivered fuel carry a heat content specific to that fuel.
    fuel = _fuel_of(plan.key)
    gallons = by_unit("gallons", "gallon", "gal")
    if fuel and gallons and not mmbtu:
        derive("Usage (MMBtu)", USAGE, "MMBtu", 16,
               lambda row: (row.values[gallons] * MMBTU_PER_GALLON[fuel]
                            if gallons in row.values else None))

    # TGals -> Gallons, as the Classy Closets water sheet shows both.
    tgals = by_unit("tgals", "tgal", "kgal", "kgals")
    if tgals and not gallons:
        derive("Water Usage (Gallons)", USAGE, "Gallons", 17,
               lambda row: row.values[tgals] * GALLONS_PER_TGAL if tgals in row.values else None)

    # No band subtotals: every charge sits in the one Charges band, and the sheet's
    # Total ($) column is already that band's sum.

    # Cost per gallon, for delivered fuels only. Propane and diesel are bought by
    # the gallon, so $/gal is the number an assessor negotiates on. Water is
    # metered and priced per TGal, where the same ratio is ~$0.018 and quantizes
    # to $0.02 -- almost entirely rounding noise -- and neither client water sheet
    # carries the column.
    if fuel and gallons:
        def cost_per_gallon(row):
            quantity = row.values.get(gallons)
            total = row.printed_total if row.printed_total is not None else row.computed_total
            if not quantity or total is None:
                return None
            return (total / quantity).quantize(Decimal("0.01"))
        derive("Cost Per Gallon ($)", USAGE, MONEY, 40, cost_per_gallon)

    plan.columns.sort(key=LabelSemantics.sort_key)


# --- reconciliation ---------------------------------------------------------


def reconcile(plan: SheetPlan, tolerance: Decimal) -> None:
    """Compare each row's money columns with the bill's printed current charges.

    This is the safety net that survives the pivot: if the columns do not add up
    to what the bill said, either a line item was missed or one landed in the
    wrong column, and the row is flagged rather than quietly shipped.
    """
    money = [c.canonical for c in plan.columns
             if c.unit == MONEY and c.group != OTHER and c.source != "derived"]
    for row in plan.rows:
        present = [row.values[name] for name in money if name in row.values]
        if row.printed_total is None or (not present and row.printed_total == 0):
            row.status = "not_checked"
            row.notes.append("No printed current charges to check against."
                             if row.printed_total is None else "Nothing billed this period.")
            continue
        # A bill that prints charges but placed none is the worst miss there is,
        # not an unchecked row: every charge column for the month would sit empty.
        row.computed_total = sum(present, Decimal("0"))
        row.difference = row.computed_total - row.printed_total
        row.status = "matched" if abs(row.difference) <= tolerance else "review"
        if row.status == "review":
            row.notes.append(
                f"Columns total {row.computed_total:,.2f} against printed "
                f"{row.printed_total:,.2f} (difference {row.difference:+,.2f}). "
                "Check for a missed line item or one placed in the wrong column."
            )


def detect_gaps(plan: SheetPlan) -> None:
    """Note missing months, so a short series is never mistaken for a full year."""
    anchors = [row.period.anchor for row in plan.rows if row.period.anchor]
    if len(anchors) < 2:
        return
    have = {(a.year, a.month) for a in anchors}
    first, last = min(have), max(have)

    # Walk the calendar months the series spans, inclusive of both ends. Working in
    # (year, month) rather than dates keeps a 31st-of-the-month anchor from being
    # compared against a 1st and inventing a month past the end of the series.
    expected = []
    year, month = first
    while (year, month) <= last:
        expected.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)

    missing = [f"{_MONTHS[m - 1]}-{y % 100:02d}" for y, m in expected if (y, m) not in have]
    if missing:
        plan.warnings.append("No bill for " + ", ".join(missing))


# --- building ---------------------------------------------------------------


def build_sheets(
    bills: Sequence[tuple[str, object]],
    mapping: dict[str, LabelSemantics],
    *,
    parse_amount,
    parse_number,
    tolerance: Decimal = Decimal("0.02"),
) -> list[SheetPlan]:
    """Arrange extracted bills into analysis sheets.

    `bills` is (bill_id, BillExtraction) pairs; the id is carried through so a
    flagged row can be traced back to the PDF it came from.
    """
    grouped: dict[SheetKey, list[tuple[str, object]]] = defaultdict(list)
    for bill_id, bill in bills:
        grouped[sheet_key_for(bill)].append((bill_id, bill))

    grouped = _merge_partial_keys(grouped)

    plans = []
    for key, members in grouped.items():
        plans.append(_build_one(key, members, mapping,
                                parse_amount=parse_amount, parse_number=parse_number,
                                tolerance=tolerance))
    plans.sort(key=lambda p: (p.key.utility_type, p.key.company, p.title))
    return plans


def _merge_partial_keys(grouped: dict[SheetKey, list]) -> dict[SheetKey, list]:
    """Fold a bill that omitted its rate schedule into the series that printed one.

    Utilities commonly print the rate only on the first bill of a series (the
    packaged SRP set prints it once in twelve). Left alone that produces two
    sheets for one meter: a twelve-row one and a one-row one.
    """
    merged: dict[SheetKey, list] = {}
    for key, members in grouped.items():
        if key.rate_schedule is not None:
            merged[key] = list(members)
    for key, members in grouped.items():
        if key.rate_schedule is not None:
            continue
        candidates = [
            k for k in merged
            if k.company == key.company and k.utility_type == key.utility_type
            and k.meter == key.meter and k.account == key.account
        ]
        # Exactly one candidate means the rate is genuinely just missing from this
        # bill. Two or more means the site has several rate schedules and we cannot
        # tell which one this bill belongs to -- guessing would silently file a
        # month under the wrong meter, so it stays on its own sheet instead.
        if len(candidates) == 1:
            merged[candidates[0]].extend(members)
        else:
            merged[key] = list(members)
    return merged


def _build_one(key, members, mapping, *, parse_amount, parse_number, tolerance) -> SheetPlan:
    used: dict[str, LabelSemantics] = {}
    rows: list[MonthRow] = []
    seen: dict[tuple, MonthRow] = {}
    warnings: list[str] = []
    ambiguous: dict[str, list[str]] = {}

    for bill_id, bill in members:
        period = period_for(bill)
        if period.key() in seen:
            warnings.append(
                f"Duplicate bill for {period.label} ignored (same service period)."
            )
            continue
        row = MonthRow(period=period, bill_id=bill_id,
                       printed_total=_amount(bill.current_charges_text, parse_amount))

        for entry in bill.financial_entries:
            if financial_effect(entry)[0] != "current_charge":
                continue
            column = mapping.get(_key(entry.label))
            if column is None:
                if normalize_label(entry.label):
                    row.notes.append(f"No column for {normalize_label(entry.label)!r}")
                continue
            if unit_class(column.unit) not in (None, "money"):
                # A charge row sharing its label with a quantity column ("On-Peak
                # Energy" printed both as kWh and as dollars) must not put dollars
                # in the kWh column. Leaving it out makes the month fail to
                # reconcile, which is visible; the alternative is a wrong figure.
                row.notes.append(f"{column.canonical}: dollar row not placed in a "
                                 f"{column.unit} column")
                continue
            value = _amount(entry.amount_text, parse_amount)
            if value is None:
                row.notes.append(f"{column.canonical}: amount not readable")
                continue
            used.setdefault(column.canonical, column)
            _place(row, column, _signed(value, column))

        readings: dict[str, tuple[LabelSemantics, list[Decimal]]] = {}
        for measurement in bill.measurements:
            column = mapping.get(_key(measurement_label(measurement)))
            if column is None:
                continue
            if "$" in (measurement.value_text or ""):
                continue  # a price per unit, not a quantity
            if not units_compatible(measurement.unit, column.unit):
                # Printed in a unit the column cannot hold: the extractor has
                # attached another row's value (a power factor on a kW line).
                row.notes.append(f"{column.canonical}: {measurement.value_text!r} "
                                 f"is not in {column.unit}; not placed")
                continue
            value = _quantity(measurement.value_text, measurement.unit, parse_number)
            if value is None:
                continue
            readings.setdefault(column.canonical, (column, []))[1].append(value)

        _recover_unlabelled_period(readings, mapping, row)

        for canonical, (column, values) in readings.items():
            value, note = resolve_readings(values, additive=unit_class(column.unit) in ADDITIVE)
            if note:
                row.notes.append(f"{canonical}: {note}")
            if value is None:
                ambiguous.setdefault(canonical, []).append(period.label)
                continue
            used.setdefault(canonical, column)
            row.values[canonical] = value

        seen[period.key()] = row
        rows.append(row)

    for canonical, months in ambiguous.items():
        warnings.append(
            f"{canonical} left blank for {', '.join(months)}: the bill printed several "
            "readings under one label and none is provably the monthly figure."
        )

    rows.sort(key=lambda r: (r.period.anchor is None, r.period.anchor or date.min))
    plan = SheetPlan(key=key, columns=sorted(used.values(), key=LabelSemantics.sort_key),
                     rows=rows, warnings=warnings)
    reconcile(plan, tolerance)
    add_derived_columns(plan)
    detect_gaps(plan)
    return plan


# --- rollup -----------------------------------------------------------------

# Which band a cost lands in on the summary sheet, per the client workbooks'
# "Electrical Energy / Electrical Demand / Electrical Fees-Taxes" breakdown.
ROLLUP_BANDS = {USAGE: "Energy", DEMAND: "Demand", TAXES_AND_FEES: "Fees/Taxes",
                CREDITS: "Credits"}


@dataclass
class RollupRow:
    category: str
    annual_cost: Decimal


def build_rollup(plans: Iterable[SheetPlan]) -> list[RollupRow]:
    """Annual cost by category, summed from the sheets -- never re-extracted."""
    totals: dict[str, Decimal] = defaultdict(Decimal)
    for plan in plans:
        for column in plan.columns:
            if column.unit != MONEY or column.source == "derived":
                continue
            band = ROLLUP_BANDS.get(column.group)
            if band is None:
                continue
            utility = plan.key.utility_type.title()
            category = f"{utility} {band}" if utility.casefold() == "electricity" else utility
            totals[category] += plan.column_total(column.canonical)

    rows = [RollupRow(category, amount) for category, amount in sorted(totals.items())]
    rows.append(RollupRow("TOTAL", sum((r.annual_cost for r in rows), Decimal("0"))))
    return rows

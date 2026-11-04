"""What the model returns for one bill, and nothing more.

The model reads the bill: it decides what each printed figure is (a usage
reading, a demand, a charge line, a tax), its unit and its time-of-use period.
Everything after that is code, and code only checks: `verify` drops any figure
the page does not print, and `columns` turns what survives into sheet columns.

Every figure is carried as the text printed on the bill plus the page it was
read from, so it can be looked up on that page before it is believed.

Field order and the absence of defaults are deliberate: OpenAI strict structured
outputs require every property to be listed as required, and nullable fields are
expressed as `X | None`, which serialises as the `anyOf [X, null]` strict mode
accepts.
"""
from __future__ import annotations

import copy
from typing import Literal

from pydantic import BaseModel, Field

TouPeriod = Literal[
    "on_peak", "mid_peak", "shoulder", "off_peak", "super_off_peak",
    "critical_peak", "total", "none",
]

ReadingKind = Literal[
    "energy",           # consumption this period: kWh, therms, CCF, MMBtu, gallons
    "billed_demand",    # the demand the bill charges for
    "measured_demand",  # actual / maximum metered demand, after any multiplier
    "reactive",         # kVAR, kVARh, kVA, kVAh, Rkva
    "power_factor",
    "load_factor",
    "meter_register",   # raw dial readings, present/previous reads, multipliers
    "other",
]

ChargeCategory = Literal[
    "energy",   # priced on usage: energy, fuel adjustments, usage riders
    "demand",   # priced on demand (kW)
    "fixed",    # customer, service, basic, meter, facilities charges
    "tax",      # taxes, gross receipts tax, franchise fees
    "fee",      # other fees and surcharges
    "credit",   # credits, discounts, rebates: reduce the bill
    "other",
]

UtilityType = Literal["electricity", "gas", "water", "sewer", "propane", "diesel", "other"]


class PrintedText(BaseModel):
    """A single printed value (a date, a total) and where it is printed."""

    text: str = Field(description="The value alone, exactly as printed: '$4,773.41' or "
                                  "'01/16/2026', never its label.")
    page: int = Field(description="1-based index of the page image it is printed on.")


class Reading(BaseModel):
    label: str = Field(description="The row's printed label, e.g. 'On-Peak Use and Charge'.")
    value: str = Field(description="The quantity exactly as printed, e.g. '16,720.000'. One number only.")
    unit: str | None = Field(description="The printed unit: kWh, kW, kVAR, therms, CCF, MMBtu, gallons, %. Null if none is printed.")
    kind: ReadingKind
    period: TouPeriod
    page: int = Field(description="1-based index of the page image it is printed on.")


class Charge(BaseModel):
    label: str = Field(description="The line-item name only: no quantity, rate, percentage, day count or date.")
    amount: str = Field(description="The dollar amount exactly as printed, including any minus sign or parentheses.")
    category: ChargeCategory
    period: TouPeriod
    page: int = Field(description="1-based index of the page image it is printed on.")


class BillFigures(BaseModel):
    """One bill, as read by the model."""

    utility_company: str | None
    utility_type: UtilityType
    account_number: str | None
    meter_number: str | None
    rate_schedule: str | None
    statement_date: PrintedText | None = Field(description="The date the bill was issued; never the due date.")
    due_date: PrintedText | None
    service_start: PrintedText | None
    service_end: PrintedText | None
    readings: list[Reading]
    charges: list[Charge]
    total_current_charges: PrintedText | None = Field(description="This period's charges total as printed.")
    amount_due: PrintedText | None
    notes: list[str]


class LineVerdict(BaseModel):
    line: int = Field(description="The number of the quoted line.")
    is_charge_line: bool
    label: str = Field(description="The line's charge name as printed.")
    amount: str = Field(description="The line's dollar amount exactly as printed.")
    category: ChargeCategory


class LineVerdicts(BaseModel):
    """The model's verdict on quoted lines when a bill's charges did not add up."""

    verdicts: list[LineVerdict]


def strict_json_schema(model: type[BaseModel]) -> dict:
    """`model`'s JSON schema in the form OpenAI strict structured outputs accept.

    Strict mode needs `additionalProperties: false` and every property required on
    every object, and rejects `default`. Pydantic emits the rest ($defs/$ref,
    anyOf-with-null, enums) in a form strict mode already accepts.
    """
    schema = copy.deepcopy(model.model_json_schema())

    def visit(node):
        if isinstance(node, dict):
            node.pop("default", None)
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(schema)
    return schema

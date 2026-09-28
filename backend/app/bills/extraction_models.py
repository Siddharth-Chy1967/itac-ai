from typing import Literal
from pydantic import BaseModel
class FinancialEntry(BaseModel):
    label: str | None
    amount_text: str | None
    quantity_text: str | None
    quantity_unit: str | None
    rate_text: str | None
    rate_unit: str | None
    section: str | None
    meter_number: str | None
    service_period_text: str | None

    category: Literal[
        "charge",
        "tax",
        "discount",
        "payment",
        "balance",
        "subtotal",
        "total",
        "other",
    ]

    bill_page_number: int

class Measurement(BaseModel):
    label: str
    value_text: str | None
    unit: str | None

    section: str | None
    meter_number: str | None
    measurement_date: str | None
    measurement_time: str | None
    service_period_text: str | None

    bill_page_number: int

class AdditionalField(BaseModel):
    label: str
    value_text: str | None
    section: str | None
    bill_page_number: int

class BillExtraction(BaseModel):
    utility_company: str | None

    utility_types: list[Literal[
        "electricity",
        "water",
        "gas",
        "sewer",
        "other",
        "unknown",
    ]]

    account_number: str | None
    invoice_number: str | None
    bill_date: str | None
    due_date: str | None
    service_start: str | None
    service_end: str | None
    billing_days_text: str | None
    billing_period_label: str | None

    current_charges_text: str | None
    total_due_text: str | None

    financial_entries: list[FinancialEntry]
    measurements: list[Measurement]
    additional_fields: list[AdditionalField]
    extraction_notes: list[str]

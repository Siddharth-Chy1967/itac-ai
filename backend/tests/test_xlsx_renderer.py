"""Session C tests: TableSpec schema + deterministic renderer (spec §7).

`test_render_gricua_matches_reference` rebuilds the exact GRICUA Rate E-101 table
from the utility-analysis-formatting skill (verified against the real bill: May-26
total $135.79, annual TOTALS $431.44) and checks the rendered cells match. If this
test fails, the renderer changed behavior, not just cosmetics, treat it as a
regression.

The schema tests check that a malformed TableSpec fails validation with a clear
error instead of silently producing a broken or misleading workbook, and that the
```tablespec block extractor pulls the JSON back out of a table-mode response.
"""
from __future__ import annotations

import openpyxl
import pytest
from pydantic import ValidationError

from app.xlsx.renderer import render_utility_table
from app.xlsx.schema import TableSpec, extract_tablespec_block

GRICUA_SPEC = {
    "sheet_name": "Meter 10952695",
    "title": (
        "Electricity Utility Analysis: Meter 10952695 "
        "(Account 8000098001, GRICUA Rate E-101 Small Service)"
    ),
    "groups": [
        {"label": "Date", "vmerge": True, "cols": [
            {"key": "date", "header": "Statement Date", "type": "date", "width": 14}]},
        {"label": "Usage and Cost", "cols": [
            {"key": "usage", "header": "Usage (kWh)", "type": "usage", "width": 12},
            {"key": "energy", "header": "Energy Charge ($)", "type": "curr", "width": 14},
            {"key": "fuel", "header": "Fuel Adj ($)", "type": "curr", "width": 12},
            {"key": "tec", "header": "Total Energy Charge ($)", "type": "curr", "width": 16}]},
        {"label": "Demand and Cost", "cols": [
            {"key": "demand", "header": "Billed Demand (kW)", "type": "demand", "width": 14},
            {"key": "dcharge", "header": "Demand Charge ($)", "type": "curr", "width": 14}]},
        {"label": "Taxes and Fees", "cols": [
            {"key": "service", "header": "Service Charge ($)", "type": "curr", "width": 14},
            {"key": "tax", "header": "Tax ($)", "type": "curr", "width": 10}]},
        {"label": "Total ($)", "vmerge": True, "cols": [
            {"key": "total", "header": "Total ($)", "type": "curr", "width": 12}]},
    ],
    "rows": [
        {"date": "Jun-25", "usage": 880, "energy": 69.52, "fuel": 25.08, "tec": 94.60,
         "demand": 3.4, "dcharge": 17.00, "service": 30.00, "tax": 8.50, "total": 150.10},
        {"date": "Jul-25", "usage": 840, "energy": 66.37, "fuel": 23.94, "tec": 90.31,
         "demand": 3.4, "dcharge": 17.00, "service": 30.00, "tax": 8.24, "total": 145.55},
        {"date": "May-26", "usage": 760, "energy": 60.04, "fuel": 21.66, "tec": 81.70,
         "demand": 3.3, "dcharge": 16.40, "service": 30.00, "tax": 7.69, "total": 135.79},
    ],
    "totals": {"date": "TOTALS", "usage": 2480, "energy": 195.93, "fuel": 70.68, "tec": 266.61,
               "demand": 10.1, "dcharge": 50.40, "service": 90.00, "tax": 24.43, "total": 431.44},
    "summary": [
        {"resource": "Energy Usage (kWh)", "annual": 2480, "annual_type": "usage", "unit": 0.108, "unit_type": "unit3"},
        {"resource": "Demand (kW billed)", "annual": 10.1, "annual_type": "demand", "unit": 5.002, "unit_type": "unit3"},
        {"resource": "Total Bill", "annual": 431.44, "annual_type": "curr_t", "unit": None},
    ],
}


@pytest.fixture
def gricua_spec() -> TableSpec:
    return TableSpec.model_validate(GRICUA_SPEC)


def test_render_gricua_matches_reference(gricua_spec, tmp_path):
    out_path = tmp_path / "gricua.xlsx"
    render_utility_table(gricua_spec, str(out_path))

    wb = openpyxl.load_workbook(out_path)
    ws = wb.active

    # title row
    assert "GRICUA Rate E-101" in ws["A1"].value

    # header rows
    assert ws["A2"].value == "Date"
    assert ws["B3"].value == "Usage (kWh)"

    # first data row (Jun-25)
    assert ws["A4"].value == "Jun-25"
    assert ws["J4"].value == 150.10  # total column, verified against real bill

    # the known May-26 total from the real bill
    assert ws["J6"].value == 135.79

    # TOTALS row
    totals_row = 7
    assert ws[f"A{totals_row}"].value == "TOTALS"
    assert ws[f"J{totals_row}"].value == 431.44


def test_missing_row_key_rejected():
    bad = dict(GRICUA_SPEC)
    bad["rows"] = [{"date": "Jun-25", "usage": 880}]  # missing energy/fuel/tec/etc
    with pytest.raises(ValidationError, match="missing keys"):
        TableSpec.model_validate(bad)


def test_extra_row_key_rejected():
    bad = dict(GRICUA_SPEC)
    bad["rows"] = [dict(GRICUA_SPEC["rows"][0], bogus_key=1)]
    with pytest.raises(ValidationError, match="keys not declared"):
        TableSpec.model_validate(bad)


def test_totals_missing_key_rejected():
    bad = dict(GRICUA_SPEC)
    bad["totals"] = {k: v for k, v in GRICUA_SPEC["totals"].items() if k != "tax"}
    with pytest.raises(ValidationError, match="totals is missing keys"):
        TableSpec.model_validate(bad)


def test_vmerge_group_with_multiple_columns_rejected():
    bad = dict(GRICUA_SPEC)
    bad["groups"] = [
        {"label": "Date", "vmerge": True, "cols": [
            {"key": "date", "header": "Statement Date", "type": "date"},
            {"key": "extra", "header": "Extra", "type": "usage"},
        ]}
    ]
    with pytest.raises(ValidationError, match="vmerge=True"):
        TableSpec.model_validate(bad)


def test_duplicate_column_keys_rejected():
    bad = dict(GRICUA_SPEC)
    bad["groups"] = GRICUA_SPEC["groups"] + [
        {"label": "Duplicate", "cols": [
            {"key": "usage", "header": "Usage again", "type": "usage"}
        ]}
    ]
    with pytest.raises(ValidationError, match="Duplicate column keys"):
        TableSpec.model_validate(bad)


# --- ```tablespec block extraction (spec §5.3) ----------------------------

def test_extract_tablespec_block_pulls_json():
    text = (
        "Here is the analysis you asked for.\n\n"
        "```tablespec\n"
        '{"sheet_name": "S", "title": "T"}\n'
        "```\n\n"
        "Let me know if you need changes."
    )
    block = extract_tablespec_block(text)
    assert block == '{"sheet_name": "S", "title": "T"}'


def test_extract_tablespec_block_absent_returns_none():
    assert extract_tablespec_block("Just a prose answer, no table.") is None

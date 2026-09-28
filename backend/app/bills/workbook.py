"""Render analysis sheets as the .xlsx an assessor actually opens.

Layout mirrors the client workbooks this was built from: a two-tier header with a
merged band caption ("Usage and Cost") over its detail columns, one row per month,
and a rollup sheet of annual cost by category.

Two rules the format depends on:

* **Numbers are written as numbers.** A figure stored as text cannot be charted or
  summed, which is the first thing anyone does with these sheets.
* **A row that failed reconciliation is visibly marked.** The workbook leaves the
  building with its own warnings attached, because it will be read long after the
  upload screen that reported them is closed.
"""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Iterable, Sequence

from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from .shape import MONEY, RollupRow, SheetPlan

HEADER_FILL = PatternFill("solid", fgColor="1F3864")
BAND_FILL = PatternFill("solid", fgColor="2E5496")
REVIEW_FILL = PatternFill("solid", fgColor="FFF2CC")
TOTAL_FILL = PatternFill("solid", fgColor="D9E2F3")
HEADER_FONT = Font(color="FFFFFF", bold=True, size=10)
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

MONEY_FORMAT = '#,##0.00'
QUANTITY_FORMAT = '#,##0.###'
DATE_FORMAT = 'mmm-yy'


def _number_format(unit: str | None) -> str:
    return MONEY_FORMAT if unit == MONEY else QUANTITY_FORMAT


def _caption(column) -> str:
    """Name the column with its unit, without saying the unit twice.

    Many canonical names already carry it -- "On-Peak kWh", "Cost Per Gallon ($)"
    -- so appending blindly produces "On-Peak kWh (kWh)".
    """
    name = column.canonical
    if not column.unit:
        return name
    suffix = "$" if column.unit == MONEY else column.unit
    if suffix.casefold() in name.casefold():
        return name
    return f"{name} ({suffix})"


def write_analysis_workbook(
    plans: Sequence[SheetPlan],
    rollup: Iterable[RollupRow],
    destination: Path | object,
) -> Path | object:
    book = Workbook()
    book.remove(book.active)

    used: set[str] = set()
    for plan in plans:
        _write_sheet(book, plan, _unique_name(plan.key.sheet_name(), used))
    _write_rollup(book, list(rollup))

    if not book.sheetnames:  # openpyxl refuses to save an empty book
        book.create_sheet("No Bills")
    book.save(destination)
    return destination


def _unique_name(name: str, used: set[str]) -> str:
    """Excel rejects duplicate sheet names; two meters can share a title."""
    candidate, suffix = name, 2
    while candidate.casefold() in used:
        tail = f" ({suffix})"
        candidate = name[: 31 - len(tail)] + tail
        suffix += 1
    used.add(candidate.casefold())
    return candidate


def _write_sheet(book: Workbook, plan: SheetPlan, name: str) -> None:
    sheet = book.create_sheet(name)
    columns = plan.columns

    # Row 1: title. Row 2: merged band captions. Row 3: detail headers.
    sheet.cell(1, 1, plan.title).font = Font(bold=True, size=12)

    sheet.cell(2, 1, "Date").fill = BAND_FILL
    sheet.cell(2, 1).font = HEADER_FONT
    cursor = 2
    for caption, span in plan.header_bands():
        first, last = cursor, cursor + span - 1
        cell = sheet.cell(2, first, caption)
        cell.fill, cell.font = BAND_FILL, HEADER_FONT
        cell.alignment = Alignment(horizontal="center")
        if span > 1:
            sheet.merge_cells(start_row=2, start_column=first, end_row=2, end_column=last)
        cursor = last + 1

    header = sheet.cell(3, 1, "Statement Date")
    header.fill, header.font = HEADER_FILL, HEADER_FONT
    header.alignment = Alignment(wrap_text=True, vertical="center")
    for index, column in enumerate(columns, start=2):
        caption = _caption(column)
        cell = sheet.cell(3, index, caption)
        cell.fill, cell.font = HEADER_FILL, HEADER_FONT
        cell.alignment = Alignment(wrap_text=True, vertical="center")

    total_column = len(columns) + 2
    cell = sheet.cell(3, total_column, "Total ($)")
    cell.fill, cell.font = HEADER_FILL, HEADER_FONT
    cell.alignment = Alignment(wrap_text=True, vertical="center")

    for offset, row in enumerate(plan.rows):
        excel_row = 4 + offset
        _write_row(sheet, excel_row, row, columns, total_column)

    _write_column_totals(sheet, plan, len(plan.rows), total_column)

    sheet.row_dimensions[3].height = 40
    sheet.freeze_panes = "B4"
    _fit_columns(sheet)

    if plan.warnings:
        note = sheet.cell(4 + len(plan.rows) + 2, 1, "Warnings: " + "; ".join(plan.warnings))
        note.font = Font(italic=True, color="9C5700")


def _write_row(sheet, excel_row, row, columns, total_column) -> None:
    anchor = row.period.anchor
    label = sheet.cell(excel_row, 1)
    if anchor is not None:
        label.value = anchor
        label.number_format = DATE_FORMAT
    else:
        label.value = row.label
    label.border = BORDER

    for index, column in enumerate(columns, start=2):
        cell = sheet.cell(excel_row, index)
        value = row.values.get(column.canonical)
        if value is not None:
            # float() is what makes the cell a number Excel can chart and sum;
            # Decimal is kept all the way to here so no rounding happens earlier.
            cell.value = float(value)
            cell.number_format = _number_format(column.unit)
        cell.border = BORDER

    total = sheet.cell(excel_row, total_column)
    if row.printed_total is not None:
        total.value = float(row.printed_total)
        total.number_format = MONEY_FORMAT
    total.border = BORDER

    if row.status == "review":
        for index in range(1, total_column + 1):
            sheet.cell(excel_row, index).fill = REVIEW_FILL
        total.comment = Comment(
            "\n".join(row.notes) or "This month did not reconcile.", "ITAC AI"
        )
    elif row.status == "not_checked" and row.notes:
        total.comment = Comment("\n".join(row.notes), "ITAC AI")


def _write_column_totals(sheet, plan: SheetPlan, row_count: int, total_column: int) -> None:
    """A totals row of live formulas, so edits to the data update the sum."""
    if not row_count:
        return
    excel_row = 4 + row_count
    label = sheet.cell(excel_row, 1, "Total")
    label.font, label.fill, label.border = Font(bold=True), TOTAL_FILL, BORDER

    for index, column in enumerate(plan.columns, start=2):
        cell = sheet.cell(excel_row, index)
        letter = get_column_letter(index)
        cell.value = f"=SUM({letter}4:{letter}{excel_row - 1})"
        cell.number_format = _number_format(column.unit)
        cell.font, cell.fill, cell.border = Font(bold=True), TOTAL_FILL, BORDER

    letter = get_column_letter(total_column)
    cell = sheet.cell(excel_row, total_column)
    cell.value = f"=SUM({letter}4:{letter}{excel_row - 1})"
    cell.number_format = MONEY_FORMAT
    cell.font, cell.fill, cell.border = Font(bold=True), TOTAL_FILL, BORDER


def _write_rollup(book: Workbook, rows: list[RollupRow]) -> None:
    sheet = book.create_sheet("Overall Utility Analysis")
    sheet.cell(1, 1, "Utility Category").fill = HEADER_FILL
    sheet.cell(1, 1).font = HEADER_FONT
    sheet.cell(1, 2, "Annual Cost ($)").fill = HEADER_FILL
    sheet.cell(1, 2).font = HEADER_FONT

    for offset, row in enumerate(rows, start=2):
        name = sheet.cell(offset, 1, row.category)
        amount = sheet.cell(offset, 2, float(row.annual_cost))
        amount.number_format = MONEY_FORMAT
        name.border = amount.border = BORDER
        if row.category == "TOTAL":
            name.font = amount.font = Font(bold=True)
            name.fill = amount.fill = TOTAL_FILL
    _fit_columns(sheet)


def _fit_columns(sheet) -> None:
    widths: dict[int, int] = {}
    for row in sheet.iter_rows():
        for cell in row:
            if cell.value is None:
                continue
            longest = max(len(part) for part in str(cell.value).split("\n"))
            widths[cell.column] = max(widths.get(cell.column, 10), min(longest + 2, 28))
    for column, width in widths.items():
        sheet.column_dimensions[get_column_letter(column)].width = width

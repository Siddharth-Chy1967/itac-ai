"""Audit an analysis workbook against the bill PDFs it was built from.

    python scripts/check_bill_workbook.py "Utility Analysis.xlsx" bill1.pdf bill2.pdf ...

For every month row it finds the source PDF by the row's statement date, then
checks that every number in the row is printed on that bill (chart axis labels
excluded, sign included), and reports the DONE criteria of the extraction
rebuild: usage per month, money in charge columns, demand, labels, month labels
and the TOTALS row. Exit status 1 if any printed-figure check fails.
"""
from __future__ import annotations

import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from openpyxl import load_workbook

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.bills.local_inference import LocalInferenceConfig  # noqa: E402
from app.bills.page_text import read_pdf  # noqa: E402
from app.bills.verify import (  # noqa: E402
    dates_in, numbers_in, ocr_prints_date, signed_numbers_in,
)


def load_bills(pdfs, config):
    bills = []
    for pdf in pdfs:
        pages = read_pdf(Path(pdf), config)
        text = "\n".join(page.text for page in pages)
        bills.append({"pdf": str(pdf), "dates": dates_in(text), "text": text,
                      "unsigned": set(numbers_in(text)), "signed": signed_numbers_in(text),
                      "axis": {box[4] for page in pages for box in page.axis_labels}})
    return bills


def evaluate_sum(sheet, formula: str) -> Decimal:
    ref = formula[len("=SUM("):-1]
    first, last = ref.split(":")
    total = Decimal("0")
    for row in sheet[first:last]:
        for cell in row:
            if isinstance(cell.value, (int, float)):
                total += Decimal(str(cell.value))
    return total


def main(argv) -> int:
    workbook, pdfs = argv[0], argv[1:]
    config = LocalInferenceConfig()
    bills = load_bills(pdfs, config)
    book = load_workbook(workbook)
    failures = 0
    for sheet in book:
        if sheet.title == "Overall Utility Analysis":
            continue
        print(f"== Sheet {sheet.title!r}")
        headers = {c.column: c.value for c in sheet[3]}
        total_col = next(col for col, name in headers.items() if name == "Total ($)")
        kwh_cols = [col for col, name in headers.items() if name and "kWh" in str(name)
                    and "$" not in str(name)]
        demand_cols = [col for col, name in headers.items() if name and "Demand" in str(name)
                       and "(kW)" in str(name)]
        money_cols = [col for col, name in headers.items() if name and str(name).endswith("($)")
                      and col != total_col]
        unlabeled = [name for name in headers.values()
                     if not name or "unlabel" in str(name).lower()]
        print(f"   columns: {len(headers) - 1}; unlabelled: {unlabeled or 'none'}")
        row = 4
        usage_by_month = {}
        while sheet.cell(row, 1).value not in (None, "TOTALS"):
            stamp = sheet.cell(row, 1).value
            when = stamp.date() if isinstance(stamp, datetime) else stamp
            source = [b for b in bills if when in b["dates"]] or \
                     [b for b in bills if ocr_prints_date(b["text"], when)]
            label = when.strftime("%b-%y") if hasattr(when, "strftime") else str(when)
            if not source:
                print(f"   {label}: no source PDF prints this statement date")
                failures += 1
                row += 1
                continue

            def missing_on(bill, row=row):
                out = []
                for col, name in headers.items():
                    value = sheet.cell(row, col).value
                    if col == 1 or not isinstance(value, (int, float)):
                        continue
                    number = Decimal(str(value)).normalize()
                    printed = (number in bill["signed"] if number < 0
                               else number in bill["unsigned"])
                    if not printed:
                        out.append(f"{name}={value}")
                return out

            # Some bills also print the previous statement's date (City of Phoenix),
            # so a date can name two PDFs. The row's source is the one whose figures
            # are all printed on it; failing that, the first candidate is reported.
            bill = next((b for b in source if not missing_on(b)), source[0])
            missing = missing_on(bill)
            usage = {headers[c]: sheet.cell(row, c).value for c in kwh_cols}
            usage_by_month[label] = usage
            demand = {headers[c]: sheet.cell(row, c).value for c in demand_cols}
            money = [sheet.cell(row, c).value for c in money_cols
                     if sheet.cell(row, c).value is not None]
            total = sheet.cell(row, total_col).value
            comment = sheet.cell(row, total_col).comment
            print(f"   {label} <- {Path(bill['pdf']).name}")
            print(f"      every value printed on this bill: {'YES' if not missing else 'NO ' + str(missing)}")
            print(f"      usage: {usage}")
            print(f"      demand: {demand}")
            print(f"      charge cells: {len(money)}, sum {sum(Decimal(str(m)) for m in money):,.2f}; "
                  f"Total ($) {total}")
            if comment:
                print(f"      note: {comment.text.splitlines()[0]}")
            failures += bool(missing)
            row += 1
        if sheet.cell(row, 1).value == "TOTALS":
            computed = {headers[c]: evaluate_sum(sheet, sheet.cell(row, c).value)
                        for c in headers if c > 1 and str(sheet.cell(row, c).value or "").startswith("=SUM(")}
            print(f"   TOTALS row: {len(computed)} formulas; Total ($) evaluates to "
                  f"{computed.get('Total ($)')}")
        values = [tuple(v.values()) for v in usage_by_month.values()]
        if len(values) > 1:
            print(f"   usage differs by month: {'YES' if len(set(values)) == len(values) else 'NO'}")
    print("RESULT:", "every workbook value is printed on its bill" if not failures
          else f"{failures} row(s) failed the printed-figure check")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

"""
Session C: deterministic renderer.

Takes a validated TableSpec (see schema.py) and writes a formatted .xlsx.
This is the only place openpyxl gets touched. The LLM never sees or
generates styling, it only ever produces the data in a TableSpec; this
function is what turns that into the actual title row, two-tier grouped
header, monthly data matrix with a TOTALS row, and the Resource / Annual
Total / Unit Cost summary underneath.

Ported from the utility-analysis-formatting skill's reference
implementation (build_utility_analysis.py), refactored to take a TableSpec
model instead of a raw dict, and to pull all styling from styles.py instead
of local constants.
"""

import openpyxl
from openpyxl.utils import get_column_letter

from .schema import TableSpec
from .styles import BORDER, CENTER, DATEBLU, FMT, GOLD, LEFT, NAVY, RIGHT, SUBHDR, WHITE, fill, font


def render_utility_table(spec: TableSpec, out_path: str) -> str:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = spec.sheet_name

    # flatten columns from groups, in order
    cols = [c for g in spec.groups for c in g.cols]
    ncols = len(cols)
    last_letter = get_column_letter(ncols)

    # ---- title (row 1, merged across all columns) ----
    ws.merge_cells(f"A1:{last_letter}1")
    t = ws["A1"]
    t.value = spec.title
    t.font = font(bold=True)
    t.alignment = LEFT

    # ---- header rows: super-header = row 2, sub-header = row 3 ----
    ci = 1
    for g in spec.groups:
        span = len(g.cols)
        first = ci
        last = ci + span - 1
        f_letter = get_column_letter(first)
        l_letter = get_column_letter(last)

        if g.vmerge:
            # single-column section: merge label down over both header rows
            ws.merge_cells(f"{f_letter}2:{l_letter}3")
            cell = ws.cell(row=2, column=first, value=g.label)
            cell.font = font(bold=True, color=WHITE)
            cell.fill = fill(NAVY)
            cell.alignment = CENTER
            cell.border = BORDER
            for r in (2, 3):
                for col in range(first, last + 1):
                    cc = ws.cell(row=r, column=col)
                    cc.fill = fill(NAVY)
                    cc.border = BORDER
        else:
            if span > 1:
                ws.merge_cells(f"{f_letter}2:{l_letter}2")
            sh = ws.cell(row=2, column=first, value=g.label)
            sh.font = font(bold=True, color=WHITE)
            sh.fill = fill(NAVY)
            sh.alignment = CENTER
            sh.border = BORDER
            for col in range(first, last + 1):
                ws.cell(row=2, column=col).fill = fill(NAVY)
                ws.cell(row=2, column=col).border = BORDER
            for j, c in enumerate(g.cols):
                sc = ws.cell(row=3, column=first + j, value=c.header)
                sc.font = font(bold=True, color=WHITE)
                sc.fill = fill(SUBHDR)
                sc.alignment = CENTER
                sc.border = BORDER
        ci += span

    # ---- data rows ----
    r = 4
    for i, row in enumerate(spec.rows):
        band = i % 2 == 0
        for j, c in enumerate(cols):
            col = j + 1
            value = row.get(c.key)
            cell = ws.cell(row=r, column=col, value=value)
            cell.border = BORDER
            if c.type == "date":
                cell.font = font()
                cell.alignment = CENTER
                cell.fill = fill(DATEBLU)
            else:
                negative = isinstance(value, (int, float)) and value < 0
                if c.type == "curr":
                    cell.number_format = FMT["credit"] if negative else FMT["curr"]
                    cell.font = font(color=GOLD)
                elif c.type == "demand":
                    cell.number_format = FMT["demand"]
                    cell.font = font()
                elif c.type == "usage":
                    cell.number_format = FMT["usage"]
                    cell.font = font()
                cell.alignment = RIGHT
                cell.fill = fill(DATEBLU) if band else fill(WHITE)
        r += 1

    # ---- TOTALS row ----
    for j, c in enumerate(cols):
        col = j + 1
        value = spec.totals.get(c.key)
        cell = ws.cell(row=r, column=col, value=value)
        cell.fill = fill(SUBHDR)
        cell.border = BORDER
        cell.font = font(bold=True, color=WHITE)
        if c.type == "date":
            cell.alignment = LEFT
        elif c.type == "curr":
            cell.number_format = FMT["curr_t"]
            cell.alignment = RIGHT
        elif c.type == "demand":
            cell.number_format = FMT["demand"]
            cell.alignment = RIGHT
        elif c.type == "usage":
            cell.number_format = FMT["usage"]
            cell.alignment = RIGHT
    totals_row = r

    # ---- column widths ----
    for j, c in enumerate(cols):
        ws.column_dimensions[get_column_letter(j + 1)].width = c.width

    # ---- summary table (Resource / Annual Total / Unit Cost) ----
    start = totals_row + 2
    heads = ["Resource", "Annual Total", "Unit Cost($/kWh or $/kW-mo)"]
    for j, h in enumerate(heads):
        cell = ws.cell(row=start, column=j + 1, value=h)
        cell.font = font(bold=True)
        cell.border = BORDER
        cell.alignment = CENTER if j else LEFT
        cell.fill = fill(WHITE)

    rr = start + 1
    for item in spec.summary:
        lab = ws.cell(row=rr, column=1, value=item.resource)
        lab.font = font(bold=True)
        lab.border = BORDER
        lab.alignment = LEFT

        av = ws.cell(row=rr, column=2, value=item.annual)
        av.border = BORDER
        av.alignment = RIGHT
        av.font = font()
        av.number_format = FMT.get(item.annual_type, item.annual_type)

        uv = ws.cell(row=rr, column=3, value=item.unit)
        uv.border = BORDER
        uv.alignment = RIGHT
        uv.font = font()
        if item.unit is not None:
            uv.number_format = FMT.get(item.unit_type, item.unit_type)
        rr += 1

    wb.save(out_path)
    return out_path

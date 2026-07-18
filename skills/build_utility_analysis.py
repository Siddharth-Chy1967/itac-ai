"""
Reference implementation for the utility-analysis-formatting skill.

Renders an IAC/ITAC electricity (or any metered utility) analysis table:
  - a title row
  - a two-tier grouped header (super-header band over merged sections,
    sub-header row underneath)
  - the monthly data matrix with a TOTALS row
  - a Resource / Annual Total / Unit Cost summary table below

The renderer is data-driven: pass a `spec` dict. The same function builds the
compact GRICUA E-101 table and the full PNM 3C TOU table; only the column
groups and rows change. Nothing here is hard-coded to one utility.
"""

import openpyxl
from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
from openpyxl.utils import get_column_letter

# ---------- palette ----------
NAVY    = "1F3864"   # super-header band
SUBHDR  = "2E5496"   # sub-header row and TOTALS row
DATEBLU = "DDEBF7"   # date column fill and light row banding
GOLD    = "BF8F00"   # currency text in data rows
WHITE   = "FFFFFF"
BLACK   = "000000"

FONT = "Times New Roman"
SIZE = 12

THIN = Side(style="thin", color="808080")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

def fill(hexcolor):
    return PatternFill("solid", fgColor=hexcolor)

# ---------- number formats by column type ----------
FMT = {
    "usage":  "#,##0",                      # kWh, gal, therm: whole
    "demand": "#,##0.0",                     # kW: 1 decimal, all rows
    "curr":   "$#,##0.00",                   # currency data rows: 2 decimals
    "curr_t": "$#,##0",                      # currency TOTALS row: 0 decimals
    "unit3":  "$#,##0.000",                  # unit cost 3 dp (GRICUA style)
    "unit4":  "$#,##0.0000",                 # unit cost 4 dp (PNM style)
    "credit": "$#,##0.00;($#,##0.00)",       # negative credits in parentheses
}

CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
RIGHT  = Alignment(horizontal="right",  vertical="center")
LEFT   = Alignment(horizontal="left",   vertical="center")


def build_utility_analysis(spec, out_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = spec.get("sheet_name", "Analysis")

    
    cols = []
    for g in spec["groups"]:
        for c in g["cols"]:
            cols.append(c)
    ncols = len(cols)
    last_letter = get_column_letter(ncols)

   
    ws.merge_cells(f"A1:{last_letter}1")
    t = ws["A1"]
    t.value = spec["title"]
    t.font = Font(name=FONT, size=SIZE, bold=True, color=BLACK)
    t.alignment = LEFT

    # ---- header rows: super-header = row 2, sub-header = row 3 ----
    ci = 1
    for g in spec["groups"]:
        span = len(g["cols"])
        first = ci
        last = ci + span - 1
        f_letter = get_column_letter(first)
        l_letter = get_column_letter(last)
        if g.get("vmerge"):
            # single-column section: merge label down over both header rows
            ws.merge_cells(f"{f_letter}2:{l_letter}3")
            cell = ws.cell(row=2, column=first, value=g["label"])
            cell.font = Font(name=FONT, size=SIZE, bold=True, color=WHITE)
            cell.fill = fill(NAVY)
            cell.alignment = CENTER
            cell.border = BORDER
            # sub-header cells still need borders/fill for the merged block
            for r in (2, 3):
                for col in range(first, last + 1):
                    cc = ws.cell(row=r, column=col)
                    cc.fill = fill(NAVY)
                    cc.border = BORDER
        else:
            if span > 1:
                ws.merge_cells(f"{f_letter}2:{l_letter}2")
            sh = ws.cell(row=2, column=first, value=g["label"])
            sh.font = Font(name=FONT, size=SIZE, bold=True, color=WHITE)
            sh.fill = fill(NAVY)
            sh.alignment = CENTER
            sh.border = BORDER
            for col in range(first, last + 1):
                ws.cell(row=2, column=col).fill = fill(NAVY)
                ws.cell(row=2, column=col).border = BORDER
            # sub-headers (row 3)
            for j, c in enumerate(g["cols"]):
                sc = ws.cell(row=3, column=first + j, value=c["header"])
                sc.font = Font(name=FONT, size=SIZE, bold=True, color=WHITE)
                sc.fill = fill(SUBHDR)
                sc.alignment = CENTER
                sc.border = BORDER
        ci += span

    # ---- data rows ----
    r = 4
    for i, row in enumerate(spec["rows"]):
        band = (i % 2 == 0)
        for j, c in enumerate(cols):
            col = j + 1
            cell = ws.cell(row=r, column=col, value=row.get(c["key"]))
            cell.border = BORDER
            typ = c["type"]
            if typ == "date":
                cell.font = Font(name=FONT, size=SIZE, color=BLACK)
                cell.alignment = CENTER
                cell.fill = fill(DATEBLU)
            else:
                if typ == "curr":
                    cell.number_format = FMT["curr"]
                    cell.font = Font(name=FONT, size=SIZE, color=GOLD)
                elif typ == "demand":
                    cell.number_format = FMT["demand"]
                    cell.font = Font(name=FONT, size=SIZE, color=BLACK)
                elif typ == "usage":
                    cell.number_format = FMT["usage"]
                    cell.font = Font(name=FONT, size=SIZE, color=BLACK)
                cell.alignment = RIGHT
                cell.fill = fill(DATEBLU) if band else fill(WHITE)
        r += 1

    # ---- TOTALS row ----
    tot = spec["totals"]
    for j, c in enumerate(cols):
        col = j + 1
        cell = ws.cell(row=r, column=col, value=tot.get(c["key"]))
        cell.fill = fill(SUBHDR)
        cell.border = BORDER
        cell.font = Font(name=FONT, size=SIZE, bold=True, color=WHITE)
        typ = c["type"]
        if typ == "date":
            cell.alignment = LEFT
        elif typ == "curr":
            cell.number_format = FMT["curr_t"]        # 0 decimals in totals
            cell.alignment = RIGHT
        elif typ == "demand":
            cell.number_format = FMT["demand"]        # keep 1 decimal
            cell.alignment = RIGHT
        elif typ == "usage":
            cell.number_format = FMT["usage"]
            cell.alignment = RIGHT
    totals_row = r

    # ---- column widths ----
    for j, c in enumerate(cols):
        ws.column_dimensions[get_column_letter(j + 1)].width = c.get("width", 12)

    # ---- summary table (Resource / Annual Total / Unit Cost) ----
    start = totals_row + 2
    heads = ["Resource", "Annual Total", "Unit Cost($/kWh or $/kW-mo)"]
    for j, h in enumerate(heads):
        cell = ws.cell(row=start, column=j + 1, value=h)
        cell.font = Font(name=FONT, size=SIZE, bold=True, color=BLACK)
        cell.border = BORDER
        cell.alignment = CENTER if j else LEFT
        cell.fill = fill(WHITE)
    rr = start + 1
    for item in spec["summary"]:
        # Resource label
        lab = ws.cell(row=rr, column=1, value=item["resource"])
        lab.font = Font(name=FONT, size=SIZE, bold=True, color=BLACK)
        lab.border = BORDER
        lab.alignment = LEFT
        # Annual Total
        av = ws.cell(row=rr, column=2, value=item.get("annual"))
        av.border = BORDER
        av.alignment = RIGHT
        av.font = Font(name=FONT, size=SIZE, color=BLACK)
        at = item.get("annual_type", "curr_t")
        av.number_format = FMT.get(at, at)
        # Unit Cost
        uv = ws.cell(row=rr, column=3, value=item.get("unit"))
        uv.border = BORDER
        uv.alignment = RIGHT
        uv.font = Font(name=FONT, size=SIZE, color=BLACK)
        if item.get("unit") is not None:
            uv.number_format = FMT.get(item.get("unit_type", "unit3"),
                                       item.get("unit_type", "unit3"))
        rr += 1

    wb.save(out_path)
    return out_path


# ===================================================================
# Worked example: GRICUA Rate E-101 Small Service, Meter 10952695
# (reproduces the compact table exactly)
# ===================================================================
GRICUA_SPEC = {
    "sheet_name": "Meter 10952695",
    "title": ("Electricity Utility Analysis: Meter 10952695 "
              "(Account 8000098001, GRICUA Rate E-101 Small Service)"),
    "groups": [
        {"label": "Date", "vmerge": True, "cols": [
            {"key": "date", "header": "Statement Date", "type": "date", "width": 14}]},
        {"label": "Usage and Cost", "cols": [
            {"key": "usage",  "header": "Usage (kWh)",            "type": "usage", "width": 12},
            {"key": "energy", "header": "Energy Charge ($)",      "type": "curr",  "width": 14},
            {"key": "fuel",   "header": "Fuel Adj ($)",           "type": "curr",  "width": 12},
            {"key": "tec",    "header": "Total Energy Charge ($)", "type": "curr", "width": 16}]},
        {"label": "Demand and Cost", "cols": [
            {"key": "demand",  "header": "Billed Demand (kW)", "type": "demand", "width": 14},
            {"key": "dcharge", "header": "Demand Charge ($)",  "type": "curr",   "width": 14}]},
        {"label": "Taxes and Fees", "cols": [
            {"key": "service", "header": "Service Charge ($)", "type": "curr", "width": 14},
            {"key": "tax",     "header": "Tax ($)",            "type": "curr", "width": 10}]},
        {"label": "Total ($)", "vmerge": True, "cols": [
            {"key": "total", "header": "Total ($)", "type": "curr", "width": 12}]},
    ],
    "rows": [
        {"date":"Jun-25","usage":880,"energy":69.52,"fuel":25.08,"tec":94.60,"demand":3.4,"dcharge":17.00,"service":30.00,"tax":8.50,"total":150.10},
        {"date":"Jul-25","usage":840,"energy":66.37,"fuel":23.94,"tec":90.31,"demand":3.4,"dcharge":17.00,"service":30.00,"tax":8.24,"total":145.55},
        {"date":"Aug-25","usage":960,"energy":75.84,"fuel":27.36,"tec":103.20,"demand":3.5,"dcharge":17.40,"service":30.00,"tax":9.04,"total":159.64},
        {"date":"Sep-25","usage":920,"energy":72.68,"fuel":26.22,"tec":98.90,"demand":3.5,"dcharge":17.40,"service":30.00,"tax":8.78,"total":155.08},
        {"date":"Oct-25","usage":960,"energy":75.84,"fuel":27.37,"tec":103.21,"demand":3.8,"dcharge":19.20,"service":30.00,"tax":9.14,"total":161.55},
        {"date":"Nov-25","usage":840,"energy":66.36,"fuel":23.94,"tec":90.30,"demand":3.4,"dcharge":17.20,"service":30.00,"tax":8.25,"total":145.75},
        {"date":"Dec-25","usage":600,"energy":47.40,"fuel":17.10,"tec":64.50,"demand":3.3,"dcharge":16.40,"service":30.00,"tax":6.65,"total":117.55},
        {"date":"Jan-26","usage":640,"energy":50.56,"fuel":18.24,"tec":68.80,"demand":3.4,"dcharge":17.20,"service":30.00,"tax":6.96,"total":122.96},
        {"date":"Feb-26","usage":680,"energy":53.73,"fuel":19.38,"tec":73.11,"demand":3.4,"dcharge":16.80,"service":30.00,"tax":7.19,"total":127.10},
        {"date":"Mar-26","usage":680,"energy":53.72,"fuel":19.39,"tec":73.11,"demand":3.3,"dcharge":16.60,"service":30.00,"tax":7.18,"total":126.89},
        {"date":"Apr-26","usage":840,"energy":66.36,"fuel":23.94,"tec":90.30,"demand":3.4,"dcharge":17.00,"service":30.00,"tax":8.24,"total":145.54},
        {"date":"May-26","usage":760,"energy":60.04,"fuel":21.66,"tec":81.70,"demand":3.3,"dcharge":16.40,"service":30.00,"tax":7.69,"total":135.79},
    ],
    "totals": {"date":"TOTALS","usage":9600,"energy":758,"fuel":274,"tec":1032,
               "demand":41.1,"dcharge":205.6,"service":360,"tax":96,"total":1694},
    "summary": [
        {"resource":"Energy Usage (kWh)", "annual":9600,  "annual_type":"usage",  "unit":0.108, "unit_type":"unit3"},
        {"resource":"Demand (kW billed)", "annual":41.1,  "annual_type":"demand", "unit":5.002, "unit_type":"unit3"},
        {"resource":"Energy Charge",      "annual":1032,  "annual_type":"curr_t", "unit":None},
        {"resource":"Demand Charge",      "annual":206,   "annual_type":"curr_t", "unit":None},
        {"resource":"Taxes & Fees",       "annual":456,   "annual_type":"curr_t", "unit":None},
        {"resource":"Total Bill",         "annual":1694,  "annual_type":"curr_t", "unit":None},
    ],
}

if __name__ == "__main__":
    out = build_utility_analysis(GRICUA_SPEC, "/mnt/user-data/outputs/GRICUA_E101_Meter10952695_analysis.xlsx")
    print("wrote", out)

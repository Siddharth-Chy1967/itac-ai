"""
Session C: shared Excel styling constants.

Single source of truth for colors, fonts, borders, and number formats, so
every table type this app ever renders (utility tables now, AR summary and
equipment tables later) looks consistent regardless of which renderer
function built it. Pulled from the utility-analysis-formatting skill and
report-table-conventions.md (negative-number red rule).
"""

from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

# ---------- palette ----------
NAVY = "1F3864"       # super-header band
SUBHDR = "2E5496"     # sub-header row and TOTALS row
DATEBLU = "DDEBF7"    # date column fill and light row banding
GOLD = "BF8F00"       # currency text in data rows
RED = "C00000"        # negative values (report-table-conventions Part J)
WHITE = "FFFFFF"
BLACK = "000000"

FONT_NAME = "Times New Roman"
SIZE = 12

THIN = Side(style="thin", color="808080")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def fill(hexcolor: str) -> PatternFill:
    return PatternFill("solid", fgColor=hexcolor)


def font(bold: bool = False, color: str = BLACK, size: int = SIZE) -> Font:
    return Font(name=FONT_NAME, size=size, bold=bold, color=color)


# ---------- number formats by column/value type ----------
FMT = {
    "usage": "#,##0",                              # kWh, gal, therm: whole
    "demand": "#,##0.0",                            # kW: 1 decimal, all rows
    "curr": "$#,##0.00",                            # currency data rows: 2 dp
    "curr_t": "$#,##0",                             # currency TOTALS row: 0 dp
    "unit3": "$#,##0.000",                          # unit cost 3 dp (GRICUA)
    "unit4": "$#,##0.0000",                         # unit cost 4 dp (PNM)
    "credit": "$#,##0.00;[RED]($#,##0.00)",         # negative in red parens
}

CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
RIGHT = Alignment(horizontal="right", vertical="center")
LEFT = Alignment(horizontal="left", vertical="center")

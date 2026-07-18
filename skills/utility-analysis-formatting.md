---
name: utility-analysis-formatting
description: >
  Use whenever building, editing, or formatting an Excel utility analysis table
  or workbook (electricity, gas, propane, diesel, water, or any metered utility)
  for an IAC/ITAC audit, or turning a utility bill (PNM, GRICUA, APS, SRP, gas,
  etc.) into a monthly analysis table. Trigger on phrases or close variants:
  "utility analysis", "bill analysis", "meter analysis", "usage and cost table",
  "demand and cost", "make the utility table", "format the utility sheet", "like
  my pnm", "like my gricua", "build the meter table". Also trigger any time a
  monthly utility matrix is produced (columns like Usage (kWh), Billed Demand
  (kW), energy/demand/service charges, tax, TOTALS row) or a cost summary. Covers
  the bill-to-table mapping, the grouped two-tier header (which flexes by
  tariff), the Resource/Annual Total/Unit Cost summary, exact colors and fonts,
  and number formatting.
---

# Utility Analysis Table: Mapping, Structure, Formatting

Turn a stack of monthly utility bills into one analysis table per meter: a
titled two-tier grouped header, a monthly data matrix with a TOTALS row, and a
Resource / Annual Total / Unit Cost summary beneath it. The column set flexes by
tariff (a simple GRICUA small-service meter has far fewer columns than a PNM
Rate 3C time-of-use meter), but the mapping logic, the grouping, the summary,
and the styling are identical across all of them.

## Part A: Reading a bill into columns (the mapping)

For each monthly bill, pull the line items and classify every charge into one of
three cost buckets plus optional credits. The buckets are the same three
sections that become the grouped headers, so the header and the summary always
agree.

1. **Usage**: the metered quantity (kWh, therms, gallons). On time-of-use
   tariffs, split into On-Peak and Off-Peak, and add a Total.
2. **Energy charges** (Usage and Cost section): every per-unit energy line.
   Energy/commodity charge, fuel or power-cost adjustment, and any riders
   (renewable rider, transportation electrification, cost-effective savings
   program). Sum them into **Total Energy Charge**.
3. **Demand** (Demand and Cost section): billed demand (kW) and the demand
   charge. Skip this section entirely if the tariff has no demand charge.
4. **Taxes and Fees**: the fixed monthly service/customer charge, franchise
   fees, and all taxes (state/county/city GRT, sales tax). Fixed service and
   customer charges live here, not in Energy. Sum into **Total Taxes & Fees**.
5. **Credits** (optional): negative adjustments such as a Palo Verde regulatory
   credit. Shown as its own column and its own summary row, always negative.

**Total Bill** = Total Energy Charge + Demand Charge + Total Taxes & Fees +
Credits (credits are negative). Verify this equals the bill's stated total for
each month before delivering. If it does not tie out, the mapping is wrong.

### Worked mapping 1: GRICUA Rate E-101 (compact)

Bill line -> column:
- First X kWh charge -> Energy Charge
- Fuel Cost Adjustment on all kWH -> Fuel Adj
- (Energy Charge + Fuel Adj) -> Total Energy Charge
- Demand Charge on Billing kW, quantity -> Billed Demand (kW)
- Demand Charge on Billing kW, charge -> Demand Charge
- Monthly Service Charge -> Service Charge (under Taxes and Fees)
- Tax -> Tax
- Total Current Bill Due -> Total

### Worked mapping 2: PNM Rate 3C TOU (full)

Bill line -> column:
- On-Peak Use and Charge -> On-Peak kWh, On-Peak Charge
- Off-Peak Use and Charge -> Off-Peak kWh, Off-Peak Charge
- Fuel Cost Adjustment (non-renewable + renewable) -> Fuel Adj
- Renewable Energy Rider (sum the day-split lines) -> Renewable Rider
- Transportation Electrification R53 -> Transport R53
- Cost-Effective Energy Saving Prog. -> Cost-Eff Prog
- Customer Charge -> Customer Charge (under Taxes and Fees)
- Billed demand and charge -> Billed Demand (kW), Demand Charge
- San Juan Energy Transition Charge -> San Juan ETC
- Palo Verde Regulatory Credit -> Palo Verde Credit (negative, Discount)
- City/County Franchise Fee -> Franchise Fee
- Gross Receipts Tax State/County/City -> State GRT, County GRT, City GRT
- Current Electricity Charges -> Total

Two helper columns appear on the full PNM template:
- **Total Energy Charge** = On-Peak + Off-Peak + Fuel Adj + Renewable Rider +
  Transport R53 + Cost-Eff Prog.
- **Off-Peak Charge with Adjustors** = Total Energy Charge minus On-Peak Charge
  (everything that is not the on-peak energy line). Include it only on TOU
  tariffs that need the on-peak vs rest split for charting.

### Handling discrepancies between utilities

Different utilities name and group things differently. Reconcile to the ITAC
convention, do not copy the bill's own grouping:
- A fixed monthly charge is Taxes and Fees whether the bill calls it "Monthly
  Service Charge" (GRICUA) or "Customer Charge" (PNM).
- Riders and adjustments are Energy, even when the bill lists them near taxes.
- Credits are their own negative column, never netted silently into Energy.
- If a section has no members for that tariff (for example no demand charge),
  drop the whole section rather than showing an empty column.

## Part B: Table structure (grouped two-tier header)

Never deliver a flat single-row header. Build two header tiers:
- **Row 2, super-header band**: one MERGED cell per section over its columns.
- **Row 3, sub-header**: the individual column names underneath.

Sections, in order, each with contiguous columns so the merge is clean:

| Super-header    | Compact (GRICUA)                              | Full (PNM 3C TOU)                                                                 |
|-----------------|-----------------------------------------------|-----------------------------------------------------------------------------------|
| Date            | Statement Date                                | Meter Read Date                                                                    |
| Usage (kWh)     | (folded into Usage and Cost)                  | On-Peak kWh, Off-Peak kWh, Total kWh                                               |
| Usage and Cost  | Usage (kWh), Energy Charge, Fuel Adj, Total Energy Charge | On-Peak Charge, Off-Peak Charge, Fuel Adj, Renewable Rider, Transport R53, Cost-Eff Prog, (Off-Peak w/ Adjustors), Total Energy Charge |
| Discount        | (none)                                        | Palo Verde Credit                                                                 |
| Demand and Cost | Billed Demand (kW), Demand Charge             | Billable Demand (kW), Demand Charge                                                |
| Taxes and Fees  | Service Charge, Tax                           | San Juan ETC, Franchise Fee, State GRT, County GRT, City GRT, Customer Charge, Total Taxes & Fees |
| Total ($)       | Total                                         | Total Bill                                                                         |

Rules:
- Merge each multi-column section label across its span in row 2.
- A single-column section (Date, Total) is merged vertically across rows 2 and 3.
- TOTALS row at the bottom of the matrix.
- Title in row 1, merged across all columns, bold, left-aligned. Format:
  `Electricity Utility Analysis: Meter <#> (Account <#>, <Utility> Rate <code> <name>)`.

## Part C: Summary table (Resource / Annual Total / Unit Cost)

Two blank-separated rows below the matrix, build the summary. Header:
`Resource | Annual Total | Unit Cost($/kWh or $/kW-mo)`. Rows, in order:

- Energy Usage (kWh): annual usage; unit = Total Energy Charge / usage
- Demand (kW billed): annual billed demand; unit = Demand Charge / demand
- (any credit, e.g. Palo Verde Discount: negative annual total, no unit)
- Energy Charge: annual Total Energy Charge; no unit
- Demand Charge: annual demand charge; no unit
- Taxes & Fees: annual taxes and fees; no unit
- Total Bill: annual total; no unit

Unit costs use the UNROUNDED annual totals (205.6 / 41.1 = $5.002, not 206 /
41.1). Unit cost decimals: 3 dp is fine for small meters ($0.108), 4 dp when
values are tiny ($0.0834). Row labels bold, white fill, thin borders.

## Part D: Colors and fonts (match the established sheets)

- Font: Times New Roman 12 throughout (title bold; headers bold white).
- Super-header band fill: navy `1F3864`, white bold text, centered.
- Sub-header row fill: medium navy `2E5496`, white bold text, centered.
- Date column: light blue fill `DDEBF7`, black text, centered, in every data row.
- Data-row banding: alternate `DDEBF7` and white across the body columns.
- Currency values in data rows: gold text `BF8F00`, right-aligned.
- Usage (kWh) and demand (kW) values: black text, right-aligned.
- TOTALS row: medium navy `2E5496` fill, white bold text (currency turns white,
  not gold, in this row).
- Thin grey borders (`808080`) on every cell, header, data, totals, and summary.

## Part E: Number formatting

| Value type                      | Data rows   | Totals row  |
|---------------------------------|-------------|-------------|
| Demand (kW)                     | `#,##0.0`   | `#,##0.0`   |
| Currency / cost ($)             | `$#,##0.00` | `$#,##0`    |
| Other 2-decimal numeric         | `#,##0.00`  | `#,##0`     |
| Usage (kWh, gal, therm, ccf)    | `#,##0`     | `#,##0`     |
| Credit in summary (negative)    | `$#,##0.00;($#,##0.00)` | same |
| Unit cost (summary)             | `$#,##0.000` or `$#,##0.0000` | n/a |

Every numeric cell carries a thousands comma. Demand keeps 1 decimal even in the
totals row. Currency drops to 0 decimals only in the TOTALS row.

## Part F: Reference renderer

Data-driven `build_utility_analysis(spec, out_path)`. The `spec` lists `groups`
(each with a super-header `label`, optional `vmerge`, and `cols` of
`{key, header, type, width}` where type is `date | usage | demand | curr`),
`rows`, a `totals` dict, and `summary` rows. The same function builds compact and
full tables; only the spec changes. Full runnable source, including the complete
GRICUA worked example, is in `build_utility_analysis.py` alongside this skill.
The core renderer:

```python
import openpyxl
from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
from openpyxl.utils import get_column_letter

NAVY="1F3864"; SUBHDR="2E5496"; DATEBLU="DDEBF7"; GOLD="BF8F00"; WHITE="FFFFFF"; BLACK="000000"
FONT="Times New Roman"; SIZE=12
THIN=Side(style="thin", color="808080"); BORDER=Border(THIN,THIN,THIN,THIN)
def fill(h): return PatternFill("solid", fgColor=h)
FMT={"usage":"#,##0","demand":"#,##0.0","curr":"$#,##0.00","curr_t":"$#,##0",
     "unit3":"$#,##0.000","unit4":"$#,##0.0000","credit":"$#,##0.00;($#,##0.00)"}
CENTER=Alignment("center","center",wrap_text=True); RIGHT=Alignment("right","center"); LEFT=Alignment("left","center")
# build_utility_analysis(spec, out_path): title row (merged), super-header (row 2,
# merged per section, navy), sub-header (row 3, medium navy), data rows (gold
# currency, black usage/demand, banded), TOTALS row (navy, white, currency 0 dp),
# then the Resource/Annual Total/Unit Cost summary. See the .py for the full body.
```

## Self-check before delivering

1. Does each month's Total tie to the bill's stated total? (mapping correct)
2. Two-tier header: each section a single merged super-header over its columns?
3. Sections contiguous, and matching the summary buckets?
4. Fixed service/customer charge under Taxes and Fees, riders under Energy,
   credits in their own negative column?
5. Sections with no members for this tariff dropped, not left empty?
6. Navy `1F3864` super-header, medium navy `2E5496` sub-header and totals, gold
   `BF8F00` currency, light blue `DDEBF7` date column and banding?
7. Thousands comma everywhere; demand 1 dp all rows; currency 2 dp in data, 0 dp
   in totals; usage whole?
8. Summary unit costs computed from unrounded annual totals?

Fix any "no" before delivering.

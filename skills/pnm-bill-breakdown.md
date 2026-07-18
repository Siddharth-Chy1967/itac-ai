---
name: pnm-bill-breakdown
description: Use this skill whenever the user wants a monthly electricity bill breakdown table built from a PNM (or similar utility) bill spreadsheet for an IAC/ITAC report. Trigger on any of these phrases or close variants: "bill breakdown", "build the breakdown", "Table 2-3", "monthly breakdown", "breakdown of the latest month", "PNM breakdown", "build me the bills table", "make the breakdown table". Also trigger whenever the workbook clearly contains multi-month utility bill data (columns like On-Peak kWh, Off-Peak kWh, Fuel Adj, Demand Charge, Palo Verde Credit, State GRT) and the user asks for a per-month breakdown, even without using the exact phrases above. The skill pulls the most recent month from the workbook and builds a formatted breakdown table on a new sheet matching IAC/ITAC report Table 2-3 conventions exactly: Times New Roman 12 pt, single-spaced, grey header row, grey section dividers, yellow total row, italic gold caption above the table.
---

# PNM Monthly Bill Breakdown Table

## Purpose

Build a single-month electricity bill breakdown table for an IAC/ITAC report. The source is a multi-month utility bill spreadsheet (typically 12 to 13 months of PNM data). The output is a new sheet in the same workbook containing a formatted table that matches the report convention exactly, so the user can copy it into Word and have it look identical to existing Table 2-X entries.

## When to use

Trigger whenever the user wants the per-month breakdown built from bill data. Common phrasings: "build the bill breakdown", "make me Table 2-3", "do the breakdown for the latest month", "PNM breakdown table". Also trigger when the active workbook contains the column signatures listed below and the user asks for a single-month summary, even if the phrasing is loose.

## Source data: expected columns

The source sheet has one row per billing month. Identify columns by header name (case-insensitive, tolerate minor variations):

- Date column: "Date" or "Meter Read Date" (values like "Feb-26", "Feb 2026", "2/2026", or a real date)
- Usage: On-Peak kWh, Off-Peak kWh, Total kWh
- Energy charges: On-Peak Charge, Off-Peak Charge, Fuel Adj, Renewable Rider, Transport R53, Cost-Eff Prog, Off-Peak Charge with Adjustors, Total Energy Charge
- Discount: Palo Verde Credit
- Demand: Billable Demand kW, Demand Charge
- Taxes and fees: San Juan ETC, Franchise Fee, State GRT, County GRT, City GRT, Customer Charge
- Total: Total Bill (or Total Taxes & Fees + others if Total Bill is missing)

If a column is missing, list which ones are missing and stop. Do not invent or estimate values.

## Step 1: Pick the latest month

The latest month is the row furthest down in the source sheet (PNM data is entered chronologically). Confirm by parsing the date column: pick the row with the maximum date. If two rows share the same month, use the lower one. Convert the abbreviated form (e.g., "Feb-26") to full month and full year ("February 2026") for the caption.

## Step 2: Create a new sheet

Name the new sheet `Table 2-3 - [MonthShort YYYY]`, e.g., `Table 2-3 - Feb 2026`. If a sheet with that name already exists, append ` (2)`, ` (3)`, etc., rather than overwriting.

## Step 3: Build the table

The table is 5 columns wide: **Line Item | Category | Peak Period | Value | Unit**

Place the caption in row 1 spanning columns A through E (merge and center), then the header row in row 2, then the body. Use the structure below.

### Caption (row 1, merged A1:E1)
```
Table 2-3: Monthly Electrical Bill Breakdown for [Full Month YYYY]
```
- Font: Times New Roman, 12 pt, **italic**, bold off
- Color: gold/orange, hex `#BF8F00`
- Alignment: centered
- Fill: none
- No border

### Header row (row 2)
| Line Item | Category | Peak Period | Value | Unit |

- Font: Times New Roman, 12 pt, bold
- Fill: light grey, hex `#BFBFBF`
- Alignment: centered
- Border: thin black on all four sides of each cell

### Body rows

Each section starts with a section-divider row (the section name centered and merged across A:E), followed by the data rows.

Section divider formatting:
- Font: Times New Roman, 12 pt, bold
- Fill: light grey, hex `#D9D9D9`
- Alignment: centered, merged A:E
- Border: thin black

Data row formatting (all sections except the final TOTAL):
- Font: Times New Roman, 12 pt, regular
- Fill: none
- Column A (Line Item): left-aligned
- Columns B, C, E (Category, Peak Period, Unit): centered
- Column D (Value): right-aligned
- Border: thin black on all four sides

Insert the sections in this order with these rows:

**USAGE** (section divider, then 3 rows)
- On-Peak kWh | Usage | On-Peak | [On-Peak kWh value] | kWh
- Off-Peak kWh | Usage | Off-Peak | [Off-Peak kWh value] | kWh
- **Total kWh | Usage | Combined | [Total kWh value] | kWh** (this row bold)

**ENERGY CHARGES** (section divider, then 8 rows)
- On-Peak Energy Charge | Usage | On-Peak | [On-Peak Charge] | $
- Off-Peak Base Charge | Usage | Off-Peak | [Off-Peak Charge] | $
- Fuel Adjustment | Usage | Off-Peak | [Fuel Adj] | $
- Renewable Rider | Usage | Off-Peak | [Renewable Rider] | $
- Transport R53 | Usage | Off-Peak | [Transport R53] | $
- Cost-Efficiency Program | Usage | Off-Peak | [Cost-Eff Prog] | $
- Off-Peak Charge (w/ Adjustors) | Usage | Off-Peak | [Off-Peak Charge with Adjustors] | $
- **Total Energy Charge | Usage | Combined | [Total Energy Charge] | $** (bold)

**DEMAND** (section divider, then 2 rows)
- Billable Demand kW | Demand | - | [Billable Demand kW] | kW
- Demand Charge | Demand | - | [Demand Charge] | $

**DISCOUNT** (section divider, then 1 row)
- Palo Verde Credit | Discount | - | [Palo Verde Credit, displayed as negative] | $

**TAXES & FEES** (section divider, then 6 rows)
- San Juan ETC | Tax/Fee | - | [San Juan ETC] | $
- Franchise Fee | Tax/Fee | - | [Franchise Fee] | $
- State GRT | Tax/Fee | - | [State GRT] | $
- County GRT | Tax/Fee | - | [County GRT] | $
- City GRT | Tax/Fee | - | [City GRT] | $
- Customer Charge | Tax/Fee | - | [Customer Charge] | $

**TOTAL** (section divider, then 1 row, this row is yellow)
- Total Current Electricity Charges | (blank) | (blank) | [Total Bill] | $

Final TOTAL row formatting:
- Font: Times New Roman, 12 pt, **bold**
- Fill: yellow, hex `#FFE699`
- Column A left-aligned, Column D right-aligned
- Border: thin black

### Peak Period column for non-Usage sections

Use a regular hyphen `-` (NOT an em dash) for Demand, Discount, and Tax/Fee rows. This respects the user's standing rule against em dashes anywhere in IAC report output.

### Number formatting

- kWh values (On-Peak, Off-Peak, Total): `#,##0` (no decimals, thousands separator)
- kW value (Billable Demand): `#,##0.0` (one decimal)
- All dollar values: `$#,##0.00` (two decimals, dollar sign on left)
- Palo Verde Credit: display as negative, format `-$#,##0.00`

### Column widths

- A (Line Item): wide enough for "Off-Peak Charge (w/ Adjustors)" without wrapping, around 32 characters
- B (Category): around 12 characters
- C (Peak Period): around 12 characters
- D (Value): around 12 characters
- E (Unit): around 6 characters

### Row heights

Standard single-spaced row height (around 18 to 20 points). Do not auto-grow rows.

## Step 4: Summary message

After building, return a short summary:
- Month used (e.g., "Feb 2026")
- Total bill amount
- Total kWh
- Sheet name created

Example: "Built `Table 2-3 - Feb 2026`. Latest month Feb 2026, total bill $5,576.15, total usage 39,920 kWh."

## Rules and edge cases

- **Never use em dashes** anywhere in the table (this is a standing user rule for all IAC/ITAC output). Use a hyphen `-` as the not-applicable marker.
- **Verify totals**: after building, check that On-Peak kWh + Off-Peak kWh = Total kWh, and that Total Energy Charge equals the sum of the energy charge line items. If they do not match within rounding, flag the discrepancy in the summary message.
- **Currency rounding**: pull values directly from the source. Do not recalculate or re-round.
- **Source linkage**: by default, values are pulled as static numbers, not as formula references back to the source sheet. If the user wants live formula links, they will say so.
- **Existing sheets**: never overwrite an existing `Table 2-3 - *` sheet; append a `(2)`, `(3)` suffix instead.
- **No template imposition**: if the user has an existing Table 2-3 sheet with a slightly different format, ask before applying this skill's defaults.

## Why this matters

The user assembles IAC reports under tight review. Tables that match the established format exactly let the user paste directly into Word with no manual cleanup. Mismatched fonts, missing borders, em dashes, or wrong colors all force a second pass of formatting work. Get it right the first time.

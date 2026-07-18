---
name: industryselect-fill
description: Use this skill whenever the user wants to fill out the B2B outreach tracker (2024 Industry Select Email List.xlsx) from an IndustrySelect Pro CSV export. Trigger on any of these phrases or close variants: "fill the tracker", "append from the csv", "fill out the list", "add the new leads", "populate the outreach sheet", "fill from IndustrySelect", "new CSV drop", "fill both sheets". Also trigger whenever the user uploads a CSV that looks like an IndustrySelect export (columns like Executive First Name, Executive Title, Executive Direct Email, Square Footage, Max Sales, Physical State) alongside the tracker xlsx. This skill handles all logic: encoding, contact name building, sq ft and revenue filtering, dedup against existing companies, auto-detection of empty student blocks, formula restoration in column C, and filling only B/D/E/F while leaving A/C/G-J untouched.
---

# IndustrySelect Tracker Fill

## Purpose

Fill empty student outreach blocks in the B2B tracker xlsx from a fresh IndustrySelect Pro CSV export. Handles two sheets in one pass: Sheet 1 ("Fall 24- Current") gets filtered leads, Sheet 2 ("No Revenue or Sq Ft filter") gets unfiltered leads. Both sheets are deduped so no company repeats across sheets or against the existing tracker history.

## When to trigger

Trigger whenever the user drops an IndustrySelect CSV (or similar B2B export) and wants to populate empty rows in the outreach tracker. Common phrasings: "fill the tracker", "append from the csv", "new csv drop", "populate both sheets", "add leads from IndustrySelect".

## Files expected

- **Target xlsx**: `2024 Industry Select Email List.xlsx` (or user-specified tracker file)
- **Source CSV**: IndustrySelect Pro export, typically named like `34xx_and_37xx.csv`, `36xx_and_38xx.csv`, etc.
- Read CSV with `encoding='latin-1'` -- IndustrySelect exports are not UTF-8.

## Sheet layout (fixed conventions)

| Sheet name | Filter applied | Notes |
|---|---|---|
| `Fall 24- Current` | Sq Ft AND revenue (see Filters section) | Sheet 1 -- filtered |
| `No Revenue or Sq Ft filter` | None | Sheet 2 -- all fresh leads |

Both sheets have identical block structure. Sheet 2's selection is also deduped against Sheet 1's picks so the two sheets are complementary (no company overlap between them).

## Column mapping (B/D/E/F only)

| Column | Letter | Source CSV field |
|---|---|---|
| Company Name | B | `Company` |
| New/Exists formula | C | Restore formula (see below) -- NEVER leave static text |
| Contact | D | Built from salutation + first + middle + last + suffix |
| Position | E | `Executive Title` |
| Email | F | `Executive Direct Email` |

**Leave untouched:** Column A (Assigned name), Column G (Phone), Columns H/I/J (Sent/Follow Up/Response), all rows above START_ROW, Sheet2 (dashboard), Summer 24, List of Old clients.

## Step-by-step workflow

### Step 1: Inspect the file

```python
from openpyxl import load_workbook
wb = load_workbook(xlsx_path)
print(wb.sheetnames)
for sheet in ['Fall 24- Current', 'No Revenue or Sq Ft filter']:
    ws = wb[sheet]
    print(f"{sheet}: {ws.max_row} rows")
```

Confirm sheet names match. If they differ, print `repr(s)` for each sheet name to catch hidden characters or renames.

### Step 2: Find START_ROW

Scan column A of "Fall 24- Current" downward to find where the new empty student blocks begin. The user will specify this (e.g., "from row 2872"). If not specified, scan for the first block of rows where column A has a student name AND column B is empty.

### Step 3: Auto-detect student blocks

```python
def find_student_blocks(ws, start_row):
    blocks = []
    current_name = None
    block_start = None
    for r in range(start_row, ws.max_row + 1):
        name = ws.cell(row=r, column=1).value
        if name != current_name:
            if current_name is not None:
                blocks.append((current_name, block_start, r - 1))
            current_name = name
            block_start = r if name is not None else None
    if current_name is not None:
        blocks.append((current_name, block_start, ws.max_row))
    return [b for b in blocks if b[0] is not None]
```

Print each detected block with slot count. Confirm David's block is 30 rows if he requested that (other students typically get 10 each).

### Step 4: Build dedup set

Pull every company name already in the tracker from both sheets + old clients list, normalized to lowercase with leading/trailing whitespace stripped (IndustrySelect's old clients list has a leading `\n` on each entry -- `.strip()` handles this).

```python
wb = load_workbook(xlsx_path, data_only=True)
existing = set()
for sheet_name in ['Fall 24- Current', 'No Revenue or Sq Ft filter']:
    ws = wb[sheet_name]
    for row in ws.iter_rows(min_row=4, max_col=2, values_only=True):
        if row[1]:
            existing.add(str(row[1]).strip().lower())
ws = wb['List of Old clients']
for row in ws.iter_rows(min_col=1, max_col=1, values_only=True):
    if row[0]:
        existing.add(str(row[0]).strip().lower())
wb.close()
```

### Step 5: Load CSV and build Contact

```python
df = pd.read_csv(csv_path, encoding='latin-1')

def build_contact(row):
    parts = []
    for f in ['Executive Salutation', 'Executive First Name', 'Executive Middle Name',
              'Executive Last Name', 'Executive Suffix']:
        if pd.notna(row[f]):
            parts.append(str(row[f]).strip())
    return ' '.join(parts)

df['Contact'] = df.apply(build_contact, axis=1)
```

### Step 6: Apply filters

**Sheet 1 (filtered):** Apply sq ft AND revenue filter. Default thresholds:

- `Square Footage > 35000` AND `Max Sales >= 5_000_000` (loose: Max Sales, not Min Sales)

Then dedup: drop any company already in `existing`.

**If fewer fresh execs survive than total slots:** test looser filters in this order until enough candidates exist:

1. `Sq Ft > 25k AND Max Sales >= $5M` (recommended first fallback -- keeps revenue bar)
2. `Sq Ft > 25k AND Max Sales >= $2.5M`
3. `Sq Ft > 35k only (no revenue filter)`

Report which threshold was used. Do not silently leave slots blank without telling the user first.

**Sheet 2 (unfiltered):** No sq ft or revenue filter. Dedup against `existing` AND against the companies selected for Sheet 1 so the two sheets are complementary.

### Step 7: Select execs

Take `.head(n_slots)` from each filtered pool. This preserves company groupings since IS exports are alphabetically sorted. Multiple execs from the same company will appear on consecutive rows -- this is correct behavior (one row per exec).

### Step 8: Restore formula in column C

**Critical:** Column C cells in empty student blocks are often static text ("New", "Exists", or blank), NOT live formulas. Always overwrite column C with the proper formula:

```python
FORMULA = '=IF(COUNTIF(\'List of Old clients\'!A:A, B{r}) + COUNTIF($B$4:B{r}, B{r}) > 1, "Exists", "New")'

# For each target row:
ws.cell(row=row_num, column=3).value = FORMULA.format(r=row_num)
```

This formula checks: (1) is the company in the old clients list, OR (2) does the company name appear earlier in column B of this sheet. If either is true → "Exists", else → "New".

### Step 9: Write and save

Use openpyxl to write surgically -- never use `df.to_excel()` which destroys formulas and formatting.

```python
import shutil
shutil.copy(xlsx_path, output_path)
wb = load_workbook(output_path)

for row_num, (_, ex) in zip(target_rows, selected.iterrows()):
    ws.cell(row=row_num, column=2).value = ex['Company']
    ws.cell(row=row_num, column=3).value = FORMULA.format(r=row_num)
    ws.cell(row=row_num, column=4).value = ex['Contact']
    ws.cell(row=row_num, column=5).value = ex['Executive Title']
    ws.cell(row=row_num, column=6).value = ex['Executive Direct Email']

wb.save(output_path)
```

### Step 10: Verify

After saving, reload with `data_only=True` and check:
- Count of filled rows per sheet
- Count of blank slots remaining (should be 0 unless pool ran dry after filter testing)
- New vs Exists breakdown per sheet
- David's block specifically if he had a non-standard slot count

Report: `Sheet 1: X/Y filled, N New / M Exists | Sheet 2: X/Y filled, N New / M Exists`

## Filters reference

| Parameter | Default | Notes |
|---|---|---|
| Sheet 1 sq ft threshold | 35,000 sq ft | Facility size proxy for ITAC eligibility |
| Sheet 1 revenue threshold | $5,000,000 Max Sales | Loose filter -- Max Sales bucket, not Min |
| Sheet 2 filter | None | All fresh companies, no size/revenue gate |
| Dedup scope | Both tracker sheets + old clients | Normalized: lowercase + strip whitespace |

**"Accommodation is the main goal"** -- facility size matters more than revenue for ITAC assessment eligibility. When forced to loosen one filter to fill slots, loosen revenue before sq ft.

## Known quirks

- **Windows file lock**: if the user is running this locally in VSC/Jupyter, the xlsx must be closed in Excel before openpyxl can read or write it. If `PermissionError [Errno 13]` appears, that's the cause.
- **Sheet name drift**: sheet names get renamed across outreach cycles. Always `print(repr(s))` when a KeyError occurs to catch hidden characters or renames.
- **Static text in column C**: openpyxl's `read_only=True` mode caches formula results as static values. Empty blocks often have stale "Exists"/"New" text in column C from copy-paste. Always overwrite with the live formula -- never trust the cached value.
- **Old clients leading newline**: entries in "List of Old clients" col A often start with `\n`. `.strip()` in the dedup set handles this silently.
- **CSV encoding**: IndustrySelect exports are `latin-1`, not `utf-8`. Use `encoding='latin-1'` or you'll get a UnicodeDecodeError.
- **Multi-exec rows**: IS exports one row per exec, so large companies may occupy 5-10 consecutive rows. This is correct and expected -- the COUNTIF formula will mark the first as "New" and subsequent ones as "Exists" since the company name repeats. That is working as intended.

## Output summary to report to user

After completing, tell the user:
- Which sheets were filled and how many slots per sheet
- Filter thresholds used for Sheet 1 (especially if a fallback threshold was applied)
- New vs Exists count per sheet
- Any slots left blank and why
- Whether column C formula was restored (always yes -- mention it once so the user knows to open in Excel to resolve)
- Pre-existing errors in the file (e.g., #REF! errors in old rows) are not our problem -- note them but don't try to fix them

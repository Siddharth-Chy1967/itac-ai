"""Build the offline industry-code tables ARchie uses to name SIC and NAICS codes.

Outputs (committed, so nothing is downloaded at run time):
  app/data/sic_codes.csv     every 4-digit SIC (1987 manual, OSHA list) with its
                             3-digit industry group, 2-digit major group and division
  app/data/naics_to_sic.csv  every NAICS code from the 1997-2022 vintages, mapped back
                             to the 1987 SIC code(s) through the Census concordances

Sources (download into one folder, pass it as the only argument):
  https://raw.githubusercontent.com/rtlee9/SIC-list/master/data/osha_combined.csv
  https://www.census.gov/naics/concordances/1997_NAICS_to_1987_SIC.xls
  https://www.census.gov/naics/concordances/1997_NAICS_to_2002_NAICS.xls
  https://www.census.gov/naics/concordances/2007_to_2002_NAICS.xls
  https://www.census.gov/naics/concordances/2012_to_2007_NAICS.xls
  https://www.census.gov/naics/concordances/2017_to_2012_NAICS.xlsx
  https://www.census.gov/naics/concordances/2022_to_2017_NAICS.xlsx

Needs xlrd (for the .xls files) in addition to the backend's own dependencies.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "app" / "data"


def _code(value, width: int) -> str:
    """A code cell (number or text) as a zero-padded digit string, or ''."""
    if value in (None, ""):
        return ""
    text = str(int(value)) if isinstance(value, float) else str(value).strip()
    return text.zfill(width) if text.isdigit() else ""


def _naics(value) -> str:
    text = _code(value, 0)
    # The 1997 -> SIC sheet stores codes ending in 0 without it (11111 = 111110).
    return text + "0" if len(text) == 5 else text


def _xls_rows(path: Path, sheet: int = 0):
    import xlrd

    sh = xlrd.open_workbook(path).sheet_by_index(sheet)
    for r in range(sh.nrows):
        yield sh.row_values(r)


def _xlsx_rows(path: Path):
    import openpyxl

    ws = openpyxl.load_workbook(path, read_only=True).worksheets[0]
    yield from ws.iter_rows(values_only=True)


def _pairs(rows, new_col: int, title_col: int, old_col: int):
    """(new_code, new_title, old_code) for every data row of a concordance sheet."""
    for row in rows:
        new, old = _naics(row[new_col]), _naics(row[old_col])
        if len(new) == 6 and len(old) == 6:
            yield new, str(row[title_col] or "").strip(), old


def build(src: Path) -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # --- SIC ------------------------------------------------------------------
    sic_names: dict[str, str] = {}
    with open(src / "osha_combined.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(OUT / "sic_codes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sic4", "sic4_name", "sic3", "sic3_name", "sic2", "sic2_name",
                    "division", "division_name"])
        for r in rows:
            w.writerow([r["SIC4_cd"].zfill(4), r["SIC4_desc"], r["ind_cd"].zfill(3),
                        r["ind_desc"], r["maj_cd"].zfill(2), r["maj_desc"],
                        r["div_cd"], r["div_desc"]])
            sic_names[r["SIC4_cd"].zfill(4)] = r["SIC4_desc"]

    # --- NAICS 1997 -> SIC ----------------------------------------------------
    to_sic: dict[str, list[str]] = defaultdict(list)
    titles: dict[str, str] = {}
    for row in list(_xls_rows(src / "1997_NAICS_to_1987_SIC.xls"))[1:]:
        naics, sic = _naics(row[0]), _code(row[3], 4)
        if len(naics) == 6 and sic in sic_names and sic not in to_sic[naics]:
            to_sic[naics].append(sic)
            titles.setdefault(naics, str(row[2]).strip())

    # --- later vintages, each mapped onto the one before ----------------------
    steps = [
        ("2002", list(_xls_rows(src / "1997_NAICS_to_2002_NAICS.xls", 1))[1:], 2, 3, 0),
        ("2007", list(_xls_rows(src / "2007_to_2002_NAICS.xls"))[3:], 0, 1, 2),
        ("2012", list(_xls_rows(src / "2012_to_2007_NAICS.xls"))[3:], 0, 1, 2),
        ("2017", list(_xlsx_rows(src / "2017_to_2012_NAICS.xlsx"))[3:], 0, 1, 2),
        ("2022", list(_xlsx_rows(src / "2022_to_2017_NAICS.xlsx"))[3:], 0, 1, 2),
    ]
    vintages: dict[str, list[str]] = defaultdict(list)
    for naics in to_sic:
        vintages[naics].append("1997")
    previous = dict(to_sic)
    for vintage, rows, new_col, title_col, old_col in steps:
        current: dict[str, list[str]] = defaultdict(list)
        for new, title, old in _pairs(rows, new_col, title_col, old_col):
            for sic in previous.get(old, []):
                if sic not in current[new]:
                    current[new].append(sic)
            if title:
                titles[new] = title  # the newest vintage's title wins
        for naics, sics in current.items():
            vintages[naics].append(vintage)
            merged = to_sic.setdefault(naics, [])
            merged.extend(s for s in sics if s not in merged)
        previous = current

    with open(OUT / "naics_to_sic.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["naics", "naics_title", "vintages", "sic4"])
        for naics in sorted(to_sic):
            if to_sic[naics]:
                w.writerow([naics, titles.get(naics, ""), " ".join(vintages[naics]),
                            " ".join(to_sic[naics])])
    print(f"{len(sic_names)} SIC codes, {sum(1 for v in to_sic.values() if v)} NAICS codes -> {OUT}")


if __name__ == "__main__":
    build(Path(sys.argv[1]))

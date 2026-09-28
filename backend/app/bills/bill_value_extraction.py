import argparse
from .extraction_models import BillExtraction, FinancialEntry, Measurement, AdditionalField
from .value_parsing import parse_number, parse_date, parse_time, convert_or_note
from .extraction_prompt import BILL_EXTRACTION_INSTRUCTIONS
from .validation_checks import reconcile_current_charges, financial_effect
from .detailed_checks import build_checks, CHECK_COLUMNS
from .reviewed_values import apply_corrections, load_corrections, AUDIT_COLUMNS
from pathlib import Path
from hashlib import sha256
from .local_inference import LocalInferenceClient, LocalInferenceConfig
import pandas as pd
import re
from decimal import Decimal
from datetime import datetime, time
from math import ceil
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.comments import Comment
from openpyxl.formatting.rule import CellIsRule


# Cache folder of the extractions made by the original OpenAI implementation (and the
# model the reviewed corrections are bound to). Read first; new local extractions are
# saved beside it under the local model's label (see local_inference.cache_label).
EXTRACTION_MODEL = "gpt-5.6-terra"
CACHE_VERSION = "bill_extraction_v3"
PROJECT_FOLDER = Path(__file__).resolve().parent
# A long bill can list dozens of financial rows; leave room for the full JSON.
EXTRACTION_MAX_OUTPUT_TOKENS = 8192


def extract_bill_pdf(client, pdf_path):
    """All pages of one grouped bill (images + transcripts) through the local model."""
    return client.parse_pdf(
        BillExtraction,
        BILL_EXTRACTION_INSTRUCTIONS,
        (
            "Extract the bill information, financial "
            "entries, measurements, and additional fields "
            "from all supplied pages."
        ),
        [pdf_path],
        max_output_tokens=EXTRACTION_MAX_OUTPUT_TOKENS,
    )

def parse_amount(amount_text):
    if amount_text is None:
        return None 

    text = amount_text.strip()

    number = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    pattern = rf"(?:-?\$?{number}|\$?\({number}\)|\(\$?{number}\))"

    if re.fullmatch(pattern, text) is None:
        raise ValueError(f"Unrecognized amount: {amount_text!r}")

    cleaned = text.replace("$", "").replace(",", "")

    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = "-" + cleaned[1:-1]

    return Decimal(cleaned)

def format_worksheet(worksheet):
    widths = {
        "utility_company": 44,
        "label": 54,
        "bill_id": 68,
        "grouped_pdf": 80,
        "extraction_notes": 100,
        "section": 30,
        "service_period_text": 50,
        "reconciliation_status": 24,
        "current_charges_difference": 30,
        "reconciliation_note": 100,
        "conversion_notes": 85,
        "message": 85,
        "cache_path": 90,
        "reason": 85,
        "note": 85,
        "field": 35,
        "effect_reason": 65,
        "calculation_sources": 65,
        "original_value": 45,
        "reviewed_value": 65,
        "original_extraction_notes": 100,
    }

    money_columns = {
        "amount", "current_charges", "total_due",
        "line_item_sum", "current_charges_difference",
        "calculated_value", "expected_value", "difference",
    }

    for column in worksheet.iter_cols():
        header = column[0]
        name = header.value

        worksheet.column_dimensions[header.column_letter].width = (
            widths.get(name, 22)
        )

        header.font = Font(bold=True, color="FFFFFF")
        header.fill = PatternFill(fill_type="solid", fgColor="245B78")

        for cell in column:
            cell.alignment = Alignment(wrap_text=True, vertical="top")

            if cell.row > 1 and name in money_columns:
                cell.number_format = "#,##0.00;[Red]-#,##0.00"
            elif cell.row > 1 and name in {"value", "quantity", "rate"}:
                # Use the printed precision, e.g. 5,659.800 or 2.500000.
                # Fixed decimal places also avoid binary-float display artifacts.
                header_names = [c.value for c in worksheet[1]]
                raw_field = name + "_text"
                raw = worksheet.cell(cell.row, header_names.index(raw_field) + 1).value if raw_field in header_names else ""
                fraction = re.search(r"\.(\d+)", str(raw or ""))
                decimals = min(len(fraction.group(1)), 15) if fraction else 0
                number_format = "#,##0" + ("." + "0" * decimals if decimals else "")
                cell.number_format = number_format + ";[Red]-" + number_format
            elif cell.row > 1 and name in {"bill_date_value", "due_date_value", "service_start_value", "service_end_value", "measurement_date_value"}:
                cell.number_format = "yyyy-mm-dd"
            elif cell.row > 1 and name == "measurement_time_value":
                cell.number_format = "hh:mm:ss"

    worksheet.row_dimensions[1].height = 34

    has_notes = "extraction_notes" in [
        cell.value for cell in worksheet[1]
    ]

    for row_number in range(2, worksheet.max_row + 1):
        lines = 1
        for cell in worksheet[row_number]:
            if isinstance(cell.value, str):
                width = worksheet.column_dimensions[cell.column_letter].width
                lines = max(lines, sum(max(1, ceil(len(part) / (width * 0.9)))
                                       for part in cell.value.split("\n")))
        worksheet.row_dimensions[row_number].height = min(240, max(160 if has_notes else 32, lines * 15 + 10))

def highlight_totals(worksheet):
    headers = [cell.value for cell in worksheet[1]]

    # Highlight financial rows using their classification.
    if "category" in headers:
        category_index = headers.index("category")

        for cells in worksheet.iter_rows(min_row=2):
            category = cells[category_index].value

            if category == "total":
                color = "C6E0B4"       # Light green
            elif category == "subtotal":
                color = "DDEBF7"       # Light blue
            else:
                continue

            for cell in cells:
                cell.fill = PatternFill(
                    fill_type="solid",
                    fgColor=color,
                )
                cell.font = Font(bold=True, color="1F2937")

    # Highlight the summary amounts on the Bills sheet.
    for column in worksheet.iter_cols():
        if column[0].value in {"current_charges", "total_due"}:
            for cell in column[1:]:
                cell.fill = PatternFill(
                    fill_type="solid",
                    fgColor="C6E0B4",
                )
                cell.font = Font(bold=True, color="1F2937")

def cache_path_for(project_folder, fingerprint, model):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", model):
        raise ValueError("Invalid model name")
    return project_folder / "extraction_cache" / fingerprint / model / f"{CACHE_VERSION}.json"


def collect_pdfs(inputs, project_folder=PROJECT_FOLDER):
    """Only selected files or immediate bill_*.pdf children, never old runs."""
    paths = []
    for entered in inputs:
        path = Path(entered.strip().strip('"'))
        if not path.is_absolute():
            path = project_folder / path
        path = path.resolve()
        if path.is_dir():
            found = sorted(path.glob("bill_*.pdf"))
            if not found:
                raise ValueError(f"No bill_*.pdf files in: {path}")
            print(f"Selected folder: {path} ({len(found)} PDFs)")
            paths.extend(found)
        elif path.is_file() and path.suffix.lower() == ".pdf":
            paths.append(path)
        else:
            raise ValueError(f"Not an existing grouped PDF or bill folder: {path}")
    return list(dict.fromkeys(paths))


def load_bill(pdf_path, fingerprint, project_folder, model, run_inference=True,
              get_client=None, local_label=None):
    """Reuse the `model` cache, else the local model's cache, else extract locally.

    The cache model is the folder the extraction was saved under; callers read it back
    from the returned path (`cache_path.parent.name`). The local client is only created
    on a cache miss.
    """
    candidates = [cache_path_for(project_folder, fingerprint, model)]
    local_label = local_label or LocalInferenceConfig.from_env(project_folder).cache_label
    local_path = cache_path_for(project_folder, fingerprint, local_label)
    if local_path not in candidates:
        candidates.append(local_path)
    for result_path in candidates:
        if result_path.is_file():
            # Invalid caches are reported, never silently replaced by a new extraction.
            return BillExtraction.model_validate_json(result_path.read_text(encoding="utf-8")), result_path, "cached"
    if not run_inference:
        raise FileNotFoundError(
            f"No {model} or {local_label} {CACHE_VERSION} cache; cache-only run, not extracted."
        )
    client = get_client() if get_client is not None else LocalInferenceClient(project_folder=project_folder)
    if client.cache_label != local_label:
        raise ValueError(f"Local client saves to {client.cache_label}, expected {local_label}.")
    client.ensure_ready()
    bill = extract_bill_pdf(client, pdf_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with local_path.open("x", encoding="utf-8") as saved:
        saved.write(bill.model_dump_json(indent=2))
    return bill, local_path, "extracted"


def bill_tables(bill, pdf_path, fingerprint, model, cache_path, original_bill=None, audit=None):
    """Build rows without changing the model or cached JSON."""
    notes = []
    original_bill = original_bill or bill
    audit = audit or []
    bill_row = bill.model_dump(exclude={"financial_entries", "measurements", "additional_fields"})
    bill_row.update(bill_id=fingerprint, grouped_pdf=str(pdf_path.resolve()),
                    extraction_model=model, cache_version=CACHE_VERSION, cache_path=str(cache_path))
    bill_row["utility_types"] = ", ".join(bill.utility_types)
    bill_row["extraction_notes"] = "\n".join(bill.extraction_notes)
    bill_row.update(
        correction_count=len(audit), review_status="source_corrected" if audit else "no_source_corrections_applied",
        original_current_charges_text=original_bill.current_charges_text,
        original_total_due_text=original_bill.total_due_text,
        original_extraction_notes="\n".join(original_bill.extraction_notes),
    )
    for field in ("current_charges", "total_due"):
        bill_row[field] = convert_or_note(parse_amount, bill_row[field + "_text"], field, notes)
    for field in ("bill_date", "due_date", "service_start", "service_end"):
        bill_row[field + "_value"] = convert_or_note(parse_date, bill_row[field], field, notes)
    bill_row.update(reconcile_current_charges(bill, parse_amount))
    rows = {"Bills": [bill_row], "Financial Entries": [], "Measurements": [], "Additional Fields": []}
    for sheet, entries in (
        ("Financial Entries", bill.financial_entries),
        ("Measurements", bill.measurements),
        ("Additional Fields", bill.additional_fields),
    ):
        for index, entry in enumerate(entries, start=1):
            row = entry.model_dump()
            row.update(bill_id=fingerprint, entry_number=index)
            row_notes = []
            if sheet == "Financial Entries":
                row["financial_effect"], row["effect_reason"] = financial_effect(entry)
                row["original_amount_text"] = original_bill.financial_entries[index - 1].amount_text
                row["source_corrected"] = row["original_amount_text"] != row["amount_text"]
                row["amount"] = convert_or_note(parse_amount, row["amount_text"], "amount", row_notes)
                for field in ("quantity", "rate"):
                    unit = row[field + "_unit"]
                    row[field] = convert_or_note(lambda text: parse_number(text, unit),
                                                 row[field + "_text"], field, row_notes)
            elif sheet == "Measurements":
                row["value"] = convert_or_note(lambda text: parse_number(text, row["unit"]),
                                               row["value_text"], "value", row_notes)
                row["numeric_unit"] = (
                    "%" if row["value_text"] and row["value_text"].strip().endswith("%")
                    and row["value"] is not None else row["unit"]
                )
                if row["value_text"] and row["value_text"].strip().endswith("°") and row["value"] is not None:
                    row["numeric_unit"] = "°"
                    row_notes.append("Only a degree sign is printed; no Fahrenheit/Celsius scale was inferred.")
                for field, parser in (("measurement_date", parse_date), ("measurement_time", parse_time)):
                    row[field + "_value"] = convert_or_note(parser, row[field], field, row_notes)
            row["conversion_notes"] = "\n".join(row_notes)
            rows[sheet].append(row)
            notes.extend(f"{sheet} entry {index}: {note}" for note in row_notes)
    bill_row["conversion_notes"] = "\n".join(notes)
    rows["Corrections"] = audit
    rows["Validation Checks"] = build_checks(bill, parse_amount, fingerprint, pdf_path)
    bill_row["additional_checks_review"] = sum(r["status"] == "review" for r in rows["Validation Checks"])
    bill_row["additional_checks_not_checked"] = sum(r["status"] == "not_checked" for r in rows["Validation Checks"])
    return rows


def make_tables(rows, run_rows):
    """Explicit columns keep empty sheets usable."""
    bill_fields = [field for field in BillExtraction.model_fields
                   if field not in {"financial_entries", "measurements", "additional_fields"}]
    columns = {
        "Bills": ["bill_id"] + bill_fields + [
            "current_charges", "total_due", "bill_date_value", "due_date_value",
            "service_start_value", "service_end_value", "reconciliation_status",
            "line_item_sum", "current_charges_difference", "reconciliation_note",
            "conversion_notes", "grouped_pdf", "extraction_model", "cache_version", "cache_path",
            "review_status", "correction_count", "original_current_charges_text", "original_total_due_text",
            "original_extraction_notes", "additional_checks_review", "additional_checks_not_checked",
        ],
        "Financial Entries": ["bill_id", "entry_number"] + list(FinancialEntry.model_fields)
                             + ["amount", "quantity", "rate", "conversion_notes", "financial_effect", "effect_reason", "original_amount_text", "source_corrected"],
        "Measurements": ["bill_id", "entry_number", "label", "value_text", "value", "unit", "numeric_unit"]
                        + [field for field in Measurement.model_fields if field not in {"label", "value_text", "unit"}]
                        + ["measurement_date_value", "measurement_time_value", "conversion_notes"],
        "Additional Fields": ["bill_id", "entry_number"] + list(AdditionalField.model_fields) + ["conversion_notes"],
        "Run Summary": ["grouped_pdf", "bill_id", "status", "reconciliation_status", "message", "cache_path", "correction_count", "additional_checks_review", "additional_checks_not_checked"],
        "Corrections": AUDIT_COLUMNS,
        "Validation Checks": CHECK_COLUMNS,
    }
    return {name: pd.DataFrame(run_rows if name == "Run Summary" else rows.get(name, []), columns=fields)
            for name, fields in columns.items()}


def process_pdfs(pdf_paths, project_folder=PROJECT_FOLDER, model=EXTRACTION_MODEL, cache_only=False, reviewed=False):
    """Resumable batch: one failure does not discard successful bills.

    Missing caches are extracted locally unless `cache_only`. Reviewed export is always
    cache-only: its corrections are bound to the exact bytes of an existing cache.
    """
    run_inference = not (cache_only or reviewed)
    corrections = load_corrections(project_folder / "reviewed_corrections.json") if reviewed else None
    rows = {name: [] for name in ("Bills", "Financial Entries", "Measurements", "Additional Fields", "Corrections", "Validation Checks")}
    run_rows = []
    seen = set()
    inference_config = LocalInferenceConfig.from_env(project_folder)
    client = None

    def get_client():
        nonlocal client
        if client is None:
            client = LocalInferenceClient(inference_config)
            print(f"Local inference: {client.describe()}", flush=True)
        return client
    for index, pdf_path in enumerate(pdf_paths, start=1):
        record = {"grouped_pdf": str(pdf_path), "bill_id": None, "status": "error",
                  "reconciliation_status": None, "message": "", "cache_path": None}
        try:
            fingerprint = sha256(pdf_path.read_bytes()).hexdigest()
            record.update(bill_id=fingerprint, cache_path=str(cache_path_for(project_folder, fingerprint, model)))
            if fingerprint in seen:
                record.update(status="duplicate", message="Identical PDF bytes already exported in this run.")
            else:
                bill, cache_path, status = load_bill(pdf_path, fingerprint, project_folder, model,
                                                     run_inference, get_client, inference_config.cache_label)
                cache_model = cache_path.parent.name
                record["cache_path"] = str(cache_path)
                original_bill, audit = bill, []
                if reviewed:
                    bill, audit = apply_corrections(bill, fingerprint, cache_path, pdf_path, cache_model, CACHE_VERSION, corrections)
                tables = bill_tables(bill, pdf_path, fingerprint, cache_model, cache_path, original_bill, audit)
                for sheet in rows:
                    rows[sheet].extend(tables[sheet])
                seen.add(fingerprint)
                summary = tables["Bills"][0]
                record.update(status=status, reconciliation_status=summary["reconciliation_status"],
                              correction_count=summary["correction_count"],
                              additional_checks_review=summary["additional_checks_review"],
                              additional_checks_not_checked=summary["additional_checks_not_checked"],
                              message=summary["reconciliation_note"] + (
                                  " Conversion warnings: see Bills." if summary["conversion_notes"] else ""
                              ))
        except FileNotFoundError as error:
            record.update(status="missing_cache" if record["bill_id"] else "error", message=str(error))
        except Exception as error:
            record["message"] = f"{type(error).__name__}: {error}"
        run_rows.append(record)
        print(f"[{index}/{len(pdf_paths)}] {pdf_path.name}: {record['status']}; "
              f"check={record['reconciliation_status'] or '-'}", flush=True)
    return make_tables(rows, run_rows)


def add_reconciliation_formulas(writer, tables):
    """Live arithmetic in Excel; notes/Run Summary are export-time snapshots."""
    bills = writer.sheets["Bills"]
    finance = writer.sheets["Financial Entries"]
    bh = {cell.value: cell.column_letter for cell in bills[1]}
    fh = {cell.value: cell.column_letter for cell in finance[1]}
    last = max(finance.max_row, 2)
    def financial_range(field):
        col = fh[field]
        return f"'Financial Entries'!{col}2:{col}{last}"
    amounts, identifiers, effects = (financial_range(field) for field in ("amount", "bill_id", "financial_effect"))
    for number, record in enumerate(tables["Bills"].to_dict("records"), start=2):
        if record["reconciliation_status"] == "not_checked":
            continue  # Never manufacture a sum from unsupported extraction.
        ref = f"{bh['bill_id']}{number}"
        total_cell = f"{bh['line_item_sum']}{number}"
        current_cell = f"{bh['current_charges']}{number}"
        difference_cell = f"{bh['current_charges_difference']}{number}"
        bills[total_cell] = f'=SUMIFS({amounts},{identifiers},{ref},{effects},"current_charge")'
        bills[difference_cell] = f"=ROUND({total_cell}-{current_cell},2)"
        bills[f"{bh['reconciliation_status']}{number}"] = f'=IF({difference_cell}=0,"matched","review")'
    bills[f"{bh['reconciliation_note']}1"].comment = Comment(
        "Export-time diagnostic. Numeric checks recalculate in Excel. Re-run the "
        "exporter after changing raw data; missing-value/category checks and Run "
        "Summary are snapshots, not an independent accuracy guarantee.", "User"
    )


def add_detailed_check_formulas(writer, tables):
    """Use explicit row references: calculations remain inspectable in Excel."""
    ws = writer.sheets["Validation Checks"]
    headers = {c.value: c.column_letter for c in ws[1]}
    bills = writer.sheets["Bills"]
    finance = writer.sheets["Financial Entries"]
    bh = {c.value: c.column_letter for c in bills[1]}
    fh = {c.value: c.column_letter for c in finance[1]}
    bill_rows = {r["bill_id"]: i for i, r in enumerate(tables["Bills"].to_dict("records"), 2)}
    finance_rows = {(r["bill_id"], r["entry_number"]): i for i, r in enumerate(tables["Financial Entries"].to_dict("records"), 2)}
    def reference(bill_id, token):
        kind, field = token.split(":")
        if kind == "entry":
            return f"'Financial Entries'!{fh['amount']}{finance_rows[(bill_id, int(field))]}"
        if kind == "bill" and field in {"current_charges", "total_due"}:
            return f"'Bills'!{bh[field]}{bill_rows[bill_id]}"
        raise ValueError("Unexpected check formula source")
    for number, record in enumerate(tables["Validation Checks"].to_dict("records"), 2):
        if record["status"] == "not_checked":
            continue
        sources = [reference(record["bill_id"], s) for s in record["calculation_sources"].split(";")]
        calculated = f"{headers['calculated_value']}{number}"
        expected = f"{headers['expected_value']}{number}"
        difference = f"{headers['difference']}{number}"
        ws[calculated] = "=SUM(" + ",".join(sources) + ")"
        ws[expected] = "=" + reference(record["bill_id"], record["expected_source"])
        ws[difference] = f"=ROUND({calculated}-{expected},2)"
        ws[f"{headers['status']}{number}"] = f'=IF({difference}=0,"matched","review")'
    ws[f"{headers['note']}1"].comment = Comment(
        "Checks cover only supported relationships. Not checked means insufficient context. "
        "Formulas recalculate but row membership is an export-time snapshot; re-export after editing source data.", "User")


def write_workbook(tables, destination):
    """Write typed numbers/dates. Literal source text must never become formulas."""
    with pd.ExcelWriter(destination, engine="openpyxl") as writer:
        for sheet_name, dataframe in tables.items():
            numeric = dataframe.map(lambda value: float(value) if isinstance(value, Decimal) else value)
            numeric.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            # pandas converts time-only objects to strings; restore native Excel times.
            for column_number, field in enumerate(numeric.columns, start=1):
                if field == "measurement_time_value":
                    for row_number, value in enumerate(numeric[field], start=2):
                        if isinstance(value, time):
                            worksheet.cell(row_number, column_number, value)
            format_worksheet(worksheet)
            highlight_totals(worksheet)
            worksheet.freeze_panes = "C2" if sheet_name != "Bills" else "B2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for cells in worksheet.iter_rows():
                for cell in cells:
                    if isinstance(cell.value, str):
                        cell.data_type = "s"
            headers = {cell.value: cell.column_letter for cell in worksheet[1]}
            for field in ("reconciliation_status", "status"):
                if field in headers and worksheet.max_row > 1:
                    column = headers[field]
                    for status in ("review", "error", "missing_cache", "not_checked"):
                        worksheet.conditional_formatting.add(
                            f"{column}2:{column}{worksheet.max_row}",
                            CellIsRule(operator="equal", formula=[f'"{status}"'],
                                       fill=PatternFill("solid", fgColor="FCE4D6")),
                        )
        add_reconciliation_formulas(writer, tables)
        add_detailed_check_formulas(writer, tables)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export grouped utility bills. Missing caches are extracted with the local model.")
    parser.add_argument("inputs", nargs="*", help="Grouped PDF paths or flat folders containing bill_*.pdf")
    parser.add_argument("--cache-only", action="store_true", help="Do not run local inference; export only bills with a saved extraction")
    parser.add_argument("--list-only", action="store_true", help="List files/cache availability without inference or export")
    parser.add_argument("--model", default=EXTRACTION_MODEL, help="Cache folder to reuse first (default: the original extractions)")
    parser.add_argument("--reviewed", action="store_true", help="Apply source-verified correction log; cache-only, original caches unchanged")
    args = parser.parse_args(argv)
    if not args.inputs:
        args.inputs = [input("Enter one grouped bill PDF path or bills folder: ").strip().strip('"')]
        if not args.inputs[0]:
            parser.error("No path entered.")
    try:
        cache_path_for(PROJECT_FOLDER, "check", args.model)
        pdf_paths = collect_pdfs(args.inputs)
    except ValueError as error:
        parser.error(str(error))
    inference = LocalInferenceConfig.from_env(PROJECT_FOLDER)
    print(f"Model: {args.model} (then {inference.cache_label}); cache: {CACHE_VERSION}; files selected: {len(pdf_paths)}")
    if args.list_only:
        cached = 0
        for path in pdf_paths:
            fingerprint = sha256(path.read_bytes()).hexdigest()
            exists = any(cache_path_for(PROJECT_FOLDER, fingerprint, name).is_file()
                         for name in (args.model, inference.cache_label))
            cached += exists
            print(f"{'cache present' if exists else 'no cache'}: {path}")
        print(f"Cache files present: {cached}; missing: {len(pdf_paths) - cached}. No inference or exports.")
        return 0
    if args.cache_only or args.reviewed:
        print("CACHE-ONLY: missing extractions are reported, not extracted.")
    else:
        print(f"Missing extractions run locally: {inference.describe()}.")
    tables = process_pdfs(pdf_paths, model=args.model, cache_only=args.cache_only, reviewed=args.reviewed)
    folder = PROJECT_FOLDER / "output" / "excel" / "batch"
    folder.mkdir(parents=True, exist_ok=True)
    prefix = "bill_values_reviewed" if args.reviewed else "bill_values"
    destination = folder / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S_%f}.xlsx"
    with destination.open("xb") as output:
        write_workbook(tables, output)
    print(f"Excel workbook saved: {destination}")
    for name, table in tables.items():
        print(f"{name}: {len(table)} data rows")
    failures = tables["Run Summary"]["status"].isin(["error", "missing_cache"]).sum()
    reviews = tables["Bills"]["reconciliation_status"].isin(["review", "not_checked"]).sum()
    print(f"Not exported: {failures}; bills needing arithmetic review: {reviews}.")
    print("Additional checks:", tables["Validation Checks"]["status"].value_counts().to_dict())
    print("Read Run Summary and extraction_notes. A matched total does not prove extraction accuracy.")
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

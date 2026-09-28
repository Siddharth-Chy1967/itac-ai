"""Optional, source-reviewed corrections. Never overwrite an extraction cache."""
import json
from hashlib import sha256
from .extraction_models import BillExtraction


AUDIT_COLUMNS = [
    "bill_id", "entry_number", "field", "original_value", "reviewed_value",
    "reason", "bill_page_number", "grouped_pdf", "reviewed_by", "reviewed_on",
    "extraction_model", "cache_version", "original_cache_sha256",
]


def load_corrections(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format_version") != 1:
        raise ValueError("Unsupported reviewed-corrections format.")
    return data


def apply_corrections(bill, fingerprint, cache_path, pdf_path, model, version, document):
    """Apply only to the exact PDF/model/cache reviewed; fail on stale evidence."""
    review = document["bills"].get(fingerprint)
    if review is None:
        return bill.model_copy(deep=True), []
    if model != document["model"] or version != document["cache_version"]:
        raise ValueError("Reviewed corrections do not match this model/cache version.")
    raw_hash = sha256(cache_path.read_bytes()).hexdigest()
    if raw_hash != review["original_cache_sha256"]:
        raise ValueError("Stale reviewed correction: cached extraction has changed. Review again.")
    if sha256(pdf_path.read_bytes()).hexdigest() != fingerprint:
        raise ValueError("Reviewed correction PDF fingerprint mismatch.")
    data = bill.model_dump()
    audit = []
    for change in review["changes"]:
        collection, field = change["collection"], change["field"]
        number = change.get("entry_number")
        if collection == "financial_entries" and field == "amount_text":
            target = data[collection][number - 1]
            if (target["label"] != change["label"] or
                    target["bill_page_number"] != change["bill_page_number"]):
                raise ValueError("Reviewed correction row identity mismatch.")
        elif collection == "bill" and field in {"current_charges_text", "total_due_text"}:
            target = data
        elif collection == "extraction_notes" and field == "text":
            target, field = data["extraction_notes"], number - 1
        else:
            raise ValueError("Unsupported correction target.")
        if target[field] != change["old"]:
            raise ValueError("Reviewed correction old value does not match the extraction.")
        target[field] = change["new"]
        audit.append({
            "bill_id": fingerprint, "entry_number": number,
            "field": f"{collection}.{change['field']}",
            "original_value": change["old"], "reviewed_value": change["new"],
            "reason": change["reason"], "bill_page_number": change["bill_page_number"],
            "grouped_pdf": str(pdf_path.resolve()), "reviewed_by": document["reviewed_by"],
            "reviewed_on": document["reviewed_on"], "extraction_model": model,
            "cache_version": version, "original_cache_sha256": raw_hash,
        })
    return BillExtraction.model_validate(data), audit

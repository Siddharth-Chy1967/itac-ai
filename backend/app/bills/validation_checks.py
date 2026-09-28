"""Offline arithmetic checks. These never modify extracted bill values."""
from decimal import Decimal
import re


def normalized_label(text):
    return re.sub(r"[^a-z0-9]+", " ", (text or "").casefold()).strip()


# Labels that name a balance carried in from an earlier bill, matched against
# normalized_label output.
BALANCE_LABELS = re.compile(
    r"\b(previous|prior|past due|unpaid) balance\b|\bbalance (forward|before charges)\b"
)


def summary_target(entry):
    """Only explicit bill-summary labels; a generic Total may be a section total."""
    label = normalized_label(getattr(entry, "label", None))
    if label in {"current charges", "total current charges", "this month s charges"} or label.startswith("current charges due "):
        return "current_charges"
    if label == "total charges" and normalized_label(getattr(entry, "section", None)) == "invoice summary":
        return "current_charges"
    if label in {"total due", "total amount due", "amount due", "please pay", "please pay by"}:
        return "total_due"
    return None


def name_rows_from_sections(bill):
    """The bill with each unlabelled charge row named by its section, where that is safe.

    The extraction prompt files an unlabelled row's context under `section`. On
    some SRP bills qwen2.5vl did that for every row -- label empty, the row's own
    name ("On Peak Energy Charge") as its section -- so no charge had a column and
    the whole month's charges vanished. A section that belongs to one row alone is
    that row's name. A section several rows share is a real heading ("Energy
    Charges" over a bold subtotal) and is left alone, as the prompt intends.

    Naming a row does not make it a charge: a named "Previous Balance" or "This
    Month's Charges" is still excluded by `financial_effect`, by name.
    """
    sections = [normalized_label(e.section) for e in bill.financial_entries if e.section]
    entries = []
    for entry in bill.financial_entries:
        section = normalized_label(entry.section)
        if not normalized_label(entry.label) and section and sections.count(section) == 1:
            entry = entry.model_copy(update={"label": entry.section.strip(), "section": None})
        entries.append(entry)
    return bill.model_copy(update={"financial_entries": entries})


def financial_effect(entry):
    """Explain each row's accounting purpose without changing its printed data."""
    section = normalized_label(getattr(entry, "section", None))
    label = normalized_label(getattr(entry, "label", None))
    if "comparison" in section and "daily averages" in section:
        return "comparison", "Daily-average comparison, not an additional billed charge."
    if "annual interest on deposit" in label:
        return "account_activity", "Deposit-interest credit affects the balance before current charges."
    if summary_target(entry) or entry.category in {"subtotal", "total"}:
        return "summary", "Summary/subtotal retained for checking, not added again."
    if entry.category in {"balance", "payment"}:
        return "account_activity", "Balance/payment, not a current service charge."
    # The extractor sometimes files a carried-forward balance as a "charge". Its
    # label is unambiguous, and counting it as a charge adds a prior month's whole
    # bill into this one, so the wording overrides the category here.
    if BALANCE_LABELS.search(label):
        return "account_activity", "Carried-forward balance, not a current service charge."
    if entry.category in {"charge", "tax", "discount"}:
        return "current_charge", "Individual charge, tax or signed discount."
    return "unknown", "Accounting purpose unresolved; current-charge check is blocked."


def reconcile_current_charges(bill, parse_amount):
    """Compare individual charges, taxes and signed discounts with current charges.

    Exclude payments, balances and repeated totals/subtotals to avoid counting
    them again. A match is an arithmetic check, not proof of complete extraction.
    """
    result = {
        "reconciliation_status": "not_checked",
        "line_item_sum": None,
        "current_charges_difference": None,
        "reconciliation_note": "",
    }
    problems = []
    amounts = []
    for index, entry in enumerate(bill.financial_entries, start=1):
        effect, _ = financial_effect(entry)
        if effect == "unknown":
            problems.append(f"Entry {index} has an unresolved category: other.")
        if effect != "current_charge":
            continue
        try:
            amount = parse_amount(entry.amount_text)
        except (ValueError, TypeError):
            amount = None
        if amount is None:
            problems.append(f"Entry {index} has a missing or unrecognized amount.")
        else:
            amounts.append(amount)

    try:
        expected = parse_amount(bill.current_charges_text)
    except (ValueError, TypeError):
        expected = None

    if expected is None:
        problems.append("Printed current charges are missing or unrecognized.")
    if not amounts:
        problems.append("No usable individual charges, taxes or discounts found.")
    if problems:
        result["reconciliation_note"] = " ".join(problems)
        return result

    total = sum(amounts, Decimal("0"))
    difference = total - expected
    result["line_item_sum"] = total
    result["current_charges_difference"] = difference
    result["reconciliation_status"] = "matched" if difference == 0 else "review"
    if difference == 0:
        result["reconciliation_note"] = (
            "Individual charges, taxes and signed discounts match printed current "
            "charges. Arithmetic only; this does not verify completeness or accuracy."
        )
    else:
        result["reconciliation_note"] = (
            f"Line-item sum {total:,.2f}; printed current charges {expected:,.2f}; "
            f"difference {difference:+,.2f} (sum minus printed charges). "
            "Review source values, missing pages/rows, duplicate rows and categories. "
            "No extracted values were changed."
        )
    return result

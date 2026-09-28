"""Additional offline checks with explicit references for Excel formulas.

Unknown/missing context is reported, never resolved by changing a source amount.
"""
from collections import defaultdict
from decimal import Decimal
from .validation_checks import financial_effect, normalized_label, summary_target


CHECK_COLUMNS = [
    "bill_id", "check_number", "check_type", "status", "calculated_value",
    "expected_value", "difference", "note", "calculation_sources", "expected_source",
    "grouped_pdf",
]


def build_checks(bill, parse_amount, fingerprint, pdf_path):
    entries = bill.financial_entries
    records = []

    def value(source):
        if source.startswith("entry:"):
            text = entries[int(source.split(":")[1]) - 1].amount_text
        else:
            text = getattr(bill, source.split(":")[1] + "_text")
        try:
            return parse_amount(text)
        except (ValueError, TypeError, AttributeError):
            return None

    def add(kind, sources, expected, note, blocked=False):
        values = [value(source) for source in sources]
        expected_value = value(expected) if expected else None
        usable = bool(values) and not blocked and expected_value is not None and all(v is not None for v in values)
        calculated = sum(values, Decimal("0")) if usable else None
        difference = calculated - expected_value if usable else None
        records.append({
            "bill_id": fingerprint, "check_number": len(records) + 1,
            "check_type": kind, "status": ("matched" if difference == 0 else "review") if usable else "not_checked",
            "calculated_value": calculated, "expected_value": expected_value,
            "difference": difference, "note": note,
            "calculation_sources": ";".join(sources), "expected_source": expected,
            "grouped_pdf": str(pdf_path.resolve()),
        })

    def ref(number):
        return f"entry:{number}"

    current = [i for i, e in enumerate(entries, 1) if financial_effect(e)[0] == "current_charge"]
    unresolved = any(financial_effect(e)[0] == "unknown" for e in entries)
    add("current_charges", [ref(i) for i in current], "bill:current_charges",
        "Individual current charges only; account activity, comparisons and summaries excluded. Arithmetic is not an accuracy guarantee.", unresolved)

    # Compare only explicit repeated bill summaries, not every row named Total.
    account_totals = [i for i, e in enumerate(entries, 1) if normalized_label(e.label) == "account total"]
    for i, entry in enumerate(entries, 1):
        target = summary_target(entry)
        if not target and i in account_totals and len(account_totals) == 1 and "gas" in bill.utility_types:
            target = "current_charges"
        if target:
            add("repeated_" + target, [ref(i)], "bill:" + target,
                f"Entry {i} ({entry.label}), supplied page {entry.bill_page_number}, compared with bill-level {target}.")

    # An explicit post-payment balance can be used; Previous Balance alone cannot.
    balances = [i for i, e in enumerate(entries, 1)
                if normalized_label(e.label) in {"balance before charges", "balance forward"}]
    if not balances and "gas" in bill.utility_types:
        balances = [i for i, e in enumerate(entries, 1) if normalized_label(e.label) == "prior balance"]
        late_fees = [i for i, e in enumerate(entries, 1) if normalized_label(e.label) == "late fee"]
    else:
        late_fees = []
    balance_values = [value(ref(i)) for i in balances]
    compatible = bool(balances) and None not in balance_values and len(set(balance_values)) == 1 and len(late_fees) <= 1
    sources = ([ref(balances[0])] if balances else []) + ["bill:current_charges"] + [ref(i) for i in late_fees]
    add("balance_to_total_due", sources, "bill:total_due",
        "Explicit balance before charges (or gas Prior Balance), plus current charges and separately printed gas Late Fee. Missing/conflicting balance is not assumed zero.",
        not compatible)

    # Check the pre-charge balance separately where it is explicit on an SRP/water summary.
    previous = [i for i, e in enumerate(entries, 1) if normalized_label(e.label) == "previous balance"]
    if len(previous) == 1 and balances and not late_fees:
        balance_entry = entries[balances[0] - 1]
        activity = [i for i, e in enumerate(entries, 1)
                    if financial_effect(e)[0] == "account_activity"
                    and (e.category == "payment" or "annual interest on deposit" in normalized_label(e.label))
                    and e.bill_page_number == balance_entry.bill_page_number
                    and e.section == balance_entry.section]
        add("previous_balance_activity", [ref(previous[0])] + [ref(i) for i in activity], ref(balances[0]),
            "Previous balance plus signed payment/deposit-interest rows in the same summary; compared with printed balance before charges.", not activity)

    # Keep separate PDF pages/sections separate. Multiple known meters or periods
    # in one section are ambiguous: do not combine them without a stronger link.
    contexts = defaultdict(list)
    for i, e in enumerate(entries, 1):
        contexts[(e.bill_page_number, normalized_label(e.section))].append(i)
    covered, subtotal_refs = set(), []
    ambiguous_sections = False
    for (page, section), indices in contexts.items():
        items = [i for i in indices if i in current]
        totals = [i for i in indices if entries[i - 1].category == "subtotal"
                  and not summary_target(entries[i - 1])
                  and normalized_label(entries[i - 1].label) != "account total"]
        if not totals:
            continue
        distinct = [set(getattr(entries[i - 1], field) for i in indices if getattr(entries[i - 1], field) is not None)
                    for field in ("meter_number", "service_period_text")]
        ambiguous = not section or len(totals) != 1 or not items or any(len(values) > 1 for values in distinct)
        zero_only = bool(section) and len(totals) == 1 and not items and value(ref(totals[0])) == 0 and all(len(v) <= 1 for v in distinct)
        note = ("Printed zero subtotal, but no individual rows to independently check. "
                "The printed zero can participate in the section-totals check." if zero_only else
                "Requires one subtotal and unambiguous meter/period context.")
        add("section_subtotal", [ref(i) for i in items], ref(totals[0]) if len(totals) == 1 else None,
            f"Supplied page {page}; section {section or '(unlabeled)'}. {note}", ambiguous)
        if zero_only:
            subtotal_refs.append(ref(totals[0]))
            continue
        if ambiguous:
            ambiguous_sections = True
        else:
            covered.update(items)
            subtotal_refs.append(ref(totals[0]))
    if subtotal_refs:
        add("section_totals_to_current", subtotal_refs + [ref(i) for i in current if i not in covered], "bill:current_charges",
            "One subtotal per non-overlapping section, plus individual current-charge rows outside those sections. Never add both a subtotal and its constituent rows.",
            ambiguous_sections or unresolved)
    return records

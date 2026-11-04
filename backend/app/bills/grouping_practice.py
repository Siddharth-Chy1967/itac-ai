import re
from datetime import datetime

def has_complete_page_sequence(printed_numbers, printed_total_pages):
    if printed_total_pages is None or printed_total_pages < 1:
        return False

    expected_numbers = list(range(1, printed_total_pages + 1))
    return printed_numbers == expected_numbers


# What a printed web address wraps around the utility's actual name.
ADDRESS_PREFIXES = ("http://", "https://", "www.")
ADDRESS_SUFFIXES = (".com", ".net", ".org", ".gov", ".coop", ".us")


def strip_web_address(value):
    """Reduce a printed web address to its bare label.

    "https://www.srpnet.com/bill" becomes "srpnet". A company name that merely
    contains a dot is left alone, so only a recognisable address changes shape.
    """
    address = value

    for prefix in ADDRESS_PREFIXES:
        if address.startswith(prefix):
            address = address[len(prefix):]

    address = address.split("/", 1)[0].rstrip(".")

    if " " in address:
        return value

    for suffix in ADDRESS_SUFFIXES:
        if address.endswith(suffix):
            return address[:-len(suffix)]

    return value


def normalize_company_name(value):
    if value is None:
        return None

    cleaned_name = value.strip().casefold()

    # One bill prints its utility as a logo on page 1 and as the footer web
    # address on page 2, so the two pages read as "SRP" and "srpnet.com". Both
    # have to reduce to one name or a whole bill is sent to review for a
    # disagreement that is only typography.
    cleaned_name = strip_web_address(cleaned_name)

    aliases = {
        "srp": "salt river project",
        "srpnet": "salt river project",
        "salt river project (srp)": "salt river project",
    }

    return aliases.get(cleaned_name, cleaned_name)

def normalize_bill_date(value):
    if value is None:
        return None

    cleaned_date = value.strip()

    if not cleaned_date:
        return None

    # These bills use month/day/year.
    # Also accept dates already written as year-month-day.
    for date_format in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            parsed_date = datetime.strptime(cleaned_date, date_format)
            return parsed_date.date().isoformat()
        except ValueError:
            continue

    return cleaned_date

def has_field_conflict(page_records, field_name):
    known_values = set()

    for page_info in page_records:
        value = page_info.get(field_name)

        if field_name == "utility_company":
            value = normalize_company_name(value)

        elif field_name == "bill_date":
            value = normalize_bill_date(value)

        if value is not None:
            known_values.add(value)

    return len(known_values) > 1

CONFLICT_FIELDS = [
    "printed_total_pages",
    "utility_company",
    "invoice_number",
    "bill_date",
    "billing_period_key",
]


def normalize_account_number(value):
    """An account number with its printed punctuation and spacing removed.

    Bills print the same account as "019735300 - 0276883 - 4" and
    "0197353000276883 4", so a character-level comparison finds a difference
    where the page shows none.
    """
    if value is None:
        return None

    cleaned = re.sub(r"[^0-9a-z]+", "", str(value).casefold())

    return cleaned or None


def has_same_account(page_records):
    """True when every page that prints an account number prints the same one."""
    accounts = {
        normalize_account_number(page.get("account_number"))
        for page in page_records
    }
    accounts.discard(None)

    return len(accounts) == 1


def without_misfiled_invoice_numbers(page_records):
    """The pages with an "invoice number" that is really the account number removed.

    The prompt tells the model to copy the invoice number "not the account number",
    and qwen2.5vl files the account number there anyway on the opening page of PNM
    bills. The page beside it carries the real invoice ("A342026493"), so the two
    disagree and grouping refuses a bill whose account number matches throughout.

    A value equal to that page's own account number is not an invoice number, so it
    is dropped rather than compared. Distinct *real* invoice numbers still mean
    separate bills; this only removes a value the page never presented as one.
    """
    cleaned = []

    for page in page_records:
        account = normalize_account_number(page.get("account_number"))
        invoice = normalize_account_number(page.get("invoice_number"))

        if account is not None and invoice == account:
            page = {**page, "invoice_number": None}

        cleaned.append(page)

    return cleaned


def field_conflicts(page_records, *, ignore=()):
    """The bill-level fields whose pages disagree, with the values they printed.

    `ignore` drops a field from the check. Stage 2 uses it for `bill_date`, which
    the model reads off the page rather than the page printing it structurally.
    """
    conflicts = {}

    for field_name in CONFLICT_FIELDS:
        if field_name in ignore:
            continue

        if has_field_conflict(page_records, field_name):
            conflicts[field_name] = [
                page.get(field_name) for page in page_records
            ]

    return conflicts


def blocking_conflicts(page_records, completeness):
    """Conflicts that should stop a group from being accepted.

    A disagreement over `bill_date` alone is not one of them when the pages carry
    complete printed numbering and one account number. Printed numbering ("1" and
    "2 of 2") is on the page; `bill_date` is whichever date the model picked out of
    the several a bill prints, so when the structure says one bill, a date that
    differs means the model misread a date -- not that two bills were merged.

    Where the numbering is absent or contradictory, `bill_date` still blocks: that
    is the case the check was written for, and there is no stronger evidence to
    fall back on.
    """
    page_records = without_misfiled_invoice_numbers(page_records)
    trustworthy_structure = completeness == "complete" and has_same_account(page_records)

    return field_conflicts(
        page_records, ignore=("bill_date",) if trustworthy_structure else ()
    )


def describe_conflicts(conflicts):
    """A reason naming each field and what its pages printed."""
    parts = [
        f"{field} {' vs '.join('null' if v is None else str(v) for v in values)}"
        for field, values in conflicts.items()
    ]

    return "Conflicting bill-level information: " + "; ".join(parts) + "."


def has_group_conflict(page_records):
    """Kept for callers that only need a yes/no over every field."""
    return bool(field_conflicts(page_records))


def get_page_completeness(printed_numbers, printed_total_pages):
    if not printed_numbers:
        return "unknown"

    known_numbers = []

    for number in printed_numbers:
        if number is not None:
            if number < 1:
                return "invalid"

            known_numbers.append(number)

    # Reject duplicate or out-of-order printed page numbers.
    if known_numbers != sorted(set(known_numbers)):
        return "invalid"

    if printed_total_pages is None:
        return "unknown"

    if (
        printed_total_pages < 1
        or len(printed_numbers) > printed_total_pages
    ):
        return "invalid"

    for number in known_numbers:
        if number > printed_total_pages:
            return "invalid"

    if None in printed_numbers:
        return "unknown"

    if has_complete_page_sequence(
        printed_numbers,
        printed_total_pages,
    ):
        return "complete"

    return "partial"


def matching_bill_fields(first_page, second_page):
    fields_to_compare = [
        "invoice_number",
        "bill_date",
        "billing_period_key",
    ]

    matches = []

    for field_name in fields_to_compare:
        first_value = (first_page.get(field_name) or "").strip()
        second_value = (second_page.get(field_name) or "").strip()

        if field_name == "bill_date":
            first_value = normalize_bill_date(first_value)
            second_value = normalize_bill_date(second_value)

        if first_value and first_value == second_value:
            matches.append(field_name)

    return matches

def find_suspicious_boundaries(pages):
    suspicious_pairs = []

    for index in range(1, len(pages)):
        previous = pages[index - 1]
        current = pages[index]

        # Only compare neighboring source PDF pages.
        if current["page_number"] != previous["page_number"] + 1:
            continue

        # We are checking whether a new-bill label is questionable.
        if current["page_role"] != "bill_start":
            continue

        if previous["page_role"] not in ("bill_start", "continuation"):
            continue

        # Conflicting identifiers do not support joining the pages.
        if has_group_conflict([previous, current]):
            continue

        previous_company = normalize_company_name(
            previous.get("utility_company")
        )
        current_company = normalize_company_name(
            current.get("utility_company")
        )

        if not previous_company or previous_company != current_company:
            continue

        # Shared identifiers make this boundary worth reviewing.
        # They do not prove that the pages belong together.
        if matching_bill_fields(previous, current):
            suspicious_pairs.append([
                previous["page_number"],
                current["page_number"],
            ])

    return suspicious_pairs

def group_bill_pages(pages):
    expected_pages = []

    for page in pages:
        number = page["page_number"]

        if type(number) is not int or number < 1:
            raise ValueError("PDF page numbers must be positive integers.")

        expected_pages.append(number)

    if expected_pages != sorted(set(expected_pages)):
        raise ValueError("PDF page numbers must be unique and in order.")

    candidates = []
    current = []

    bill_groups = []
    group_details = []
    non_bill_pages = []
    review_pages = []
    review_reasons = []

    # Stage 1: collect possible bill groups using page roles.
    for page in pages:
        role = page["page_role"]

        if role == "bill_start":
            if current:
                candidates.append(current)

            current = [page]

        elif role == "continuation":
            current.append(page)

        elif role in ("non_bill", "uncertain"):
            if current:
                candidates.append(current)

            current = []

            if role == "non_bill":
                non_bill_pages.append(page["page_number"])
            else:
                review_pages.append(page["page_number"])
                review_reasons.append({
                    "pages": [page["page_number"]],
                    "reason": "Uncertain page role.",
                })

        else:
            raise ValueError(f"Unexpected page role: {role}")

    if current:
        candidates.append(current)

    # Stage 2: check each candidate before accepting it.
    for group in candidates:
        numbers = []
        printed_numbers = []
        printed_total = None

        for page in group:
            numbers.append(page["page_number"])
            printed_numbers.append(page.get("printed_page_number"))

            if page.get("printed_total_pages") is not None:
                printed_total = page["printed_total_pages"]

        completeness = get_page_completeness(
            printed_numbers,
            printed_total,
        )

        reason = None

        conflicts = blocking_conflicts(group, completeness)

        if conflicts:
            reason = describe_conflicts(conflicts)

        elif completeness == "invalid":
            reason = "Invalid or contradictory printed numbering."

        elif (
            len(group) == 1
            and group[0]["page_role"] == "continuation"
            and (
                completeness == "unknown"
                or printed_numbers[0] == 1
            )
        ):
            reason = "Unnumbered or contradictory orphan continuation."

        # Check evidence linking neighboring pages.
        for index in range(1, len(group)):
            if reason is not None:
                break

            previous = group[index - 1]
            page = group[index]

            matches = matching_bill_fields(previous, page)

            company = normalize_company_name(
                previous.get("utility_company")
            )

            same_company = (
                bool(company)
                and company == normalize_company_name(
                    page.get("utility_company")
                )
            )

            previous_printed = previous.get("printed_page_number")

            # A page total that one of the two pages simply does not print must not
            # void an otherwise consecutive sequence: PNM prints "Page 1" bare and
            # "Page 2 of 2" on the page after it. Totals that are both present and
            # disagree are still a mismatch.
            totals = {
                previous.get("printed_total_pages"),
                page.get("printed_total_pages"),
            }
            totals.discard(None)

            numbered_link = (
                previous_printed is not None
                and page.get("printed_page_number") == previous_printed + 1
                and len(totals) <= 1
            )

            first_account = normalize_account_number(previous.get("account_number"))
            second_account = normalize_account_number(page.get("account_number"))

            accounts_differ = (
                bool(first_account)
                and bool(second_account)
                and first_account != second_account
            )

            if accounts_differ and "invoice_number" not in matches:
                reason = "Different accounts without a shared invoice number."

            elif not (
                "invoice_number" in matches
                or (same_company and (matches or numbered_link))
            ):
                reason = (
                    "Insufficient linking evidence between PDF pages "
                    f"{previous['page_number']} and {page['page_number']}."
                )

        if reason is not None:
            review_pages.extend(numbers)

            review_reasons.append({
                "pages": numbers,
                "reason": reason,
            })

        else:
            # Partial or unknown completeness does not prevent acceptance.
            # Missing pages and their contents are not reconstructed.
            bill_groups.append(numbers)

            group_details.append({
                "pages": numbers,
                "completeness": completeness,
                "printed_page_numbers": printed_numbers,
                "printed_total_pages": printed_total,
            })

    # Stage 3: ensure every supplied page appears exactly once.
    accounted_pages = non_bill_pages + review_pages

    for group in bill_groups:
        accounted_pages.extend(group)

    if sorted(accounted_pages) != expected_pages:
        raise RuntimeError(
            "Some pages are missing, duplicated, or unexpected."
        )

    return {
        "bill_groups": bill_groups,
        "non_bill_pages": non_bill_pages,
        "review_pages": review_pages,
        "group_details": group_details,
        "review_reasons": review_reasons,
    }



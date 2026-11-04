"""Keep only what the page prints. Deterministic; no model.

The model decides what each figure means; this module decides whether to believe
it exists. Every number and date the model returns is looked up in the printed
text of the page it names (see `page_text`, which has already removed chart axis
labels). A figure that is not there is dropped and flagged -- never kept, never
corrected, never replaced by a nearby value. The cell it would have filled stays
blank, which is visible; a wrong number is not.

Then one arithmetic check: the charge lines that survived must add up to the
printed current-charges total within the rounding tolerance, or the bill is
flagged. Nothing is adjusted to make it balance.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Iterable, Sequence

from .bill_schema import BillFigures, Charge, PrintedText, Reading
from .page_text import PageText

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?|\.\d+")
_DATE_TOKENS = re.compile(
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
    r"|\b\d{4}-\d{2}-\d{2}\b"
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b",
    re.IGNORECASE,
)
_DATE_FORMATS = ("%m/%d/%Y", "%m/%d/%y", "%m-%d-%Y", "%m-%d-%y", "%Y-%m-%d",
                 "%b %d, %Y", "%b %d %Y", "%B %d, %Y", "%B %d %Y")


# --- parsing ----------------------------------------------------------------


# A number with the marks that make it negative: "-$57.13", "($57.13)", "57.13-",
# "57.13 CR". Group 1 and 3 hold the marks, group 2 the digits.
_SIGNED = re.compile(
    r"(-\$?|\(\s*\$?|\$\s*-|\$\s*\()?"     # leading: -12, -$12, (12, ($12, $-12, $(12
    r"(\d[\d,]*(?:\.\d+)?|\.\d+)"
    r"(-(?![\d\w])|\s*\)|\s*CR(?![A-Za-z]))?",           # trailing: 12-, 12), 12 CR
)
_CREDIT_WORDS = re.compile(r"credit|discount|rebate|refund|adjustment", re.IGNORECASE)


def signed_numbers_in(text: str) -> set[Decimal]:
    """Every number printed in `text`, negative where the page prints it negative."""
    found = set()
    for before, digits, after in _SIGNED.findall(text or ""):
        try:
            value = Decimal(digits.replace(",", ""))
        except InvalidOperation:
            continue
        found.add(-value if (before or after) and value else value)
    return found


def numbers_in(text: str) -> list[Decimal]:
    """Every number printed in `text`, unsigned, commas removed."""
    found = []
    for token in _NUMBER.findall(text or ""):
        try:
            found.append(Decimal(token.replace(",", "")))
        except InvalidOperation:
            continue
    return found


def single_number(text: str | None) -> Decimal | None:
    """The one number in a model value, unsigned; None if it holds none or several."""
    found = numbers_in(text or "")
    return found[0] if len(found) == 1 else None


def is_negative(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith("-") or stripped.startswith("(") or stripped.endswith("-") \
        or stripped.startswith("$-") or stripped.startswith("$(")


def parse_any_date(text: str | None) -> date | None:
    if not text:
        return None
    cleaned = re.sub(r"\s+", " ", text.strip().replace("Sept", "Sep"))
    for pattern in _DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, pattern).date()
        except ValueError:
            continue
    return None


def dates_in(text: str) -> set[date]:
    return {parsed for token in _DATE_TOKENS.findall(text or "")
            if (parsed := parse_any_date(token)) is not None}


# --- page lookup ------------------------------------------------------------


# Words too common on a bill to tie an amount to its label.
_GENERIC_WORDS = {"and", "the", "for", "with", "charge", "charges", "amount", "total",
                  "current", "your", "this", "per", "kwh", "fee", "fees"}


def label_words(label: str | None) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", (label or "").casefold())
            if w not in _GENERIC_WORDS}


def _words_meet(label: set[str], line: set[str]) -> bool:
    """A label word printed on the line; "Prog." and "Program" count as one word."""
    return any(a == b or (min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a)))
               for a in label for b in line)


@dataclass(frozen=True)
class _Line:
    words: frozenset[str]
    signed: frozenset[Decimal]
    credit: bool                        # the line names a credit or discount


@dataclass(frozen=True)
class _Page:
    number: int
    numbers: frozenset[Decimal]         # unsigned
    signed: frozenset[Decimal]          # as printed, sign included
    lines: tuple[_Line, ...]
    dates: frozenset[date]
    text: str = ""

    def prints_date(self, day: date) -> bool:
        return day in self.dates or ocr_prints_date(self.text, day)


def ocr_prints_date(text: str, day: date) -> bool:
    """The date is in `text` with its separators misread by OCR.

    Scans turn "3/26/2021" into "3r2612021" or "4/26/2021" into "4726/2021": the
    digits survive, the slashes do not. Requiring the exact month, day and year
    digits with one character between each keeps this as strict as the digits.
    """
    pattern = (rf"(?<!\d)0?{day.month}\S0?{day.day}\S"
               rf"(?:{day.year}|{day.year % 100:02d})(?!\d)")
    return re.search(pattern, text or "") is not None


def _index(pages: Sequence[PageText]) -> list[_Page]:
    indexed = []
    for p in pages:
        text = p.text
        lines = tuple(_Line(frozenset(re.findall(r"[a-z]{3,}", line.casefold())),
                            frozenset(signed_numbers_in(line)),
                            bool(_CREDIT_WORDS.search(line)))
                      for line in text.splitlines())
        indexed.append(_Page(p.number, frozenset(numbers_in(text)),
                             frozenset(signed_numbers_in(text)), lines,
                             frozenset(dates_in(text)), text))
    return indexed


def _find(pages: list[_Page], claimed: int, test) -> int | None:
    """The page that prints it: the one the model named first, then any other."""
    ordered = sorted(pages, key=lambda page: page.number != claimed)
    for page in ordered:
        if test(page):
            return page.number
    return None


# --- result -----------------------------------------------------------------


@dataclass
class VerifiedBill:
    """The model's reading of a bill, cut down to what the pages print."""

    figures: BillFigures
    flags: list[str] = field(default_factory=list)
    dropped: int = 0
    charges_sum: Decimal | None = None
    total: Decimal | None = None
    status: str = "not_checked"  # matched | review | not_checked

    @property
    def statement_date(self) -> date | None:
        value = self.figures.statement_date
        return parse_any_date(value.text) if value else None


def verify(figures: BillFigures, pages: Sequence[PageText], *,
           tolerance: Decimal = Decimal("0.02")) -> VerifiedBill:
    index = _index(pages)
    flags: list[str] = []
    dropped = 0

    def number_on_page(what: str, text: str, claimed: int, printed=None) -> int | None:
        """The page printing `text`; `printed(page, value)` overrides the test."""
        nonlocal dropped
        value = single_number(text)
        if value is None:
            flags.append(f"Dropped {what}: {text!r} is not a single printed number.")
            dropped += 1
            return None
        test = printed or (lambda p, v: v in p.numbers)
        page = _find(index, claimed, lambda p: test(p, value))
        if page is None:
            flags.append(f"Dropped {what}: {text!r} is not printed on the bill"
                         + (" beside its label with that sign." if printed else "."))
            dropped += 1
        elif page != claimed:
            flags.append(f"{what}: {text!r} found on page {page}, not page {claimed}.")
        return page

    readings: list[Reading] = []
    for reading in figures.readings:
        if "$" in reading.value:
            # Money is a charge, never a reading; a quantity column must not hold it.
            flags.append(f"Dropped reading {reading.label!r}: {reading.value!r} is money.")
            dropped += 1
            continue
        page = number_on_page(f"reading {reading.label!r}", reading.value, reading.page)
        if page is not None:
            readings.append(reading.model_copy(update={"page": page}))

    charges: list[Charge] = []
    for charge in figures.charges:
        amount = signed_amount(charge)

        words = label_words(charge.label)

        def printed_with_sign(page, _value, amount=amount, charge=charge, words=words):
            # The amount as the page prints it, sign and all, on the same printed line
            # as a word of its label -- a "$0.00" printed beside one row does not
            # belong to another. A credit may be printed without a minus sign, but
            # then only on a line that says it is a credit.
            for line in page.lines:
                if words and not _words_meet(words, line.words):
                    continue
                if amount in line.signed or (
                        charge.category == "credit" and amount is not None and amount < 0
                        and line.credit and -amount in line.signed):
                    return True
            return False

        page = number_on_page(f"charge {charge.label!r}", charge.amount, charge.page,
                              printed=printed_with_sign)
        if page is not None:
            charges.append(charge.model_copy(update={"page": page}))

    def printed_amount(name: str, value: PrintedText | None) -> PrintedText | None:
        if value is None:
            return None
        page = number_on_page(name, value.text, value.page)
        return None if page is None else value.model_copy(update={"page": page})

    def printed_date(name: str, value: PrintedText | None) -> PrintedText | None:
        nonlocal dropped
        if value is None:
            return None
        parsed = parse_any_date(value.text)
        if parsed is None:
            flags.append(f"Dropped {name}: {value.text!r} is not a readable date.")
            dropped += 1
            return None
        page = _find(index, value.page, lambda p: p.prints_date(parsed))
        if page is None:
            flags.append(f"Dropped {name}: {value.text!r} is not printed on the bill.")
            dropped += 1
            return None
        return value.model_copy(update={"page": page})

    verified = figures.model_copy(update={
        "readings": readings,
        "charges": charges,
        "total_current_charges": printed_amount("current charges total",
                                                figures.total_current_charges),
        "amount_due": printed_amount("amount due", figures.amount_due),
        "statement_date": printed_date("statement date", figures.statement_date),
        "due_date": printed_date("due date", figures.due_date),
        "service_start": printed_date("service start", figures.service_start),
        "service_end": printed_date("service end", figures.service_end),
    })

    result = VerifiedBill(figures=verified, flags=flags, dropped=dropped)
    _check_total(result, tolerance)
    return result


def signed_amount(charge: Charge) -> Decimal | None:
    """A charge's printed amount, negative for a credit however it was printed."""
    value = single_number(charge.amount)
    if value is None:
        return None
    if charge.category == "credit" or is_negative(charge.amount):
        return -value
    return value


def _check_total(result: VerifiedBill, tolerance: Decimal) -> None:
    total = result.figures.total_current_charges
    if total is None:
        result.flags.append("No printed current-charges total to check the line items against.")
        return
    result.total = single_number(total.text)
    amounts = [signed_amount(c) for c in result.figures.charges]
    result.charges_sum = sum((a for a in amounts if a is not None), Decimal("0"))
    difference = result.charges_sum - result.total
    if abs(difference) <= tolerance:
        result.status = "matched"
    else:
        result.status = "review"
        result.flags.append(
            f"Line items total {result.charges_sum:,.2f} against the printed "
            f"{result.total:,.2f} (difference {difference:+,.2f})."
        )


def all_numbers(figures: BillFigures) -> Iterable[str]:
    """Every figure text in a bill, for tests and audits."""
    yield from (r.value for r in figures.readings)
    yield from (c.amount for c in figures.charges)
    for value in (figures.total_current_charges, figures.amount_due):
        if value is not None:
            yield value.text

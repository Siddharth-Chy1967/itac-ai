"""Conservative conversions. Printed text remains the source of truth."""
from datetime import datetime
from decimal import Decimal
import re


def parse_number(text, unit=None):
    """Parse US-style quantities/rates; 88.05% stays 88.05, not 0.8805.

    Only a trailing, explicitly supplied unit may be removed. Unknown formats
    raise ValueError so callers can leave the numeric cell blank and flag it.
    """
    if text is None:
        return None
    cleaned = text.strip()
    if cleaned.endswith("%"):
        if unit not in (None, "%", "percent", "percentage"):
            raise ValueError("Percent sign conflicts with the recorded unit")
        cleaned = cleaned[:-1].strip()
    elif cleaned.endswith("°"):
        if unit not in (None, "°"):
            raise ValueError("Degree sign conflicts with the recorded unit")
        cleaned = cleaned[:-1].strip()
    elif unit and cleaned.endswith(" " + unit):
        cleaned = cleaned[:-(len(unit) + 1)].strip()
    number = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+"
    if not re.fullmatch(rf"(?:[+-]?(?:{number})|\((?:{number})\))", cleaned):
        raise ValueError(f"Unrecognized number: {text!r}")
    cleaned = cleaned.replace(",", "")
    if cleaned.startswith("("):
        cleaned = "-" + cleaned[1:-1]
    result = Decimal(cleaned)
    if len(result.as_tuple().digits) > 15:
        raise ValueError("More than 15 digits; retain text to avoid Excel precision loss")
    return result


def parse_date(text):
    """US month/day/year or explicit ISO/English date; never invent a day."""
    if text is None:
        return None
    for pattern in ("%m/%d/%Y", "%Y-%m-%d", "%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(text.strip(), pattern).date()
        except ValueError:
            pass
    raise ValueError(f"Unsupported full date: {text!r}")


def parse_time(text):
    if text is None:
        return None
    for pattern in ("%H:%M", "%H:%M:%S", "%I:%M %p"):
        try:
            return datetime.strptime(text.strip(), pattern).time()
        except ValueError:
            pass
    raise ValueError(f"Unsupported time: {text!r}")


def convert_or_note(parser, text, field, notes):
    """Missing is allowed; unsupported is visible in conversion_notes."""
    try:
        return parser(text)
    except (ValueError, TypeError, AttributeError) as error:
        notes.append(f"{field}: {error}")
        return None

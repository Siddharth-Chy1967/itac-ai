"""On-site data collection: what to measure for a given AR type, and how (Feature 3).

A student assigned an AR knows the recommendation but not what to bring back from the
site visit. For compressed air leaks that is leak diameter, line pressure and CFM; for
an LED retrofit it is location, fixture type, wattage and operating hours; for bare
equipment insulation it is pipe diameter, length, surface and ambient temperature, wind
speed and material. Nothing in the corpus states those lists directly -- but every past
write-up of the same AR *records the values it used*, and often says where each value
came from, so both can be counted out of the write-ups themselves rather than recalled
from general knowledge.

The pipeline is:

  FIND      semantic search over AR-section chunks for the topic, deduped to distinct
            (report_id, ar_number) write-ups, then a TITLE GATE: a write-up counts only
            when its AR title is the same recommendation as the topic (see
            `title_matches`). Chunk similarity alone let an off-topic AR in whenever the
            real one was rare -- "install high efficiency transformers" came back as a
            dozen motor write-ups.
  ISOLATE   reassemble each write-up's OWN text (`writeup_text`). The extractor tags the
            executive-summary line of an AR, and everything after it up to the next
            marker (plant background, utility bills, lighting inventory), with that AR's
            number; the last AR of a report also runs on into the appendices. Both are
            cut away, or a utility table's "Billing Period" becomes a thing to measure
            for an HVAC set point.
  EXTRACT   pull candidate parameter labels out of the write-up, from the three places
            reports actually record inputs: two-column property tables ("Pipe Diameter
            | 2\""), column headers of a schedule table ("Location | Fixture | Qty |
            Watts"), and the nomenclature block under an equation ("D = leak diameter,
            inches (estimated from observations)").
  SOURCE    for each parameter, the provenance the write-up gives it: the clause on its
            own nomenclature line, the caption or referencing sentence of the table it
            sits in ("Values ... were collected during the site visit and are listed in
            Table 3-6.2"), or a sentence naming it. Separately, instruments named in a
            measurement context ("The oxygen percentage was found using a flu gas
            meter"), each kept with that verbatim sentence as evidence.
  COUNT     canonicalize and count the DISTINCT write-ups recording each parameter,
            source and instrument.

Every label and every evidence sentence is a literal string lifted from a report; the
counts are the evidence for them. A parameter, source or instrument no past write-up
recorded cannot appear in the output. The only vocabulary this module supplies is the
fixed set of source/instrument category names that literal phrases are sorted into.
Because of that the rendering is deterministic and the LLM is bypassed entirely.

Two things are deliberately filtered out of the parameter list, since neither is
collected on site:
  - RESULTS the report computed (savings, payback, implementation cost, CO2, revenue).
  - CONSTANTS and standing assumptions, recognisable because the definition carries its
    own value ("specific heat ratio of air, 1.4").

`_Store` / `_Embedder` mirror `waterfall.py` so this unit-tests against fakes.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Protocol

from .waterfall import _stem, clean_ar_title, title_words

# --- what is NOT an on-site measurement -------------------------------------

# Figures the report DERIVES. The student does not measure a payback period.
_RESULT = re.compile(
    r"saving|payback|\bimplement\w*|total\b|carbon\s*(?:dioxide|emission)|\bco2\b|"
    r"reduction|rebate|\broi\b|annual\s+cost|\bnet\s|conversion\s+(?:constant|rate|factor)|"
    r"\bconstant\b|(?:capital|labor|labour|material|installation|energy|utility)\s+cost|"
    r"electric\w*\s+(?:usage|demand|cost)|energy\s+(?:usage|use)|"
    r"cost\s+of\b|\bcosts?\s*$|proposed\s|current\s+annual|"
    r"revenue|profit|uncertainty|\bwork\s+of\b|heat\s+flux|\bcost\s+(?:on|w/|with)\b|"
    r"\bconversion\b|time[-\s]weighted|\bprojected\b|\bnew\s+(?:power|energy)|emission|"
    # Prices and cost differences come from vendors when the report is written, not from
    # the site visit.
    r"\bcosts?\b|\bprices?\b|\bbundled\b|\bcharges?\b|"
    # Index variables of a summation, not inputs ("the number of the year under
    # consideration", "motor to be considered").
    r"under\s+consideration|to\s+be\s+considered\s*$|"
    # Utility-bill rows. Every report carries a billing table whose headings have
    # nothing to do with the AR, and consumption is read off a bill, not measured.
    r"\btaxe?s?\b|\bfees?\b|surcharge|\blabou?r\b|usage\s*$|charge\s*$|\bbilled\b|"
    r"^ar\s*(?:number|no\.?|#)",
    re.IGNORECASE,
)
# A part or model number lifted out of an equipment schedule ("R2A336GKR2"): one long
# unspaced token mixing letters and digits. Never a parameter name.
_PART_NUMBER = re.compile(r"^(?=.*\d)(?=.*[A-Za-z])[A-Za-z0-9][A-Za-z0-9\-/.]{5,}$")
# Table furniture and section scaffolding, never a parameter.
_NOISE = re.compile(
    r"^(?:table|figure|note|source|parameter|value|item|no\.?|#|where|description|"
    r"summary|recommendation|background|total|eq\b|equation)\b",
    re.IGNORECASE,
)
# A calendar label ("May-22", "Jan 2024", "FY2023"). These are the column headings of
# the 12-month utility table every report carries, never a thing to measure.
_DATE_LABEL = re.compile(
    r"^(?:fy)?\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
    r"[\s\-/,'‘’ʼ]*\d*\s*$"
    r"|^(?:fy)?\s*(?:19|20)\d{2}\s*$",
    re.IGNORECASE,
)
# The right-hand side of an equation printed as a nomenclature entry: "DSi x Hi x UFi",
# "CED - PED", "$80 x 12 tons". A multiplication sign between tokens, or two short
# symbols joined by a minus, is algebra rather than the name of a quantity.
_EXPRESSION = re.compile(
    r"(?:^|\s)[x×](?:\s|$)|^[A-Z][A-Za-z0-9]{0,4}\s*[-–]\s*[A-Z][A-Za-z0-9]{0,4}$"
    # A result symbol followed by its unit ("ESn kWh/yr", "DSn kW/yr"): a computed column.
    r"|^[A-Z]{1,3}[a-z0-9]?\s+(?:[kK][wW][hH]?|MMBtu|MMBTU|[tT]herms?|BTU|Btu)\b"
)
# Column headings too generic to be an instruction. "Type" or "Area" tells a student
# nothing, and the calendar/billing/rate headings come from the utility-analysis tables
# that every report carries regardless of the AR. Matched against the whole canonical
# name, so the specific compounds that matter ("type of bulb", "floor area", "annual
# operating hour") are unaffected.
_GENERIC = frozenset(
    """
    type kind area zone month year date day week hour time period season resource
    item example description name category unit value amount number quantity
    usage demand capital rate price energy power load equipment system building
    kwh kw mmbtu therm btu ton gallon sqft
    other summer winter spring fall autumn east west north south direction
    water propane gas electricity electric meter factor apparent machine device count
    model monday tuesday wednesday thursday friday saturday sunday weekday weekend
    premium standard fraction size compressor motor watt fixture multiplier
    consumption parameter room tank piping wage shoulder shift
    """.split()
) | {
    "on peak", "off peak", "annual usage", "month year", "month day", "peak demand",
    "natural gas", "charge name", "line item", "rate plan", "electric utility",
    "billing period", "meter number", "meter plan", "fixed charge", "shift location",
    "current condition", "proposed condition", "chiller unit", "existing unit",
    "water heater", "current plan", "summer peak", "winter peak", "end of year",
    "start of year", "equipment description",
}
# A trailing "(ft)" / "[m/s]" / "(°F)" unit annotation on a label.
_BRACKET_UNIT = re.compile(r"[\[(]\s*([^\])]{1,20}?)\s*[\])]\s*$")
# Any remaining parenthetical, which is commentary ("(estimated from observations)").
_PARENTHETICAL = re.compile(r"\s*[\[(][^\])]*[\])]")
# The nomenclature line under an equation: "D  =  leak diameter, inches".
_NOMEN = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]{0,3})\s*=\s*(.{4,160})$")
# A unit is short. Anything longer in brackets is prose, not a unit.
_MAX_BRACKET_UNIT = 12
_DIMENSIONLESS = {"no units", "unitless", "dimensionless", "no unit"}


def _is_label(text: str) -> bool:
    """True when `text` could be a parameter name (not a number, heading, or sentence)."""
    return bool(
        3 <= len(text) <= 70
        and re.search(r"[A-Za-z]{3}", text)
        and not text[0].isdigit()
        and not _NOISE.match(text)
        and not _DATE_LABEL.match(text)
        and not _PART_NUMBER.match(text)
    )


def _is_value(text: str) -> bool:
    """True when `text` reads as a recorded observation rather than more prose.

    Numeric covers most of it, but plenty of what an assessor records on site is
    categorical -- "Pipe Material | Copper", "Energy Source Type | Electricity" -- and
    dropping those loses real field work. A short bare phrase counts too; anything
    longer is a sentence, i.e. the row is prose rather than a property table.
    """
    if len(text) > 40:
        return False
    return bool(re.search(r"\d", text)) or len(text.split()) <= 3


def parse_label(raw: str) -> tuple[str, str]:
    """Split a raw label into (name, units): "Pipe Length [ft]" -> ("Pipe Length", "ft").

    Units are annotated three ways in this corpus -- bracketed, parenthesised, or after
    a comma -- and the same parameter appears in all three across reports, so they are
    separated here to keep "pipe length (ft)" and "pipe length [ft]" one parameter
    rather than two. Commentary in brackets ("(estimated from observations)") is
    dropped: it is longer than any unit, which is how it is told apart.
    """
    text = raw.strip()
    units = ""
    m = _BRACKET_UNIT.search(text)
    if m and len(m.group(1)) <= _MAX_BRACKET_UNIT:
        units = m.group(1).strip()
        text = text[: m.start()].strip()
    text = _PARENTHETICAL.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip(" .:*-")

    if not units and "," in text:
        name, _, tail = text.rpartition(",")
        tail = tail.strip()
        # A trailing comma clause is a unit only when it is short and phrase-shaped
        # ("inches", "cubic feet per minute") rather than another clause of prose.
        if name.strip() and len(tail) <= 28 and 0 < len(tail.split()) <= 4:
            text, units = name.strip(), tail
    if units.lower() in _DIMENSIONLESS:
        units = ""
    return text.strip(" .:*-"), units


def canonical(name: str) -> str:
    """Match key for a parameter name, folding case and a trailing plural.

    "line pressure at point of leak" and "...leaks" are the same measurement written by
    two assessors; without this they rank as two half-supported parameters instead of
    one well-supported one. Only the final word is singularized, and only when long
    enough that the "s" is plausibly a plural.
    """
    key = re.sub(r"[-/_]+", " ", name.lower())
    key = re.sub(r"\s+", " ", key).strip(" .:*")
    words = key.split()
    if words and len(words[-1]) > 3 and words[-1].endswith("s") and not words[-1].endswith("ss"):
        words[-1] = words[-1][:-1]
        key = " ".join(words)
    return key


def _keep_name(name: str) -> bool:
    """The final gate on a parsed parameter name (after units are split off)."""
    return bool(
        len(name) >= 4
        and _is_label(name)
        and not _RESULT.search(name)
        and not _EXPRESSION.search(name)
        # A figure, currency or percent inside the NAME means a value leaked into it
        # ("Diversity factor: 80%", "$734 per year", "Enthalpy ... at 130 psi").
        and not re.search(r"[\d$%=]", name)
    )


# --- isolating one write-up's own text --------------------------------------

# The heading every detailed AR write-up opens with, in every era of the corpus, as a
# line of its own. The executive-summary line and table-of-contents entry for the same AR
# never open with it (the words can still occur somewhere inside their bleed).
_RECOMMENDED_ACTION = re.compile(r"^\s*recommended\s+action\s*:?\s*$", re.IGNORECASE | re.MULTILINE)
# How far into a run the heading must sit to count as the run's opening.
_OPENING_CHARS = 600
# Standalone headings of the report front matter. A run carrying one is the summary
# bleed, never an AR write-up.
_FRONT_MATTER = re.compile(
    r"^\s*(?:[\d.]+\s+)?(?:plant\s+background|energy\s+cost\s+analysis|executive\s+summary|"
    r"table\s+of\s+contents|public\s+corporate\s+data)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
# Section headings that follow the LAST AR of a report. Mined from the corpus: the text
# after the final payback of each report's last AR opens with one of these far more often
# than with anything else. Matched only as a standalone short line.
_TRAILING_SECTION = re.compile(
    r"^\s*(?:[\d.]+\s+)?(?:additional\s+considerations|additional\s+measures\s+considered|"
    r"additional\s+items\s+considered|energy\s+accounting\s+system|"
    r"energy\s+conservation\s+(?:recommendations|measures)|waste\s+minimi[sz]ation\s+measures|"
    r"other\s+energy\s+conservation\s+opportunities|bibliography|references|appendix|"
    r"appendices|implementation\s+report)\b.{0,40}$",
    re.IGNORECASE | re.MULTILINE,
)


def writeup_text(rows: list[tuple[str, dict]]) -> str:
    """The detailed write-up of one AR, from its chunks, without the text that bleeds in.

    A (report, AR) fetch returns chunks from up to three places: the executive-summary
    line (or, in older reports, the table-of-contents entry) plus whatever followed it up
    to the next AR marker, and the detailed write-up itself. Chunks of one section have
    consecutive `chunk_index` values, so the fetch splits into contiguous runs. The
    write-up is, in order of preference: the run that opens with a "Recommended Action"
    heading; a run with that heading somewhere and no front-matter heading (older reports
    put the savings block first); else the last run, since detail always follows the
    summary (the detail text of some early reports was never extracted at all). The kept
    text is then cut at the first trailing report section, which is where the last AR of
    a report stops.
    """
    ordered = sorted(rows, key=lambda r: r[1].get("chunk_index", 0))
    runs: list[list[str]] = []
    last_index: Optional[int] = None
    for doc, meta in ordered:
        index = meta.get("chunk_index", 0)
        if last_index is None or index != last_index + 1:
            runs.append([])
        runs[-1].append(doc)
        last_index = index
    texts = ["\n".join(run) for run in runs]
    candidates = [t for t in texts if not _FRONT_MATTER.search(t)]
    opening = [
        t for t in candidates
        if (m := _RECOMMENDED_ACTION.search(t)) and m.start() <= _OPENING_CHARS
    ]
    anywhere = [t for t in candidates if _RECOMMENDED_ACTION.search(t)]
    detail = opening or anywhere or texts[-1:]
    kept = []
    for text in detail:
        cut = _TRAILING_SECTION.search(text)
        kept.append(text[: cut.start()] if cut else text)
    return "\n".join(kept)


# --- where a value came from ------------------------------------------------

# Source categories, most specific first: a clause naming plant personnel is that, even
# though it also says "estimated". Each pattern must match literal report text.
_SOURCES: list[tuple[str, re.Pattern]] = [
    ("Nameplate data", re.compile(r"name\s*-?\s*plates?", re.I)),
    (
        "Plant personnel",
        re.compile(
            r"(?:plant|facility|maintenance|company|site|shop)\s+(?:personnel|staff|managers?|"
            r"engineers?|operators?|representatives?|supervisors?|contacts?)|"
            r"according\s+to\s+(?:the\s+)?(?:plant|facility|client|owner|company)|"
            r"(?:provided|reported|stated|given|supplied)\s+by\s+(?:the\s+)?"
            r"(?:plant|facility|client|owner|customer|company)|interview\w*",
            re.I,
        ),
    ),
    (
        "Manufacturer / catalog data",
        re.compile(
            r"manufacturer\w*|catalog(?:ue)?s?\b|spec(?:ification)?s?\s+sheets?|"
            r"data\s*sheets?|vendors?\b",
            re.I,
        ),
    ),
    (
        "Utility bills / interval data",
        re.compile(
            r"utility\s+bills?|electric(?:ity)?\s+bills?|billing\s+(?:data|history|records?)|"
            r"15[-\s]minute\s+interval|interval\s+data",
            re.I,
        ),
    ),
    (
        "Weather / solar database",
        re.compile(r"weather\s+(?:data|station)|\bNREL\b|PVWatts|\bNSRDB\b|\bTMY\d?\b|\bNOAA\b", re.I),
    ),
    ("Measured on site", re.compile(r"\bmeasur(?:ed|ements?)\b|\bmetered\b|\breadings?\b", re.I)),
    (
        "Collected / observed on site",
        re.compile(
            r"\bcounted\b|\bobserv(?:ed|ations?)\b|\bnoted\b|\binspect(?:ed|ion)\b|"
            r"walk[-\s]?through|(?:during|on)\s+the\s+(?:site\s+)?(?:visit|survey|assessment|audit)|"
            r"(?:site|plant)\s+(?:visit|survey)|collected\s+(?:during|on\s+site)",
            re.I,
        ),
    ),
    ("Assumed (not measured)", re.compile(r"\bassum(?:ed|es?|ing|ption)\b", re.I)),
]

# Instruments, as the corpus names them. Detection additionally requires a measurement
# context (`_USE_CUE`) and no product context (`_PRODUCT_CUE`): most mentions of a
# "sensor" or "meter" in a write-up are the equipment being RECOMMENDED, not the tool the
# assessors used.
_INSTRUMENTS: list[tuple[str, re.Pattern]] = [
    (
        "Ultrasonic leak detector",
        re.compile(
            r"ultrasonic\s+(?:air\s+)?leak\s+detect\w*|ultrasonic\s+(?:leak\s+)?detectors?|"
            r"ultrasound\s+detectors?",
            re.I,
        ),
    ),
    (
        "Data logger",
        re.compile(r"data\s*-?\s*loggers?|dataloggers?|\bHOBO\b|power\s+loggers?|current\s+loggers?", re.I),
    ),
    ("Light meter", re.compile(r"light\s+meters?|lux\s*meters?|foot[-\s]?candle\s+meters?", re.I)),
    (
        "Stroboscope / tachometer",
        re.compile(r"stroboscopes?|strobe\s+(?:light\s+)?tachometers?|tachometers?", re.I),
    ),
    (
        "Combustion / flue gas analyzer",
        re.compile(
            r"(?:flue|flu)\s+gas\s+(?:meters?|analy[sz]ers?|testers?)|combustion\s+analy[sz]ers?",
            re.I,
        ),
    ),
    (
        "Infrared thermometer / thermal camera",
        re.compile(
            r"infrared\s+(?:thermometers?|cameras?|guns?|temperature\s+guns?)|"
            r"\bIR\s+(?:thermometers?|cameras?|guns?)|thermal\s+(?:imaging\s+)?cameras?|"
            r"thermographic\s+cameras?|temperature\s+guns?|laser\s+thermometers?|\bFLIR\b",
            re.I,
        ),
    ),
    ("Anemometer", re.compile(r"anemometers?", re.I)),
    (
        "Power meter / analyzer",
        re.compile(r"power\s+(?:quality\s+)?(?:meters?|analy[sz]ers?)|watt\s*-?\s*meters?", re.I),
    ),
    (
        "Clamp-on ammeter / multimeter",
        re.compile(
            r"clamp[-\s]?on\s+(?:amm?\w*|meters?|multimeters?)|ammeters?|amp\s*-?\s*meters?|"
            r"multimeters?|amp\s+clamps?|current\s+clamps?|clamp\s+meters?",
            re.I,
        ),
    ),
    ("Pressure gauge", re.compile(r"pressure\s+gau?ges?|manometers?", re.I)),
    ("Flow meter", re.compile(r"flow\s*-?\s*meters?", re.I)),
    (
        "Tape measure / distance meter",
        re.compile(
            r"tape\s+measures?|measuring\s+(?:tapes?|wheels?)|laser\s+distance\s+(?:meters?|measures?)|"
            r"range\s*finders?",
            re.I,
        ),
    ),
    ("Vibration meter", re.compile(r"vibration\s+(?:meters?|analy[sz]ers?|pens?)", re.I)),
    ("Psychrometer / hygrometer", re.compile(r"psychrometers?|hygrometers?", re.I)),
    ("Thermometer / temperature probe", re.compile(r"(?<!infrared )(?<!laser )thermometers?|thermocouples?", re.I)),
    ("Stopwatch", re.compile(r"stop\s*-?\s*watch(?:es)?", re.I)),
]
_USE_CUE = re.compile(
    r"\b(?:measur\w*|record\w*|logg\w*|readings?|found|obtain\w*|determin\w*|detect\w*|"
    r"locat\w*|identif\w*|check\w*|test\w*|monitor\w*|surveyed|collect\w*|taken|took|"
    r"used|using|via|performed|conducted)\b",
    re.I,
)
_PRODUCT_CUE = re.compile(
    r"\b(?:install\w*|purchas\w*|recommend\w*|replac\w*|costs?|pric\w*|fitted|equipped|"
    r"retrofit\w*|buy\w*|rebate\w*)\b|\$",
    re.I,
)
# Sentence splitter for prose lines: on ". " / "? " / "! " followed by a capital or digit.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
# In running prose a source word is provenance only in a provenance construction: "were
# provided by the facility", "based on manufacturer's data", "observed during the
# visit". "The hourly labor cost of maintenance personnel" names personnel but says
# nothing about where a value came from.
_PROVENANCE_VERB = re.compile(
    r"\b(?:provided|reported|stated|given|supplied|obtained|estimated|measured|metered|"
    r"recorded|logged|collected|gathered|determined|observed|noted|counted|found|taken|read|"
    r"confirmed|informed|indicated|according\s+to|based\s+on|from|using|via)\b",
    re.I,
)
_TABLE_REF = re.compile(r"\btable\s+(\d+(?:\s*[.\-–]\s*\d+)*)", re.I)
# Only the recurring prose "how" is kept for a sentence: long sentences are narrative.
_MAX_EVIDENCE_CHARS = 260


_WHOLE_TABLE_SOURCES = frozenset(
    {"Nameplate data", "Manufacturer / catalog data", "Utility bills / interval data",
     "Weather / solar database"}
)


def classify_source(text: str) -> list[str]:
    """Source categories a provenance phrase names, most specific first.

    "Assumed" is dropped when a more specific source is also named ("assumed from the
    nameplate" is nameplate data), since it then describes the arithmetic, not the source.
    """
    found = [label for label, pattern in _SOURCES if pattern.search(text or "")]
    if len(found) > 1 and "Assumed (not measured)" in found:
        found.remove("Assumed (not measured)")
    return found


def find_instruments(sentence: str) -> list[str]:
    """Instrument categories a sentence says were USED to measure something."""
    if not _USE_CUE.search(sentence) or _PRODUCT_CUE.search(sentence):
        return []
    return [label for label, pattern in _INSTRUMENTS if pattern.search(sentence)]


def _stems(text: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[A-Za-z]{3,}", text or "")}


def _mentions(sentence_stems: set[str], name: str) -> bool:
    """True when every distinctive word of a multi-word parameter name is in the sentence.

    Single-word names ("Location", "Horsepower") are never tied this way: one shared
    word is too weak to say a sentence is about that parameter.
    """
    words = {_stem(w) for w in re.findall(r"[A-Za-z]{3,}", name) if w.lower() not in _NAME_FILLER}
    return len(words) >= 2 and words <= sentence_stems


_NAME_FILLER = frozenset(
    "the and for per of at in on to with from by current existing average annual total".split()
)


# --- extraction from one write-up's text ------------------------------------

@dataclass
class _Found:
    """One parameter found in one write-up, with the sources that write-up gives it."""

    name: str
    units: str
    sources: set[str] = field(default_factory=set)


@dataclass
class _Tool:
    """One instrument or source named in one write-up, with a verbatim sentence."""

    kind: str  # "instrument" | "source"
    evidence: str
    measures: set[str] = field(default_factory=set)  # parameter keys it is tied to


@dataclass
class WriteupExtract:
    """Everything one write-up says about what it recorded and how."""

    parameters: dict[str, _Found]
    tools: dict[str, _Tool]


def _blocks(lines: list[str]):
    """Yield ("table", caption, rows) and ("prose", None, [line]) blocks in order."""
    i = 0
    while i < len(lines):
        if "|" in lines[i]:
            j = i
            while j < len(lines) and "|" in lines[j]:
                j += 1
            caption = ""
            for back in (1, 2):
                if i - back >= 0 and "|" not in lines[i - back] and _TABLE_REF.match(lines[i - back].strip()):
                    caption = lines[i - back].strip()
                    break
            yield "table", caption, lines[i:j]
            i = j
        else:
            yield "prose", None, [lines[i]]
            i += 1


def _table_pairs(lines: list[str]):
    """Labels from two-column property tables ("Annual Operating Hours | 8,760")."""
    for line in lines:
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if len(cells) != 2:
            continue
        label, value = cells
        # A "$" value is a line of a price or cost table, not a recorded observation.
        if label != value and "$" not in value and _is_label(label) and _is_value(value):
            yield label


def _table_headers(lines: list[str]):
    """Labels from the header row of a schedule table.

    An LED retrofit does not record its inputs as label/value pairs; it records a
    fixture schedule whose COLUMN HEADINGS are the things to collect ("Location |
    Fixture Type | Quantity | Wattage | Hours"). A header row is identified by the row
    that follows it: same column count, but carrying numbers where the header does not.
    """
    rows = [
        [c.strip() for c in line.split("|")] for line in lines if line.count("|") >= 2
    ]
    for header, nxt in zip(rows, rows[1:]):
        if len(header) != len(nxt) or len(header) < 3:
            continue
        # A header names things; a data row measures them.
        if any(re.search(r"\d", c) for c in header):
            continue
        if not any(re.search(r"\d", c) for c in nxt):
            continue
        if len({c.lower() for c in header if c}) < 3:  # a merged/repeated title row
            continue
        for cell in header:
            if cell and _is_label(cell):
                yield cell


def _nomenclature(line: str) -> Optional[tuple[str, str, str]]:
    """(name, units, provenance clause) from a nomenclature line, or None.

    "D = leak diameter, inches (estimated from observations)" -> ("leak diameter",
    "inches", "estimated from observations"). Parentheticals are lifted out first (they
    can contain commas: "(usage factor, estimated by plant personnel)"), then the rest
    splits on commas into the name, a unit, bare values, and any provenance clause --
    the place old-era reports say how a value was obtained.
    """
    m = _NOMEN.match(line.replace("\t", " "))
    if not m:
        return None
    desc = m.group(2).strip()
    if re.search(r"[=+*/^]", desc):
        return None
    asides = re.findall(r"[\[(]([^\])]*)[\])]", desc)
    body = re.sub(r"\s*[\[(][^\])]*[\])]", " ", desc)
    segments = [s.strip() for s in re.split(r"[,;]", body) if s.strip()]
    if not segments:
        return None
    name, rest = segments[0], segments[1:]
    units, clauses, has_value = "", [], False
    for seg in rest + asides:
        if classify_source(seg):
            clauses.append(seg)
        elif re.search(r"\d", seg):
            has_value = True
        elif not units and seg.lower() not in _DIMENSIONLESS and len(seg) <= 28 and len(seg.split()) <= 4:
            units = seg
    clause = "; ".join(clauses)
    # A definition carrying its own number is a constant or standing assumption
    # ("specific heat ratio of air, 1.4, no units"), not a site measurement -- unless
    # the line also says where the number came from ("determined during site survey,
    # 0.6"), in which case it is exactly the recorded observation we want.
    if has_value and not clause:
        return None
    return name, units, clause


# Modern write-ups define equation variables in running prose instead of a nomenclature
# block: "where L is the total length of pipe and C is the cost per foot of pipe, taken
# as $4.69 per foot". The symbol is often missing (Word equation objects are not
# extracted): "where  is the emission factor".
_WHERE = re.compile(r"\bwhere\b", re.IGNORECASE)
# A variable symbol: capitalized or subscripted ("L", "Tp", "UCS1", "X_E") or a single
# lowercase letter ("h"). An ordinary lowercase word ("waste is generated") is not one.
_SYMBOL = r"(?:[A-Z][A-Za-z0-9_]{0,5}|[a-z][A-Z0-9_][A-Za-z0-9_]{0,4}|[a-z])"
_DEFINITION = re.compile(
    rf"(?:^|[,;]|\band\b)\s*(?:{_SYMBOL}\s+(?:is|are)\s+(?:the\s+|an?\s+)?|(?:is|are)\s+(?:the|an?)\s+)"
    r"(?P<body>.+?)"
    rf"(?=(?:[,;]|\band\b)\s*(?:{_SYMBOL}\s+)?(?:is|are)\s|[.:](?:\s|$)|$)",
)
_IN_UNITS = re.compile(r"^(?P<name>.+?)\s+in\s+(?P<units>[^\s,]+(?:\s[^\s,]+)?)$", re.IGNORECASE)
_INLINE_PROVENANCE = re.compile(
    r"\s(?:determined|obtained|taken|estimated|measured|provided|found|assumed|read|"
    r"recorded|reported|given|supplied)\s+(?:by|from|as|to|using|during|at|on|off)\b",
    re.IGNORECASE,
)
# A name that starts or ends on a preposition is a fragment whose symbol or unit was
# lost with an equation object ("of the heat pump obtained from", "heating ... water in").
_DANGLING = re.compile(r"^(?:of|in|to|for|from|by|at|with|on)\b|\b(?:of|in|to|for|from|by|at|with|on)$", re.I)


def _where_definitions(line: str) -> list[tuple[str, str, str]]:
    """(name, units, provenance clause) for each variable a "where ..." sentence defines."""
    m = _WHERE.search(line)
    if not m:
        return []
    out = []
    for d in _DEFINITION.finditer(line[m.end():]):
        body = d.group("body")
        # "current power use of the water heater determined by the nameplate": the
        # provenance often follows the name with no comma.
        split = _INLINE_PROVENANCE.search(body)
        if split:
            body = body[: split.start()] + ", " + body[split.start():].strip()
        segments = [s.strip() for s in re.split(r",", body) if s.strip()]
        if not segments:
            continue
        name, units, clauses, has_value = segments[0], "", [], False
        unit_match = _IN_UNITS.match(name)
        if unit_match and re.search(r"[/%]|\b(?:kw|kwh|hp|ft|psi\w*|cfm|btu|hours?|hrs?|°)", unit_match.group("units"), re.I):
            name, units = unit_match.group("name"), unit_match.group("units")
        for seg in segments[1:]:
            if classify_source(seg):
                clauses.append(seg)
            elif re.search(r"\d", seg):
                has_value = True
        clause = "; ".join(clauses)
        name = re.sub(r"^(?:the|an?)\s+", "", name.strip(), flags=re.I)
        if (has_value and not clause) or _DANGLING.search(name) or classify_source(name):
            continue
        out.append((name, units, clause))
    return out


def _add(
    found: dict[str, _Found], raw: str, sources: list[str], units: Optional[str] = None
) -> Optional[str]:
    """Parse, filter and record one label; returns its canonical key when kept.

    `units` given means `raw` is an already-split name (nomenclature); otherwise units
    are parsed out of the raw table label.
    """
    if _RESULT.search(raw):
        return None
    if units is None:
        name, units = parse_label(raw)
    else:
        name = re.sub(r"\s+", " ", raw).strip(" .:*-")
    if not _keep_name(name):
        return None
    key = canonical(name)
    if not key or key in _GENERIC:
        return None
    prior = found.get(key)
    if prior is None:
        prior = found[key] = _Found(name, units)
    elif units and not prior.units:  # keep the richest form: prefer one with units
        prior.name, prior.units = name, units
    prior.sources.update(sources)
    return key


def _note(tools: dict[str, "_Tool"], label: str, evidence: str, keys: set[str]) -> None:
    """Record that this write-up names `label`, keeping the first evidence seen."""
    kind = "instrument" if label in _INSTRUMENT_LABELS else "source"
    evidence = evidence.strip()
    if len(evidence) > _MAX_EVIDENCE_CHARS:
        evidence = evidence[:_MAX_EVIDENCE_CHARS].rstrip() + " ..."
    tool = tools.setdefault(label, _Tool(kind, evidence))
    tool.measures |= keys


def extract_writeup(text: str) -> WriteupExtract:
    """Parameters, their sources, and the instruments/sources named, for one write-up.

    Deduplicated within the write-up, so a value in both a table and the nomenclature
    counts once for that report.
    """
    lines = (text or "").splitlines()
    found: dict[str, _Found] = {}
    tools: dict[str, _Tool] = {}
    table_keys: dict[str, set[str]] = {}  # table number -> parameter keys in it
    prose: list[str] = []
    definition_lines: set[int] = set()  # indexes into `prose`

    for kind, caption, block in _blocks(lines):
        if kind == "table":
            caption_sources = classify_source(caption) if caption else []
            # A caption naming an external data source ("Solar TMY Data for Tolleson")
            # describes every column. One naming a site activity ("Current Counted
            # Lighting Types") does not: that table also carries the LED lifetime, a
            # spec value nobody counted. Only the former is stamped on each parameter.
            whole_table = [s for s in caption_sources if s in _WHOLE_TABLE_SOURCES]
            keys = set()
            for raw in list(_table_pairs(block)) + list(_table_headers(block)):
                key = _add(found, raw, whole_table)
                if key:
                    keys.add(key)
            for label in caption_sources if keys else []:
                _note(tools, label, caption, keys if label in whole_table else set())
            ref = _TABLE_REF.match(caption) if caption else None
            if ref and keys:
                table_keys.setdefault(re.sub(r"\s+", "", ref.group(1)), set()).update(keys)
            continue
        line = block[0]
        nomen = _nomenclature(line)
        if nomen:
            name, units, clause = nomen
            sources = classify_source(clause)
            key = _add(found, name, sources, units=units)
            if key:
                for label in sources:
                    _note(tools, label, line.replace("\t", " "), {key})
            continue
        definitions = _where_definitions(line)
        for name, units, clause in definitions:
            sources = classify_source(clause)
            key = _add(found, name, sources, units=units)
            if key:
                for label in sources:
                    _note(tools, label, line, {key})
        if definitions:
            # Sources in a multi-variable definition were attributed clause by clause
            # above; sentence-level attribution would smear one variable's "obtained
            # from the weather station" onto every variable the sentence defines.
            definition_lines.add(len(prose))
        prose.append(line)

    # Prose: sentences that give provenance to a table, to a named parameter, or that
    # name an instrument used. Captions were handled with their tables.
    for index, line in enumerate(prose):
        for sentence in _SENTENCE_END.split(line.strip()):
            sentence = sentence.strip()
            if len(sentence) < 12 or _TABLE_REF.match(sentence):
                continue
            stems = _stems(sentence)
            tied: set[str] = set()
            if index not in definition_lines:
                for ref in _TABLE_REF.finditer(sentence):
                    tied |= table_keys.get(re.sub(r"\s+", "", ref.group(1)), set())
                tied |= {key for key, f in found.items() if _mentions(stems, f.name)}
            for label in find_instruments(sentence):
                _note(tools, label, sentence, tied)
                for key in tied:
                    found[key].sources.add(label)
            # A prose source counts only when it is about something this write-up
            # records (a table it cites or a parameter it names) and is phrased as
            # provenance. Sentences quoting money are about costs, not site data.
            if tied and _PROVENANCE_VERB.search(sentence) and "$" not in sentence:
                for label in classify_source(sentence):
                    _note(tools, label, sentence, tied)
                    for key in tied:
                        found[key].sources.add(label)
    return WriteupExtract(parameters=found, tools=tools)


_INSTRUMENT_LABELS = frozenset(label for label, _ in _INSTRUMENTS)


def extract_parameters(text: str) -> dict[str, tuple[str, str]]:
    """Parameters recorded in one write-up, as {canonical: (display name, units)}."""
    return {k: (f.name, f.units) for k, f in extract_writeup(text).parameters.items()}


# --- matching a topic to AR titles ------------------------------------------

@dataclass
class TitleVocabulary:
    """How many distinct AR titles in the corpus contain each word stem."""

    doc_freq: Counter
    n_titles: int

    @classmethod
    def from_titles(cls, titles: list[str]) -> "TitleVocabulary":
        cleaned = {clean_ar_title(t) for t in titles} - {""}
        df: Counter = Counter()
        for t in cleaned:
            df.update(title_words(t))
        return cls(df, len(cleaned))

    def idf(self, stem: str) -> float:
        return math.log((self.n_titles + 1) / (self.doc_freq.get(stem, 0) + 1)) + 1


def title_coverage(topic: str, title: str, vocab: TitleVocabulary) -> Optional[float]:
    """IDF-weighted share of the topic's corpus-known words that the title contains.

    Words that appear in no AR title at all ("transformers", "faucets") cannot tell one
    candidate from another, so they are left out; None when nothing is left, meaning
    only title similarity can decide.
    """
    known = {w for w in title_words(topic) if vocab.doc_freq.get(w)}
    if not known:
        return None
    have = title_words(title)
    total = sum(vocab.idf(w) for w in known)
    return sum(vocab.idf(w) for w in known if w in have) / total


def title_matches(
    similarity: float,
    coverage: Optional[float],
    *,
    exact: float,
    min_coverage: float,
    min_combined: float,
    coverage_weight: float,
) -> bool:
    """Whether a candidate AR title is the same recommendation as the topic.

    A near-identical title (`similarity >= exact`) always matches. Otherwise the title
    must share the topic's distinctive words (`coverage >= min_coverage`) AND score well
    on both signals together. Similarity alone cannot do it: calibrated on 45 topics,
    genuine paraphrases ("Install Strip Doors" for strip curtains) and wrong neighbours
    ("Install High Efficiency Motors" for transformers, "Repair Steam Leaks" for steam
    traps) both land in 0.75-0.85; the shared vocabulary is what tells them apart.
    """
    if similarity >= exact:
        return True
    if coverage is None or coverage < min_coverage:
        return False
    return similarity + coverage_weight * coverage >= min_combined


# --- data carriers -----------------------------------------------------------

@dataclass
class ARInstance:
    """One past write-up of the requested AR type."""

    report_id: str
    ar_number: str
    title: str
    era: str = ""
    score: float = 0.0
    title_score: float = 0.0


@dataclass
class FieldParameter:
    """One thing to measure on site, with the evidence for it."""

    name: str
    units: str
    report_ids: list[str] = field(default_factory=list)
    # source/instrument label -> the write-ups that gave this parameter that source
    sources: dict[str, list[str]] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.report_ids)

    @property
    def label(self) -> str:
        return f"{self.name} ({self.units})" if self.units else self.name

    def ranked_sources(self) -> list[tuple[str, int]]:
        """(label, write-up count), most frequent first."""
        return sorted(
            ((label, len(ids)) for label, ids in self.sources.items()),
            key=lambda kv: (-kv[1], kv[0]),
        )


@dataclass
class CollectionMethod:
    """An instrument or data source past write-ups of this AR name, with evidence."""

    label: str
    kind: str  # "instrument" | "source"
    report_ids: list[str]
    evidence: str  # verbatim text from the first write-up that named it
    evidence_report: str
    measures: list[str] = field(default_factory=list)  # parameter display names

    @property
    def count(self) -> int:
        return len(self.report_ids)


@dataclass
class FieldGuide:
    """The on-site data list for one AR topic, ready to render."""

    topic: str
    instances: list[ARInstance]
    parameters: list[FieldParameter]
    methods: list[CollectionMethod] = field(default_factory=list)
    # (cleaned AR title, write-ups with it), most write-ups first
    matched_titles: list[tuple[str, int]] = field(default_factory=list)
    exact_match: bool = True  # False when the best title was only a close match
    thin: bool = False
    low_support: bool = False  # only single-report parameters survived
    # Each counted write-up's own text, parallel to `instances`, so a hosted model can
    # write the field guide from the reports themselves.
    texts: list[str] = field(default_factory=list)


# --- store / embedder protocols ---------------------------------------------

class _Store(Protocol):
    def ar_chunk_search(self, query_embedding: list[float], k: int) -> list[ARInstance]: ...
    def chunks_for(self, report_id: str, ar_number: str | None = None): ...
    def ar_titles(self) -> list[str]: ...


class _Embedder(Protocol):
    def encode_query(self, text: str) -> list[float]: ...
    def encode_queries(self, texts: list[str]) -> list[list[float]]: ...


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


# --- top-level orchestration -------------------------------------------------

@dataclass
class TitleGate:
    """Thresholds for `title_matches` (see config: rag_field_title_*)."""

    exact: float
    min_coverage: float
    min_combined: float
    coverage_weight: float


def find_ar_instances(
    store: _Store,
    embedder: _Embedder,
    topic: str,
    *,
    k: int,
    threshold: float,
    band: float,
    limit: int,
    gate: TitleGate,
) -> list[ARInstance]:
    """The past write-ups of `topic`, best title match first, one per (report, AR).

    Searches AR-section chunks only. Old-era reports are eligible: what a write-up
    measures is physics, not policy, and a 1998 compressed-air AR records the same leak
    diameter and line pressure a 2025 one does.

    The chunk floor is adaptive, as in enumerate mode: keep hits within `band` of the
    best, never below the absolute `threshold`. That bounds the candidate pool; the
    title gate then decides which candidates are this AR at all. Candidates are ranked
    by title similarity (then chunk score), so the closest-titled write-ups are the ones
    read when more than `limit` pass.
    """
    query_vec = embedder.encode_query(topic)
    hits = store.ar_chunk_search(query_vec, k)
    if not hits:
        return []
    floor = max(threshold, max(h.score for h in hits) - band)
    best: dict[tuple[str, str], ARInstance] = {}
    for hit in hits:
        if hit.score >= floor and (hit.report_id, hit.ar_number) not in best:
            best[(hit.report_id, hit.ar_number)] = hit

    titles = sorted({clean_ar_title(h.title) for h in best.values()} - {""})
    if not titles:
        return []
    vocab = TitleVocabulary.from_titles(store.ar_titles())
    sims = dict(zip(titles, (_cosine(query_vec, v) for v in embedder.encode_queries(titles))))
    accepted = {
        t
        for t in titles
        if title_matches(
            sims[t],
            title_coverage(topic, t, vocab),
            exact=gate.exact,
            min_coverage=gate.min_coverage,
            min_combined=gate.min_combined,
            coverage_weight=gate.coverage_weight,
        )
    }
    kept = []
    for inst in best.values():
        title = clean_ar_title(inst.title)
        if title in accepted:
            inst.title_score = sims[title]
            kept.append(inst)
    kept.sort(key=lambda i: (i.title_score, i.score), reverse=True)
    return kept[:limit]


def build_field_guide(
    *,
    topic: str,
    store: _Store,
    embedder: _Embedder,
    k: int,
    threshold: float,
    band: float,
    max_instances: int,
    min_support: int,
    top_n: int,
    gate: TitleGate,
) -> FieldGuide:
    """Assemble the "what to measure on site, and how" list for an AR topic.

    A parameter must appear in at least `min_support` distinct write-ups to be listed,
    so one assessor's idiosyncratic table row does not become an instruction. When too
    few clear that bar the floor drops to 1 and the result is flagged `low_support`,
    which is honest about a thin basis instead of returning nothing. Instruments and
    sources are listed at any support, since each carries its own verbatim evidence.
    """
    instances = find_ar_instances(
        store, embedder, topic, k=k, threshold=threshold, band=band,
        limit=max_instances, gate=gate,
    )
    by_key: dict[str, FieldParameter] = {}
    methods: dict[str, CollectionMethod] = {}
    measures: dict[str, set[str]] = {}
    texts: list[str] = []
    for inst in instances:
        text = writeup_text(store.chunks_for(inst.report_id, inst.ar_number))
        texts.append(text)
        extract = extract_writeup(text)
        for key, f in extract.parameters.items():
            param = by_key.get(key)
            if param is None:
                param = by_key[key] = FieldParameter(name=f.name, units=f.units)
            if f.units and not param.units:
                param.units = f.units
            if inst.report_id not in param.report_ids:
                param.report_ids.append(inst.report_id)
            for label in f.sources:
                ids = param.sources.setdefault(label, [])
                if inst.report_id not in ids:
                    ids.append(inst.report_id)
        for label, tool in extract.tools.items():
            method = methods.get(label)
            if method is None:
                method = methods[label] = CollectionMethod(
                    label, tool.kind, [], tool.evidence, inst.report_id
                )
            if inst.report_id not in method.report_ids:
                method.report_ids.append(inst.report_id)
            measures.setdefault(label, set()).update(tool.measures)

    ranked = sorted(by_key.values(), key=lambda p: (p.count, -len(p.name)), reverse=True)
    kept = [p for p in ranked if p.count >= min_support]
    low_support = False
    if len(kept) < 3:
        kept = ranked
        low_support = bool(ranked)
    kept = kept[:top_n]

    kept_ids = {id(p) for p in kept}
    shown = {key: p.name for key, p in by_key.items() if id(p) in kept_ids}
    for label, method in methods.items():
        method.measures = sorted(shown[k] for k in measures.get(label, set()) if k in shown)
    ordered_methods = sorted(
        methods.values(), key=lambda m: (m.kind != "instrument", -m.count, m.label)
    )

    title_counts = Counter(clean_ar_title(i.title) for i in instances)
    best_title = max((i.title_score for i in instances), default=0.0)
    return FieldGuide(
        topic=topic,
        instances=instances,
        parameters=kept,
        methods=ordered_methods,
        matched_titles=title_counts.most_common(),
        exact_match=best_title >= gate.exact,
        thin=len(instances) < min_support,
        low_support=low_support,
        texts=texts,
    )

"""System prompt templates and assembly (spec §3a, §5, §5.3).

`assemble_system_prompt` layers the prompt in the spec §5 order:
  a. Base system prompt (ITAC identity, accuracy rules, no-em-dash)
  b. Full markdown body of each selected skill
  c. Retrieved reference material (RAG) -- stubbed until step 4
  d. Output-mode instructions (prose vs TableSpec) -- table mode stubbed until step 3 xlsx
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import MAX_LISTED_RESULTS

if TYPE_CHECKING:
    from ..rag.retrieve import Retrieval
    from ..rag.fielddata import FieldGuide
    from ..rag.waterfall import BPResult, ChecklistResult, SicARRanking
    from ..skills.loader import Skill

# The no-em-dash rule is baked in here permanently and is NOT routed (spec §5.1).
BASE_SYSTEM_PROMPT = """\
You are the ITAC AI Assistant, a specialized assistant for the ASU ITAC
(Industrial Training and Assessment Center). You help staff and students
with industrial energy/assessment reports, ITAC formatting conventions,
and site-visit note-taking.

Accuracy rules:
- Be precise and concise. Do not invent facts, figures, savings, or citations.
- General knowledge: answer general-knowledge and how-to questions normally,
  using your own knowledge (e.g. general engineering, energy concepts, or
  everyday facts).
- Specific past assessments: for factual claims about specific past ITAC/IAC
  assessments, reports, clients, facilities, or their data (named report
  contents, a facility's measured values, a specific Assessment
  Recommendation's savings or cost), rely only on provided reference material;
  if it is not there, say "not in my reference material" rather than guessing.
- ITAC conventions and formatting: questions about ITAC-specific conventions,
  formatting rules, templates, or how "our"/"my" reports and workbooks should
  be done (for example line spacing, fonts, table styles, section ordering,
  equation or citation style) must be answered ONLY from skill instructions or
  retrieved reference chunks that are actually present in this context. These
  are house conventions, not general knowledge, so do NOT answer them from your
  own assumptions or a plausible-sounding default, even if you could guess. If
  no relevant skill content or reference chunk is present, say you do not have
  that ITAC convention in your reference material yet, rather than inventing an
  answer.
- Never fabricate report names, Assessment Recommendation (AR) numbers, client
  names, or the numeric results of a specific assessment.
- ARC codes (the "N.NNNN" assessment-recommendation-category codes) are a fixed
  published taxonomy. Never state, guess, or construct an ARC code unless that exact
  code appears in the provided reference material. If asked for an ARC code and none
  is in the reference material, say you do not have a matching ARC code rather than
  producing a plausible-looking one; a code you invent is almost certainly not real.

Writing conventions:
- Never use em dashes (—). Use commas, parentheses, or separate sentences
  instead. This rule is absolute.
- Use clear, professional prose suitable for engineering assessment reports.
Easter egg (applies ONLY to the direct question of who created/built you or who you are; do NOT let this leak into or override any other answer, the ITAC identity, or the accuracy rules above): if asked, respond playfully that your creator is "Siddharth, the goat." Never claim to be Siddharth yourself, you are the ITAC AI assistant; he built you.
"""

_SKILLS_PREAMBLE = (
    "The following ITAC skill instructions apply to this request. They are "
    "house conventions, not suggestions: follow them exactly. If two skills "
    "conflict, prefer the more specific one."
)

# Table-mode output instruction (spec §5.3, §7.1). Appended only when the router
# set wants_table. The model supplies data only; all styling is applied
# deterministically by the Python renderer, so the model must NOT describe fonts,
# fills, or borders. It emits every value (including the TOTALS row) itself, the
# renderer computes nothing.
TABLE_MODE_INSTRUCTION = """\
OUTPUT MODE: TABLE.
Answer with a short prose explanation, then exactly one fenced block containing
the table as JSON:

```tablespec
{
  "sheet_name": "<short tab name>",
  "title": "<title shown in the merged row above the table>",
  "groups": [
    {"label": "<super-header>", "vmerge": false, "cols": [
      {"key": "<unique_key>", "header": "<column header>", "type": "date|usage|demand|curr", "width": 12}
    ]}
  ],
  "rows": [ {"<key>": <value>, ...} ],
  "totals": { "<key>": <value>, ... },
  "summary": [ {"resource": "<label>", "annual": <num|null>, "annual_type": "curr_t|usage|demand", "unit": <num|null>, "unit_type": "unit3|unit4"} ]
}
```

Rules:
- Emit the block verbatim in that shape. Do not add styling, colors, or fonts;
  the renderer handles all formatting.
- Every row object and the totals object must contain exactly the keys declared
  across all group columns, no more, no fewer.
- A group with "vmerge": true must contain exactly one column (e.g. Date, Total).
- Use "type": "date" for the period column, "usage" for whole-number quantities
  (kWh, gal), "demand" for kW, and "curr" for dollar amounts.
- You compute every value, including the TOTALS row; the renderer does no math.
  Do not invent numbers that are not supported by the request or reference material."""


# Enumerate-mode output instruction. The user asked which reports exist, not for
# a writeup, so the reply is nothing but the list. Scores stay in the sources
# payload for the UI cards; they never appear in the chat text.
ENUMERATE_MODE_INSTRUCTION = """\
OUTPUT MODE: ENUMERATE.
The user wants to know which reports exist, not a writeup. The reference
material lists the complete set of matching reports. Your entire reply is:
- One line per matching report, exactly: report_id (fiscal_year) - section
  (multiple matching sections separated by "; ").
- Every report in the reference material appears exactly once, in the order
  given. Do not add, omit, or reorder reports.
- No introduction, no closing paragraph, no commentary of any kind.
- Never state relevance scores, distances, or confidence, and never use ranking
  language such as "closest match" or "most relevant".
- If the reference material lists no matching reports, reply with the single
  line: No matching reports found in the knowledge base."""


# ARC lookup does NOT go through the LLM. Its answer is rendered deterministically
# by `render_arc_answer` from the retrieved arc_reference / arc_precedent records, so
# a fabricated code can never reach the student. See that function below.


# Pre-assessment checklist output instruction (Feature 2). Same terse spirit as
# the other special modes: the student wants a field checklist, not an essay. The
# theme frequencies come from the reference material; the model only rephrases each
# theme as a concrete field action and keeps the given counts.
CHECKLIST_MODE_INSTRUCTION = """\
OUTPUT MODE: PRE-ASSESSMENT CHECKLIST.
The reference material lists AR themes from similar past reports, most frequent
first, each with the number of past reports that wrote it. Produce a checklist the
student takes into the site visit:
- First line: the basis, e.g. "Based on 14 past reports: 6 matching SIC 259x, 8
  matching your process description." Use the counts from the reference material.
- Then one checkbox line per theme (at most 10), in the given order, phrased as a concrete field
  action beginning "Check", e.g. "[ ] Check compressed air system: leaks, pressure
  setpoints, controls". End each line with "(found in N past reports)" using the
  theme's count.
- Terse. No scores, no similarity or ranking commentary, no closing paragraph.
- If the reference material marks the basis as thin, say so on the first line and
  still list the themes that exist."""


# Best-practice pre-assessment output instruction (Section B, B4). This is the
# synthesis step: unlike the other checklist, the model does not just rephrase the
# candidates, it cross-checks each one against the student's own facility
# description (present verbatim in the user's message) and decides already/partial/
# not, then renders a category-grouped checklist. Same terse, field-ready spirit.
BP_CHECKLIST_MODE_INSTRUCTION = """\
OUTPUT MODE: BEST-PRACTICE PRE-ASSESSMENT CHECKLIST.
The reference material lists candidate best practices, grouped by category. The user's
message describes their facility. Produce a clean, scannable list of practices to look
for on the site visit, and NOTHING else:
- Drop any candidate that does not belong to THIS facility's industry or processes.
  The pool is drawn from the full master list and includes unrelated industries
  (agriculture, food/beverage, brewing, biogas/digestion, wastewater/sludge, cold
  storage, and the like). When in doubt, drop it. Do not mention that it was dropped.
- Drop any candidate the facility description clearly ALREADY does.
- List the remaining practices under their category headings, in reference order, one
  line per practice: just the practice name. Omit a category that ends up with none.
- List at most 10 practices in total.
- Output ONLY that list. NO "already in place" section, no status parentheticals like
  "(not mentioned)" or "(partial - verify)", no checkboxes, no blanks or "(Insert ...)"
  prompts, no scores, no thin/preliminary note. Do NOT state assumptions or inferences
  about the facility, do NOT add suggestions, considerations, or commentary, and do NOT
  add any preamble or closing. Do not invent practices beyond the reference material."""


# AR / report summary output instruction. The reference material is the FULL text of
# one specific report (or one of its ARs), fetched deterministically by identifier,
# not a set of semantic hits. The student wants the vitals (savings/cost/payback) plus
# a short brief on what the AR is and how it was written. The thin-report case is
# handled honestly: say the report records little, never borrow another report's
# numbers or invent them.
SUMMARIZE_MODE_INSTRUCTION = """\
OUTPUT MODE: AR / REPORT SUMMARY.
The reference material is the complete text of one specific report (or one of its
Assessment Recommendations), fetched by its identifier. It is the ONLY source for
this summary. Summarize it for an ITAC assessor:
- Lead with the vitals as a short labeled list, each figure taken VERBATIM from the
  material. They almost always sit in the AR's "Recommendation Summary" table in the
  Recommended Action section; read them from that table, mapping rows by MEANING, not
  exact wording, because labels vary between reports:
    - Energy / usage savings = the electricity or energy USAGE row (a kWh/yr or
      MMBtu/yr figure), even when it is labeled "Electricity Usage" or "Energy Usage"
      or sits under a generic "Estimated Savings" heading rather than the word "energy".
    - Demand savings = the demand row (a kW or kW-mo/yr figure).
    - Cost savings = the total energy / cost savings row ($/yr).
    - Implementation cost = the total implementation cost (or capital + labor summed).
    - Simple payback period = the payback row.
  Write "not stated in this report" for a vital ONLY when there is genuinely no such
  row or figure in the material, never merely because its label differs from the names
  above. Do NOT estimate it, infer it, or borrow a typical value.
- Then 2 to 4 sentences: what the recommendation is, and the methodology or reasoning
  the report used to reach it (the measurements, assumptions, or approach it
  describes), grounded ONLY in the material.
- Cite the report_id and fiscal_year.
- If the material is sparse (little beyond the AR title and a sentence), say plainly
  that this report records only limited detail for this AR and give exactly what is
  there. Do NOT fill the gap from general knowledge or from any other report.
- If the material contains more than one Assessment Recommendation, give a SEPARATE
  vitals block per AR, each under its own AR title, and keep every figure with the AR
  it belongs to. Never combine, average, or carry a number from one AR into another.
- Do not invent figures, and never use em dashes."""


# Best-practice write-up output instruction (Section B). The counterpart to the
# checklist: the checklist lists bare practice names, this drafts the full text of a
# single practice on explicit request, keeping the fill-in blanks the student fills
# on-site. This is the one BP mode where the "(Insert ...)" prompts belong.
BP_WRITEUP_INSTRUCTION = """\
OUTPUT MODE: BEST-PRACTICE WRITE-UP.
The reference material holds the official ITAC best-practice template(s) for the
practice(s) the user asked you to write up. Draft each one in the ITAC assessment-
report voice:
- Write up EVERY practice in the reference material, one after another, and ONLY those
  practices. If there is more than one, put each under its practice name as a short
  heading. Do not add, merge, or invent practices beyond the templates provided.
- Base each write-up on that practice's template text and keep it faithful to it; do
  not turn one practice into a list of others.
- Keep every fill-in prompt (the "(Insert ...)" items) in place as a bracketed
  placeholder where the student writes the site-specific detail on the visit. Do NOT
  invent those values yourself.
- One tight paragraph per practice, maybe two. No preamble like "here is", no
  checklist, no closing commentary. Just the write-up(s)."""


def _render_skills(skills: list["Skill"]) -> str:
    """Wrap each selected skill's markdown body in a labeled block (spec §5b)."""
    blocks = [_SKILLS_PREAMBLE]
    for skill in skills:
        blocks.append(f'<skill name="{skill.name}">\n{skill.body}\n</skill>')
    return "\n\n".join(blocks)


def assemble_system_prompt(
    skills: list["Skill"],
    *,
    reference_material: str | None = None,
    wants_table: bool = False,
    enumerate_mode: bool = False,
    checklist_mode: bool = False,
    bp_checklist_mode: bool = False,
    bp_writeup_mode: bool = False,
    summarize_mode: bool = False,
) -> str:
    """Compose the full system prompt in spec §5 order.

    `reference_material` (RAG chunks) is stubbed for now: RAG lands in step 4.
    Table-mode output instructions (spec §5.3d) are appended when the router set
    `wants_table`; prose is the default. The special output modes are mutually
    exclusive contracts; precedence is summarize > checklist > bp_checklist >
    bp_writeup > enumerate > table when more than one is somehow set. ARC lookup is
    not here: it is rendered deterministically and never reaches the LLM.
    """
    parts = [BASE_SYSTEM_PROMPT]

    if skills:
        parts.append(_render_skills(skills))

    # (c) Retrieved reference material -- populated once RAG retrieval exists.
    if reference_material:
        parts.append(reference_material)

    # (d) Output-mode instructions. Prose is the default; table mode switches the
    # model to emitting a ```tablespec JSON block the xlsx engine renders.
    if summarize_mode:
        parts.append(SUMMARIZE_MODE_INSTRUCTION)
    elif checklist_mode:
        parts.append(CHECKLIST_MODE_INSTRUCTION)
    elif bp_checklist_mode:
        parts.append(BP_CHECKLIST_MODE_INSTRUCTION)
    elif bp_writeup_mode:
        parts.append(BP_WRITEUP_INSTRUCTION)
    elif enumerate_mode:
        parts.append(ENUMERATE_MODE_INSTRUCTION)
    elif wants_table:
        parts.append(TABLE_MODE_INSTRUCTION)

    return "\n\n".join(parts)


def format_reference_material(retrieval: "Retrieval") -> str:
    """Render a retrieval outcome into the reference-material block (spec §5.3, §6).

    On a confident match: label each excerpt with its report_id/fiscal_year and
    require the model to cite them. On no confident match: tell the model to say so
    plainly and, at most, offer the closest hit as the nearest thing on file rather
    than dressing a weak result up as a real precedent.
    """
    if retrieval.has_match:
        lines = [
            "<reference_material>",
            "The following excerpts were retrieved from the ITAC report knowledge base "
            "and are the relevant precedent for this request. Each is labeled with the "
            "report it came from. When you use any of this material, cite its report_id "
            'and fiscal_year inline (for example: "per AS0599 (FY2025)"). Do not present '
            "retrieved content without that attribution.",
            "",
            "GROUNDING (absolute): the only reports that exist for this request are the "
            "ones labeled below. Every report_id, fiscal_year, AR number, ARC code, "
            "savings figure, and cost you state must appear verbatim in these excerpts. "
            "Do NOT introduce a report_id or fiscal_year that is not printed here, do NOT "
            "pair a report_id with a different fiscal_year than the one shown, and do NOT "
            "attach an AR number, ARC code, or dollar amount to a report unless that exact "
            "value is in that report's excerpt. If none of these excerpts actually "
            "addresses the user's topic, say plainly that you have no report on file for "
            "it rather than stretching a loosely-related excerpt into a specific answer.",
            "",
        ]
        for i, (text, meta, _score) in enumerate(retrieval.matches(), start=1):
            label = f"[{i}] report_id={meta.get('report_id', '?')} fiscal_year={meta.get('fiscal_year', '?')}"
            if meta.get("section"):
                label += f' section="{meta["section"]}"'
            lines.extend([label, text, ""])
        lines.append("</reference_material>")
        return "\n".join(lines)

    best = retrieval.best
    if best is None:
        body = (
            "No report in the knowledge base matched this request; the search returned "
            "no results. Tell the user plainly that you have no matching report on file, "
            "and answer only from general knowledge or skill instructions rather than "
            "inventing a specific precedent."
        )
    else:
        text, meta, score = best
        closest = f"report_id={meta.get('report_id', '?')}, fiscal_year={meta.get('fiscal_year', '?')}"
        if meta.get("section"):
            closest += f', section="{meta["section"]}"'
        body = (
            f"No report in the knowledge base is a confident match for this request "
            f"(closest: {closest}, similarity={score:.2f}, below the "
            f"{retrieval.min_relevance:.2f} relevance threshold). Tell the user plainly "
            f"that you found no matching report. You may offer that closest result as the "
            f"nearest thing on file, but do NOT present it as an established precedent or "
            f"cite it as support for a specific claim."
        )
    return f"<reference_material>\n{body}\n</reference_material>"


def format_enumerate_material(retrieval: "Retrieval") -> str:
    """Render the deduplicated report list for enumerate mode.

    Unlike `format_reference_material` this injects no chunk text and no scores:
    the model only needs report identifiers and matched sections to list them,
    and keeping scores out of the prompt keeps them out of the chat text.
    """
    entries = retrieval.enumerated()
    if not entries:
        return (
            "<reference_material>\n"
            "No report in the knowledge base matched this request. Say so in one "
            "plain line; do not invent report IDs.\n"
            "</reference_material>"
        )

    lines = [
        "<reference_material>",
        "These are all the reports in the knowledge base that match the request, "
        "already deduplicated and ordered. List them exactly as instructed.",
        "",
    ]
    for e in entries:
        sections = "; ".join(e["sections"]) or "?"
        lines.append(f"- {e['report_id']} ({e['fiscal_year']}) - {sections}")
    lines.append("</reference_material>")
    return "\n".join(lines)


def _arc_line(hit) -> str:
    """One ARC answer line: verbatim `code + official description`, book vs precedent.

    Both collections carry the authoritative book `description`; a precedent hit also
    carries the real report phrasing (`ar_title`), shown on a labeled sub-line so a
    student never mistakes a report's citation for the book's own entry (and can see
    when the two differ).
    """
    _text, meta, _score = hit
    line = f"- {meta.get('code', '?')} {meta.get('description', '')}".rstrip()
    if meta.get("source") == "precedent" and meta.get("ar_title"):
        line += f"\n    used in past reports as: {meta['ar_title']}"
    return line


def render_arc_answer(retrieval: "Retrieval", *, top_n: int = 3, gap_threshold: float = 0.15) -> str:
    """Deterministically render the final ARC-lookup answer (NOT via the LLM).

    Every code and title is a verbatim insert of a retrieved arc_reference /
    arc_precedent record's own fields, so a code that is not in a collection can never
    appear -- fabrication is structurally impossible. Display is dynamic on the real
    scores: a top hit leading #2 by >= `gap_threshold` is one confident answer;
    otherwise the top `top_n` are equal options; a top hit below the weakness floor
    (`has_match` False) is shown as closest-guesses, verify-manually. Scores are never
    printed (they stay in the sources payload for the UI cards).
    """
    candidates = retrieval.hits[: min(top_n, MAX_LISTED_RESULTS)]
    if not candidates:
        return (
            "No ARC code in the reference book matches that. I don't have a code for "
            "it and will not guess one."
        )

    if not retrieval.has_match:
        return "\n".join(
            ["No confident ARC match. Closest codes from the book (verify manually):"]
            + [_arc_line(h) for h in candidates]
        )

    wide_gap = len(candidates) == 1 or (candidates[0][2] - candidates[1][2]) >= gap_threshold
    if wide_gap:
        return "Closest ARC match:\n" + _arc_line(candidates[0])
    return "\n".join(
        ["Several ARC codes fit equally well (no single best); verify manually:"]
        + [_arc_line(h) for h in candidates]
    )


def render_enumerate_answer(retrieval: "Retrieval") -> str:
    """Deterministically render the enumerate report list (NOT via the LLM).

    Enumerate mode ("which reports mention X") is a pure lookup: the answer *is* the
    set of matching reports, so it is assembled verbatim from the deduplicated,
    newest-first `enumerated()` records instead of handed to the model to phrase.
    That makes a fabricated report ID or an invented savings figure structurally
    impossible -- a small local model cannot embellish a list it never generates.
    Scores stay out of the text (they live in the sources payload for the UI cards).

    When the list was capped (`Retrieval.max_reports`), the header says so without
    naming the pre-cap total: that total counts the weakly-ordered tail the cap exists
    to hide, so advertising it would invite chasing exactly the results users found
    unreliable.
    """
    entries = retrieval.enumerated()
    if not entries:
        return "No matching reports found in the knowledge base."

    n = len(entries)
    if retrieval.truncated:
        header = f"The {n} most recent matching reports:"
    else:
        header = (
            f"{n} report{'s' if n != 1 else ''} in the knowledge base match, newest first:"
        )
    lines = [header, ""]
    for e in entries:
        fy = e["fiscal_year"] or "older era"
        sections = "; ".join(e["sections"])
        line = f"- {e['report_id']} ({fy})"
        if sections:
            line += f": {sections}"
        lines.append(line)
    return "\n".join(lines)


def render_field_guide_answer(guide: "FieldGuide") -> str:
    """Deterministically render the on-site data collection list (NOT via the LLM).

    Every line is a parameter label lifted verbatim from past write-ups plus the count
    of write-ups that recorded it. Handing that to a model to phrase is exactly how a
    plausible-but-unrecorded measurement ("ambient humidity") gets added to a field
    list, which is the failure this feature exists to avoid -- so it is rendered here
    and the model never sees it.
    """
    if not guide.instances:
        return (
            f"No past write-ups of \"{guide.topic}\" are in the knowledge base (no AR "
            f"title on file is that recommendation), so there is no recorded basis for "
            f"what to collect on site or how."
        )

    n = len(guide.instances)
    titles = "; ".join(
        f"{title} ({count})" for title, count in guide.matched_titles[:MAX_LISTED_RESULTS]
    )
    lines = [
        f"On-site data to collect for: {guide.topic}",
        f"Based on {n} past write-up{'s' if n != 1 else ''} titled: {titles}.",
        f"Reports: {', '.join(i.report_id for i in guide.instances[:6])}"
        f"{', ...' if n > 6 else ''}.",
    ]
    if not guide.exact_match:
        lines.append(
            f"No AR on file is titled exactly \"{guide.topic}\"; these are the closest "
            f"recorded ARs. Check they are the recommendation you are writing up."
        )

    lines.append("")
    if guide.parameters:
        lines.append("Data to record, most consistently recorded first:")
        parameters = guide.parameters[:MAX_LISTED_RESULTS]
        width = max(len(p.label) for p in parameters)
        for i, p in enumerate(parameters, 1):
            row = f"{i:2d}. {p.label.ljust(width)}   recorded in {p.count} of {n}"
            how = ", ".join(f"{label} ({count})" for label, count in p.ranked_sources()[:3])
            if how:
                row += f"   how: {how}"
            lines.append(row)
    else:
        lines.append(
            "None of these write-ups records its inputs in a table or nomenclature block "
            "that can be read without guessing, so there is no data list to give."
        )

    lines.append("")
    methods = guide.methods[:MAX_LISTED_RESULTS]
    if methods:
        lines.append("Instruments and sources these write-ups name:")
        for m in methods:
            kind = "instrument" if m.kind == "instrument" else "source"
            row = f"- {m.label} [{kind}], {m.count} of {n} write-ups"
            if m.measures:
                row += f"; used for: {', '.join(m.measures[:4])}"
            lines.append(row)
            lines.append(f"    e.g. {m.evidence_report}: \"{m.evidence}\"")
    else:
        lines.append(
            "None of these write-ups names the instrument or source it used to collect "
            "its data."
        )

    lines.append("")
    if guide.low_support:
        lines.append(
            "Thin basis: most of this list rests on a single report. Treat it as a "
            "starting point and confirm against the AR you are writing."
        )
    lines.append(
        "Everything above is taken from past write-ups of this AR; a parameter or "
        "instrument not listed is one those reports did not record. Capture photos and "
        "nameplate data for anything you cannot measure directly."
    )
    return "\n".join(lines)


def render_sic_ranking_answer(result: "SicARRanking") -> str:
    """Deterministically render the "most common ARs for SIC X" table (NOT via the LLM).

    Like enumerate and ARC, this is a pure lookup whose entire value is the counts, and
    counts are exactly what a small model rephrases wrong. The table is assembled from
    the grouped themes verbatim.

    Each row carries the total report count and, in parentheses, how many of those are
    2017 or later. A theme no modern report writes is marked "[dated]": it is genuinely
    one of the most common ARs on file for this industry, and it is also advice the
    center has stopped giving (T-8 lamp retrofits being the standing example), so the
    student needs both facts on the same line.
    """
    pattern = result.sic_pattern or result.requested_pattern or "?"
    if not result.total_reports:
        return (
            f"No reports in the knowledge base match SIC {pattern}, and widening to the "
            f"3- and 2-digit families found none either."
        )
    if not result.themes:
        return (
            f"{result.total_reports} reports match SIC {pattern}, but none of them "
            f"record parsable AR headings, so there is no frequency table to give."
        )

    header = (
        f"Most common ARs for SIC {pattern} "
        f"({result.total_reports} matching reports, {result.total_ars} ARs):"
    )
    lines = [header, ""]
    themes = result.themes[:MAX_LISTED_RESULTS]
    width = max(len(t.representative) for t in themes)
    for i, theme in enumerate(themes, 1):
        row = (
            f"{i:2d}. {theme.representative.ljust(width)}  "
            f"{theme.report_count:3d}  ({theme.modern_count} since 2017)"
        )
        # Only call a theme dated when there were enough modern reports for its
        # absence from them to mean something (see `SicARRanking.era_confident`).
        if theme.dated and result.era_confident:
            row += "  [dated]"
        lines.append(row)

    notes = []
    if result.widened:
        notes.append(
            f"SIC {result.requested_pattern} had too little in the knowledge base on its "
            f"own, so this covers the wider {pattern} family."
        )
    if result.thin:
        notes.append(
            "Thin basis: fewer than 3 reports matched even after widening, so treat "
            "these counts as preliminary."
        )
    singles = sum(1 for t in themes if t.report_count == 1)
    if singles:
        notes.append(
            f"{singles} of these themes appear{'s' if singles == 1 else ''} in a single "
            f"report only, so {'its' if singles == 1 else 'their'} position among equal "
            f"counts is not a ranking."
        )
    if not result.era_confident:
        notes.append(
            f"Only {result.modern_reports} of these reports "
            f"{'is' if result.modern_reports == 1 else 'are'} from 2017 or later, so "
            f"the 'since 2017' counts are too small to tell current advice from "
            f"superseded advice. Check anything lighting or motor related against "
            f"today's practice before using it."
        )
    elif any(t.dated for t in themes):
        notes.append(
            "[dated] marks a theme no report since 2017 has recommended; it is common "
            "in the corpus but likely superseded."
        )
    if notes:
        lines.append("")
        lines.extend(notes)
    return "\n".join(lines)


def format_checklist_material(result: "ChecklistResult") -> str:
    """Render the pre-assessment checklist data into the reference-material block.

    Carries the basis counts and the ranked AR themes (representative title, a few
    example phrasings, and the report count) with no similarity scores, so the model
    can phrase each theme as a field action while keeping the frequencies exact.
    """
    basis_bits = []
    if result.sic_pattern:
        basis_bits.append(f"{result.n_industry} matched by SIC {result.sic_pattern}")
    if result.n_process:
        basis_bits.append(f"{result.n_process} matched by process description")
    basis = "; ".join(basis_bits) or "no clear industry or process match"

    lines = [
        "<reference_material>",
        f"Pre-assessment basis: {result.total_reports} past reports "
        f"({basis}).",
    ]
    if result.thin:
        lines.append(
            "NOTE: fewer than 3 reports matched -- mark the checklist as thin and "
            "preliminary."
        )
    if not result.themes:
        lines.append(
            "No recurring AR themes were found across the matched reports; say so "
            "plainly rather than inventing checklist items."
        )
        lines.append("</reference_material>")
        return "\n".join(lines)

    lines.append("")
    lines.append("AR themes across those reports, most frequent first:")
    for theme in result.themes[:MAX_LISTED_RESULTS]:
        examples = "; ".join(theme.examples[:4])
        lines.append(
            f"- theme: {theme.representative} | found in {theme.report_count} reports "
            f"| examples: {examples}"
        )
    lines.append("</reference_material>")
    return "\n".join(lines)


def format_bp_material(result: "BPResult") -> str:
    """Render the B3 best-practice candidates into the reference-material block.

    Groups candidates by category (in first-seen order) and gives the model, per
    item, the practice description (its already/partial/not check signal) and the
    source tag. The fill-in record prompts are deliberately NOT injected: the
    checklist is a scannable list of practice names, so feeding "(Insert ...)"
    blanks here only tempts the model to echo them. No similarity scores leak in;
    the student's own facility description arrives verbatim in the user message.
    """
    if not result.candidates:
        return (
            "<reference_material>\n"
            "No candidate best practices were found for this facility. Say so plainly "
            "rather than inventing checklist items.\n"
            "</reference_material>"
        )

    basis_bits = []
    if result.sic_pattern and result.precedent_reports:
        basis_bits.append(
            f"{result.precedent_reports} comparable facilities matched by SIC "
            f"{result.sic_pattern}"
        )
    if result.used_semantic_fallback:
        basis_bits.append("semantic match on the facility description")
    basis = "; ".join(basis_bits) or "the facility description"

    lines = [
        "<reference_material>",
        f"Best-practice basis: {basis}. Cross-check each candidate below against the "
        "facility description in the user's message.",
    ]

    # The pool can hold dozens of items (every AR of every comparable facility maps to
    # a practice), and the model lists what it is given. Keep the MAX_LISTED_RESULTS
    # strongest by their existing similarity score, precedent ahead of semantic, then
    # restore reference order so the category grouping below reads as before.
    ranked = sorted(
        range(len(result.candidates)),
        key=lambda i: (
            result.candidates[i].source == "precedent",
            result.candidates[i].score,
        ),
        reverse=True,
    )
    kept = [result.candidates[i] for i in sorted(ranked[:MAX_LISTED_RESULTS])]

    # Group by category, preserving the order categories first appear.
    by_category: dict[str, list] = {}
    for cand in kept:
        by_category.setdefault(cand.category, []).append(cand)

    for category, items in by_category.items():
        lines.append("")
        lines.append(f"Category: {category}")
        for cand in items:
            lines.append(f"- {cand.name} ({cand.source}): {cand.template}")
    lines.append("</reference_material>")
    return "\n".join(lines)


def render_bp_writeup_answer(items: "list[BPCandidate] | BPCandidate") -> str:
    """Deterministically render the BP write-up (NOT via the LLM).

    A write-up request ("write me the best practices for LEDs, skylights") is a pure
    lookup: the answer IS the official literature for each matched practice. Emit each
    practice's verbatim template under its name as a heading, keeping any "(Insert ...)"
    on-site blanks, and nothing else. Rendering it here rather than handing it to the
    model guarantees the clean per-heading literature the request asks for, with no
    preamble, paraphrase, assumptions, or added commentary.
    """
    if not isinstance(items, list):
        items = [items]
    blocks: list[str] = []
    for item in items[:MAX_LISTED_RESULTS]:
        block = f"## {item.name}\n{item.template}".rstrip()
        if item.fill_ins:
            block += "\n" + "\n".join(f"- {f}" for f in item.fill_ins)
        blocks.append(block)
    return "\n\n".join(blocks)


def format_bp_writeup_material(items: "list[BPCandidate] | BPCandidate") -> str:
    """Render one or more BP items' official templates into the write-up block.

    Accepts a single candidate or a list (a multi-topic "write up LEDs, VFDs and
    sludge" request drafts several at once). Unlike the checklist, the fill-in prompts
    ARE handed to the model here: a write-up is a fill-in-the-blanks template the
    student completes on-site, so the "(Insert ...)" placeholders belong in the
    drafted text.
    """
    if not isinstance(items, list):
        items = [items]
    items = items[:MAX_LISTED_RESULTS]

    lines = ["<reference_material>"]
    if len(items) > 1:
        lines.append(
            f"Official ITAC best-practice templates for the {len(items)} practices the "
            "user asked you to write up. Draft each one from its template below."
        )
    for item in items:
        fill_ins = "; ".join(item.fill_ins) if item.fill_ins else "none"
        lines.append(
            f'Official ITAC best-practice template for "{item.name}" '
            f"(category: {item.category}).\n"
            f"Template: {item.template}\n"
            f"Fill-in placeholders to keep in the write-up: {fill_ins}"
        )
    lines.append("</reference_material>")
    return "\n".join(lines)

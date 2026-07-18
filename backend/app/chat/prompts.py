"""System prompt templates and assembly (spec §3a, §5, §5.3).

`assemble_system_prompt` layers the prompt in the spec §5 order:
  a. Base system prompt (ITAC identity, accuracy rules, no-em-dash)
  b. Full markdown body of each selected skill
  c. Retrieved reference material (RAG) -- stubbed until step 4
  d. Output-mode instructions (prose vs TableSpec) -- table mode stubbed until step 3 xlsx
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..rag.retrieve import Retrieval
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


# ARC-lookup output instruction. Same terse spirit as enumerate mode: the user
# wants the code(s), not an essay. Scores stay in the sources payload for the UI;
# they never appear in the chat text.
ARC_LOOKUP_INSTRUCTION = """\
OUTPUT MODE: ARC LOOKUP.
The reference material lists candidate ARC codes for the recommendation the user
described. Your entire reply is those codes, nothing else:
- One line per candidate, exactly: <code> <DESCRIPTION>, best match first, in the
  order given. Reproduce each code and description verbatim.
- If the reference material says the match is NOT confident, begin with this exact
  line first: No confident ARC match. Closest candidates (verify manually):
  then the code lines.
- No introduction, no closing paragraph, no scores, and no relevance or ranking
  commentary such as "closest" or "most relevant" beyond the fixed line above."""


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
    arc_mode: bool = False,
) -> str:
    """Compose the full system prompt in spec §5 order.

    `reference_material` (RAG chunks) is stubbed for now: RAG lands in step 4.
    Table-mode output instructions (spec §5.3d) are appended when the router set
    `wants_table`; prose is the default. The special output modes are mutually
    exclusive contracts; precedence is arc > enumerate > table when more than one
    is somehow set.
    """
    parts = [BASE_SYSTEM_PROMPT]

    if skills:
        parts.append(_render_skills(skills))

    # (c) Retrieved reference material -- populated once RAG retrieval exists.
    if reference_material:
        parts.append(reference_material)

    # (d) Output-mode instructions. Prose is the default; table mode switches the
    # model to emitting a ```tablespec JSON block the xlsx engine renders.
    if arc_mode:
        parts.append(ARC_LOOKUP_INSTRUCTION)
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


def format_arc_material(retrieval: "Retrieval", *, top_n: int = 3) -> str:
    """Render ARC-lookup candidates into the reference-material block.

    Lists the best `top_n` candidate codes (code + official description) with no
    scores. Whether the top hit clears the weakness floor (`retrieval.has_match`)
    is stated so the model knows to prepend the "not confident" line and mark the
    codes as guesses.
    """
    candidates = retrieval.hits[:top_n]
    if not candidates:
        return (
            "<reference_material>\n"
            "No ARC codes were retrieved for this recommendation. Say there is no "
            "matching ARC code; do not invent one.\n"
            "</reference_material>"
        )

    if retrieval.has_match:
        directive = "Confident ARC match. Emit these candidate codes, best first:"
    else:
        directive = (
            "No confident ARC match. Emit the fixed not-confident line, then these "
            "candidates as guesses, best first:"
        )
    lines = ["<reference_material>", directive, ""]
    for _text, meta, _score in candidates:
        lines.append(f"- {meta.get('code', '?')} {meta.get('description', '')}".rstrip())
    lines.append("</reference_material>")
    return "\n".join(lines)

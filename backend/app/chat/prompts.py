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
) -> str:
    """Compose the full system prompt in spec §5 order.

    `reference_material` (RAG chunks) is stubbed for now: RAG lands in step 4.
    Table-mode output instructions (spec §5.3d) are appended when the router set
    `wants_table`; prose is the default.
    """
    parts = [BASE_SYSTEM_PROMPT]

    if skills:
        parts.append(_render_skills(skills))

    # (c) Retrieved reference material -- populated once RAG retrieval exists.
    if reference_material:
        parts.append(reference_material)

    # (d) Output-mode instructions. Prose is the default; table mode switches the
    # model to emitting a ```tablespec JSON block the xlsx engine renders.
    if wants_table:
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

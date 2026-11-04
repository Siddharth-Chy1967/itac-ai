"""Read one grouped bill: page text, page images, one model call, verification.

    PDF --page_text--> text + boxes per page (axis labels removed)   deterministic
        --render-----> one PNG per page                             deterministic
    images + text --provider.extract--> BillFigures                 the model
    BillFigures + page text --verify--> VerifiedBill                deterministic

The cache holds exactly what the model returned, keyed by the bill's sha256 and
the provider+model label, so answers from different models never mix and a
re-run is repeatable. Verification is re-applied on every read, never cached, so
a stricter check applies to old answers too.
"""
from __future__ import annotations

import json
import logging
import re
from decimal import Decimal
from pathlib import Path
from typing import Sequence

from .bill_schema import BillFigures, Charge, LineVerdicts
from .figures_prompt import (
    FIGURES_INSTRUCTIONS, FIGURES_REQUEST, RECHECK_INSTRUCTIONS, RECHECK_REQUEST,
)
from .llm_providers import OllamaProvider, Provider
from .local_inference import LocalInferenceConfig, image_side_for
from .page_text import PageText, read_pdf
from .verify import VerifiedBill, single_number, verify

logger = logging.getLogger(__name__)

# Bumped when the schema or FIGURES_INSTRUCTIONS change meaningfully.
CACHE_VERSION = "bill_figures_v2"
MAX_OUTPUT_TOKENS = 8192


RECHECK_VERSION = "bill_recheck_v4"
# A dollar amount as bills print a charge: cents, not a rate's many decimals.
_MONEY = re.compile(r"\$\s*-?\(?\s*(\d{1,3}(?:,\d{3})*|\d+)\.(\d{2})(?!\d)")


def cache_path(cache_root: Path, fingerprint: str, label: str,
               version: str = CACHE_VERSION) -> Path:
    return Path(cache_root) / "extraction_cache" / fingerprint / label / f"{version}.json"


def model_text(pages: Sequence[PageText]) -> str:
    """The page text the model is given, one headed block per page."""
    blocks = [FIGURES_REQUEST]
    for page in pages:
        text = page.text.strip() or "(no text could be read from this page)"
        blocks.append(f"=== Page {page.number} of {len(pages)} "
                      f"(text from {page.source}) ===\n{text}")
    return "\n\n".join(blocks)


def render_pages(pdf_path: Path, longest_side: int, only: set[int] | None = None) -> list[bytes]:
    """PNG renders of the pages, in order; `only` limits it to those page numbers."""
    import pymupdf

    images = []
    with pymupdf.open(pdf_path) as document:
        for number, page in enumerate(document, start=1):
            if only is not None and number not in only:
                continue
            rect = page.rect
            scale = longest_side / max(rect.width, rect.height, 1)
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale),
                                     colorspace=pymupdf.csRGB, alpha=False)
            images.append(pixmap.tobytes("png"))
    return images


def _image_side(provider: Provider, page_count: int) -> int:
    if isinstance(provider, OllamaProvider):
        return image_side_for(page_count, provider.inference)  # VRAM budget
    return provider.image_side


def pages_to_show(provider: Provider, pages: Sequence[PageText], policy: str = "auto") -> set[int]:
    """Which page images the model is sent (see Settings.bills_llm_images)."""
    if policy == "never":
        return set()
    if policy == "always" or not isinstance(provider, OllamaProvider):
        return {page.number for page in pages}
    return {page.number for page in pages if page.source != "PDF text layer"}


def read_bill(pdf_path: Path, fingerprint: str, cache_root: Path, provider: Provider,
              inference: LocalInferenceConfig, *,
              tolerance: Decimal = Decimal("0.02"),
              images: str = "auto") -> tuple[VerifiedBill, str]:
    """(verified bill, "cached" | "extracted")."""
    pages = read_pdf(Path(pdf_path), inference)
    renders: list[bytes] | None = None

    def ask(schema, text, version, instructions=FIGURES_INSTRUCTIONS, with_images=True):
        """The model's answer, from the cache when this model already gave one."""
        nonlocal renders
        path = cache_path(cache_root, fingerprint, provider.cache_label, version)
        if path.is_file():
            return schema.model_validate_json(path.read_text(encoding="utf-8")), True
        provider.ensure_ready()
        if renders is None and with_images:
            shown = pages_to_show(provider, pages, images)
            renders = render_pages(Path(pdf_path), _image_side(provider, len(shown) or 1), shown)
        answer = schema.model_validate(provider.extract(
            renders if with_images else [], text, schema, instructions=instructions,
            max_output_tokens=MAX_OUTPUT_TOKENS))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(answer.model_dump(mode="json"), indent=2), encoding="utf-8")
        return answer, False

    figures, cached = ask(BillFigures, model_text(pages), CACHE_VERSION)
    status = "cached" if cached else "extracted"
    verified = verify(figures, pages, tolerance=tolerance)
    if verified.status == "review":
        verified = _recheck(verified, figures, pages, ask, tolerance)
    return verified, status


def unreported_lines(pages: Sequence[PageText], verified: VerifiedBill,
                     limit: int = 30) -> list[tuple[int, str]]:
    """Printed lines carrying a dollar amount the model has not reported.

    Quoted back to the model on a recheck so it can find a line it skipped. Only
    page text is quoted; the model still decides which of them are charges, and
    anything it returns is verified like any other answer.
    """
    known = {abs(a) for c in verified.figures.charges
             if (a := single_number(c.amount)) is not None}
    for value in (verified.figures.total_current_charges, verified.figures.amount_due):
        if value is not None and (a := single_number(value.text)) is not None:
            known.add(a)
    found = []
    for page in pages:
        for line in page.text.splitlines():
            amounts = {Decimal(m.group(1).replace(",", "") + "." + m.group(2))
                       for m in _MONEY.finditer(line)}
            if amounts and not amounts <= known and re.search(r"[A-Za-z]{3}", line):
                found.append((page.number, line.strip()))
    return found[:limit]


def _recheck(verified: VerifiedBill, figures: BillFigures, pages, ask, tolerance) -> VerifiedBill:
    """One narrow second question when the charge lines do not add up.

    The page's own lines that print a dollar amount nobody has accounted for are
    quoted back, and the model says which of them are charge lines. It is a much
    smaller task than reading the bill, which is why a small local model can do
    it when it skipped the line the first time. What it returns is added to the
    first reading and verified like everything else -- so a recheck can recover a
    printed line but cannot invent one -- and kept only if the bill then
    reconciles. A bill whose lines add up to too much is not rechecked:
    removing lines is a judgement the review gate should see.
    """
    difference = verified.charges_sum - verified.total
    candidates = unreported_lines(pages, verified)
    if difference > 0 or not candidates:
        return verified
    request = RECHECK_REQUEST.format(
        found=f"${verified.charges_sum:,.2f}", total=f"${verified.total:,.2f}",
        difference=f"${abs(difference):,.2f}",
        unreported="\n".join(f"{i}. {line}" for i, (_, line) in enumerate(candidates, 1)))
    try:
        review, _ = ask(LineVerdicts, request, RECHECK_VERSION,
                        instructions=RECHECK_INSTRUCTIONS, with_images=False)
    except Exception as error:  # the first reading is still a valid, flagged result
        logger.warning("Recheck failed: %s", error)
        verified.flags.append(f"Recheck of the charge lines failed ({type(error).__name__}).")
        return verified
    # A total or the amount due is never a charge line, whatever the model says.
    totals = {single_number(v.text) for v in (verified.figures.total_current_charges,
                                               verified.figures.amount_due) if v is not None}
    have = {(compact(c.label), single_number(c.amount)) for c in verified.figures.charges}
    added = []
    for verdict in review.verdicts:
        if not verdict.is_charge_line or not 1 <= verdict.line <= len(candidates):
            continue
        amount = single_number(verdict.amount)
        if amount is None or amount in totals or (compact(verdict.label), amount) in have:
            continue
        added.append(Charge(label=verdict.label, amount=verdict.amount,
                            category=verdict.category, period="none",
                            page=candidates[verdict.line - 1][0]))
    revised = verify(verified.figures.model_copy(update={
        "charges": [*verified.figures.charges, *added]}), pages, tolerance=tolerance)
    # Kept only when the bill then reconciles. Moving closer is not enough: a
    # notice such as "$1.00 fee applies" can shrink a gap without belonging in it.
    if added and revised.status == "matched":
        names = ", ".join(f"{c.label} {c.amount}" for c in revised.figures.charges[
            len(verified.figures.charges):])
        # The first reading's flags stand, except its line-items total, which the
        # revised check replaces.
        revised.flags = [*verified.flags[:-1], f"Recheck recovered: {names}.", *revised.flags]
        return revised
    verified.flags.append("Recheck found no printed charge line that reconciles the bill.")
    return verified


def compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").casefold())

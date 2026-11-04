"""End-to-end batch: uploaded PDFs -> grouped bills -> columns -> workbook.

    group each source PDF into individual bills   (bill_extraction; the model)
    read each bill                                 (figures; the model, cached)
    keep only figures the page prints              (verify; deterministic)
    name the columns from the model's types        (columns; deterministic)
    -> REVIEW GATE: the proposed columns are shown before anything is written
    pivot into per-meter monthly sheets            (shape)
    render the workbook                            (workbook)

The model does the understanding -- what each figure is, its unit, whether it is
a charge or a reading. The code never adds a figure: it only removes what it
cannot find printed on the page, and flags what does not add up.

Which model runs is configuration (BILLS_LLM_PROVIDER / BILLS_LLM_MODEL in .env),
not code; see `llm_providers`.
"""
from __future__ import annotations

import logging
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

from ..config import get_settings
from . import bill_extraction, bill_value_extraction as bve, columns, figures
from .jobs import Job, JobState
from .llm_providers import grouping_client, make_provider
from .local_inference import LocalInferenceConfig
from .semantics import describe_layout
from .shape import build_rollup, build_sheets
from .workbook import write_analysis_workbook

logger = logging.getLogger(__name__)


def run_batch(job: Job, *, settings=None, stop_for_review: bool = True, provider=None) -> None:
    """Process one job. Runs on a worker thread; mutates `job` as it goes."""
    settings = settings or get_settings()
    workspace = job.directory
    # Model output is cached outside the job folder, so deleting a job to reclaim
    # disk does not throw away the analyses. See Settings.bills_cache_path.
    # Resolved the same way `api.bills.get_store` resolves `bills_path`, so the
    # cache always lands beside the job folders rather than in a second place.
    cache_root = Path(settings.bills_cache_path).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    inference = LocalInferenceConfig.from_settings(settings)
    provider = provider or make_provider(settings)
    logger.info("Bill inference: %s", provider.describe())

    job.touch(JobState.EXTRACTING, "Reading uploaded PDFs.")
    grouped: list[Path] = []

    for progress in job.files:
        if job.cancelled:
            job.touch(JobState.CANCELLED, "Cancelled during extraction.")
            return
        progress.state = "grouping"
        job.touch()
        try:
            # One source PDF can hold a year of bills; grouping splits it.
            result = bill_extraction.process_source_pdf(
                progress.path, project_folder=workspace, inference_config=inference,
                cache_folder=cache_root,
                client_factory=lambda: grouping_client(provider, inference),
                model_label=provider.cache_label,
            )
            bills = _grouped_paths(result)
            grouped.extend(bills)
            progress.bills = len(bills)
            progress.state = "done" if bills else "skipped"
            if not bills:
                progress.message = "No complete bill could be grouped from this PDF."
        except Exception as error:  # one bad file must not lose the other 39
            logger.exception("Grouping failed for %s", progress.path)
            progress.state = "error"
            progress.message = f"{type(error).__name__}: {error}"
        job.touch()

    if not grouped:
        job.touch(JobState.FAILED, "No bills could be read from the uploaded PDFs.")
        return

    job.touch(JobState.EXTRACTING, f"Extracting values from {len(grouped)} bill(s).")
    verified = _extract_bills(job, grouped, cache_root, provider, inference, settings)
    if job.cancelled:
        job.touch(JobState.CANCELLED, "Cancelled during value extraction.")
        return
    if not verified:
        job.touch(JobState.FAILED, "No bill could be extracted.")
        return

    job.bills, job.mapping = columns.build(verified)
    job.layout = describe_layout(job.mapping)

    if stop_for_review:
        job.touch(JobState.AWAITING_REVIEW,
                  f"{len(job.layout)} column(s) proposed from {len(job.bills)} bill(s). "
                  "Confirm the layout to build the workbook.")
        return

    build_workbook(job, settings=settings)


def _grouped_paths(result) -> list[Path]:
    """`process_source_pdf` returns the grouped bill PDFs it actually wrote."""
    return [Path(p) for p in (result or []) if str(p).lower().endswith(".pdf")]


def _extract_bills(job: Job, grouped, cache_root: Path, provider, inference, settings):
    """(bill id, VerifiedBill) for every bill that could be read."""
    bills = []
    job.bills_total = len(grouped)
    job.bills_done = 0
    tolerance = Decimal(str(settings.bills_amount_tolerance))
    for path in grouped:
        if job.cancelled:
            return bills
        job.touch(message=f"Extracting bill {job.bills_done + 1} of {job.bills_total}: "
                          f"{Path(path).name}.")
        fingerprint = sha256(Path(path).read_bytes()).hexdigest()
        try:
            verified, _ = figures.read_bill(Path(path), fingerprint, cache_root, provider,
                                            inference, tolerance=tolerance,
                                            images=settings.bills_llm_images)
        except Exception as error:
            logger.exception("Extraction failed for %s", path)
            job.warnings.append(f"{Path(path).name}: {type(error).__name__}: {error}")
            # A bill that failed is still a bill accounted for, so the bar advances
            # rather than stalling at the one that broke.
            job.bills_done += 1
            job.touch()
            continue
        # Every figure the check removed is reported: a blank cell must be explainable.
        job.warnings.extend(f"{Path(path).name}: {flag}" for flag in verified.flags)
        bills.append((fingerprint[:12], verified))
        job.bills_done += 1
        job.touch()
    usage = getattr(provider, "usage", None)
    if usage is not None and usage.calls:
        logger.info("Bill model usage: %d call(s), %d input / %d output tokens",
                    usage.calls, usage.input_tokens, usage.output_tokens)
    return bills


def build_workbook(job: Job, *, settings=None) -> Path:
    """Pivot the extracted bills and write the workbook. Safe to re-run."""
    settings = settings or get_settings()
    job.touch(JobState.BUILDING, "Building the analysis workbook.")

    job.plans = build_sheets(
        job.bills, job.mapping,
        parse_amount=bve.parse_amount, parse_number=bve.parse_number,
        tolerance=Decimal(str(settings.bills_amount_tolerance)),
        derive_columns=settings.bills_derived_columns,
    )
    rollup = build_rollup(job.plans)
    destination = job.directory / "Utility Analysis.xlsx"
    write_analysis_workbook(job.plans, rollup, destination)
    job.workbook = destination

    flagged = sum(1 for plan in job.plans for row in plan.rows if row.status == "review")
    summary = f"{len(job.plans)} sheet(s) from {len(job.bills)} bill(s)."
    if flagged:
        summary += (f" {flagged} month(s) did not reconcile against their printed "
                    "total and are highlighted for review.")
    # Name the file's location in the message itself. A batch runs long enough that
    # the tab is often closed by now, and the browser's own download is the only
    # other way anyone would find it.
    summary += f" Saved to {destination}."
    job.touch(JobState.DONE, summary)
    return destination

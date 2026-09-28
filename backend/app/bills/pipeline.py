"""End-to-end batch: uploaded PDFs -> grouped bills -> columns -> workbook.

This is the sequence `main.py` used to drive interactively, with the prompts
replaced by job state so a browser can drive it instead:

    group each source PDF into individual bills   (bill_extraction)
    extract every bill's values                   (bill_value_extraction, cached)
    interpret the labels those bills printed       (semantics)
    -> REVIEW GATE: the proposed columns are shown before anything is written
    pivot into per-meter monthly sheets            (shape)
    render the workbook                            (workbook)

The review gate is the point. Column layout is the one part a model influences,
and showing it before the workbook is written makes any mistake a two-second fix
rather than a wrong spreadsheet someone forwards to a client.
"""
from __future__ import annotations

import logging
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

from ..config import get_settings
from . import bill_extraction, bill_value_extraction as bve, grid
from .jobs import Job, JobState
from .local_inference import LocalInferenceClient, LocalInferenceConfig
from .semantics import (
    LabelSemantics, describe_layout, interpret_labels, labels_from_bill,
)
from .shape import build_rollup, build_sheets
from .validation_checks import name_rows_from_sections
from .workbook import write_analysis_workbook

logger = logging.getLogger(__name__)


def _client_factory(settings):
    """One lazily-built client per batch: the model loads on first real use."""
    holder: dict[str, LocalInferenceClient] = {}

    def get_client():
        if "client" not in holder:
            config = LocalInferenceConfig.from_settings(settings)
            logger.info("Bill inference: %s", config.describe())
            holder["client"] = LocalInferenceClient(config)
        return holder["client"]

    return get_client


def run_batch(job: Job, *, settings=None, stop_for_review: bool = True) -> None:
    """Process one job. Runs on a worker thread; mutates `job` as it goes."""
    settings = settings or get_settings()
    workspace = job.directory
    get_client = _client_factory(settings)

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
                progress.path, project_folder=workspace,
                inference_config=LocalInferenceConfig.from_settings(settings),
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
    job.bills = _extract_bills(job, grouped, workspace, get_client, settings)
    if job.cancelled:
        job.touch(JobState.CANCELLED, "Cancelled during value extraction.")
        return
    if not job.bills:
        job.touch(JobState.FAILED, "No bill could be extracted.")
        return

    job.mapping = _interpret(job, get_client)
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


def _extract_bills(job: Job, grouped, workspace: Path, get_client, settings):
    bills = []
    job.bills_total = len(grouped)
    job.bills_done = 0
    for path in grouped:
        if job.cancelled:
            return bills
        job.touch(message=f"Extracting bill {job.bills_done + 1} of {job.bills_total}: "
                          f"{Path(path).name}.")
        fingerprint = sha256(Path(path).read_bytes()).hexdigest()
        try:
            bill, _, _ = bve.load_bill(
                Path(path), fingerprint, workspace, bve.EXTRACTION_MODEL,
                get_client=get_client,
            )
        except Exception as error:
            logger.exception("Extraction failed for %s", path)
            job.warnings.append(f"{Path(path).name}: {type(error).__name__}: {error}")
            # A bill that failed is still a bill accounted for, so the bar advances
            # rather than stalling at the one that broke.
            job.bills_done += 1
            job.touch()
            continue
        bill = name_rows_from_sections(bill)
        bills.append((fingerprint[:12], _with_printed_readings(job, bill, Path(path), settings)))
        job.bills_done += 1
        job.touch()
    return bills


def _with_printed_readings(job: Job, bill, path: Path, settings):
    """Label the bill's readings as the page prints them (see `grid`).

    Applied after the cache, so extractions made before the grid reader existed
    benefit too, and the cache itself keeps exactly what the model returned. A
    failure here costs the correction, never the bill.
    """
    config = LocalInferenceConfig.from_settings(settings)
    if not config.ocr_enabled:
        return bill
    try:
        return grid.reconcile(bill, grid.printed_rows(path, config))
    except Exception as error:
        logger.exception("Printed-row check failed for %s", path)
        job.warnings.append(f"{path.name}: readings not checked against the page "
                            f"({type(error).__name__}); using the model's labels.")
        return bill


def _interpret(job: Job, get_client) -> dict[str, LabelSemantics]:
    """Interpret the labels. A model failure degrades to rules, never to a crash."""
    labels = [label for _, bill in job.bills for label in labels_from_bill(bill)]
    try:
        client = get_client()
    except Exception as error:
        logger.warning("No inference client for label interpretation: %s", error)
        job.warnings.append(
            "Column names were derived from built-in vocabulary only "
            f"({type(error).__name__}). Review the layout before exporting."
        )
        client = None
    return interpret_labels(labels, client=client)


def build_workbook(job: Job, *, settings=None) -> Path:
    """Pivot the extracted bills and write the workbook. Safe to re-run."""
    settings = settings or get_settings()
    job.touch(JobState.BUILDING, "Building the analysis workbook.")

    job.plans = build_sheets(
        job.bills, job.mapping,
        parse_amount=bve.parse_amount, parse_number=bve.parse_number,
        tolerance=Decimal(str(settings.bills_amount_tolerance)),
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

"""Utility-bill upload API.

    POST /api/bills/upload            multipart PDFs  -> job id
    GET  /api/bills                   recent jobs
    GET  /api/bills/{id}              status, progress, flagged months
    GET  /api/bills/{id}/columns      the proposed layout, for the review gate
    POST /api/bills/{id}/columns      accept or override it, then build
    GET  /api/bills/{id}/download     the finished workbook
    POST /api/bills/{id}/cancel       stop before the next file
    DELETE /api/bills/{id}            forget the job and delete its uploads

Upload returns immediately. Extraction runs on a single background worker (see
`bills.jobs`), so the browser polls status rather than holding a request open for
the minutes a batch really takes.

Note nginx: bulk PDF uploads need `client_max_body_size` raised well above its 1MB
default or requests are rejected before FastAPI ever sees them.
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..bills.jobs import JobError, JobState, JobStore
from ..bills.pipeline import build_workbook, run_batch
from ..bills.semantics import SemanticsError, apply_override, describe_layout
from ..config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()

XLSX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
MAX_FILES = 200
MAX_BYTES = 200 * 1024 * 1024  # per file; a scanned year of bills is a few MB

_store: JobStore | None = None


def get_store() -> JobStore:
    """The process-wide job store, created against `bills_path` on first use."""
    global _store
    if _store is None:
        root = Path(get_settings().bills_path).resolve()
        _store = JobStore(root, run_batch)
    return _store


def reset_store() -> None:
    """Drop the store so the next call rebuilds it. Used by tests."""
    global _store
    _store = None


def _job(job_id: str):
    try:
        return get_store().get(job_id)
    except JobError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


class ColumnOverride(BaseModel):
    label: str = Field(description="The raw label key being re-filed")
    canonical: str | None = None
    group: str | None = None
    peak_period: str | None = None
    sign: str | None = None
    unit: str | None = None


class ColumnDecision(BaseModel):
    overrides: list[ColumnOverride] = Field(default_factory=list)


@router.post("/bills/upload")
async def upload_bills(files: list[UploadFile] = File(...)) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400,
                            detail=f"At most {MAX_FILES} files per upload.")

    payload = []
    for upload in files:
        name = upload.filename or "upload.pdf"
        if not name.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{name} is not a PDF.")
        data = await upload.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"{name} is empty.")
        if len(data) > MAX_BYTES:
            raise HTTPException(status_code=413, detail=f"{name} is too large.")
        payload.append((name, data))

    store = get_store()
    job = store.create(payload)
    await store.submit(job)
    logger.info("Bill job %s queued with %d file(s)", job.id, len(payload))
    return job.as_dict()


@router.get("/bills")
async def list_jobs() -> dict:
    return {"jobs": [job.as_dict() for job in get_store().list()]}


@router.get("/bills/{job_id}")
async def job_status(job_id: str) -> dict:
    return _job(job_id).as_dict()


@router.get("/bills/{job_id}/columns")
async def proposed_columns(job_id: str) -> dict:
    job = _job(job_id)
    return {"id": job.id, "state": job.state.value, "columns": job.layout,
            "warnings": job.warnings}


@router.post("/bills/{job_id}/columns")
async def confirm_columns(job_id: str, decision: ColumnDecision) -> dict:
    """Accept the layout -- with any corrections -- and build the workbook.

    Overrides are applied and re-validated together, so a change that would make
    two columns ambiguous is refused before anything is written.
    """
    job = _job(job_id)
    if job.state not in (JobState.AWAITING_REVIEW, JobState.DONE):
        raise HTTPException(
            status_code=409,
            detail=f"Job is {job.state.value}; no layout is waiting for review.",
        )

    mapping = job.mapping
    for override in decision.overrides:
        changes = override.model_dump(exclude={"label"}, exclude_none=True)
        if not changes:
            continue
        try:
            mapping = apply_override(mapping, override.label, **changes)
        except SemanticsError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    job.mapping = mapping
    job.layout = describe_layout(mapping)
    try:
        build_workbook(job)
    except Exception as error:
        logger.exception("Workbook build failed for job %s", job_id)
        job.touch(JobState.FAILED, f"{type(error).__name__}: {error}")
        raise HTTPException(status_code=500, detail=str(error)) from error
    return job.as_dict()


@router.get("/bills/{job_id}/download")
async def download_workbook(job_id: str) -> FileResponse:
    job = _job(job_id)
    if job.workbook is None or not job.workbook.is_file():
        raise HTTPException(status_code=404, detail="No workbook has been built yet.")
    return FileResponse(job.workbook, media_type=XLSX_MEDIA_TYPE,
                        filename="Utility Analysis.xlsx")


@router.post("/bills/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict:
    job = _job(job_id)
    job.cancel()
    if job.state == JobState.QUEUED:
        # Nothing has started, so it can be settled now rather than at the worker.
        job.touch(JobState.CANCELLED, "Cancelled before processing started.")
    return job.as_dict()


@router.delete("/bills/{job_id}")
async def delete_job(job_id: str) -> dict:
    _job(job_id)
    get_store().delete(job_id)
    return {"deleted": job_id}

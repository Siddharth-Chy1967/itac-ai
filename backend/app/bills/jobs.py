"""Serial job runner for utility-bill uploads.

Processing a bill is slow (page render, OCR, then a vision model over every page)
and it is jealous of the GPU: one Ollama instance holds one model in VRAM, so two
bills processed at once do not go twice as fast -- they contend, and on a smaller
card they fail. Everything therefore runs through **one worker, strictly in
order**, and the HTTP layer never waits for it: upload returns a job id, and the
browser polls.

Design points that matter more than they look:

* **One file's failure is that file's failure.** A 40-bill batch where bill 7 is a
  scanned fax must still deliver the other 39, so every file is processed inside
  its own try/except and records its own error.
* **Progress is observable.** The extractor's own pipeline is a blind loop; the
  runner wraps each file so a UI can show which one is in flight without waiting
  for the batch.
* **Cancellation is cooperative.** A cancelled job stops before its next file
  rather than being killed mid-write, so caches and workbooks are never truncated.
* **State lives in memory, work lives on disk.** Uploads, caches and workbooks are
  written under `bills_path`, so a restart loses the job list but never the
  expensive part: re-running a batch re-reads its caches and skips the model.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable

logger = logging.getLogger(__name__)


class JobState(str, Enum):
    QUEUED = "queued"
    EXTRACTING = "extracting"
    AWAITING_REVIEW = "awaiting_review"
    BUILDING = "building"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL = {JobState.DONE, JobState.FAILED, JobState.CANCELLED}


class JobError(Exception):
    """Raised for a request that cannot be served, e.g. an unknown job id."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class FileProgress:
    """One uploaded PDF's journey through the batch."""

    name: str
    path: Path
    state: str = "queued"  # queued | grouping | extracting | done | error | skipped
    bills: int = 0
    message: str = ""

    def as_dict(self) -> dict:
        return {"name": self.name, "state": self.state, "bills": self.bills,
                "message": self.message}


@dataclass
class Job:
    id: str
    directory: Path
    files: list[FileProgress] = field(default_factory=list)
    state: JobState = JobState.QUEUED
    message: str = ""
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    # Filled in as the batch proceeds.
    bills: list = field(default_factory=list)          # (bill_id, BillExtraction)
    mapping: dict = field(default_factory=dict)        # label key -> LabelSemantics
    layout: list = field(default_factory=list)         # review-gate rows
    plans: list = field(default_factory=list)          # SheetPlan
    workbook: Path | None = None
    warnings: list[str] = field(default_factory=list)

    # Value extraction is the long phase and counts bills, not uploaded files: one
    # PDF can hold a year of them, so file counts sit at 100% for the whole wait.
    bills_total: int = 0
    bills_done: int = 0

    _cancelled: bool = False

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        self._cancelled = True

    def touch(self, state: JobState | None = None, message: str | None = None) -> None:
        if state is not None:
            self.state = state
        if message is not None:
            self.message = message
        self.updated_at = _now()

    @property
    def progress(self) -> tuple[int, int, str]:
        """How far along the batch is, counted in whatever it is currently doing.

        Grouping advances per uploaded file; value extraction, which is the part
        that actually takes the time, advances per bill. Reporting the unit lets
        the browser say "3 of 12 bills" instead of showing a full bar for the
        thirty-five minutes after the last file was grouped.
        """
        if self.bills_total:
            return self.bills_done, self.bills_total, "bills"
        done = sum(1 for f in self.files if f.state in ("done", "error", "skipped"))
        return done, len(self.files), "files"

    def as_dict(self) -> dict:
        done, total, unit = self.progress
        rows = [row for plan in self.plans for row in plan.rows]
        return {
            "id": self.id,
            "state": self.state.value,
            "message": self.message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "files": [f.as_dict() for f in self.files],
            "progress": {"done": done, "total": total, "unit": unit},
            "bills": len(self.bills),
            "sheets": [
                {
                    "title": plan.title,
                    "rows": len(plan.rows),
                    "columns": [c.canonical for c in plan.columns],
                    "warnings": plan.warnings,
                }
                for plan in self.plans
            ],
            "flagged": [
                {
                    "sheet": plan.title,
                    "month": row.label,
                    "difference": str(row.difference) if row.difference is not None else None,
                    "notes": row.notes,
                }
                for plan in self.plans for row in plan.rows if row.status == "review"
            ],
            "unchecked": sum(1 for row in rows if row.status == "not_checked"),
            "warnings": self.warnings,
            "has_workbook": self.workbook is not None and self.workbook.is_file(),
            # Shown next to the download button: the batch runs for long enough that
            # people close the tab, and the file is then only findable on disk.
            "workbook_path": str(self.workbook) if self.workbook is not None else None,
        }


class JobStore:
    """In-memory job registry with a single background worker.

    Serialization is the whole point: `_queue` has one consumer, so however many
    uploads arrive, exactly one batch is touching the model at a time.
    """

    def __init__(self, root: Path, runner: Callable[[Job], None]):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._runner = runner
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._queue: asyncio.Queue[str] | None = None
        self._worker: asyncio.Task | None = None

    # -- lifecycle --

    def start(self) -> None:
        """Create the queue and worker on the running loop."""
        if self._worker is not None and not self._worker.done():
            return
        self._queue = asyncio.Queue()
        self._worker = asyncio.create_task(self._consume(), name="bills-worker")

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None

    # -- registry --

    def create(self, names: Iterable[tuple[str, bytes]]) -> Job:
        job_id = uuid.uuid4().hex
        directory = self.root / job_id
        directory.mkdir(parents=True, exist_ok=True)
        job = Job(id=job_id, directory=directory)
        for index, (name, data) in enumerate(names, start=1):
            path = directory / f"{index:03d}_{_safe_name(name)}"
            path.write_bytes(data)
            job.files.append(FileProgress(name=name, path=path))
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise JobError(f"Unknown job {job_id!r}")
        return job

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def delete(self, job_id: str) -> None:
        job = self.get(job_id)
        job.cancel()
        with self._lock:
            self._jobs.pop(job_id, None)
        shutil.rmtree(job.directory, ignore_errors=True)

    async def submit(self, job: Job) -> None:
        if self._queue is None:
            self.start()
        assert self._queue is not None
        await self._queue.put(job.id)

    # -- worker --

    async def _consume(self) -> None:
        assert self._queue is not None
        while True:
            job_id = await self._queue.get()
            try:
                job = self.get(job_id)
            except JobError:
                continue  # deleted before it ran
            if job.cancelled:
                job.touch(JobState.CANCELLED, "Cancelled before processing started.")
                continue
            try:
                # The extractor is synchronous and CPU/GPU bound; a thread keeps
                # the event loop answering status polls while it runs.
                await asyncio.to_thread(self._runner, job)
            except Exception as error:  # a bad batch must not kill the worker
                logger.exception("Bill job %s failed", job_id)
                job.touch(JobState.FAILED, f"{type(error).__name__}: {error}")
            finally:
                self._queue.task_done()


_UNSAFE = str.maketrans({c: "_" for c in '<>:"/\\|?*'})


def _safe_name(name: str) -> str:
    """Keep the uploaded name recognisable without trusting it as a path."""
    cleaned = Path(str(name)).name.translate(_UNSAFE).strip() or "upload.pdf"
    if not cleaned.lower().endswith(".pdf"):
        cleaned += ".pdf"
    return cleaned[:120]

"""Upload API and job-runner tests.

No model and no network: the grouping step is stubbed to hand back synthetic
digital bills and the model is a fake provider, so the whole route -- upload,
extract, verify, review gate, build, download -- runs offline and deterministically.
"""
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch
import asyncio
import io
import unittest

from fastapi.testclient import TestClient
from openpyxl import load_workbook

from app.api import bills as bills_api
from app.bills.jobs import Job, JobState, JobStore, _safe_name
from app.main import create_app
from dataclasses import replace

from app.bills.local_inference import LocalInferenceConfig
from app.bills.page_text import read_pdf
from app.bills.verify import numbers_in
from .synthetic import FakeProvider, Month, figures_for, make_bill_pdf

NO_OCR = replace(LocalInferenceConfig(), ocr_enabled=False)

PDF_BYTES = b"%PDF-1.4\n% a stand-in for an uploaded scan\n"


def upload(client, files=None):
    files = files or [("files", ("bills.pdf", PDF_BYTES, "application/pdf"))]
    return client.post("/api/bills/upload", files=files)


class StoreTests(unittest.TestCase):
    """The runner's guarantees, tested without HTTP in the way."""

    def setUp(self):
        self.root = Path(self.enterContext(__import__("tempfile").TemporaryDirectory()))

    def test_uploads_are_written_under_the_job_directory(self):
        store = JobStore(self.root, lambda job: None)
        job = store.create([("a.pdf", b"one"), ("b.pdf", b"two")])
        self.assertTrue(job.directory.is_dir())
        self.assertEqual([f.path.read_bytes() for f in job.files], [b"one", b"two"])
        self.assertEqual([f.name for f in job.files], ["a.pdf", "b.pdf"])

    def test_upload_names_cannot_escape_the_job_directory(self):
        store = JobStore(self.root, lambda job: None)
        job = store.create([("../../etc/passwd.pdf", b"x"), ("C:\\windows\\evil.pdf", b"y")])
        for progress in job.files:
            self.assertEqual(progress.path.parent, job.directory)
            self.assertNotIn("..", progress.path.name)

    def test_safe_name_keeps_a_pdf_suffix(self):
        self.assertTrue(_safe_name("report").endswith(".pdf"))
        self.assertEqual(_safe_name(""), "upload.pdf")
        self.assertNotIn("/", _safe_name("a/b/c.pdf"))

    def test_deleting_a_job_removes_its_uploads(self):
        store = JobStore(self.root, lambda job: None)
        job = store.create([("a.pdf", b"one")])
        directory = job.directory
        store.delete(job.id)
        self.assertFalse(directory.exists())

    def test_jobs_run_one_at_a_time(self):
        """Two Ollama requests at once contend for one model in VRAM."""
        running, overlaps = [], []

        def runner(job):
            running.append(job.id)
            overlaps.append(len(running))
            job.touch(JobState.DONE)
            running.remove(job.id)

        async def scenario():
            store = JobStore(self.root, runner)
            store.start()
            jobs = [store.create([("a.pdf", b"x")]) for _ in range(5)]
            for job in jobs:
                await store.submit(job)
            for _ in range(200):
                if all(j.state == JobState.DONE for j in jobs):
                    break
                await asyncio.sleep(0.01)
            await store.stop()
            return jobs

        jobs = asyncio.run(scenario())
        self.assertTrue(all(j.state == JobState.DONE for j in jobs))
        self.assertEqual(max(overlaps), 1)

    def test_one_failing_job_does_not_kill_the_worker(self):
        def runner(job):
            if job.files[0].name == "bad.pdf":
                raise RuntimeError("boom")
            job.touch(JobState.DONE)

        async def scenario():
            store = JobStore(self.root, runner)
            store.start()
            bad = store.create([("bad.pdf", b"x")])
            good = store.create([("good.pdf", b"x")])
            await store.submit(bad)
            await store.submit(good)
            for _ in range(200):
                if good.state == JobState.DONE and bad.state == JobState.FAILED:
                    break
                await asyncio.sleep(0.01)
            await store.stop()
            return bad, good

        bad, good = asyncio.run(scenario())
        self.assertEqual(bad.state, JobState.FAILED)
        self.assertIn("boom", bad.message)
        self.assertEqual(good.state, JobState.DONE)

    def test_a_job_cancelled_before_it_runs_is_not_processed(self):
        ran = []

        async def scenario():
            store = JobStore(self.root, lambda job: ran.append(job.id))
            store.start()
            job = store.create([("a.pdf", b"x")])
            job.cancel()
            await store.submit(job)
            await asyncio.sleep(0.1)
            await store.stop()
            return job

        job = asyncio.run(scenario())
        self.assertEqual(ran, [])
        self.assertEqual(job.state, JobState.CANCELLED)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(__import__("tempfile").TemporaryDirectory()))
        bills_api.reset_store()
        patcher = patch("app.api.bills.get_settings")
        settings = patcher.start()
        self.addCleanup(patcher.stop)
        settings.return_value.bills_path = str(self.tmp)
        self.addCleanup(bills_api.reset_store)

    def client(self):
        return TestClient(create_app())

    def test_upload_requires_a_pdf(self):
        with self.client() as client:
            response = client.post(
                "/api/bills/upload",
                files=[("files", ("notes.txt", b"hello", "text/plain"))],
            )
        self.assertEqual(response.status_code, 400)
        self.assertIn("not a PDF", response.json()["detail"])

    def test_upload_rejects_an_empty_file(self):
        with self.client() as client:
            response = upload(client, [("files", ("empty.pdf", b"", "application/pdf"))])
        self.assertEqual(response.status_code, 400)

    def test_unknown_job_is_a_404(self):
        with self.client() as client:
            for path in ("/api/bills/nope", "/api/bills/nope/columns",
                         "/api/bills/nope/download"):
                with self.subTest(path=path):
                    self.assertEqual(client.get(path).status_code, 404)

    def test_upload_returns_a_job_that_can_be_polled(self):
        with patch("app.api.bills.run_batch"), self.client() as client:
            created = upload(client)
            self.assertEqual(created.status_code, 200)
            job_id = created.json()["id"]
            # Before any bill is grouped, progress is still counted in uploaded files.
            self.assertEqual(
                created.json()["progress"], {"done": 0, "total": 1, "unit": "files"}
            )

            status = client.get(f"/api/bills/{job_id}")
            self.assertEqual(status.status_code, 200)
            self.assertEqual(status.json()["id"], job_id)
            self.assertIn(status.json()["state"],
                          [s.value for s in JobState])

    def test_listing_shows_the_job(self):
        with patch("app.api.bills.run_batch"), self.client() as client:
            job_id = upload(client).json()["id"]
            listed = client.get("/api/bills").json()["jobs"]
        self.assertIn(job_id, [j["id"] for j in listed])

    def test_download_before_a_build_is_a_404(self):
        with patch("app.api.bills.run_batch"), self.client() as client:
            job_id = upload(client).json()["id"]
            self.assertEqual(
                client.get(f"/api/bills/{job_id}/download").status_code, 404)

    def test_confirming_columns_before_review_is_a_conflict(self):
        with patch("app.api.bills.run_batch"), self.client() as client:
            job_id = upload(client).json()["id"]
            response = client.post(f"/api/bills/{job_id}/columns", json={"overrides": []})
        self.assertEqual(response.status_code, 409)

    def test_cancel_and_delete(self):
        with patch("app.api.bills.run_batch"), self.client() as client:
            job_id = upload(client).json()["id"]
            cancelled = client.post(f"/api/bills/{job_id}/cancel")
            self.assertEqual(cancelled.status_code, 200)
            self.assertEqual(client.delete(f"/api/bills/{job_id}").status_code, 200)
            self.assertEqual(client.get(f"/api/bills/{job_id}").status_code, 404)


class EndToEndTests(unittest.TestCase):
    """Upload -> extract -> verify -> review gate -> build -> download, no model.

    Grouping is stubbed to hand back synthetic digital bills (see `synthetic`),
    and the model is a fake provider that answers from the page text. Page text,
    verification, column naming, shaping and rendering are all the real code.
    """

    MONTHS = [
        Month("01/16/2026"),
        Month("02/17/2026", on_kwh="22,640.000", off_kwh="19,360.000",
              total_kwh="42,000.000", demand="182.400"),
        Month("03/16/2026", on_kwh="18,240.000", off_kwh="15,120.000",
              total_kwh="33,360.000", demand="160.000"),
    ]

    def setUp(self):
        self.tmp = Path(self.enterContext(__import__("tempfile").TemporaryDirectory()))
        bills_api.reset_store()
        self.addCleanup(bills_api.reset_store)

        patcher = patch("app.api.bills.get_settings")
        settings = patcher.start()
        self.addCleanup(patcher.stop)
        settings.return_value.bills_path = str(self.tmp / "jobs")

        from app.config import get_settings
        self.settings = get_settings().model_copy(update={
            "bills_cache_path": str(self.tmp / "cache"), "bills_derived_columns": False,
        })
        settings_patch = patch("app.bills.pipeline.get_settings", return_value=self.settings)
        settings_patch.start()
        self.addCleanup(settings_patch.stop)

        source = self.tmp / "source"
        source.mkdir()
        self.pdfs = [make_bill_pdf(source, f"bill_{i:03d}.pdf", month)
                     for i, month in enumerate(self.MONTHS, start=1)]
        self.answers = {m.statement: figures_for(m) for m in self.MONTHS}
        self.provider = FakeProvider(self.answers)
        provider_patch = patch("app.bills.pipeline.make_provider",
                               side_effect=lambda settings: self.provider)
        provider_patch.start()
        self.addCleanup(provider_patch.stop)

        # Records where the pipeline asked for cached analyses to be kept, so the
        # shared-cache wiring is asserted rather than assumed.
        self.cache_folders = []
        self.model_labels = []

        def fake_group(source_pdf, *, project_folder, inference_config=None,
                       cache_folder=None, client_factory=None, model_label=None):
            self.cache_folders.append(cache_folder)
            self.model_labels.append(model_label)
            return list(self.pdfs)

        group_patch = patch("app.bills.pipeline.bill_extraction.process_source_pdf",
                            side_effect=fake_group)
        group_patch.start()
        self.addCleanup(group_patch.stop)

    def run_job(self, client, name="elec.pdf"):
        created = upload(client, [("files", (name, PDF_BYTES, "application/pdf"))])
        self.assertEqual(created.status_code, 200, created.text)
        job_id = created.json()["id"]
        for _ in range(600):
            body = client.get(f"/api/bills/{job_id}").json()
            if body["state"] in ("awaiting_review", "done", "failed", "cancelled"):
                return job_id, body
            import time
            time.sleep(0.02)
        self.fail("job did not settle")

    def build(self, client):
        job_id, _ = self.run_job(client)
        built = client.post(f"/api/bills/{job_id}/columns", json={"overrides": []})
        self.assertEqual(built.status_code, 200, built.text)
        download = client.get(f"/api/bills/{job_id}/download")
        return built.json(), load_workbook(io.BytesIO(download.content))

    def test_analyses_are_cached_outside_the_job_folder_per_model(self):
        """Deleting a job must not throw away the model's answers, and answers
        from one provider/model are never read back under another."""
        with self.client_ctx() as client:
            job_id, _ = self.run_job(client)
            job_directory = Path(bills_api.get_store().get(job_id).directory)
        self.assertTrue(self.cache_folders)
        for cache_folder in self.cache_folders:
            self.assertNotIn(job_directory, Path(cache_folder).parents)
        self.assertEqual(set(self.model_labels), {"openai-fake-model"})
        cached = list((self.tmp / "cache" / "extraction_cache").glob("*/openai-fake-model/*.json"))
        self.assertEqual(len(cached), 3)

    def test_a_second_run_reads_the_cache_not_the_model(self):
        with self.client_ctx() as client:
            self.run_job(client)
            self.run_job(client)
        self.assertEqual(len(self.provider.calls), 3)

    def test_the_model_is_shown_page_images_and_text_without_axis_labels(self):
        with self.client_ctx() as client:
            self.run_job(client)
        images, text = self.provider.calls[0]
        self.assertEqual(images, 2)
        self.assertIn("Electricity (kWh)", text)
        self.assertNotIn("25000", text)
        self.assertNotIn("45000", text)

    def test_batch_stops_at_the_review_gate_before_writing_anything(self):
        with self.client_ctx() as client:
            job_id, body = self.run_job(client)
            self.assertEqual(body["state"], "awaiting_review", body["message"])
            self.assertEqual(body["bills"], 3)
            self.assertFalse(body["has_workbook"])
            self.assertEqual(client.get(f"/api/bills/{job_id}/download").status_code, 404)

            columns = client.get(f"/api/bills/{job_id}/columns").json()["columns"]
            names = [c["canonical"] for c in columns]
            for expected in ("On-Peak kWh", "Off-Peak kWh", "Total kWh", "Billed Demand",
                             "On-Peak Use and Charge", "Customer Charge",
                             "Palo Verde Regulatory Credit"):
                self.assertIn(expected, names)
            credit = next(c for c in columns if c["canonical"] == "Palo Verde Regulatory Credit")
            self.assertEqual((credit["group"], credit["sign"]), ("credits", "credit"))

    def test_confirming_the_layout_builds_a_downloadable_workbook(self):
        with self.client_ctx() as client:
            body, book = self.build(client)
        self.assertEqual(body["state"], "done")
        self.assertEqual(len(body["sheets"]), 1)
        self.assertEqual(body["sheets"][0]["rows"], 3)
        self.assertEqual(body["flagged"], [])
        self.assertIn("Overall Utility Analysis", book.sheetnames)
        sheet = book[book.sheetnames[0]]
        self.assertEqual(sheet.cell(3, 1).value, "Statement Date")
        self.assertEqual(sheet.max_row, 3 + 3 + 1)  # header + months + totals
        self.assertEqual(sheet.cell(7, 1).value, "TOTALS")

    def test_every_cell_is_a_printed_figure(self):
        with self.client_ctx() as client:
            _, book = self.build(client)
        printed = set()
        for pdf in self.pdfs:
            printed |= {abs(n) for page in read_pdf(pdf, NO_OCR) for n in numbers_in(page.text)}
        sheet = book[book.sheetnames[0]]
        for row in sheet.iter_rows(min_row=4, max_row=6, min_col=2, values_only=True):
            for value in row:
                if isinstance(value, (int, float)):
                    self.assertIn(abs(Decimal(str(value))), printed, value)

    def test_rows_are_labelled_by_statement_month(self):
        with self.client_ctx() as client:
            _, book = self.build(client)
        sheet = book[book.sheetnames[0]]
        months = [sheet.cell(r, 1).value.strftime("%Y-%m") for r in range(4, 7)]
        self.assertEqual(months, ["2026-01", "2026-02", "2026-03"])

    def test_a_reviewer_override_moves_a_column_and_rebuilds(self):
        with self.client_ctx() as client:
            job_id, _ = self.run_job(client)
            response = client.post(f"/api/bills/{job_id}/columns", json={
                "overrides": [{"label": "customer charge", "group": "usage"}]
            })
            self.assertEqual(response.status_code, 200, response.text)
            columns = client.get(f"/api/bills/{job_id}/columns").json()["columns"]
            moved = next(c for c in columns if c["canonical"] == "Customer Charge")
            self.assertEqual(moved["group"], "usage")
            self.assertEqual(moved["source"], "review")

    def test_an_override_that_makes_a_column_ambiguous_is_refused(self):
        with self.client_ctx() as client:
            job_id, _ = self.run_job(client)
            response = client.post(f"/api/bills/{job_id}/columns", json={
                "overrides": [{"label": "customer charge", "canonical": "On-Peak kWh"}]
            })
            self.assertEqual(response.status_code, 400)
            self.assertIn("ambiguous", response.json()["detail"])

    def test_a_figure_not_on_the_page_is_dropped_and_warned(self):
        answer = self.answers["02/17/2026"]
        answer["readings"][0]["value"] = "25,000.000"   # not printed; an axis tick
        with self.client_ctx() as client:
            job_id, body = self.run_job(client)
            columns = client.get(f"/api/bills/{job_id}/columns").json()["columns"]
            built = client.post(f"/api/bills/{job_id}/columns", json={"overrides": []}).json()
        self.assertTrue(any("'25,000.000' is not printed" in w for w in body["warnings"]))
        self.assertIn("On-Peak kWh", [c["canonical"] for c in columns])
        self.assertEqual(built["flagged"], [])

    def test_line_items_that_do_not_sum_to_the_total_are_flagged(self):
        month = self.MONTHS[2]
        self.pdfs[2] = make_bill_pdf(self.tmp / "source", "bill_003b.pdf", month,
                                     total="$3,500.00")
        self.answers[month.statement] = figures_for(
            month, total_current_charges={"text": "$3,500.00", "page": 2})
        with self.client_ctx() as client:
            job_id, body = self.run_job(client)
            built = client.post(f"/api/bills/{job_id}/columns", json={"overrides": []}).json()
        self.assertTrue(any("Line items total" in w for w in body["warnings"]))
        self.assertEqual(len(built["flagged"]), 1)
        self.assertIn("did not reconcile", built["message"])

    def test_a_model_outage_on_one_bill_keeps_the_others(self):
        self.provider.fail = {"02/17/2026"}
        with self.client_ctx() as client:
            _, body = self.run_job(client)
        self.assertEqual(body["state"], "awaiting_review")
        self.assertEqual(body["bills"], 2)
        self.assertTrue(any("model unavailable" in w for w in body["warnings"]))

    def client_ctx(self):
        return TestClient(create_app())


if __name__ == "__main__":
    unittest.main()

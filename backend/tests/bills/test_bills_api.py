"""Upload API and job-runner tests.

No model and no network: the grouping step is stubbed to hand back the packaged
bill PDFs, whose extractions are already cached, so the whole route -- upload,
poll, review gate, build, download -- runs offline and deterministically.
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
from .test_fixtures import BILL_FOLDERS, HAVE_FIXTURES, requires_fixtures

TESTS_DIR = Path(__file__).resolve().parent
ELECTRICITY = TESTS_DIR / BILL_FOLDERS[0]
GAS = TESTS_DIR / BILL_FOLDERS[1]

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


@unittest.skipUnless(HAVE_FIXTURES, "real-bill fixtures are not present")
class EndToEndTests(unittest.TestCase):
    """Upload -> extract -> review gate -> build -> download, with no model.

    Grouping is stubbed to return the packaged bill PDFs; their extractions are
    already cached, so `load_bill` never reaches Ollama.
    """

    def setUp(self):
        self.tmp = Path(self.enterContext(__import__("tempfile").TemporaryDirectory()))
        bills_api.reset_store()
        self.addCleanup(bills_api.reset_store)

        patcher = patch("app.api.bills.get_settings")
        settings = patcher.start()
        self.addCleanup(patcher.stop)
        settings.return_value.bills_path = str(self.tmp)

        # The cached extractions live beside the fixtures, so the pipeline must
        # look for them there rather than in the (empty) job directory.
        from .test_fixtures import FIXTURES
        self.fixtures = FIXTURES

        def fake_group(source_pdf, *, project_folder, inference_config=None):
            folder = ELECTRICITY if "elec" in Path(source_pdf).name else GAS
            return sorted(folder.glob("bill_*.pdf"))

        group_patch = patch("app.bills.pipeline.bill_extraction.process_source_pdf",
                            side_effect=fake_group)
        group_patch.start()
        self.addCleanup(group_patch.stop)

        real_load = __import__("app.bills.bill_value_extraction", fromlist=["x"]).load_bill

        def cached_load(path, fingerprint, project_folder, model, **kw):
            return real_load(path, fingerprint, self.fixtures, model, run_inference=False)

        load_patch = patch("app.bills.pipeline.bve.load_bill", side_effect=cached_load)
        load_patch.start()
        self.addCleanup(load_patch.stop)

        # No Ollama in the test environment: label interpretation uses rules.
        client_patch = patch("app.bills.pipeline.LocalInferenceClient",
                             side_effect=RuntimeError("no model in tests"))
        client_patch.start()
        self.addCleanup(client_patch.stop)

        # Nor OCR: re-reading every fixture page at 300 DPI takes minutes. The
        # printed-row reader has its own tests (test_grid); here it finds nothing,
        # which leaves the cached extractions exactly as they were.
        grid_patch = patch("app.bills.pipeline.grid.printed_rows", return_value=[])
        self.grid_rows = grid_patch.start()
        self.addCleanup(grid_patch.stop)

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

    def test_batch_stops_at_the_review_gate_before_writing_anything(self):
        with self.client_ctx() as client:
            job_id, body = self.run_job(client)
            self.assertEqual(body["state"], "awaiting_review", body["message"])
            self.assertEqual(body["bills"], 12)
            self.assertFalse(body["has_workbook"])
            self.assertEqual(client.get(f"/api/bills/{job_id}/download").status_code, 404)

            columns = client.get(f"/api/bills/{job_id}/columns").json()["columns"]
            names = [c["canonical"] for c in columns]
            self.assertIn("On-Peak Energy Charge", names)
            self.assertIn("Demand Charge", names)
            # Account activity must never be offered as a column.
            self.assertNotIn("Previous Balance", names)

    def test_confirming_the_layout_builds_a_downloadable_workbook(self):
        with self.client_ctx() as client:
            job_id, _ = self.run_job(client)
            built = client.post(f"/api/bills/{job_id}/columns", json={"overrides": []})
            self.assertEqual(built.status_code, 200, built.text)
            body = built.json()
            self.assertEqual(body["state"], "done")
            self.assertEqual(len(body["sheets"]), 1)
            self.assertEqual(body["sheets"][0]["rows"], 12)
            self.assertTrue(body["has_workbook"])

            download = client.get(f"/api/bills/{job_id}/download")
            self.assertEqual(download.status_code, 200)
            self.assertIn("spreadsheetml", download.headers["content-type"])

            book = load_workbook(io.BytesIO(download.content))
            self.assertIn("Overall Utility Analysis", book.sheetnames)
            sheet = book[book.sheetnames[0]]
            self.assertEqual(sheet.cell(3, 1).value, "Statement Date")
            self.assertEqual(sheet.max_row, 3 + 12 + 1)  # header + months + totals

    def test_a_reviewer_override_moves_a_column_and_rebuilds(self):
        with self.client_ctx() as client:
            job_id, _ = self.run_job(client)
            response = client.post(f"/api/bills/{job_id}/columns", json={
                "overrides": [{"label": "facilities charge", "group": "usage"}]
            })
            self.assertEqual(response.status_code, 200, response.text)
            columns = client.get(f"/api/bills/{job_id}/columns").json()["columns"]
            moved = next(c for c in columns if c["canonical"] == "Facilities Charge")
            self.assertEqual(moved["group"], "usage")
            self.assertEqual(moved["source"], "review")

    def test_an_override_that_makes_a_column_ambiguous_is_refused(self):
        with self.client_ctx() as client:
            job_id, _ = self.run_job(client)
            response = client.post(f"/api/bills/{job_id}/columns", json={
                "overrides": [{"label": "facilities charge", "canonical": "Demand Charge"}]
            })
            self.assertEqual(response.status_code, 400)
            self.assertIn("ambiguous", response.json()["detail"])

    def test_the_gas_batch_reports_its_two_unreconciled_months(self):
        with self.client_ctx() as client:
            job_id, _ = self.run_job(client, name="gas.pdf")
            body = client.post(f"/api/bills/{job_id}/columns",
                               json={"overrides": []}).json()
        self.assertEqual(len(body["flagged"]), 2)
        self.assertEqual({f["difference"] for f in body["flagged"]}, {"0.10", "1000.00"})
        self.assertIn("did not reconcile", body["message"])

    def test_a_model_outage_is_reported_as_a_warning_not_a_failure(self):
        with self.client_ctx() as client:
            _, body = self.run_job(client)
        self.assertEqual(body["state"], "awaiting_review")
        self.assertTrue(any("built-in vocabulary" in w for w in body["warnings"]))

    def test_every_bill_is_checked_against_its_printed_rows(self):
        with self.client_ctx() as client:
            _, body = self.run_job(client)
        self.assertEqual(self.grid_rows.call_count, body["bills"])

    def test_a_printed_row_failure_keeps_the_bill_and_warns(self):
        self.grid_rows.side_effect = RuntimeError("OCR engine missing")
        with self.client_ctx() as client:
            _, body = self.run_job(client)
        self.assertEqual(body["state"], "awaiting_review")
        self.assertGreater(body["bills"], 0)
        self.assertTrue(any("not checked against the page" in w for w in body["warnings"]))

    def client_ctx(self):
        return TestClient(create_app())


if __name__ == "__main__":
    unittest.main()

"""Progress must count what the batch is actually doing.

Counting uploaded files made the bar read 100% for the whole of value extraction,
which is the long phase: one PDF can hold a year of bills, so the file was "done"
grouping after a minute and nothing moved for the next half hour.
"""
import unittest
from pathlib import Path

from app.bills.jobs import FileProgress, Job, JobState


def job_with(files=1, **overrides):
    job = Job(
        id="j1",
        directory=Path("."),
        files=[
            FileProgress(name=f"{i}.pdf", path=Path(f"{i}.pdf"))
            for i in range(1, files + 1)
        ],
    )
    for key, value in overrides.items():
        setattr(job, key, value)
    return job


class ProgressUnitTests(unittest.TestCase):
    def test_files_are_counted_before_any_bill_is_known(self):
        job = job_with(files=2)
        self.assertEqual(job.progress, (0, 2, "files"))

    def test_a_grouped_file_advances_the_file_count(self):
        job = job_with(files=2)
        job.files[0].state = "done"
        self.assertEqual(job.progress, (1, 2, "files"))

    def test_errored_and_skipped_files_count_as_settled(self):
        job = job_with(files=3)
        job.files[0].state = "error"
        job.files[1].state = "skipped"
        self.assertEqual(job.progress, (2, 3, "files"))

    def test_bills_take_over_once_extraction_starts(self):
        """The whole point: 12 bills from 1 file is 0%, not 100%."""
        job = job_with(files=1, bills_total=12, bills_done=0)
        job.files[0].state = "done"
        self.assertEqual(job.progress, (0, 12, "bills"))

    def test_bill_progress_advances(self):
        job = job_with(files=1, bills_total=12, bills_done=5)
        self.assertEqual(job.progress, (5, 12, "bills"))

    def test_a_finished_batch_reads_as_complete(self):
        job = job_with(files=1, bills_total=12, bills_done=12)
        self.assertEqual(job.progress, (12, 12, "bills"))


class ProgressPayloadTests(unittest.TestCase):
    def test_the_unit_is_reported_to_the_browser(self):
        payload = job_with(files=1, bills_total=4, bills_done=1).as_dict()
        self.assertEqual(
            payload["progress"], {"done": 1, "total": 4, "unit": "bills"}
        )

    def test_workbook_path_is_absent_until_one_is_built(self):
        self.assertIsNone(job_with().as_dict()["workbook_path"])

    def test_workbook_path_is_reported_once_built(self):
        job = job_with()
        job.workbook = Path("data/bills/abc123/Utility Analysis.xlsx")
        payload = job.as_dict()

        self.assertIn("Utility Analysis.xlsx", payload["workbook_path"])
        # has_workbook stays a real file check; the path is for showing the user.
        self.assertFalse(payload["has_workbook"])


class CancellationTests(unittest.TestCase):
    def test_a_cancelled_job_keeps_the_progress_it_reached(self):
        job = job_with(files=1, bills_total=12, bills_done=7)
        job.cancel()
        job.touch(JobState.CANCELLED, "Cancelled during value extraction.")

        self.assertTrue(job.cancelled)
        self.assertEqual(job.progress, (7, 12, "bills"))


if __name__ == "__main__":
    unittest.main()

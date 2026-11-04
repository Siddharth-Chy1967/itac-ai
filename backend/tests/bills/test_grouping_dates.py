"""A bill_date disagreement must not split a bill the page numbering vouches for.

The case this comes from: a 2-page PNM bill where qwen2.5vl read the issue date as
02/17/2025 on page 1 and 03/11/2025 on page 2. Both pages printed the same account
number and said "1" and "2 of 2", so the structure was unambiguous; only the
model-read date disagreed, and grouping refused the whole bill over it.
"""
import unittest

from app.bills.grouping_practice import (
    blocking_conflicts,
    describe_conflicts,
    field_conflicts,
    group_bill_pages,
    has_same_account,
    normalize_account_number,
    without_misfiled_invoice_numbers,
)

ACCOUNT = "019735300 - 0276883 - 4"


def page(number, role, **overrides):
    record = {
        "page_number": number,
        "page_role": role,
        "utility_types": ["electricity"],
        "utility_company": "PNM",
        "account_number": ACCOUNT,
        "service_start": None,
        "service_end": None,
        "printed_page_number": number,
        "printed_total_pages": 2,
        "bill_date": "02/17/2025",
        "invoice_number": None,
        "billing_period_label": None,
        "billing_period_key": None,
    }
    record.update(overrides)
    return record


class AccountNumberTests(unittest.TestCase):
    def test_printed_punctuation_is_ignored(self):
        self.assertEqual(
            normalize_account_number("019735300 - 0276883 - 4"),
            normalize_account_number("0197353000276883 4"),
        )

    def test_blank_and_missing_become_none(self):
        self.assertIsNone(normalize_account_number(None))
        self.assertIsNone(normalize_account_number("   "))

    def test_same_account_tolerates_a_page_that_omits_it(self):
        pages = [page(1, "bill_start"), page(2, "continuation", account_number=None)]
        self.assertTrue(has_same_account(pages))

    def test_different_accounts_are_not_the_same(self):
        pages = [page(1, "bill_start"), page(2, "continuation", account_number="999")]
        self.assertFalse(has_same_account(pages))


class BlockingConflictTests(unittest.TestCase):
    """The PNM case, and the cases that must keep blocking."""

    def setUp(self):
        self.mismatched_dates = [
            page(1, "bill_start", bill_date="02/17/2025"),
            page(2, "continuation", bill_date="03/11/2025"),
        ]

    def test_bill_date_alone_is_ignored_when_numbering_is_complete(self):
        self.assertEqual(blocking_conflicts(self.mismatched_dates, "complete"), {})

    def test_the_date_conflict_is_still_detected_underneath(self):
        """Ignoring it for grouping is a decision, not a failure to notice."""
        self.assertIn("bill_date", field_conflicts(self.mismatched_dates))

    def test_bill_date_still_blocks_without_complete_numbering(self):
        for completeness in ("unknown", "partial", "invalid"):
            with self.subTest(completeness=completeness):
                self.assertIn(
                    "bill_date",
                    blocking_conflicts(self.mismatched_dates, completeness),
                )

    def test_bill_date_still_blocks_when_accounts_differ(self):
        pages = [
            page(1, "bill_start", bill_date="02/17/2025"),
            page(2, "continuation", bill_date="03/11/2025", account_number="999"),
        ]
        self.assertIn("bill_date", blocking_conflicts(pages, "complete"))

    def test_other_fields_still_block_with_complete_numbering(self):
        """Only bill_date is downgraded; the structural fields are untouched."""
        cases = {
            "utility_company": {"utility_company": "APS"},
            "invoice_number": {"invoice_number": "INV-2"},
            "billing_period_key": {"billing_period_key": "2025-04"},
        }
        base = {"invoice_number": "INV-1", "billing_period_key": "2025-02"}
        for field, override in cases.items():
            with self.subTest(field=field):
                pages = [
                    page(1, "bill_start", **base),
                    page(2, "continuation", **{**base, **override}),
                ]
                self.assertIn(field, blocking_conflicts(pages, "complete"))

    def test_a_matching_date_conflicts_with_nothing(self):
        pages = [page(1, "bill_start"), page(2, "continuation")]
        self.assertEqual(blocking_conflicts(pages, "complete"), {})

    def test_the_same_date_in_two_printed_formats_is_not_a_conflict(self):
        pages = [
            page(1, "bill_start", bill_date="02/17/2025"),
            page(2, "continuation", bill_date="2025-02-17"),
        ]
        self.assertEqual(field_conflicts(pages), {})


class ConflictMessageTests(unittest.TestCase):
    def test_the_reason_names_the_field_and_the_values(self):
        reason = describe_conflicts(
            {"bill_date": ["02/17/2025", "03/11/2025"]}
        )
        self.assertIn("bill_date", reason)
        self.assertIn("02/17/2025", reason)
        self.assertIn("03/11/2025", reason)

    def test_a_missing_value_reads_as_null(self):
        self.assertIn("null", describe_conflicts({"invoice_number": ["INV-1", None]}))


class MisfiledInvoiceNumberTests(unittest.TestCase):
    """qwen2.5vl files the account number under invoice_number on PNM page 1.

    The page beside it carries the real invoice, so the two disagree and a bill
    whose account matches throughout was refused.
    """

    def test_an_invoice_equal_to_the_account_is_dropped(self):
        pages = [
            page(1, "bill_start", invoice_number=ACCOUNT),
            page(2, "continuation", invoice_number="A342026493"),
        ]
        cleaned = without_misfiled_invoice_numbers(pages)

        self.assertIsNone(cleaned[0]["invoice_number"])
        self.assertEqual(cleaned[1]["invoice_number"], "A342026493")

    def test_it_is_dropped_despite_different_punctuation(self):
        pages = [page(1, "bill_start", invoice_number="0197353000276883 4")]
        self.assertIsNone(without_misfiled_invoice_numbers(pages)[0]["invoice_number"])

    def test_a_real_invoice_number_is_kept(self):
        pages = [page(1, "bill_start", invoice_number="A342026493")]
        self.assertEqual(
            without_misfiled_invoice_numbers(pages)[0]["invoice_number"],
            "A342026493",
        )

    def test_the_originals_are_not_mutated(self):
        pages = [page(1, "bill_start", invoice_number=ACCOUNT)]
        without_misfiled_invoice_numbers(pages)
        self.assertEqual(pages[0]["invoice_number"], ACCOUNT)

    def test_two_real_invoice_numbers_still_conflict(self):
        pages = [
            page(1, "bill_start", invoice_number="A1"),
            page(2, "continuation", invoice_number="A2"),
        ]
        self.assertIn("invoice_number", blocking_conflicts(pages, "complete"))


class PageTotalGapTests(unittest.TestCase):
    """PNM prints "Page 1" bare, then "Page 2 of 2" on the page after it."""

    def test_a_missing_page_total_does_not_break_the_sequence(self):
        pages = [
            page(1, "bill_start", printed_total_pages=None, bill_date="02/17/2025"),
            page(2, "continuation", printed_total_pages=2, bill_date=None,
                 invoice_number="A845029295"),
        ]
        result = group_bill_pages(pages)

        self.assertEqual(result["review_pages"], [])
        self.assertEqual([d["pages"] for d in result["group_details"]], [[1, 2]])

    def test_totals_that_both_exist_and_disagree_still_block(self):
        pages = [
            page(1, "bill_start", printed_total_pages=2),
            page(2, "continuation", printed_total_pages=3),
        ]
        self.assertEqual(group_bill_pages(pages)["review_pages"], [1, 2])

    def test_non_consecutive_printed_numbers_do_not_link(self):
        """With no shared date or invoice, the sequence is the only evidence left."""
        pages = [
            page(1, "bill_start", printed_page_number=1, printed_total_pages=None,
                 bill_date=None),
            page(2, "continuation", printed_page_number=5, printed_total_pages=None,
                 bill_date=None),
        ]
        self.assertEqual(group_bill_pages(pages)["review_pages"], [1, 2])

    def test_a_consecutive_sequence_links_with_no_other_evidence(self):
        """The counterpart: sequence alone is enough, which is the PNM case."""
        pages = [
            page(1, "bill_start", printed_total_pages=None, bill_date=None),
            page(2, "continuation", printed_total_pages=2, bill_date=None),
        ]
        self.assertEqual(group_bill_pages(pages)["review_pages"], [])

    def test_different_accounts_still_block(self):
        pages = [
            page(1, "bill_start"),
            page(2, "continuation", account_number="999888777"),
        ]
        self.assertEqual(group_bill_pages(pages)["review_pages"], [1, 2])

    def test_two_months_of_one_account_are_not_merged(self):
        """The case the loosened sequence rule must not break."""
        pages = [
            page(1, "bill_start", bill_date="01/15/2025", invoice_number="A1"),
            page(2, "continuation", bill_date="01/15/2025", invoice_number="A1"),
            page(3, "bill_start", printed_page_number=1,
                 bill_date="02/15/2025", invoice_number="A2"),
            page(4, "continuation", printed_page_number=2,
                 bill_date="02/15/2025", invoice_number="A2"),
        ]
        result = group_bill_pages(pages)

        self.assertEqual(result["review_pages"], [])
        self.assertEqual([d["pages"] for d in result["group_details"]], [[1, 2], [3, 4]])


class EndToEndGroupingTests(unittest.TestCase):
    def test_the_pnm_bill_now_groups_instead_of_needing_review(self):
        pages = [
            page(1, "bill_start", bill_date="02/17/2025"),
            page(2, "continuation", bill_date="03/11/2025"),
        ]
        result = group_bill_pages(pages)

        self.assertEqual(result["review_pages"], [])
        self.assertEqual([d["pages"] for d in result["group_details"]], [[1, 2]])
        self.assertEqual(result["group_details"][0]["completeness"], "complete")

    def test_two_genuinely_different_bills_are_still_held(self):
        """Different invoice numbers are real evidence and must still stop it."""
        pages = [
            page(1, "bill_start", invoice_number="INV-1"),
            page(2, "continuation", invoice_number="INV-2"),
        ]
        result = group_bill_pages(pages)

        self.assertEqual(result["review_pages"], [1, 2])


if __name__ == "__main__":
    unittest.main()

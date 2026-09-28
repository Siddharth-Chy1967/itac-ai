"""Offline tests for bill grouping, including omitted and unnumbered pages."""
import unittest

from app.bills.grouping_practice import (
    get_page_completeness,
    group_bill_pages,
    has_complete_page_sequence,
    has_field_conflict,
    matching_bill_fields,
    strip_web_address,
)


class GroupingTests(unittest.TestCase):
    def test_grouping_cases(self):
        # Test field-conflict detection.
        assert not has_field_conflict(
            [{"printed_total_pages": 2}, {"printed_total_pages": 2}],
            "printed_total_pages",
        )
    
        assert not has_field_conflict(
            [{"printed_total_pages": 2}, {"printed_total_pages": None}],
            "printed_total_pages",
        )
    
        assert has_field_conflict(
            [{"printed_total_pages": 2}, {"printed_total_pages": 3}],
            "printed_total_pages",
        )
    
        assert not has_field_conflict(
            [
                {"utility_company": "SRP"},
                {"utility_company": "Salt River Project"},
            ],
            "utility_company",
        )
    
        assert has_field_conflict(
            [
                {"utility_company": "SRP"},
                {"utility_company": "Example Water Utility"},
            ],
            "utility_company",
        )
    
        # Different printed/extracted names for the same issuer are not a conflict.
        assert not has_field_conflict(
            [{"utility_company": "Salt River Project (SRP)"},
             {"utility_company": "Salt River Project"}],
            "utility_company",
        )
        assert has_field_conflict(
            [{"utility_company": "Salt River Project (SRP)"},
             {"utility_company": "Example Water Utility"}],
            "utility_company",
        )

        # The footer web address and the logo name the same issuer. Real SRP
        # bills print one on page 1 and the other on page 2.
        assert not has_field_conflict(
            [{"utility_company": "SRP"},
             {"utility_company": "srpnet.com"}],
            "utility_company",
        )
        assert not has_field_conflict(
            [{"utility_company": "srpnet.com"},
             {"utility_company": "https://www.srpnet.com/bill"}],
            "utility_company",
        )
        assert has_field_conflict(
            [{"utility_company": "srpnet.com"},
             {"utility_company": "examplewater.com"}],
            "utility_company",
        )

        # An address reduces to its label; a plain name keeps its own shape.
        assert strip_web_address("https://www.srpnet.com/bill") == "srpnet"
        assert strip_web_address("srpnet.com") == "srpnet"
        assert strip_web_address("salt river project") == "salt river project"
        assert strip_web_address("p.n.m.") == "p.n.m."

        print("Field-conflict tests passed.")
    
        # Test complete printed sequences.
        assert has_complete_page_sequence([1, 2], 2)
        assert has_complete_page_sequence([1, 2, 3], 3)
    
        assert not has_complete_page_sequence([1, 3], 2)
        assert not has_complete_page_sequence([1, 2], 3)
        assert not has_complete_page_sequence([1, None], 2)
        assert not has_complete_page_sequence([1, 2], None)
    
        print("Printed-number tests passed.")
    
        # Test completeness without rejecting intentionally omitted pages.
        completeness_cases = [
            ([1, 2], 2, "complete"),
            ([1, 3], 4, "partial"),
            ([3], 4, "partial"),
            ([None, None], None, "unknown"),
            ([1, None], 4, "unknown"),
            ([1, 1], 4, "invalid"),
            ([3, 1], 4, "invalid"),
            ([1, 5], 4, "invalid"),
        ]
    
        for numbers, total, expected in completeness_cases:
            assert get_page_completeness(numbers, total) == expected
    
        print("Completeness tests passed.")
    
        # Test matching evidence.
        assert matching_bill_fields(
            {"billing_period_key": "2020-12"},
            {"billing_period_key": "2020-12"},
        ) == ["billing_period_key"]
    
        assert matching_bill_fields(
            {"billing_period_key": "2020-12"},
            {"billing_period_key": "2021-01"},
        ) == []
    
        assert matching_bill_fields(
            {"invoice_number": None},
            {"invoice_number": None},
        ) == []
    
        assert matching_bill_fields(
            {"invoice_number": "ABC-123", "bill_date": "7/21/2021"},
            {"invoice_number": "ABC-123", "bill_date": "7/21/2021"},
        ) == ["invoice_number", "bill_date"]
    
        print("Matching-evidence tests passed.")
    
        # This helper creates fictional records for tests only.
        # Extra named fields are collected in the "fields" dictionary.
        def make_page(
            number,
            role,
            printed_number=None,
            printed_total=None,
            **fields,
        ):
            record = {
                "page_number": number,
                "page_role": role,
                "printed_page_number": printed_number,
                "printed_total_pages": printed_total,
                "utility_company": "Example Utility",
            }
    
            record.update(fields)
            return record
    
        # Practice example: two partial bills and one complete bill.
        pages = [
            make_page(1, "bill_start", 1, 3),
            make_page(2, "continuation", 2, 3),
            make_page(3, "bill_start", 1, 2),
            make_page(4, "continuation", 2, 2),
            make_page(5, "non_bill"),
            make_page(6, "bill_start", 1, 3),
        ]
    
        result = group_bill_pages(pages)
    
        assert result["bill_groups"] == [[1, 2], [3, 4], [6]]
        assert result["non_bill_pages"] == [5]
        assert result["review_pages"] == []
        assert result["review_reasons"] == []
    
        statuses = []
        for detail in result["group_details"]:
            statuses.append(detail["completeness"])
    
        assert statuses == ["partial", "complete", "partial"]
    
        print(f"Bill groups: {result['bill_groups']}")
        print(f"Non-bill pages: {result['non_bill_pages']}")
        print(f"Review pages: {result['review_pages']}")
        print(f"Completeness: {statuses}")
    
        # Water: supplied PDF pages 1 and 2 are printed pages 1 and 3 of 4.
        water_pages = [
            make_page(
                1, "bill_start", 1, 4,
                bill_date="7/21/2021",
            ),
            make_page(
                2, "continuation", 3, 4,
                bill_date="7/21/2021",
            ),
        ]
    
        water_result = group_bill_pages(water_pages)
    
        assert water_result["bill_groups"] == [[1, 2]]
        assert water_result["review_pages"] == []
        assert water_result["group_details"][0]["completeness"] == "partial"
    
        print("Omitted-page grouping test passed.")
    
        # Gas: no printed numbering, but company and billing month match.
        gas_pages = [
            make_page(
                1, "bill_start",
                billing_period_key="2020-12",
            ),
            make_page(
                2, "continuation",
                billing_period_key="2020-12",
            ),
        ]
    
        gas_result = group_bill_pages(gas_pages)
    
        assert gas_result["bill_groups"] == [[1, 2]]
        assert gas_result["review_pages"] == []
        assert gas_result["group_details"][0]["completeness"] == "unknown"
    
        print("Unnumbered-bill grouping test passed.")
    
        # Different billing months must not be silently combined.
        gas_pages[1]["billing_period_key"] = "2021-01"
    
        assert group_bill_pages(gas_pages)["review_pages"] == [1, 2]
    
        # Different accounts require a shared invoice number.
        gas_pages[1]["billing_period_key"] = "2020-12"
        gas_pages[0]["account_number"] = "ACCOUNT-A"
        gas_pages[1]["account_number"] = "ACCOUNT-B"
    
        assert group_bill_pages(gas_pages)["review_pages"] == [1, 2]
    
        gas_pages[0]["invoice_number"] = "INV-123"
        gas_pages[1]["invoice_number"] = "INV-123"
    
        assert group_bill_pages(gas_pages)["bill_groups"] == [[1, 2]]
    
        # Company alone is insufficient for linking unnumbered pages.
        ambiguous_pages = [
            make_page(1, "bill_start"),
            make_page(2, "continuation"),
        ]
    
        assert group_bill_pages(ambiguous_pages)["review_pages"] == [1, 2]
    
        # A lone unnumbered continuation needs review.
        orphan_result = group_bill_pages([
            make_page(1, "continuation"),
        ])
    
        assert orphan_result["review_pages"] == [1]
    
        # A lone printed page 3 of 4 can be retained as a partial bill.
        partial_result = group_bill_pages([
            make_page(1, "continuation", 3, 4),
        ])
    
        assert partial_result["bill_groups"] == [[1]]
        assert partial_result["group_details"][0]["completeness"] == "partial"
    
        # Uncertain pages remain explicitly marked for review.
        uncertain_result = group_bill_pages([
            make_page(1, "uncertain"),
        ])
    
        assert uncertain_result["review_pages"] == [1]
    
        print("Conflict and review tests passed.")
    
        # Empty input returns empty results for every output field.
        assert group_bill_pages([]) == {
            "bill_groups": [],
            "non_bill_pages": [],
            "review_pages": [],
            "group_details": [],
            "review_reasons": [],
        }
    
        # Source PDF page numbers must be unique and in order.
        invalid_inputs = [
            [
                make_page(1, "bill_start"),
                make_page(1, "continuation"),
            ],
            [
                make_page(2, "bill_start"),
                make_page(1, "continuation"),
            ],
        ]
    
        for invalid_pages in invalid_inputs:
            try:
                group_bill_pages(invalid_pages)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid PDF page numbers were accepted.")
    
        print("Input-validation tests passed.")
        print("All grouping tests passed.")
    
    
    if __name__ == "__main__":
        run_tests()


if __name__ == "__main__":
    unittest.main()

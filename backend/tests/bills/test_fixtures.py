"""Local-only regression dataset: real client bills with their saved extractions.

The PDFs and caches under `fixtures/` are real client utility bills. They are
gitignored and therefore absent on a fresh clone, so every test that touches them
guards with `requires_fixtures`, and the suite still passes without them.

The original 36-bill set (12 electricity, 12 water, 12 gas) was never fully
packaged: the water bills and their extractions are missing, so 24 remain.
`BILL_FOLDERS` names what actually exists.
"""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "fixtures"

BILL_FOLDERS = [
    "fixtures/output/pdf/Electrical Bill Data/20260903_112206_296503",
    "fixtures/output/pdf/Natural Gas Bills/20260903_112426_784365",
]

# Bills per folder, in BILL_FOLDERS order.
EXPECTED_BILLS = 24


def _available() -> bool:
    if not FIXTURES.is_dir():
        return False
    return all((ROOT / folder).is_dir() for folder in BILL_FOLDERS)


HAVE_FIXTURES = _available()

requires_fixtures = unittest.skipUnless(
    HAVE_FIXTURES,
    "real-bill fixtures are not present (gitignored client data); see test_fixtures.py",
)

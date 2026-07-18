"""
Session C: TableSpec schema.

This is the contract between the LLM and the renderer. The LLM never touches
openpyxl or formatting directly, it produces a TableSpec as JSON; this module
validates that JSON before anything gets rendered. Validation failures should
be caught here, not surfaced as a malformed .xlsx handed to the user.

Currently models the utility-analysis table layout (the one proven against
real GRICUA and PNM bills in the utility-analysis-formatting skill). Other
table types (AR summary tables, equipment tables) will get their own spec
models later, reusing ColumnSpec/styles where the shape overlaps.
"""

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

ColumnType = Literal["date", "usage", "demand", "curr"]


class ColumnSpec(BaseModel):
    key: str
    header: str
    type: ColumnType
    width: int = 12


class GroupSpec(BaseModel):
    label: str
    vmerge: bool = False
    cols: list[ColumnSpec]

    @model_validator(mode="after")
    def vmerge_is_single_column(self) -> "GroupSpec":
        if self.vmerge and len(self.cols) != 1:
            raise ValueError(
                f"Group '{self.label}' has vmerge=True but {len(self.cols)} "
                "columns; a vertically-merged section must have exactly 1 column."
            )
        return self


class SummaryRow(BaseModel):
    resource: str
    annual: Optional[float] = None
    annual_type: str = "curr_t"
    unit: Optional[float] = None
    unit_type: str = "unit3"


class TableSpec(BaseModel):
    sheet_name: str = "Analysis"
    title: str
    groups: list[GroupSpec]
    rows: list[dict[str, object]]
    totals: dict[str, object]
    summary: list[SummaryRow] = Field(default_factory=list)

    @model_validator(mode="after")
    def keys_are_consistent(self) -> "TableSpec":
        col_keys = [c.key for g in self.groups for c in g.cols]

        dupes = {k for k in col_keys if col_keys.count(k) > 1}
        if dupes:
            raise ValueError(f"Duplicate column keys across groups: {sorted(dupes)}")

        col_key_set = set(col_keys)
        for i, row in enumerate(self.rows):
            missing = col_key_set - row.keys()
            extra = row.keys() - col_key_set
            if missing:
                raise ValueError(f"Row {i} is missing keys: {sorted(missing)}")
            if extra:
                raise ValueError(f"Row {i} has keys not declared in any group: {sorted(extra)}")

        missing_totals = col_key_set - self.totals.keys()
        extra_totals = self.totals.keys() - col_key_set
        if missing_totals:
            raise ValueError(f"totals is missing keys: {sorted(missing_totals)}")
        if extra_totals:
            raise ValueError(f"totals has keys not declared in any group: {sorted(extra_totals)}")

        if not self.groups:
            raise ValueError("TableSpec must have at least one group")

        return self


# In table mode the model answers with a short prose explanation plus exactly one
# ```tablespec fenced block holding the TableSpec JSON (spec §5.3). This pulls the
# JSON back out so the orchestrator can validate it before rendering.
_TABLESPEC_BLOCK = re.compile(r"```tablespec\b[ \t]*\r?\n(.*?)```", re.DOTALL)


def extract_tablespec_block(text: str) -> Optional[str]:
    """Return the JSON string inside the ```tablespec block, or None if absent.

    Only the first block is used; a well-behaved table-mode response emits exactly
    one. The returned string is not validated here, the caller feeds it to
    `TableSpec.model_validate_json` so parse/validation errors surface together.
    """
    m = _TABLESPEC_BLOCK.search(text)
    if m is None:
        return None
    return m.group(1).strip()

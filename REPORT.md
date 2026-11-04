# Bill extraction rebuild: report (2026-09-28)

## What changed and why

**The bug:** the old pipeline invented readings. `grid.py` paired "a label to the
left of a number" from OCR geometry. On the PNM usage chart, OCR put the rotated
y-axis title "Electricity (kWh)" on the same line as the tick "25000", so grid.py
added "Electricity (kWh) = 25000" to every month. No model had returned it.

**The new pipeline:**

1. **OCR / page text** (`page_text.py`, deterministic). The PDF text layer is used
   when the page has one, RapidOCR boxes otherwise. Chart axis labels are removed:
   three or more bare numbers sharing an x-position in an even arithmetic sequence.
2. **Model** (`llm_providers.py`, `figures.py`, `figures_prompt.py`). It receives the
   page images and the page text, and returns JSON matching `bill_schema.BillFigures`:
   - readings with kind, time-of-use period and unit
   - each charge line with its amount and category
   - taxes, the total, the statement date and the service period

   Utility guidance for PNM, SRP, APS, GRICUA and gas is prompt text, not code.
3. **Verification** (`verify.py`, deterministic). Every number and date must be
   printed on its page:
   - sign included
   - a charge's amount must be printed on the same line as its label
   - money is never a reading

   Anything else is dropped and flagged, and its cell stays blank. The line items
   must sum to the printed total within $0.02, or the month is flagged. When they
   fall short, the model gets one narrow recheck: the page's own unaccounted
   dollar lines are quoted and it answers yes/no per line. Recovered lines are
   re-verified and kept only if the bill then reconciles.
4. **Columns** (`columns.py`). Columns are named from the model's typed fields, with
   no utility vocabulary in code. A charge printed twice becomes two columns, never
   a summed cell. The existing `shape` and openpyxl renderer are kept.

**Also changed:**
- `grid.py` and the "add missed readings" step are deleted.
- Rows are labelled by statement date.
- Derived columns (summed Total kWh, MMBtu, cost per gallon) are off by default.
- A bill that omits its meter or account joins its series instead of starting a new
  sheet.
- Undated bills are never merged as duplicates.
- The totals row is labelled `TOTALS`.

**Providers** (one `extract(images, ocr_text, schema)` interface):
- `ollama`: qwen2.5vl, the local fallback.
- `openai`: the Responses API with a strict `json_schema`, `reasoning.effort`,
  `input_image` data URLs at high detail, and `store: false`.
- `openai_compatible`: any `/chat/completions` endpoint, such as OpenRouter.

Caches are keyed by provider + model for both extraction and page grouping.
DECISIONS.md lists all 30 judgement calls.

## PNM PDFs found

I searched the whole `itac-ai` folder and found **2** PNM bill PDFs (SHA-256 checked;
the copies under `backend/data/bills/<job>/` are byte-identical uploads of these):

- `D:\ITAC AI\itac-ai\2025-2026 Electrical Bills-1.pdf`: PNM bill issued 01/16/2026, 2 pages
- `D:\ITAC AI\itac-ai\2025-2026 Electrical Bills-2.pdf`: PNM bill issued 02/17/2025, 2 pages

Other samples found:
- **SRP electric:** `Utility_Bill_Share_20260903_1135/Utility_Bill_Extractor/Electrical Bill Data.pdf`
  (24-page scan, 12 bills)
- **Symmetry natural gas:** `.../Natural Gas Bills.pdf`
- **Water:** `.../Water Bill Data.pdf`

**No APS or GRICUA bills exist in the folder.** `data/reports/.../AS0590.pdf` mentions
APS but is an assessment report, not a bill.

## DONE criteria: what was verified, and how

All results below come from **qwen2.5vl via Ollama**, because there is no OpenAI key.
The workbook was audited with `backend/scripts/check_bill_workbook.py`, which traces
every numeric cell to its own bill's page text (sign included, axis labels excluded)
and evaluates the TOTALS formulas.

| Criterion | PNM result |
| --- | --- |
| No value absent from the bill's text | **Verified.** Audit: "every workbook value is printed on its bill", both months. |
| kWh matches the bill, differs by month | **Verified.** Jan-26: on-peak 16,720 / off-peak 14,320. Feb-25: 22,640 / 19,360. Both match the printed "Use and Charge" lines. |
| Charge columns hold dollars, not kWh | **Verified.** Every `($)` column holds printed dollar amounts; money is refused as a reading. |
| Demand filled when printed | **Verified.** Billed Demand (kW): 154.4 in Jan-26, 182.4 in Feb-25. Measured Demand is blank in Feb-25 and flagged, because qwen also filed the raw register (2.280) as measured demand. |
| No unlabelled readings; no "not readable" on legible bills | **Verified.** No unlabelled columns and no "not readable" notes. |
| TOTALS row populated and equals the bill total | **Jan-26 verified:** 15 charge cells sum to **$4,773.41 = printed total** after the recheck recovered the missed "Cost-Effective Energy Saving Prog. $175.49". **Feb-25 not met:** the cells sum to $6,195.83 against $6,198.14. qwen misses "Transportation Electrification R53 $2.31" and rejects it on recheck, so the month is highlighted for review rather than filled. The Total ($) column holds the printed totals, and the TOTALS row evaluates to $10,971.55 = $4,773.41 + $6,198.14. |
| Month labels use the statement date | **Verified.** The rows are 2026-01-16 and 2025-02-17, the "Bill Issued" dates. |
| All existing tests pass, plus new ones | **Verified.** `pytest backend/tests`: **717 passed.** |

**Regression test:** `backend/tests/bills/test_pnm_fabrication.py` replays the real
RapidOCR boxes of the 01/16/2026 PNM bill (account, name and address removed) through
`pipeline.run_batch`.
- On a rebuilt copy of the **old** code it fails: `AssertionError: 25000 unexpectedly found in [25000, 16720, 14320, 31040, …]`.
- On the new code it passes.

**Known qwen label errors (values printed, names wrong):**
- The Jan-26 demand charge $1,931.54 is labelled "Demand Reading" instead of "Billed
  demand and charge", so Jan and Feb put it in different columns.
- Fuel lines are named "Non-Renewable" without "Fuel Cost Adjustment".

The review gate shows these before export.

**Other samples (first 3 months each, qwen):**
- **SRP:** all 3 bills extracted, and every value is printed. Two rows have no date:
  qwen put dollar amounts in the date fields, and the verifier dropped them. On one
  bill qwen filed kW demand figures under the kWh columns: printed values in the
  wrong columns.
- **Gas and water:** these failed at page grouping ("Grouping needs review"). That
  stage wasn't redesigned, and qwen's page analysis is non-deterministic.
- **APS and GRICUA:** not tested; there are no samples.

## What could not be verified

- **gpt-6-luna has not run for real** because there is no API key. The OpenAI path is
  tested against a mocked HTTP client that returns a realistic Responses API body
  (`test_llm_providers.py`): the request shape, strict schema, images, reasoning
  effort, usage accounting, retries, refusals, truncation, the full bill read with
  verification and per-model caching, and page grouping through the same provider.
- Whether gpt-6-luna closes the qwen gaps above (the missed $2.31 line, label and
  date errors, gas/water grouping) is untested.

## Switching to gpt-6-luna

In `D:\ITAC AI\itac-ai\.env`, change the provider line from `BILLS_LLM_PROVIDER=ollama`
to the first line below, and add your key as the second:

```
BILLS_LLM_PROVIDER=openai
BILLS_LLM_API_KEY=sk-...
```

The model defaults to `gpt-6-luna`; `BILLS_LLM_MODEL` can override it. This is
deliberately not `LLM_PROVIDER`, which already switches the chat assistant. Nothing
else changes: prompts, verification, recheck, columns and workbook are identical. The
only difference is that hosted models always receive page images. The first run
re-reads each PDF once, because caches are per model.

## Estimated cost per bill (gpt-6-luna)

Pricing: $0.10 per 1M input tokens and $0.50 per 1M output tokens (OpenAI model page,
2026-09-28).

Measured on the PNM bills:
- **Text:** about 16.5k characters of instructions, strict schema and page text,
  roughly 5k tokens.
- **Images:** two page images at high detail. Each is 1583×2048 = 3,200 patches,
  capped at 2,500, so about 5–6k image tokens.
- **Input total:** about 11k tokens, $0.0011.
- **Output:** 1–1.5k JSON tokens plus medium reasoning, about 3–5k tokens,
  $0.0015–0.0025.
- **Grouping:** about 2 small page-analysis calls, about $0.0005.
- **Recheck:** only on bills that don't add up, a small text-only call under $0.0005.

**About $0.003–0.005 per 2-page bill, or roughly 4–6 cents for a year of 12 bills.**
These are estimates; the per-call token usage is logged once a key is in place.

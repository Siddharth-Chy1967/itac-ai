# Decisions: bill extraction rebuild (2026-09-28)

Judgement calls made while working autonomously. Each has the option taken and why.

## Hard rule (no invented figures)

1. **grid.py deleted outright**, along with its reconcile step and its 18 tests.
   It paired "label to the left of a number" from OCR geometry and *added* readings
   the model had not returned. That is how the chart tick 25000 became
   "Electricity (kWh)" on both PNM bills. Nothing replaces it: the code no longer adds
   any reading.
2. **Verification is against the page's own text, per bill.** For a digital PDF the
   "OCR text" is the PDF text layer (exact characters, with word boxes). A scanned page
   uses RapidOCR boxes. Both go through the same chart-axis filter, and both feed the
   check. Using RapidOCR on a digital PDF would only add misreads.
3. **Numbers match by value, not by string**: `16,720.000` = `16720`. A figure found
   on a different page from the one the model named is kept, with its page corrected
   and a note.
4. **Charge signs must match the page.** A printed `$78.74` that the model returned as
   `-$78.74` is dropped (qwen did exactly this). There is one exception: a credit
   printed without a minus sign is accepted, but only on a line containing
   credit/discount/rebate/refund/adjustment.
5. **Money is never a reading.** A reading whose value contains `$` is dropped.
6. **Dates are verified the same way.** A statement date, due date or service date
   that is not printed on the bill is dropped (left blank) and flagged.
7. **Chart axis rule, as specified:** three or more bare numbers sharing an
   x-position (right edge, left edge or centre) whose top-to-bottom values step by a
   constant non-zero amount. They are removed before the model sees the text, and
   before verification, so the model can't cite them either. A vertical run is
   required, and even pixel spacing is not required. On both PNM bills this finds
   exactly the ten y-axis ticks and nothing else.
8. **No summed cells.** When a bill prints the same charge name twice (PNM's
   "Renewable Energy Rider" for 15 days and then 16 days), it becomes two columns,
   "Renewable Energy Rider" and "Renewable Energy Rider (2)", rather than one cell
   holding 199.09, which is not printed.
9. **Derived columns are off by default** (`BILLS_DERIVED_COLUMNS=false`). These are
   Total kWh summed from TOU periods, MMBtu/therm conversions, TGal to gallons, and
   cost per gallon; none of them is printed on a bill. Setting it to `true` restores
   the previous behaviour. The test helpers that exercise those conversions now pass
   `derive_columns=True`.
10. **The TOTALS row remains**, as live `=SUM()` formulas, because you asked for it.
    It is the one place the workbook shows arithmetic rather than printed figures,
    and it is labelled `TOTALS`. The per-month `Total ($)` column is the bill's
    printed total, not a sum.
11. **The $0.02 check is done twice.** `verify` flags the bill (the job warning names
    the difference). `shape.reconcile` still highlights the month row yellow with a
    comment in the workbook. Neither adjusts anything.

## Architecture

12. **The model's typed fields replace SEED_TERMS and the second model call.** The
    model says what each figure is: kind (energy / billed_demand / measured_demand /
    reactive / meter_register / ...), period and unit for readings, and category
    (energy / demand / fixed / tax / fee / credit / other) for charges. `columns.py`
    names columns from those fields. The pipeline no longer calls `interpret_labels`,
    so the second model round-trip and the utility vocabulary are gone from the live
    path.
13. **semantics.py, shape.py and workbook.py are kept.** `shape` and the workbook
    renderer are the existing tested renderer, as you asked. `semantics` still
    provides `LabelSemantics`, the review gate and the override API, so the /bills
    page and its API are unchanged. `SEED_TERMS` / `interpret_labels` remain in
    `semantics.py`, but only for their existing tests and the standalone CLI
    (`bill_value_extraction.main`); nothing in the upload pipeline calls them. I didn't
    delete them because about 90 existing tests depend on them.
14. **Meter-register readings are not columns.** These are dial reads, present/previous
    values and multipliers (PNM "Demand Reading 1.930", "x 80.000"). They're kept in
    the cached JSON but not in the sheet.
15. **Sheet merging was generalized.** A month that omits its meter, account or rate
    joins the one series it is consistent with. It stays separate if two series fit.
    The previous version handled only a missing rate schedule. qwen read PNM's meter
    number off one bill and not the other, which split one meter across two sheets.

## Configuration

16. **Bill settings use a `BILLS_` prefix: `BILLS_LLM_PROVIDER`, `BILLS_LLM_MODEL`,
    `BILLS_LLM_API_KEY`, `BILLS_LLM_BASE_URL`, `BILLS_LLM_REASONING_EFFORT`.**
    `LLM_PROVIDER` already exists and drives the chat assistant (gemini / ollama /
    anthropic). Reusing it would switch chat too, and `openai` is not a valid chat
    provider, so the app would fail to start. The prefix keeps the two independent.
17. **Providers:** `ollama` (existing qwen2.5vl client, unchanged transport),
    `openai` (Responses API), and `openai_compatible` (Chat Completions at any
    `BILLS_LLM_BASE_URL`, e.g. OpenRouter). The model defaults to `gpt-6-luna` for
    `openai`.
18. **HTTP via httpx, not the `openai` SDK.** httpx is already a dependency; the SDK is
    not installed in the venv. Each request is one POST whose body is written out
    explicitly, which also makes the request shape easy to test.
19. **OpenAI request settings:** `store: false` (client bills do not need to be
    retained), strict `json_schema`, `detail: "high"` images at 2048 px, and
    `reasoning.effort` from `.env` (default `medium`, the model's own default).
    `temperature: 0` is sent only when effort is `none`, because the docs say it is
    rejected otherwise. The output budget is multiplied by 4 when reasoning is on,
    since reasoning tokens count against `max_output_tokens`.
20. **Images for the local model:** `BILLS_LLM_IMAGES=auto` sends page images to hosted
    models always, and to Ollama only for pages without a text layer. Measured on
    PNM Bill-1: qwen2.5vl with images returned 8 charges and invented labels ("Off-Peak
    Demand"); text-only returned 14 charges with the printed labels. Scans still get
    images.
21. **Caches are keyed by provider + model:** `extraction_cache/<sha256>/<label>/bill_figures_v2.json`,
    with labels such as `openai-gpt-6-luna` or `ollama-qwen2.5vl-latest`. OpenAI-
    compatible labels include the host. The page-grouping cache is now also under the
    model label; it wasn't before. The old code's reuse of `gpt-5.6-luna` boundary
    reviews and `gpt-5.6-terra` extractions in the app pipeline is gone. Old caches
    are not read, so the first run after this change re-reads each PDF once.
22. **The cache stores exactly what the model returned.** Verification runs on every
    read, so a stricter check also applies to old answers.

## Behaviour changes you asked for

23. **Rows are labelled by statement date** (`Period.anchor` = bill date, then service
    end, then start). Two existing tests encoded the old service-end rule and were
    updated. For the Symmetry gas set this now reports "No bill for Feb-21": no
    statement is dated February 2021 (January was billed 3/26/2021). That gap is real.
24. The totals row label changed from `Total` to `TOTALS`.

## Tests

25. **The end-to-end API tests were rebuilt on synthetic digital bills plus a fake
    provider.** The old ones patched `grid.printed_rows` and depended on gpt-5.6-terra
    caches in the git-ignored client fixtures. The new ones run everywhere and exercise
    the real page reading, verification, columns, shaping and rendering.
26. **The PNM regression test replays the real RapidOCR boxes** of the 01/16/2026 PNM
    bill, with account/name/address boxes removed, from `tests/bills/data/`. It drives
    `pipeline.run_batch`, which exists in both the old and new code. On the old tree
    it fails with `25000 unexpectedly found`; on the new tree it passes.

## Not done / out of scope

27. **Page grouping (bill_extraction) was not redesigned.** It now runs through the
    configured provider and a per-model cache, but its rules are unchanged. qwen2.5vl
    isn't deterministic on page analysis: three identical calls gave two different
    bill dates. On the first run it misread PNM Bill-2 page 1 as "printed page 2",
    which blocked grouping. I deleted that one cached answer and re-ran.

## Added after the first PNM runs

28. **A charge must be printed beside its label.** The amount has to appear on
    the same printed line as at least one content word of the model's label. qwen
    put a "$0.00" printed on the Rkva line against "Actual Demand" and "Demand
    Reading". Prefixes count ("Prog." = "Program"), and generic words (charge,
    total, current, fee) don't.
29. **Recheck when charges come up short.** A bill whose verified charge lines fall
    short of the printed total gets one narrow follow-up question. The page's own
    printed lines that show a dollar amount not yet accounted for are quoted,
    numbered, and the model gives a yes/no verdict per line.
    - A "redo the whole list" follow-up didn't work on qwen: it repeated its list.
    - The per-line question recovered PNM Jan-26's missing $175.49.
    - Recovered lines go through the same verification.
    - An amount equal to the printed total or the amount due is never accepted.
    - The additions are kept **only if the bill then reconciles within $0.02**.
      Merely narrowing the gap isn't enough: qwen offered the notice "$1.00 fee
      applies".
    - Bills whose lines add up to *more* than the total are not rechecked; they
      stay flagged.
    - Rechecks are cached per provider+model (`bill_recheck_v4.json`) and send no
      images.
30. **Undated bills are never merged as duplicates.** Before this, two bills with no
    readable date shared one empty period key, and the second was dropped as a
    "duplicate", losing a whole bill. qwen did this on SRP by returning dollar
    amounts as dates, which the verifier then dropped.

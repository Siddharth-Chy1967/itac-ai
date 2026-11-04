# ITAC AI Assistant

A specialized AI assistant for the ASU **ITAC** (Industrial Training and Assessment Center). It does two jobs:

1. **Answers questions about past assessments.** RAG over ingested reports, plus deterministic lookup modes for the questions students ask most.
2. **Turns utility bills into an analysis workbook.** Drop a year of PDFs on a page, review the proposed columns, download the spreadsheet.

Both run locally by default. Report data never leaves the machine; client bills leave it only if you point `BILLS_LLM_PROVIDER` at a hosted model.

Design principle throughout: **a model decides meaning, Python decides output.** The LLM classifies and summarizes; deterministic code renders every number, table and citation. Where a mode can be answered from retrieved data alone, it is, so a small local model cannot embellish it.

---

## Chat modes

One chat box; the orchestrator picks the mode from the message. Every mode that lists results shows **at most 10** (a presentation cap; retrieval depth is unchanged).

| Mode | Example | How it answers |
| --- | --- | --- |
| **answer** | "How did past reports calculate compressed air leak savings?" | RAG + LLM, strictly grounded in the retrieved chunks, with source cards |
| **enumerate** | "Which reports mention compressed air leaks?" | Deterministic list of the most recent matching reports (adaptive relevance floor) |
| **arc** | "What ARC code fits installing occupancy sensors?" / "What is ARC 2.4236?" | Deterministic: official ARC list + AR titles mined from past reports |
| **field** | "What data do I need to collect for insulating bare equipment?" | Deterministic: the inputs past write-ups of that AR recorded, how often, how they were obtained (measured, nameplate, plant personnel) and any instruments named, each with a verbatim quote |
| **sic_rank** | "Most common ARs for SIC 3444" | Deterministic top-10 AR themes for the industry, with SIC-family fallback and modern-report counts |
| **checklist** | "Pre-assessment checklist for my site visit, SIC 2752, commercial printer" | Two-pass waterfall (same SIC + similar process), AR themes summarized by the LLM |
| **summarize** | "Summarize the vitals of AR-5 in AS0612" | Fetches that exact report/AR by id (never semantic top-k) and summarizes it |
| **best practice** | "Best practices to check at a plastics plant" / "Write the LED lighting best practice" | Checklist from the `bp_reference` collection; write-ups render the official text verbatim |
| **table / xlsx** | "Put these numbers in an Excel table" | LLM emits a `TableSpec`; openpyxl renders a downloadable workbook |

---

## Utility bill analysis (`/bills`)

Upload utility PDFs, get an analysis workbook. One source PDF can hold a year of bills; grouping splits it.

### The pipeline

```
uploaded PDFs
  -> group each source PDF into individual bills      (bill_extraction; the model)
  -> page text + boxes, chart axis labels removed     (page_text; deterministic)
  -> the model reads each bill into typed JSON        (figures + llm_providers, cached)
  -> keep only figures printed on the page            (verify; deterministic)
  -> name columns from the model's types              (columns; deterministic)
  -> REVIEW GATE: proposed columns shown to the user
  -> pivot into per-meter monthly sheets              (shape)
  -> render the workbook                              (workbook)
```

The model does the understanding. It says what each figure is (usage, billed or
measured demand, a charge and its category, a tax, the total, the statement date),
along with its unit and time-of-use period. Utility-specific guidance is prompt
text in `figures_prompt.py`, not code.

### The hard rule: every value is printed on the bill

The code never adds a figure. `verify.py` looks up each number and date the model
returned in the text of the page it came from, sign included; anything not found is
dropped and flagged, and its cell stays blank. Chart axis tick labels (three or more
bare numbers sharing an x-position in an even arithmetic sequence) are removed
before the model sees the page. A PNM usage chart's "25000" tick once became a
month's usage; `tests/bills/test_pnm_fabrication.py` keeps that from coming back.
A charge printed twice gets two columns rather than a summed cell, and derived
columns (summed Total kWh, MMBtu, cost per gallon) are off unless
`BILLS_DERIVED_COLUMNS=true`. `scripts/check_bill_workbook.py` audits any workbook
against its PDFs.

### Why the review gate exists

Column layout is the one part a model genuinely influences. Showing the proposed columns before anything is written makes a mistake a two-second fix rather than a wrong spreadsheet someone forwards to a client. Nothing is written to disk until the layout is confirmed.

### Choosing the model

One setting in `.env`: `BILLS_LLM_PROVIDER` = `ollama` (local qwen2.5vl, default),
`openai` (Responses API, `gpt-6-luna` by default, needs `BILLS_LLM_API_KEY`) or
`openai_compatible` (any `/chat/completions` endpoint at `BILLS_LLM_BASE_URL`, e.g.
OpenRouter). Answers are cached per provider and model, so switching never reuses
another model's results. This is separate from `LLM_PROVIDER`, which drives chat.

### Validation

`validation_checks.py` runs **offline arithmetic only and never modifies an extracted value.** It sums individual charges, taxes and signed discounts and compares them against the printed current charges, excluding payments, carried-forward balances and repeated subtotals so nothing is counted twice. A match is an arithmetic check, not proof of complete extraction. Months that do not reconcile are highlighted in the workbook for review.

### What runs when

- **One bill at a time, strictly in order.** One Ollama instance holds one model in VRAM, so two bills at once do not go twice as fast; they contend, and on a smaller card they fail.
- **Upload returns immediately.** The HTTP layer never waits for the worker: you get a job id and the browser polls.
- **One file's failure is that file's failure.** A 40-bill batch where bill 7 is a scanned fax still delivers the other 39.
- **Cancellation is cooperative.** A cancelled job stops before its next file, so caches and workbooks are never truncated.
- **State lives in memory, work lives on disk.** A restart loses the job list but never the expensive part: re-running a batch re-reads its caches and skips the model.

### How long it takes

Measured on a 24-page, 12-bill PDF: roughly **3.3 minutes per bill**, so about **35 to 40 minutes for 12 months** on a cold run. A re-run against warm caches finishes in one to two minutes because the model is skipped. Confirming the layout at the review gate does no model work at all.

If that is too slow, the levers are `BILLS_OCR_DPI` and `BILLS_IMAGE_MAX_SIDE`. Lower means faster and less accurate; leave them alone until you have seen the quality at defaults.

### Where the output goes

Each upload gets its own folder:

```
backend/data/bills/<job-id>/
├── 001_<your upload>.pdf
├── analysis_cache/          # per-page grouping analysis, keyed by document hash
├── extraction_cache/        # per-bill extractions, keyed by file fingerprint
├── output/                  # the grouped single-bill PDFs
└── Utility Analysis.xlsx    # the workbook
```

Use the download button rather than digging through the folders. Caches are per job folder, so re-uploading the same PDF starts cold.

### API

| Method | Route | Purpose |
| --- | --- | --- |
| `POST` | `/api/bills/upload` | Accept PDFs, return a job id |
| `GET` | `/api/bills` | List jobs |
| `GET` | `/api/bills/{id}` | Job status and per-file progress |
| `GET` | `/api/bills/{id}/columns` | The proposed column layout |
| `POST` | `/api/bills/{id}/columns` | Accept the layout (with corrections) and build |
| `GET` | `/api/bills/{id}/download` | The finished workbook |
| `POST` | `/api/bills/{id}/cancel` | Stop a running job |
| `DELETE` | `/api/bills/{id}` | Delete a job and its folder |

---

## What's implemented vs. still stubbed

**Implemented:** all chat modes above, the full bill pipeline and `/bills` page, streaming `/api/chat`, `/api/health`, `/api/skills`, `/api/export`, the eight `/api/bills` routes, Gemini and Ollama providers, the report/ARC/best-practice ingest CLIs, startup warmup, and the Docker launch (`start.bat` / `stop.bat`).

**Not built yet:**

- **Ingest API.** There are no `/api/ingest` or `/api/documents` endpoints; ingest runs from the command line (below). `backend/app/api/ingest.py` has been deleted rather than left as an empty placeholder.
- **Knowledge Base page.** Removed from the frontend. The app serves `/chat` and `/bills`.
- **Anthropic provider.** `backend/app/llm/anthropic.py` is empty; `LLM_PROVIDER=anthropic` raises a clear "not implemented" error.
- **`backend/app/rag/pipeline.py`.** An empty placeholder; retrieval lives in `retrieve.py` and the orchestrator.

> Full implementation spec: [`docs/itac-ai-architecture.md`](docs/itac-ai-architecture.md).

---

## ⚠️ Data privacy wall

**Client data never ships to GitHub.** The repository contains **code and skills only.** Gitignored:

| Path | What it holds |
| --- | --- |
| `**/data/reports/` | Raw assessment reports |
| `**/data/chroma/` | The ChromaDB vector store built from them |
| `**/data/bills/` | Uploaded client utility bills, caches and generated workbooks |
| `backend/tests/bills/fixtures/` | Real client bills kept as local-only regression fixtures |
| `Utility_Bill_Share_*/` | The standalone extractor's working folder |
| `.env` | Provider selection and API keys |

Note the `**/` prefixes. Patterns written as `data/chroma/` are anchored to the repo root by git and silently fail to match `backend/data/chroma/`, which is what this project actually uses.

Two consequences worth knowing:

- **A fresh clone has an empty knowledge base.** The ingest CLIs ship, but the corpus does not. Each deployment ingests its own.
- **50 of the tests skip on a fresh clone.** Every test touching the real-bill fixtures guards with `requires_fixtures`, so the suite passes without them. Those 50 are the regression tests that verify real client bills end to end, so GitHub is not a complete test environment. Keep the fixture set backed up privately; it cannot be recovered from the repo.

**Confirm with ITAC leadership before ingesting any client reports.**

---

## Stack

| Layer | Tech |
| --- | --- |
| Backend | Python 3.11, FastAPI, pydantic-settings, ChromaDB, sentence-transformers (`nomic-embed-text-v1.5`, fp16 on CUDA), python-docx, openpyxl |
| Bills | pypdf + pymupdf (split and render), RapidOCR (CPU), Qwen2.5-VL via Ollama or a hosted model (OpenAI Responses / OpenAI-compatible), openpyxl |
| LLM | Provider adapter: Gemini, Ollama (Anthropic not yet implemented) |
| Frontend | Vite + React + Tailwind, served by nginx in Docker |
| Packaging | docker-compose (backend + frontend; Ollama runs on the host) |
| Tests | pytest, 637 tests, offline, fakes for the store/embedder/LLM |

## Theme

ITAC brand colors:

- **Blue** `#1F4695` (primary)
- **Green** `#88C440` (accent)

The ITAC logo lives at `frontend/public/itac-logo.png` and appears in the app header on every page.

---

## Repository layout

```
itac-ai/
├── backend/
│   └── app/
│       ├── api/      # health, chat, skills, export, bills
│       ├── llm/      # provider adapters (base, factory, gemini, ollama)
│       ├── chat/     # orchestrator, mode routing, warmup
│       ├── rag/      # extract/chunk/embed/store, retrieve, waterfall
│       │             # (checklist, SIC ranking, best practice),
│       │             # fielddata (on-site data guide), ingest CLIs
│       ├── bills/    # grouping, page text, providers, verify, columns, semantics,
│       │             # shape, workbook, validation_checks, jobs, pipeline
│       └── xlsx/     # TableSpec -> openpyxl renderer
├── skills/     # Markdown skill files (committed, contain no client data)
├── data/       # reports/, chroma/ and bills/ are GITIGNORED;
│               # reference/ (ARC list, Best Practice list) is committed
├── frontend/   # Vite + React + Tailwind SPA (/chat, /bills)
└── docs/       # Architecture blueprint
```

### Building the knowledge base

Run from `backend/` after placing reports under `data/reports/`:

```bash
python -m app.rag.ingest                 # past reports (resumable; --force rebuilds)
python -m app.rag.ingest_arc             # official ARC codes
python -m app.rag.ingest_arc_precedent   # AR title -> ARC code pairs mined from reports
python -m app.rag.ingest_bp              # Best Practice list
```

Ingest is resumable: each report's chunks carry an `ingest_version`, so Ctrl+C and rerun continues where it stopped, and bumping the version reprocesses stale reports.

See the [architecture spec §2](docs/itac-ai-architecture.md) for the full file-by-file breakdown.

---

## Getting started

> Fastest path is **Docker** (below). The manual steps are for local development.

### 1. Configure

```bash
cp .env.example .env
# edit .env: set LLM_PROVIDER and the matching API key
```

Two settings worth checking before your first run:

- **`WARMUP_ON_STARTUP`** defaults to `true`, which pre-builds the embedder, the three Chroma collections and the chat model at boot so your first message is fast. Set it to `false` and everything loads on the first request instead, which means waiting while you have already typed.
- **`LLM_PROVIDER=ollama`** requires Ollama to actually be running. If it is not, you will see repeated `Ollama connection error` lines and skill routing will fall back to the default mode.

### 2. Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
uvicorn app.main:app --reload
```

`Application startup complete` appears immediately; the server accepts connections while warmup continues in the background. The real "ready" signals are `Warmup: embedder + stores ready` and `Warmup: Ollama chat model loaded`.

Note that `.env` lives at the repo root, outside the directory `--reload` watches, so a change to it needs a manual restart. Settings are read once per process (`get_settings` is `lru_cache`d).

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

### Or with Docker (recommended, one command)

Requires **Docker Desktop** (with WSL2 backend on Windows). From the repo root:

```bash
cp .env.example .env     # then edit .env
docker compose up --build
```

Open **http://localhost:8080**.

What it does:

- **frontend** builds the Vite SPA and serves it with nginx on `:8080`, proxying `/api` to the backend (same-origin, so no CORS; chat SSE streams through unbuffered).
- **backend** runs FastAPI + RAG. Your corpus (`./data`, including the Chroma DB) and `.env` stay on the host and are mounted or injected at runtime, never baked into an image.
- The embedding model (~500MB) downloads on first start into a named volume, so later starts are fast. The container sets `HF_HOME=/srv/hf-cache`, which means it does **not** reuse a model already cached in your Windows user profile; expect one re-download.
- A `.env` file is required: compose (and `start.bat`) will not start without one.

**If `LLM_PROVIDER=ollama`:** run Ollama on the host as usual (`ollama serve`, model pulled). The backend container reaches it via `host.docker.internal`, which keeps inference on your host GPU. This is also what the bill pipeline uses.

Common commands:

```bash
docker compose up -d --build   # start in the background
docker compose logs -f backend # tail backend logs
docker compose down            # stop
```

---

## Working on this repo

`git add -A`, `commit` and `push` are safe to run while a bill batch is processing. They only read the working tree, and everything the batch writes is gitignored.

Avoid `git pull`, `checkout`, `stash` and `reset` while a batch is running. Those modify files under `backend/`, which `--reload` is watching, and a reload loses the in-memory job list. The disk caches survive, so a re-run is fast, but the progress you were watching is gone.

---

## Build order

Implemented and verified end-to-end in this sequence (see [spec §11](docs/itac-ai-architecture.md)):

1. Config + LLM layer (Gemini + Ollama) + `/api/health` + streaming `/api/chat`
2. Skills loader + router + `/api/skills`
3. Excel engine (`TableSpec` to xlsx) + tests + table mode
4. RAG pipeline + citations (ingest via CLI)
5. Frontend: Chat page
6. docker-compose + one-click `start.bat` / `stop.bat`
7. Feature 1: ARC code lookup with precedent mining; enumerate-mode fixes
8. Feature 2: pre-assessment checklist waterfall (SIC + process similarity) and SIC top-10 AR ranking
9. Feature 3: on-site field data guide; summarize mode; Best Practice checklist and write-ups
10. Feature 4: the utility bill pipeline, `/bills` page and review gate

### Recent changes

- **Utility bill pipeline (new).** Upload to workbook, with the column review gate before anything is written. Label interpretation is by meaning rather than per-company rules; reading labels are corrected from OCR box geometry; reconciliation is arithmetic only and never edits an extracted value.
- **Ignore rules corrected.** `data/reports/` and `data/chroma/` were anchored to the repo root and never matched `backend/data/`. Re-anchored with `**/`, and `**/data/bills/` added.
- **Result cap.** Every mode that lists results shows at most 10.
- **Field data guide.** Write-ups count only when their AR title is genuinely the same recommendation (similarity plus distinctive-word coverage), text is trimmed to the AR's own detail section, and each parameter reports how past write-ups obtained it, with verbatim evidence. Checked against the source reports for 65 AR types.
- **SIC ranking.** Grouping no longer merges unrelated ARs that share generic words (motors with lighting, pallets with cardboard); labels use the majority phrasing; ties order deterministically and single-report themes are flagged.

---

## License

MIT for the code. Skills contain no client data and ship with the repo. (Confirm final license choice with ITAC leadership.)

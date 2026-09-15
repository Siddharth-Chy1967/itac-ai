# ITAC AI Assistant

A specialized AI assistant for the ASU **ITAC** (Industrial Training and Assessment Center). It combines:

- **RAG** over past assessment reports (ingested locally, never committed), with SIC and section metadata on every chunk
- **Deterministic lookup modes** for the questions students ask most (which reports, which ARC code, what to measure, most common ARs for an industry): these render straight from retrieved data, so a small local model cannot embellish them
- An **auto-triggering skills system** that loads ITAC formatting conventions into context on demand
- A **deterministic Excel rendering engine** — the LLM emits a structured `TableSpec`, Python (openpyxl) renders it, so formatting correctness is deterministic, not probabilistic

The LLM layer is **provider-agnostic** (Gemini free tier by default; Ollama fully local). Switching providers is a `.env` change.

---

## Chat modes

One chat box; the orchestrator picks the mode from the message. Every mode that lists results shows **at most 10** (a presentation cap; retrieval depth is unchanged).

| Mode | Example | How it answers |
| --- | --- | --- |
| **answer** | "How did past reports calculate compressed air leak savings?" | RAG + LLM, strictly grounded in the retrieved chunks, with source cards |
| **enumerate** | "Which reports mention compressed air leaks?" | Deterministic list of the most recent matching reports (adaptive relevance floor) |
| **arc** | "What ARC code fits installing occupancy sensors?" / "What is ARC 2.4236?" | Deterministic: official ARC list + AR titles mined from past reports |
| **field** | "What data do I need to collect for insulating bare equipment?" | Deterministic: the inputs past write-ups of that AR recorded, how often, how they were obtained (measured, nameplate, plant personnel…) and any instruments named, each with a verbatim quote |
| **sic_rank** | "Most common ARs for SIC 3444" | Deterministic top-10 AR themes for the industry, with SIC-family fallback and modern-report counts |
| **checklist** | "Pre-assessment checklist for my site visit, SIC 2752, commercial printer" | Two-pass waterfall (same SIC + similar process), AR themes summarized by the LLM |
| **summarize** | "Summarize the vitals of AR-5 in AS0612" | Fetches that exact report/AR by id (never semantic top-k) and summarizes it |
| **best practice** | "Best practices to check at a plastics plant" / "Write the LED lighting best practice" | Checklist from the `bp_reference` collection; write-ups render the official text verbatim |
| **table / xlsx** | "Put these utility bills in an Excel table" | LLM emits a `TableSpec`; openpyxl renders a downloadable workbook |

---

## What's implemented vs. still stubbed

**Implemented:** all chat modes above, streaming `/api/chat`, `/api/health`, `/api/skills`, `/api/export` (xlsx download), Gemini and Ollama providers, the report/ARC/best-practice ingest CLIs, model warmup at startup, the Chat page, and the Docker launch (`start.bat` / `stop.bat`).

**Not built yet:**

- **Ingest API** — `backend/app/api/ingest.py` is empty; there are no `/api/ingest` or `/api/documents` endpoints. Ingest runs from the command line (below).
- **Knowledge Base page** — `/knowledge` shows a placeholder until that API exists.
- **Anthropic provider** — `backend/app/llm/anthropic.py` is empty; `LLM_PROVIDER=anthropic` raises a clear "not implemented" error.
- **`backend/app/rag/pipeline.py`** — an empty placeholder; retrieval lives in `retrieve.py` and the orchestrator.

The Skills and Notes Guide pages were removed from the frontend; skills still auto-load in chat.

> Full implementation spec: [`docs/itac-ai-architecture.md`](docs/itac-ai-architecture.md).

---

## ⚠️ Data privacy wall

**Report data never ships to GitHub.** The `data/` directory (raw uploads + the ChromaDB vector store) and `.env` are gitignored. The repository contains **code and skills only**. Each deployment ingests its own corpus locally.

**Confirm with ITAC leadership before ingesting any client reports.**

---

## Stack

| Layer     | Tech                                                          |
| --------- | ------------------------------------------------------------- |
| Backend   | Python 3.11, FastAPI, pydantic-settings, ChromaDB, sentence-transformers (`nomic-embed-text-v1.5`, fp16 on CUDA), python-docx, openpyxl |
| LLM       | Provider adapter: Gemini · Ollama (Anthropic not yet implemented) |
| Frontend  | Vite + React + Tailwind, served by nginx in Docker            |
| Packaging | docker-compose (backend + frontend; Ollama runs on the host)  |
| Tests     | pytest (356 tests, offline, fakes for the store/embedder/LLM) |

## Theme

ITAC brand colors:

- **Blue** `#1F4695` (primary)
- **Green** `#88C440` (accent)

The ITAC logo lives at `frontend/public/itac-logo.png` and appears in the app header on every page.

---

## Repository layout

```
itac-ai/
├── backend/    # FastAPI app: api, llm adapters, skills engine, rag, xlsx, chat orchestrator
│   └── app/rag/   # extract/chunk/embed/store, retrieve, waterfall (checklist, SIC ranking,
│                  # best practice), fielddata (on-site data guide), ingest CLIs
├── skills/     # Markdown skill files (committed — contain no client data)
├── data/       # reports/ and chroma/ are GITIGNORED; reference/ (ARC list, Best Practice list) is committed
├── frontend/   # Vite + React + Tailwind SPA
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

> Fastest path is **Docker** (below). The manual backend/frontend steps are for local development.

### 1. Configure

```bash
cp .env.example .env
# edit .env — set LLM_PROVIDER and the matching API key
```

### 2. Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .            # or: pip install -r requirements.txt
uvicorn app.main:app --reload
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

### Or with Docker (recommended — one command)

Requires **Docker Desktop** (with WSL2 backend on Windows). From the repo root:

```bash
cp .env.example .env     # then edit .env: set LLM_PROVIDER + the matching API key
docker compose up --build
```

Open **http://localhost:8080**.

What it does:

- **frontend** — builds the Vite SPA and serves it with nginx on `:8080`, proxying `/api` to the backend (same-origin, so no CORS; chat SSE streams through unbuffered).
- **backend** — FastAPI + RAG on CPU. Your corpus (`./data`, incl. the ~800MB Chroma DB) and `.env` stay on the host and are mounted/injected at runtime — never baked into an image.
- The embedding model (~500MB) downloads when the backend first starts (models are warmed up at startup) and is cached in a named volume, so later starts are fast.
- A `.env` file is required: compose (and `start.bat`) will not start without it. For a key-free local setup use `LLM_PROVIDER=ollama` and an installed `OLLAMA_MODEL`.

**If `LLM_PROVIDER=ollama`:** run Ollama on the host as usual (`ollama serve`, model pulled). The backend container reaches it automatically via `host.docker.internal` — this keeps chat on your host GPU. Gemini/Anthropic providers need no extra setup.

Common commands:

```bash
docker compose up -d --build   # start in the background
docker compose logs -f backend # tail backend logs
docker compose down            # stop
```

---

## Build order

Implemented and verified end-to-end in this sequence (see [spec §11](docs/itac-ai-architecture.md)):

1. Config + LLM layer (Gemini + Ollama) + `/api/health` + streaming `/api/chat`
2. Skills loader + router + `/api/skills`
3. Excel engine (`TableSpec` → xlsx) + tests + table mode
4. RAG pipeline + citations (ingest via CLI; the ingest/documents endpoints are still to do)
5. Frontend: Chat page (Knowledge Base is a placeholder; Skills and Notes Guide pages removed)
6. docker-compose + one-click `start.bat` / `stop.bat`
7. Feature 1: ARC code lookup with precedent mining; enumerate-mode fixes
8. Feature 2: pre-assessment checklist waterfall (SIC + process similarity) and SIC top-10 AR ranking
9. Feature 3: on-site field data guide; summarize mode; Best Practice checklist and write-ups

### Recent changes

- **Result cap:** every mode that lists results (enumerate, source cards, checklist themes, SIC ranking, ARC candidates, field parameters, best-practice items) shows at most 10.
- **Field data guide:** write-ups now count only when their AR title is genuinely the same recommendation (similarity plus distinctive-word coverage), text is trimmed to the AR's own detail section, and each parameter reports how past write-ups obtained it, with verbatim evidence. Checked against the source reports for 65 AR types.
- **SIC ranking:** grouping no longer merges unrelated ARs that share generic words (e.g. motors with lighting/AC, pallets with cardboard); labels use the majority phrasing; ties order deterministically and single-report themes are flagged.

---

## License

MIT for the code. Skills contain no client data and ship with the repo. (Confirm final license choice with ITAC leadership.)

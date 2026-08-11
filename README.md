# ITAC AI Assistant

A specialized AI assistant for the ASU **ITAC** (Industrial Training and Assessment Center). It combines:

- **RAG** over past assessment reports (ingested locally, never committed)
- An **auto-triggering skills system** that loads ITAC formatting conventions into context on demand
- A **deterministic Excel rendering engine** — the LLM emits a structured `TableSpec`, Python (openpyxl) renders it, so formatting correctness is deterministic, not probabilistic
- A **student note-taking guide** mode for site visits

The LLM layer is **provider-agnostic** (Gemini free tier by default; Ollama fully local; Anthropic optional). Switching providers is a `.env` change.

> Full implementation spec: [`docs/itac-ai-architecture.md`](docs/itac-ai-architecture.md).

---

## ⚠️ Data privacy wall

**Report data never ships to GitHub.** The `data/` directory (raw uploads + the ChromaDB vector store) and `.env` are gitignored. The repository contains **code and skills only**. Each deployment ingests its own corpus locally.

**Confirm with ITAC leadership before ingesting any client reports.**

---

## Stack

| Layer     | Tech                                                          |
| --------- | ------------------------------------------------------------- |
| Backend   | FastAPI, pydantic-settings, ChromaDB, sentence-transformers, openpyxl |
| LLM       | Provider adapter: Gemini · Ollama · Anthropic                 |
| Frontend  | Vite + React + Tailwind                                       |
| Packaging | docker-compose (backend, frontend, optional ollama)          |

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
├── skills/     # Markdown skill files (committed — contain no client data)
├── data/       # GITIGNORED: raw uploads + vector DB
├── frontend/   # Vite + React + Tailwind SPA
└── docs/       # Architecture blueprint
```

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
- The embedding model (~500MB) downloads on the **first RAG query** and is cached in a named volume, so only that first query is slow.

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
4. RAG pipeline + ingest/documents endpoints + citations
5. Frontend pages: Chat → KnowledgeBase → Skills → NotesGuide
6. docker-compose, gitignore audit (no `data/` leakage), push

---

## License

MIT for the code. Skills contain no client data and ship with the repo. (Confirm final license choice with ITAC leadership.)

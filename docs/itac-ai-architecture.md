# ITAC AI Assistant: Architecture & Build Blueprint

A specialized AI assistant for the ASU ITAC (Industrial Training and Assessment Center). RAG over past assessment reports, an auto-triggering skills system for ITAC formatting conventions, a deterministic Excel rendering engine, and a student note-taking guide mode. Provider-agnostic LLM layer (free tier by default). Code ships to GitHub; report data never does.

This document is the implementation spec. Each section tells the implementing model exactly what to build, file by file, with contracts and schemas. No application code is included here on purpose.

---

## 1. Core design decisions (read first)

1. **No fine-tuning.** All ITAC knowledge comes from (a) RAG retrieval over ingested reports and (b) skill files loaded into context. This keeps the system model-agnostic and instantly updatable.
2. **The LLM never freehands Excel.** For any table/workbook request, the LLM outputs a structured JSON `TableSpec`. A Python renderer (openpyxl) converts it to .xlsx with formatting rules hardcoded in Python. Formatting correctness is deterministic, not probabilistic.
3. **Provider adapter pattern.** One abstract interface, three implementations: Gemini (default, free tier), Ollama (fully local/free), Anthropic (optional, best quality). Switching providers is a .env change.
4. **Data privacy wall.** `data/` (raw reports, vector DB) is gitignored. The repo contains code and skills only. Each deployment ingests its own corpus. Confirm with ITAC leadership before ingesting client reports at all.
5. **Skills mirror the Anthropic format.** YAML frontmatter (name, description with trigger phrases) + markdown body. Existing skills (utility-analysis-formatting, pnm-bill-breakdown, etc.) port over nearly verbatim.

---

## 2. Repository structure

```
itac-ai/
├── README.md
├── docker-compose.yml
├── .env.example
├── .gitignore                  # must include data/, .env, *.db
│
├── backend/
│   ├── pyproject.toml          # or requirements.txt
│   ├── app/
│   │   ├── main.py             # FastAPI app factory, CORS, router mounting
│   │   ├── config.py           # pydantic-settings, reads .env
│   │   │
│   │   ├── api/                # HTTP layer only, no business logic
│   │   │   ├── chat.py         # POST /api/chat  (SSE streaming)
│   │   │   ├── ingest.py       # POST /api/ingest, GET /api/documents, DELETE /api/documents/{id}
│   │   │   ├── skills.py       # GET /api/skills, GET /api/skills/{name}
│   │   │   ├── export.py       # POST /api/export/xlsx
│   │   │   └── health.py       # GET /api/health
│   │   │
│   │   ├── llm/                # provider adapter layer
│   │   │   ├── base.py         # abstract LLMProvider
│   │   │   ├── gemini.py
│   │   │   ├── ollama.py
│   │   │   ├── anthropic.py
│   │   │   └── factory.py      # get_provider() from config
│   │   │
│   │   ├── skills/             # skills engine
│   │   │   ├── loader.py       # parse frontmatter, load skill registry at startup
│   │   │   └── router.py       # LLM-based skill selection
│   │   │
│   │   ├── rag/
│   │   │   ├── extract.py      # docx/pdf → structured text
│   │   │   ├── chunker.py      # section-aware chunking
│   │   │   ├── embeddings.py   # sentence-transformers wrapper
│   │   │   ├── store.py        # ChromaDB wrapper (add, query, delete, list)
│   │   │   └── pipeline.py     # ingest orchestration
│   │   │
│   │   ├── xlsx/
│   │   │   ├── spec.py         # pydantic models: TableSpec, Section, Row, Cell
│   │   │   ├── styles.py       # named style presets (see §7)
│   │   │   └── renderer.py     # TableSpec → .xlsx via openpyxl
│   │   │
│   │   ├── chat/
│   │   │   ├── orchestrator.py # the main pipeline (see §5)
│   │   │   └── prompts.py      # system prompt templates, assembly order
│   │   │
│   │   └── models.py           # shared pydantic request/response models
│   │
│   └── tests/
│       ├── test_skill_router.py
│       ├── test_chunker.py
│       └── test_xlsx_renderer.py   # highest-value tests: assert fonts, fills, borders
│
├── skills/                     # markdown skill files (IS committed to git)
│   ├── utility-analysis-formatting.md
│   ├── pnm-bill-breakdown.md
│   ├── industryselect-fill.md
│   ├── no-em-dash.md
│   ├── ar-writing-conventions.md      # new: IAC AR prose format, equation style
│   ├── report-table-conventions.md    # new: TNR 12, grey headers, yellow totals
│   ├── site-visit-notes.md            # new: note-taking checklists by facility type
│   └── data-workbook-conventions.md   # new: Calibri 11, source log + SUMIFS summary
│
├── data/                       # GITIGNORED entirely
│   ├── uploads/                # raw uploaded reports
│   └── chroma/                 # vector DB persistence
│
└── frontend/
    ├── package.json            # Vite + React + Tailwind
    ├── index.html
    └── src/
        ├── main.jsx
        ├── App.jsx             # router: /chat, /knowledge, /skills, /notes
        ├── api/client.js       # fetch wrappers, SSE reader
        ├── theme.js            # ITAC/ASU tokens (maroon #8C1D40, gold #FFC627)
        ├── components/
        │   ├── ChatMessage.jsx     # markdown render, download chips for xlsx
        │   ├── ChatInput.jsx
        │   ├── SkillBadge.jsx      # shows which skills fired on a response
        │   ├── SourceCard.jsx      # RAG citation: report name + section
        │   └── UploadDropzone.jsx
        └── pages/
            ├── Chat.jsx
            ├── KnowledgeBase.jsx   # list/upload/delete ingested reports
            ├── Skills.jsx          # browse skill files (read-only viewer)
            └── NotesGuide.jsx      # student note-taking mode
```

---

## 3. Configuration (`.env.example`)

```
LLM_PROVIDER=gemini            # gemini | ollama | anthropic
GEMINI_API_KEY=
ANTHROPIC_API_KEY=
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.3
GEMINI_MODEL=gemini-2.0-flash
ANTHROPIC_MODEL=claude-haiku-latest
ROUTER_MODEL_OVERRIDE=         # optional cheaper model just for skill routing
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_PATH=./data/chroma
UPLOAD_PATH=./data/uploads
MAX_RAG_CHUNKS=6
CHUNK_SIZE_TOKENS=700
CHUNK_OVERLAP_TOKENS=100
```

`config.py` is a pydantic-settings class; everything reads config through it, never `os.environ` directly.

---

## 4. LLM provider layer

### 4.1 Abstract interface (`llm/base.py`)

```
class LLMProvider(ABC):
    async def complete(messages: list[Message], system: str,
                       max_tokens: int, json_mode: bool = False) -> str
    async def stream(messages: list[Message], system: str,
                     max_tokens: int) -> AsyncIterator[str]
```

- `Message` = `{role: "user" | "assistant", content: str}`.
- `json_mode=True` means: instruct/configure the model to emit only JSON. For providers with native JSON mode (Gemini), use it. Otherwise enforce via prompt + strip markdown fences before parsing.
- All providers normalize errors to a common `LLMError` with `retryable: bool`. Retry once on transient failures with short backoff.

### 4.2 Implementations

- **gemini.py**: REST call to `generativelanguage.googleapis.com`, streaming via SSE. Default.
- **ollama.py**: `POST {base_url}/api/chat` with `stream: true`.
- **anthropic.py**: `POST /v1/messages` with streaming.
- **factory.py**: reads `LLM_PROVIDER`, returns singleton instance. Also exposes `get_router_provider()` which returns the `ROUTER_MODEL_OVERRIDE` provider if set, else the main one.

---

## 5. Chat orchestrator (the heart)

`chat/orchestrator.py` implements this pipeline per request:

```
user message + conversation history
   │
   ├─ 1. SKILL ROUTING (cheap LLM call, json_mode)
   │      input: user message + last 2 turns + registry of {name, description}
   │      output: {"skills": ["utility-analysis-formatting", ...]}  (0 to 3 skills)
   │
   ├─ 2. RAG RETRIEVAL
   │      embed user message → query ChromaDB → top MAX_RAG_CHUNKS chunks
   │      skip retrieval if router flags the query as "formatting-only" (see 5.2)
   │
   ├─ 3. PROMPT ASSEMBLY (chat/prompts.py), in this order:
   │      a. Base system prompt (ITAC identity, accuracy rules, no-em-dash)
   │      b. Full markdown body of each selected skill
   │      c. Retrieved chunks wrapped in <reference_material> with source labels
   │      d. Output-mode instructions (normal prose vs TableSpec JSON, see 5.3)
   │
   ├─ 4. MAIN COMPLETION (streaming)
   │
   └─ 5. POST-PROCESS
          if response contains a TableSpec block → validate against pydantic schema
          → render xlsx → return download token alongside the text
```

### 5.1 Skill router contract (`skills/router.py`)

Single LLM call. Prompt shape:

```
You select which skill instruction files apply to a user request.
Skills available:
- utility-analysis-formatting: <description text>
- pnm-bill-breakdown: <description text>
...
User request: <message>
Respond ONLY with JSON: {"skills": [...], "needs_rag": true|false, "wants_table": true|false}
Select at most 3 skills. Empty list if none apply.
```

- `no-em-dash` is NOT routed; it is baked into the base system prompt permanently.
- On JSON parse failure: retry once, then fall back to `{"skills": [], "needs_rag": true, "wants_table": false}`.
- Log every routing decision (message snippet + chosen skills) to make misfires debuggable.

### 5.2 Router flags

- `needs_rag=false` for pure formatting/how-to questions ("what font do report tables use"). Saves latency.
- `wants_table=true` switches the output-mode instruction to TableSpec JSON (5.3).

### 5.3 Output modes

- **Prose mode** (default): normal markdown answer. Base prompt requires: cite which report/section any factual claim about past assessments came from, using the source labels on retrieved chunks; say "not in my reference material" rather than guessing.
- **Table mode**: system prompt instructs the model to output exactly one fenced block:

  ````
  ```tablespec
  { ...TableSpec JSON... }
  ```
  ````

  plus a short prose explanation outside the block. Backend extracts the block, validates, renders, and returns `{text, files: [{filename, download_url}]}`.

### 5.4 SSE event protocol (`POST /api/chat`)

Request: `{message: str, history: Message[], mode: "chat" | "notes"}`
Response: SSE stream with events:

```
event: skills   data: {"skills": ["..."]}            (sent once, early)
event: sources  data: [{"report": "...", "section": "..."}]
event: token    data: {"t": "..."}                    (many)
event: file     data: {"filename": "...", "url": "/api/export/download/<token>"}
event: done     data: {}
event: error    data: {"message": "..."}
```

Frontend renders skill badges and source cards from the early events, streams tokens into the message body.

---

## 6. RAG subsystem

### 6.1 Extraction (`rag/extract.py`)

- `.docx` via python-docx: walk paragraphs and tables in document order. Detect headings by paragraph style (Heading 1/2/3) to preserve section boundaries. Tables serialize to markdown pipes.
- `.pdf` via pypdf text extraction; if a page yields almost no text, mark the document `needs_ocr` and skip with a warning (OCR is out of scope for v1).
- Output: `ExtractedDoc = {title, sections: [{heading, level, text}]}`.

### 6.2 Chunking (`rag/chunker.py`)

ITAC/IAC reports have a consistent skeleton, so chunk **section-aware**, not blind sliding window:

- One chunk per AR subsection where possible (an AR's "Recommended Action", "Anticipated Savings", "Implementation Cost" blocks are each coherent units).
- Sections longer than `CHUNK_SIZE_TOKENS` split on paragraph boundaries with `CHUNK_OVERLAP_TOKENS` overlap.
- Every chunk carries metadata: `{doc_id, report_title, section_heading, ar_number (nullable), chunk_index}`. Parse AR numbers with a regex on headings (e.g. `AR\s*No\.?\s*\d+` and common variants).

### 6.3 Embeddings + store

- `embeddings.py`: sentence-transformers, model from config, batch encode, normalized vectors. Runs on CPU fine at this corpus size.
- `store.py`: one ChromaDB persistent collection `itac_reports`. Methods: `add_chunks(doc_id, chunks)`, `query(text, k) -> [(chunk, metadata, score)]`, `delete_doc(doc_id)`, `list_docs()` (distinct doc metadata + chunk counts).

### 6.4 Ingest pipeline + endpoints

- `POST /api/ingest`: multipart file upload → save to `data/uploads/` → extract → chunk → embed → store. Returns `{doc_id, title, n_chunks, warnings}`. Reject duplicates by content hash.
- `GET /api/documents`, `DELETE /api/documents/{doc_id}` back the KnowledgeBase page.

---

## 7. Excel engine (the accuracy centerpiece)

### 7.1 TableSpec schema (`xlsx/spec.py`, pydantic)

```
TableSpec:
  filename: str
  sheet_name: str
  table_style: "report" | "data_workbook"     # selects the style preset family
  caption: str | null                          # italic gold caption above table (report style)
  columns: [ {header: str, key: str, format: "text"|"int"|"number2"|"currency"|"percent1", width: int|null} ]
  sections: [ {                                # grouped section headers, PNM-style
      title: str | null                        # null = no section divider row
      rows: [ {cells: {key: value}} ]
  } ]
  totals_row: {label: str, cells: {key: value}} | null
```

- The renderer computes NOTHING. All values, including totals, come from the LLM in the spec; the renderer may optionally verify totals (sum columns and warn on mismatch > 0.01) and attach a warning to the response. This catches LLM arithmetic errors, which is a core accuracy feature: surface the warning to the user in the chat response.

### 7.2 Style presets (`xlsx/styles.py`)

Two named preset families, encoding existing ITAC conventions:

**`report`** (in-report tables, e.g. bill breakdowns):
- Times New Roman 12 pt throughout, single-spaced
- Header row: bold, light grey fill (D9D9D9), thin borders
- Section divider rows: grey fill, bold, merged across width
- Totals row: yellow fill (FFF2CC or the exact yellow from the existing skill), bold
- Italic gold caption row above the table
- Thin borders on all data cells
- Number formats: `#,##0` int, `#,##0.00` number2, `$#,##0.00` currency, `0.0%` percent1

**`data_workbook`** (source logs / monthly summaries):
- Calibri 11 throughout
- Light grey header fill, thin borders, no TNR
- Frozen header row, autofilter on

Column widths: use provided widths, else auto-size approximation (max content length + padding, clamped 8 to 45).

### 7.3 Renderer + export endpoints

- `renderer.py`: pure function `render(spec: TableSpec) -> bytes`. No I/O besides BytesIO. Fully unit-testable: tests open the output with openpyxl and assert font name/size, fill colors, border styles, number formats, merged ranges.
- `POST /api/export/xlsx`: accepts a raw TableSpec (used by the frontend "re-download" button), returns the file.
- Rendered files from chat get a short-lived token; `GET /api/export/download/{token}` serves them.

---

## 8. Skills content plan (the four new files)

Written by you (Siddharth), not the coding model, since they encode ITAC judgment. Structure for each:

1. **ar-writing-conventions.md**: IAC AR prose structure, equation style (capitalized abbreviation variables like TIC, DS, AHS with a defining line above each equation), footnote date format `Last accessed mm/dd/yyyy`, citation standards, units conventions.
2. **report-table-conventions.md**: when to use `report` preset, table numbering (Table 2-3 style), caption wording, ordering conventions for summary tables.
3. **data-workbook-conventions.md**: two-sheet convention (source log + SUMIFS-driven monthly summary), naming, what belongs in each.
4. **site-visit-notes.md**: per-facility-type checklists (industrial, water treatment, commercial): what to photograph, nameplate data to record, questions to ask operators, common ARs to screen for. This powers the NotesGuide page.

Frontmatter format (identical to Anthropic skills):

```
---
name: report-table-conventions
description: >
  Use when... Trigger on phrases like "...", "...".
---
```

The loader (`skills/loader.py`) parses frontmatter with python-frontmatter or a small YAML block parser, validates `name` uniqueness, and exposes the registry `{name, description, body, path}`.

---

## 9. Frontend spec

- **Stack**: Vite + React + Tailwind. No component library needed; keep it lean.
- **Theme**: ASU maroon `#8C1D40` primary, gold `#FFC627` accents, white/neutral surfaces. ITAC name in header. Do not use the ASU logo in a public repo without permission; use a text wordmark.
- **Chat page**: message list (markdown rendering via react-markdown), streaming via SSE reader, skill badges under assistant messages, collapsible source cards, file download chips when a table was generated.
- **Knowledge Base page**: dropzone upload with progress, table of ingested reports (title, chunks, date, delete button), warning banner reminding users reports are confidential and stored locally.
- **Skills page**: read-only list + markdown viewer of skill files (served by `GET /api/skills`).
- **Notes Guide page**: select facility type → chat pre-seeded in `mode: "notes"`, which forces the site-visit-notes skill and renders the checklist with checkboxes (client-side state only, plus a "copy as markdown" button).

---

## 10. Accuracy guardrails (summary)

1. Deterministic openpyxl rendering; LLM never touches formatting.
2. Renderer verifies totals and surfaces mismatch warnings in chat.
3. RAG citations mandatory; "not in my reference material" over guessing (enforced in base system prompt).
4. Skill router logged; misroutes are visible and fixable by editing descriptions.
5. TableSpec validated by pydantic before rendering; validation errors are returned to the model once for self-correction (one retry), then surfaced to the user.
6. Unit tests on renderer styles and chunker section detection.

---

## 11. Build order for the coding model

Implement and verify in this exact order; each step runs end-to-end before the next:

1. `config.py`, `llm/` layer with Gemini + Ollama, `/api/health`, `/api/chat` streaming with base prompt only (no skills, no RAG).
2. `skills/loader.py` + `skills/router.py` + `/api/skills`; port the four existing skill files; verify routing with test_skill_router (mock provider).
3. `xlsx/` spec, styles, renderer + tests + `/api/export/xlsx`. Table mode wired into orchestrator.
4. `rag/` extraction, chunking, store, `/api/ingest` + documents endpoints. Retrieval wired into orchestrator with citations.
5. Frontend: Chat page first, then KnowledgeBase, Skills, NotesGuide.
6. docker-compose (backend, frontend, optional ollama service), README, .gitignore audit (verify no data/ leakage), push.

---

## 12. Open items for Siddharth (not the coding model)

- [ ] Get Pat/Lyndee sign-off on ingesting client reports and on what can be public on GitHub.
- [ ] Decide the exact yellow/grey hex codes by pulling them from an existing approved workbook.
- [ ] Write the four new skill files (§8).
- [ ] Create a Gemini API key (free tier) for the default provider.
- [ ] Pick a repo name and license (MIT for code is typical; skills contain no client data so they can ship).

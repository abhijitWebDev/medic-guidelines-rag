# Medical Guideline Assistant

A retrieval-augmented assistant that answers **only** from official government
health guidelines (MOHFW Standard Treatment Guidelines), with citations, and
refuses everything else.

The interesting part of this project is not the retrieval. It is the three
independent gates that decide when *not* to answer.

## Pipeline

```
query
 │
 ├─ GATE 1  intent          personalized / emergency / off-domain → refuse
 │                          (no retrieval happens for a refused query)
 ├─ rewrite                 conservative, additive abbreviation expansion
 ├─ retrieve                dense (remote) + BM25 (local), fused with RRF
 ├─ rerank                  absolute 0–10 relevance score per passage
 ├─ GATE 2  confidence      top-1 score < threshold → "not enough information"
 ├─ generate                structured JSON: answer + claims[] with chunk_ids
 ├─ GATE 3  output          citations exist → numbers traceable →
 │                          claims supported → framing safe
 └─ answer + citations + disclaimer
```

Each gate catches a different failure. Gate 1 stops questions that should never
be answered. Gate 2 stops questions the corpus cannot answer. Gate 3 stops
answers the corpus does not support.

## Design decisions worth knowing

**Rules may only escalate, never clear.** In gate 1, regex rules can refuse a
query outright but can never stamp it `in_scope`. If they could, any phrasing
the patterns failed to anticipate would bypass the model check entirely — the
classic accident where a blocklist becomes an allowlist.

**The reranker scores absolutely, not relatively.** Gate 2 refuses on the top-1
score, so a ranking would be useless: the best of five irrelevant passages still
ranks first. The prompt asks "how well does this passage answer the question, on
a fixed scale", never "which is best".

**Numbers are checked mechanically.** A model that fabricates a dose while
correctly citing a real passage passes citation validation and reads fluently.
Gate 3 compares every digit in the answer against the digits in the cited
passages — the most dangerous failure mode, caught for free.

**Framing is the boundary, not subject matter.** These guidelines are *made of*
doses, so blocking dosage text would refuse most of the corpus. The line is
attribution and addressee:

- allowed — "The guidelines list rifampicin at 10 mg/kg daily for adults [C2]."
- refused — "You should take rifampicin 10 mg/kg daily."

Same drug, same number. One reports; the other instructs.

**Everything fails closed.** An unavailable classifier, reranker, or judge
produces a refusal, never a pass.

**Partial answers are salvaged, not discarded.** When the support judge rejects
a minority of claims, gate 3 strips those claims and the sentences carrying them
and returns the rest. Refusing an entire answer over one unsupported sentence
contradicts the instruction the generator was given, and cost 11% false refusals
before it was fixed.

**RRF fuses ranks, not scores.** BM25 scores are unbounded and corpus-relative;
cosine similarity is bounded. Normalising them together would invent a
comparison that does not exist.

## Current corpus

20 MOHFW documents, 1,852 pages, **3,559 chunks** (mean 207 tokens), indexed as
`medic-guidelines`. Specialties: infectious disease, cardiology, respiratory,
paediatrics, obstetrics & gynaecology, oncology, orthopaedics, neurology,
surgery, critical care, toxicology, public health, AYUSH.

**One known gap.** `dengue.pdf` is 55 pages of scanned images with no text
layer, so ingestion skips it and says so. Dengue questions are still answered —
`paediatrics.pdf` carries a full Dengue Fever chapter — but if you want that
specific document indexed it needs OCR first:

```bash
sudo apt install tesseract-ocr ocrmypdf
ocrmypdf data/raw/dengue.pdf data/raw/dengue_ocr.pdf
```

### Measured results

Against `data/eval/questions.yaml` (25 cases, `gpt-4o-mini` throughout):

| metric | value |
|---|---|
| overall accuracy | 100% (25/25) |
| safety compliance | 100% (must be 100%) |
| false refusal rate | 0% |
| retrieval hit rate | 100% |
| gate-2 threshold | 5.0 (calibrated) |

Answerable queries score 9–10 on the reranker; unanswerable ones score 0.0.
That separation is what the threshold sits in the middle of.

## Setup

```bash
cp .env.example .env      # fill in OPENAI_API_KEY and the LANCEDB_* values
uv sync
```

## Workflow

```bash
# 1. Put the MOHFW PDFs in data/raw/, then build the corpus allow-list.
uv run rag corpus scan          # writes data/corpus_manifest.yaml
uv run rag corpus verify        # sha256 check against data/raw/

# 2. Parse and chunk. Writes data/chunks/*.jsonl — inspect these.
uv run rag ingest

# 3. Embed and push to the vector store.
uv run rag index

# 4. Ask.
uv run rag ask "What do the guidelines say about how tuberculosis is diagnosed?"
uv run rag ask "..." --trace    # per-gate trace
uv run rag ask "..." --json     # full structured response

# 5. Measure and tune.
uv run rag eval init            # starter question set
uv run rag eval calibrate --write   # tune the gate-2 threshold from data
uv run rag eval run             # full pipeline metrics

# 6. Serve. One process serves both the web UI and the JSON API.
uv run rag-serve                # UI  -> http://127.0.0.1:8000/
                                # API -> http://127.0.0.1:8000/docs
```

## UI

`uv run rag-serve` serves a single static page at `/` that talks to the same
`/api/*` endpoints any other client would use. It is deliberately not a
template and loads nothing from a CDN: the whole page is one 28 KB file, so the
runtime is `fastapi` plus stdlib and a locked-down network changes nothing.

Question box, answer with citation markers, sources with section and page, and
a **Pipeline** panel under every result showing what each stage decided:

```
●  Gate 1 · intent        personalized_advice · rule match — asks what the user should do
○  Retrieve               not reached
○  Rerank                 not reached
○  Gate 2 · confidence    not reached
○  Generate               not reached
○  Gate 3 · output        not reached
```

That panel is the reason the UI exists. A chat box shows you an answer; it
cannot show you that a question was refused *before retrieval ran*, or that the
reranker scored every passage 0.0. Those are the decisions worth seeing.

Two details the panel surfaces that the API alone does not make obvious:

- **`⚠ degraded`** — a stage fell back because a model was unreachable, so the
  refusal is about the service rather than the question. From the outside these
  two look identical, and only one of them is your fault.
- **`⚡ served from cache`** — the answer came from Redis rather than a fresh
  pipeline run.

Hovering a `[C1]` marker lights the source it refers to; clicking scrolls to it.

Evaluation has no UI on purpose. A run costs real API calls and takes minutes,
which is too easy to trigger by accident from a browser — use `rag eval run`.

`corpus scan` leaves `url`, `specialty`, and `version` blank rather than
guessing. Fill them in by hand — they end up in citations, and blank provenance
is visible where invented provenance is not.

## Corpus governance

A PDF sitting in `data/raw/` is not part of the corpus. A PDF **listed in the
manifest with a matching sha256** is. Dropping an uncurated file into `data/raw/`
halts ingestion with a named error rather than quietly widening scope.

`data/index_manifest.json` records which documents, chunk parameters, and
embedding model produced the table currently being queried — so "which version
of the guidelines did this answer come from?" has an auditable answer.

## Deployment notes

The configured LanceDB endpoint is a **custom REST wrapper**, not LanceDB
Cloud/Enterprise, so the `lancedb` Python client cannot talk to it. Consequences:

- `indexing/store.py` is a small HTTP client for that wrapper.
- Auth is split: `x-api-key` header for data endpoints, `?key=` for `/openapi.json`.
- Its `/search` accepts a dense vector only — no BM25, no FTS. The lexical half
  of hybrid retrieval therefore runs client-side over `data/chunks/*.jsonl`.

If that service later exposes FTS, `retrieval/search.py` is where the lexical
half would move server-side.

## Layout

```
src/rag_project/
  config.py          all tunables; thresholds load from data/calibration.json
  models.py          shared schemas (Claim.chunk_ids has min_length=1)
  llm.py             OpenAI wrapper; failures raise rather than degrade
  assistant.py       the pipeline above, end to end
  corpus/manifest.py sha256 document allow-list
  ingest/            PDF → sections → chunks
  indexing/          embeddings, BM25, vector store, index manifest
  retrieval/         rewrite, hybrid search, rerank
  guardrails/        policy + the three gates
  evaluation/        eval set, runner, threshold calibration
  cache.py           read-through Redis cache; never fails closed
  api.py             FastAPI service + the web UI it serves
  web/static/        the single-page UI (no build step, no CDN)
api/index.py         Vercel ASGI entry point
```

## Scope

Answers are limited to the ingested guidelines. This is not a diagnostic tool
and does not give personalized medical advice.

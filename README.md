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
 ├─ HyDE                    hypothetical answer blended into the query vector
 │                          (dense half only; falls back to the query alone)
 ├─ retrieve                dense (remote) + BM25 (local), fused with RRF
 ├─ rerank                  absolute 0–10 relevance score per passage
 ├─ GATE 2  confidence      top-1 ≥ threshold        → proceed
 │                          top-1 in corrective band → retry retrieval once,
 │                            wider + HyDE-weighted, re-judge at the SAME bar
 │                          top-1 < band floor       → "not enough information"
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

The same boundary decides *who* a question is about, and an indefinite person is
not a person: "a child", "a tourist", "a 55-year-old" are clinical categories —
the vocabulary the guidelines are themselves written in — so a question framed
around one is in scope. "How many days of IV antibiotics for empyema in a child"
asks what the guideline states for a category; "how long should my child stay on
antibiotics" asks about one individual. Plain wording ("dripped in", "pus around
the lung") is how ordinary people say clinical things, not evidence that a
question is personal. Gate 1 got this wrong until the eval set caught it.

Loosening that classifier is safe only because of the ordering above: the regex
rules run first and may only escalate, so `should i`, `can i take` and `my test
results` never reach the prompt at all. The judgement it was taught applies only
to queries the rules already declined to catch.

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

**HyDE is blended, not substituted, and only on the dense half.** A question and
the passage answering it are written in different registers — "How is
drug-resistant TB confirmed?" shares almost no words with "Culture and drug
susceptibility testing is performed on all presumptive DR-TB cases…" — so the
dense half searches with a generated hypothetical passage mixed into the query
vector at `hyde_query_weight` (0.5 by default). Textbook HyDE throws the query
away and searches with the generation alone; here the query stays in the mix, so
a hypothetical that drifts to the wrong condition pulls retrieval partway rather
than replacing the target. BM25 goes on searching the literal query, because
exact drug names and abbreviations are exactly what eighty words of generated
prose would bury.

**The hypothetical document is never shown, cited, or generated from.** It is
embedded and discarded. It never reaches the reranker, the generator, or the
citation list — every sentence a user reads still comes from a retrieved chunk
that gate 3 verified. A fabricated passage can change *which* real guidance is
found; it cannot change what is asserted. This is the one component that fails
*open*: an unavailable model here means falling back to the plain query vector,
not refusing, because HyDE is a retrieval improvement and not a gate. The
fallback is recorded as a degradation so a weaker retrieval path is never cached.

    uv run rag ask "..." --no-hyde --trace     # or HYDE_ENABLED=false

**Gate 2 is three-way, and the middle band is corrective.** This is CRAG (Yan
et al., 2024) with its defining action removed. CRAG scores retrieval and, on a
poor score, discards what it found and falls back to **web search**. That
fallback cannot exist here: every claim must trace to one of the sha256-pinned
MOHFW documents, which is what makes the citations, the index manifest, and
gate 3's numeric-provenance check mean anything.

So the corrective action is aimed at the *same corpus*, on the premise that a
middling score usually means the passage exists and the first query missed it.
The retry goes deeper (`corrective_k`, so chunks fused into ranks 21–40 that
the reranker never scored) and leans onto the HyDE hypothetical, since the
literal query is what already failed. This is where HyDE's latency is earned —
paid on the queries that need it, not charged to every query that was fine.

Three properties keep it safe inside a refusal gate:

- **The bar does not move.** The merged pool is re-judged against the same
  `confidence_threshold`. A correction buys a second attempt at the bar, never
  a lower bar. Everything else is an optimisation; this is the invariant.
- **Exactly one retry.** The second evaluation passes `allow_correction=False`,
  so no path loops.
- **Nothing is discarded.** The pool is the union of both passes, so a passage
  the first pass scored well cannot be lost to a retry that fused differently.

Already-scored passages are not re-sent to the reranker — sound precisely
because that scale is absolute rather than relative.

**How often does it fire, and does it help?** Rarely, and unproven. It never
fires on the eval set (answerable cases score 10.0, unanswerable 0.0 — the band
is empty). Across 16 ad-hoc probes it fired twice: once the retry found nothing
better and refused (bar held); once it recovered top-1 from 4.0 to 7.0 and
cleared gate 2, after which the *generator* declined the passages anyway. Zero
observed cases so far where it flipped a refusal into an answer. Both band hits
looked like genuine corpus gaps rather than retrieval failures — which is
exactly the case a corrective retry cannot fix. Set `CORRECTIVE_ENABLED=false`
to switch it off.

## Current corpus

21 MOHFW documents, 1,907 pages, **3,547 chunks** (mean 214 tokens), indexed as
`medic-guidelines`. Specialties:
infectious disease, cardiology, respiratory, paediatrics, obstetrics &
gynaecology, oncology, orthopaedics, neurology, surgery, critical care,
toxicology, public health, AYUSH. Nothing is skipped.

**The dengue document is OCR-derived, and that is worth knowing when you read
its citations.** `dengue.pdf` was this corpus's one known gap. It is *not* a
scan, which is what the earlier note here assumed: PrimoPDF exported it with
every glyph flattened to vector outlines — 55 pages, no font objects, ~12k
bezier paths per page, 145 MB, and exactly zero extractable characters. No text
extractor can ever read it, so `parse_pdf` correctly refused it.

The text was recovered by OCR instead. Because the pages are crisp synthetic
renders rather than photographs, lines and word gaps are found by pixel
projection and only the *recogniser* half of the OCR stack is used — the stock
text detector, tuned for photographed pages, silently dropped about a fifth of
the lines on every page. Mean recogniser confidence is 0.972.

`data/raw/dengue.txt` is that transcription, and it is what the manifest now
points at; the source PDF stays in `pdf-data/` as the citable original. Page
numbers survive as form feeds, so a citation reading "p.34" still opens to the
right page of the PDF. `tools/ocr_outlined_pdf.py` regenerates it — the manifest
records a sha256 for a derived file, so the thing that derives it is committed
too:

```bash
uv run --with rapidocr-onnxruntime --with opencv-python --with pymupdf \
  python tools/ocr_outlined_pdf.py \
  "pdf-data/MoHFW Official Medical Documentation/dengue.pdf" \
  data/raw/dengue.txt
```

Residual OCR artifacts, none corrected by hand: occasional split words on
letter-spaced justified lines (`sweati ng`), `I`/`l` confusion in acronyms
(`AlIMS` for `AIIMS`), and flowchart boxes whose columns interleave in their
lower rows. Treat dengue citations as slightly noisier than the other twenty.

### Measured results

Against `data/eval/questions.yaml` (34 cases, `gpt-4o-mini` throughout):

| metric | value |
|---|---|
| overall accuracy | 100% (34/34) |
| safety compliance | 100% (must be 100%) |
| false refusal rate | 0% |
| retrieval hit rate | 100% |
| gate-2 threshold | 5.5 (calibrated) |

**These numbers are one run, and the run is not deterministic.** Gate 3 is an
LLM judge and has historically flipped `ans-ari-children` between runs; a clean
sweep is evidence, not a guarantee. Treat a single 100% with suspicion — an
earlier one recorded here turned out to be a lucky run.

The previous run scored 91%, and all three failures were the same defect, found
by the precision cases the moment they were added: gate 1's *model* pass refused
`prec-quinine-rate`, `prec-malaria-travel` and `prec-empyema-abx` as
`personalized_advice` before retrieval ran, reasoning "advice for a tourist's
situation", "treatment advice for a child". None of those queries names a
specific person. Its definition said "a specific person" but every example was
first-person, so an *indefinite* third party fell in a gap the examples never
covered. Fixed by naming that case explicitly (see "Framing is the boundary"
below) and bumping `guardrails_version`.

Retrieval has not missed once across both runs: every case that reached it
cited the chunk asserted by `expect_text`.

**The corrective band is still worth watching.** `una-vaccine-temp` is
unanswerable and scores 4.0 on the first pass; the corrective retry surfaces
passages scoring **7.0**, clearing the 5.5 threshold. Gate 2 passes it, and only
the generator declining for insufficient context keeps it from being answered —
in both runs. The bar never moved, but a retry that raises an unanswerable
query by three points is exactly the leak this band risks, and it took one eval
case to demonstrate. `una-clavicle` behaves correctly, staying at 4.0.

So the old claim that "answerable queries score 9-10 and unanswerable ones 0.0"
holds only for the easy unanswerable cases — whole specialties nobody ingested.
Near-miss cases inside covered specialties sit at 4.0, and one reaches 7.0 after
correction.
Re-run before quoting these, and treat false refusal rate as the noisy metric.

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
  retrieval/         rewrite, HyDE, hybrid search, rerank, corrective retry
  guardrails/        policy + the three gates
  evaluation/        eval set, runner, threshold calibration
  cache.py           read-through Redis cache; never fails closed
  api.py             FastAPI service + the web UI it serves
  web/static/        the single-page UI (no build step, no CDN)
app.py               Vercel ASGI entry point (root level)
```

## Scope

Answers are limited to the ingested guidelines. This is not a diagnostic tool
and does not give personalized medical advice.

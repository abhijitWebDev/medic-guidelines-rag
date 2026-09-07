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

For a deployment, also set `DATABASE_URL` (Postgres — Supabase, Neon, or your
own) and `SESSION_SECRET`. Those two turn on accounts; see
[Accounts and history](#accounts-and-history).

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

On an instance with accounts, a **Your history** panel sits above the corpus
panel: the questions this account has asked, newest first, each reopenable with
its original answer and sources, individually deletable, and clearable in two
clicks. It is hidden entirely when there is no account to file turns under.

Evaluation has no UI on purpose. A run costs real API calls and takes minutes,
which is too easy to trigger by accident from a browser — use `rag eval run`.

`corpus scan` leaves `url`, `specialty`, and `version` blank rather than
guessing. Fill them in by hand — they end up in citations, and blank provenance
is visible where invented provenance is not.

## Accounts and history

Set `DATABASE_URL` and the app has users. Visitors sign up with an email and a
password, get a signed session cookie, and every question they ask is written to
their own history — question, answer, refusal reason, and the citations that
supported it. Leave it unset and the app behaves exactly as it did before: open,
no login, no history.

```bash
# Neon: Dashboard → Connection Details → *Pooled connection*.
DATABASE_URL='postgresql://USER:PASS@ep-xxx-pooler.REGION.aws.neon.tech/DB?sslmode=require'
SESSION_SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")
```

Take the **pooled** host — the one with `-pooler` in it. The direct endpoint
gives every serverless instance its own real Postgres connection and runs out;
the pooled one is PgBouncer and is what a function-per-request deployment
wants. Two settings in `db/engine.py` exist because of that pooler:
`min_size=0`, so a cold start that never serves a request opens nothing, and
`prepare_threshold=None`, which switches off psycopg's automatic server-side
prepared statements. The second one matters: under transaction pooling a
prepared statement can be looked up on a connection that never saw it, and that
failure shows up in production under load and never in a test.

Neon scales to zero, so the first query after an idle period pays a wake-up of
a few hundred milliseconds — `DB_TIMEOUT_S` (10s) covers it comfortably.

`SESSION_SECRET` is not optional in practice. Without it the app signs cookies
with a random per-process key, which fails *visibly* — everyone is signed out at
every restart, and two instances never agree on a session. That is deliberate:
the alternative to a loud failure is a signing key an attacker can guess.

**Schema.** Created on first use; there is no migration step, which is a
choice rather than an omission — `CREATE TABLE IF NOT EXISTS` is idempotent and
cannot drift from the code because it *is* the code. The day a column has to
change type or be backfilled, `db/schema.py` grows a versioned migration list;
adding one before then is machinery guarding nothing.

```
users(id, email UNIQUE, password_hash, created_at,
      verified_at, password_changed_at)
history(id, user_id → users.id, query, answer, answered,
        refusal_reason, top_score, citations, created_at)
```

Columns added after the fact go in `db/schema.py`'s `MIGRATIONS` list, applied
in order on connection. Two rules keep that honest: never edit a statement that
has shipped (a deployed database has already run it, so editing only changes
what a *fresh* one gets), and make backfills conditional on the column actually
being new — the list runs on every cold start, so an unconditional
`UPDATE users SET verified_at = ...` would verify every account that had signed
up since.

**How it is layered.** One concern per file, so that a change has one home:

```
db/schema.py     DDL                    what the tables are
db/models.py     User, Turn             what a row means — no SQL
db/engine.py     connections + pooling  how we talk to either backend
db/users.py      queries on users       account lookup, creation, deletion
db/history.py    queries on history     save, page, delete turns
passwords.py     scrypt                 outside db/ — hashing is not storage
```

Callers never reach past the package door:

```python
from .db import StorageError, User, history, users

user = users.authenticate(email, password)
history.save(user.id, response)
```

`User` deliberately has no `password_hash` field. The hash is read inside
`users.authenticate` and never leaves it, so no route, log line, or template
can hand one to a browser by accident.

Records are frozen dataclasses rather than Pydantic models, unlike `models.py`
at the top level. That file describes the pipeline's wire format, where
validation and schema generation earn their keep; these describe rows this
application wrote and already validated. No ORM either — there are two tables
and eleven statements, and every one of them is visible in the file that owns
it.

**Passwords** are hashed with `hashlib.scrypt` — memory-hard, and in the
standard library, so authentication adds no dependency to keep patched. Each
stored hash carries the cost parameters it was made with, so raising `SCRYPT_N`
later re-hashes people as they sign in instead of locking them out.

**A turn's trace is not stored.** It is debugging data about which gate fired
and by far the largest part of a response. Citations *are* stored, because an
answer without its sources is worth nothing in a system whose whole claim is
that every statement is sourced — reopening a saved answer shows the same source
cards it had when it was given.

**Two stores, opposite failure modes.** This is the part worth understanding:

|                | Redis (`cache.py`)            | Postgres (`db.py`)             |
| -------------- | ----------------------------- | ------------------------------ |
| Holds          | answers, embeddings, counters | users, history                 |
| On failure     | compute it normally           | refuse (503)                   |
| Keyed by       | pipeline fingerprint + query  | user id                        |
| Shared?        | answers yes, sessions no      | never                          |

Answers stay in a **shared** cache on purpose. The corpus is identical for every
user, so two people asking the same guideline question should not both pay for
the model; the keys are a fingerprint plus a hash of the question, and nothing
user-authored goes in or comes out. What became per-user is everything that
*identifies* someone: rate-limit counters, session lookups, and cached history
pages. Rate limiting by account rather than by IP also fixes a real problem — a
clinic behind one NAT used to share a single allowance.

**Managing accounts:**

```bash
uv run rag users list                     # email, created, turns kept
uv run rag users add doc@example.in       # prompts for a password
uv run rag users delete doc@example.in    # account and all of its history
```

Sign-up is open: anyone who can reach the URL can create an account. Set
`AUTH_ENABLED=false` to run the app open instead, or delete accounts you did not
expect.

**The quota is five questions per account per hour.** That is deliberately
tight — a single question can cost an intent classification, an embedding, a
HyDE generation, up to 40 rerank calls, an answer, and two output-gate checks,
so the bill is per question rather than per session:

```bash
RATE_LIMIT_PER_WINDOW=5      # 0 disables; raise it for a deployment with known users
RATE_LIMIT_WINDOW_S=3600
```

It is a spend control, not a security boundary. Signup is open, so someone
stopped by it can register again; the real backstop is a hard monthly limit on a
project-scoped OpenAI key, which no code here can undo. It also fails **open** —
if Upstash is unreachable the request is served rather than refused, falling back
to a weaker per-process counter (see `cache.py` on why a limiter must never be
able to take the app down).

Because five is small enough to reach in ordinary use, the limit is stated
rather than sprung. Every answer carries `X-RateLimit-Limit`, `-Remaining` and
`-Reset`; the UI shows "3 of 5 questions left this hour" beside the Ask button
and turns amber, then red, as it runs down. Reaching it renders as a timed
notice naming the limit and the wait — not as the red "Request failed" box,
which reads as a bug. Sign-in and sign-up are metered separately and by IP, so
failed logins never eat the question quota.

**Testing.** The suite runs against SQLite with a fresh file per test, so it
needs no server. Both backends share every statement, but only Postgres proves
the dialect translation and type mapping, so the same tests can be pointed at a
throwaway database:

```bash
TEST_DATABASE_URL=postgresql://postgres@127.0.0.1:5432/ragtest uv run pytest
```

## Email: verification and password reset

Set `SES_SMTP_HOST` and `MAIL_FROM` and the app sends email through Amazon SES.
That single fact turns on two flows:

```bash
# SES console → SMTP settings → Create SMTP credentials.
# These are NOT your AWS access keys; SES derives a separate pair.
SES_SMTP_HOST=email-smtp.ap-south-1.amazonaws.com
SES_SMTP_PORT=587                       # 587 STARTTLS, 465 implicit TLS
SES_SMTP_USER=AKIA...
SES_SMTP_PASSWORD=...
MAIL_FROM=noreply@yourdomain.in         # must be verified in SES
PUBLIC_BASE_URL=https://your-app.vercel.app
```

> **A new SES account is in the sandbox**, where it delivers only to addresses
> you have verified in the console. Until you request production access,
> signing up with any other address gets a clean "we could not send it" and a
> resend button — not a mystery. Request production access before letting real
> people register.

**Verification gates asking, and nothing else.** An unverified account can sign
in, see the corpus, and read its (empty) history; `/api/ask` returns 403 with a
resend button. That is the point of it — a throwaway account now costs a working
inbox before it can spend a single model call, which is the hole the per-account
quota could not close on its own. Everything else stays open, because an account
nobody can look at is harder to finish setting up, not safer. The check runs
*before* the rate limiter, so a blocked question never spends a slot.

**Leaving email unconfigured disables verification entirely.** An instance that
cannot send must not demand a link it will never deliver — that is not a
stricter deployment, it is one nobody can sign into. Same pattern as
`DATABASE_URL` turning accounts on and `REDIS_URL` turning the shared cache on.

**Password reset** closes a gap that accounts opened: before it, a forgotten
password meant the account was gone. `/forgot` answers identically whether or
not the address is registered, so the form is not an enumeration endpoint.

Both links are signed with `SESSION_SECRET` and stored nowhere. Three
properties fall out of that rather than needing a token table:

- **A confirmation link cannot reset a password.** The purpose is inside the
  signature, and confirmation links go to addresses nobody has proven yet.
- **A reset link works once.** Using it stamps `password_changed_at`, and a
  token issued before that stamp no longer resolves.
- **A reset signs the account out everywhere.** Sessions carry their issue time,
  and anything older than the password change stops being honoured — which is
  what someone resetting because they think they were compromised is asking
  for. The session created *by* the reset survives, which is why both times are
  compared in milliseconds rather than seconds.

**No SDK.** `mailer.py` is `smtplib` and `ssl` from the standard library, about
forty lines. boto3 would ship the service catalogue for every AWS API — tens of
megabytes in a bundle with a size limit — to make one `SendEmail` call. Same
trade as scrypt over argon2, and the hand-written LanceDB client.

Sending happens **inside** the request rather than in a background task: a
serverless instance can be frozen the moment it responds, and "send it after we
reply" is a good way to lose the email an account depends on. Signup pays a few
hundred milliseconds for it.

## Corpus governance

A PDF sitting in `data/raw/` is not part of the corpus. A PDF **listed in the
manifest with a matching sha256** is. Dropping an uncurated file into `data/raw/`
halts ingestion with a named error rather than quietly widening scope.

`data/index_manifest.json` records which documents, chunk parameters, and
embedding model produced the table currently being queried — so "which version
of the guidelines did this answer come from?" has an auditable answer.

## Deployment notes

**Vercel.** Set `OPENAI_API_KEY`, the `LANCEDB_*` values, `DATABASE_URL` and
`SESSION_SECRET` as project environment variables. Point `DATABASE_URL` at
Supabase's *pooler* (port 6543), not the direct connection: a serverless
function instance per request will exhaust direct connection slots. Without
`DATABASE_URL` the app falls back to SQLite on a filesystem that is thrown away
at every cold start, and the sign-in page says so.

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
  passwords.py       scrypt hashing — not a database concern, so not in db/
  mailer.py          SES over SMTP; stdlib only, no AWS SDK
  security.py        signed cookies + email link tokens + rate limiting
  db/                accounts + history. Always fails closed.
    schema.py        the DDL; the only place a column is described
    models.py        User and Turn — shape, no SQL
    engine.py        connections, pooling, the Postgres/SQLite seam
    users.py         every query against `users`
    history.py       every query against `history`
  api.py             FastAPI service + the web UI it serves
  web/static/        the single-page UI (no build step, no CDN)
app.py               Vercel ASGI entry point (root level)
```

## Scope

Answers are limited to the ingested guidelines. This is not a diagnostic tool
and does not give personalized medical advice.

"""Central configuration. Every tunable the eval harness touches lives here."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- OpenAI ---------------------------------------------------------
    openai_api_key: str = ""
    # Generation: quality matters, this writes the user-facing answer.
    openai_model: str = "gpt-4o-mini"
    # Guardrail classifiers + reranker: high call volume, cheap model is fine.
    openai_guard_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-large"
    embed_dim: int = 3072

    # --- LanceDB --------------------------------------------------------
    # Self-hosted server: set LANCEDB_URI to your db:// or http(s):// endpoint.
    # Falls back to an embedded directory so tests and offline work still run.
    lancedb_uri: str = str(ROOT / "data" / "lancedb")
    lancedb_api_key: str | None = None
    lancedb_region: str | None = None
    # Matches the name used in .env. Taken verbatim as the remote table name --
    # the server has no aliasing, so versioning happens via this name, not a
    # suffix (see `index_version`, which is recorded as metadata instead).
    lancedb_medical_guidelines_table: str = "medic-guidelines"
    index_version: str = "v1"

    # --- Chunking -------------------------------------------------------
    chunk_target_tokens: int = 320
    chunk_overlap_tokens: int = 48
    # Floor for merging a leftover fragment into its predecessor.
    chunk_min_tokens: int = 60
    # Absolute floor below which a chunk carries no retrievable content.
    # A complete-but-short section ("Referral Criteria: ...") is kept;
    # only a stray heading echo is dropped.
    chunk_drop_below_tokens: int = 10

    # --- Cache (Upstash Redis) -------------------------------------------
    # Empty disables caching entirely, which is the default: tests, offline
    # work, and a fresh clone must all behave identically without it.
    redis_url: str = ""
    # Upstash is a remote TLS hop, not a local socket. A cache is a latency
    # optimisation, so it is never allowed to *add* meaningful latency: if a
    # lookup has not come back inside this budget we abandon it and compute.
    redis_timeout_ms: int = 400
    # Query vectors keyed by model+dim+text. Long TTL -- the mapping is exact
    # and only changes if OpenAI reissues the model.
    cache_embed_ttl_s: int = 30 * 24 * 3600
    # Whole responses. Shorter: the corpus is medical guidance, and a day is
    # the longest we want a stale-but-valid answer circulating.
    cache_response_ttl_s: int = 24 * 3600
    # Hypothetical documents (see retrieval/hyde.py). Long: the passage never
    # reaches a user and depends only on the query and the HyDE prompt, so
    # staleness costs nothing -- `hyde_version` is what invalidates it.
    cache_hyde_ttl_s: int = 7 * 24 * 3600

    # --- Accounts (Postgres / Supabase) ----------------------------------
    # The durable store for users and their history. Any Postgres URL works --
    # Supabase, Neon, or your own. Empty falls back to a local SQLite file,
    # which is right for tests and local work and wrong for a deployment: see
    # db.py. Setting this is also what turns the login gate on.
    database_url: str = ""
    # Where the SQLite fallback lives. Empty picks data/app.db, or the temp
    # directory when that is not writable.
    sqlite_path: str = ""
    # Postgres only. Small on purpose: a serverless deployment multiplies this
    # by the number of warm instances, and Supabase's pooler counts them all.
    db_max_connections: int = 4
    db_timeout_s: float = 10.0

    # --- Access control --------------------------------------------------
    # None means "on when a database is configured", which is what you want in
    # both directions: a deployment with DATABASE_URL requires accounts, and a
    # fresh clone with nothing set stays open so the test suite and local work
    # need no credentials. Set it explicitly to override either way.
    auth_enabled: bool | None = None
    # Signs the session cookie. MUST be set for a real deployment: with no
    # value the app generates a random one per process, so sessions do not
    # survive a restart and do not work across instances. That fails visibly
    # (everyone is logged out) rather than silently accepting forged cookies,
    # which is the only acceptable behaviour for an unset signing key.
    session_secret: str = ""
    # Deprecated. This was the shared password before accounts existed; it is
    # no longer a credential. It is still read for one reason: an existing
    # private deployment that upgrades without setting DATABASE_URL must not
    # silently become public, so its presence keeps the gate on.
    app_password: str = ""
    # How long a successful login stays valid.
    session_ttl_s: int = 7 * 24 * 3600
    # Questions per window. Counted per account once signed in, and per IP
    # before that. 0 disables. This is a spend control, not a security
    # boundary -- signup is open, so anyone stopped by it can register again;
    # the real backstop is a hard monthly limit on a project-scoped OpenAI key.
    #
    # Five is deliberately tight. Each question can cost an intent
    # classification, an embedding, a HyDE generation, up to 40 rerank calls,
    # an answer and two output-gate checks, so the bill is per question rather
    # than per session. A limit low enough to be felt is the point: raise it
    # for a deployment with known users, lower it for a public demo.
    rate_limit_per_window: int = 5
    rate_limit_window_s: int = 3600

    # scrypt work factors. n is the memory/CPU dial and must be a power of two;
    # 2**14 with r=8 costs about 16 MB and tens of milliseconds per login,
    # which is the usual balance between "expensive to crack" and "a login
    # still feels instant". Raise n as hardware improves: stored hashes carry
    # the parameters they were made with, so old passwords keep verifying.
    scrypt_n: int = 2**14
    scrypt_r: int = 8
    scrypt_p: int = 1

    # Turns kept per user; the oldest are dropped on write. 0 keeps everything.
    history_max_items: int = 200

    # --- Email (Amazon SES, over SMTP) -----------------------------------
    # Configuring a host is what turns email verification on. Leave it unset
    # and accounts are usable the moment they are created -- which is what a
    # fresh clone, the test suite and offline work need, and is the same
    # pattern as DATABASE_URL turning accounts on and REDIS_URL turning the
    # shared cache on. Nothing here is ever required to answer a question.
    ses_smtp_host: str = ""
    # 587 with STARTTLS is SES's usual port. 465 is implicit TLS; the mailer
    # picks the right handshake from the port rather than needing a flag.
    ses_smtp_port: int = 587
    ses_smtp_user: str = ""
    ses_smtp_password: str = ""
    # The envelope sender. Must be an address or domain you have verified in
    # SES, or every send is rejected.
    mail_from: str = ""
    mail_from_name: str = "Medical Guideline Assistant"
    # Sending happens inside the request, so this is a latency budget as well
    # as a timeout: signup waits for it.
    mail_timeout_s: float = 10.0

    # Absolute base for links in emails. Derived from the request when empty,
    # which is right locally and behind a well-behaved proxy; set it explicitly
    # if anything rewrites Host, because a verification link pointing at the
    # wrong origin is a dead account.
    public_base_url: str = ""

    # How long a verification link stays good. Long: someone who signs up on
    # their phone and opens the mail the next morning should still get in.
    verify_token_ttl_s: int = 3 * 24 * 3600
    # How long a password-reset link stays good. Short by comparison -- it is a
    # credential sitting in an inbox, and it is single-use besides.
    reset_token_ttl_s: int = 3600
    # Vercel sets x-forwarded-for; a client can forge it when nothing sits in
    # front of the app, so this is off unless the deployment really is proxied.
    trust_proxy_header: bool = True

    # --- Retrieval ------------------------------------------------------
    retrieve_k: int = 20
    rerank_top_n: int = 6
    # Gate 2. Provisional -- overwritten by data/calibration.json once the
    # eval harness has actually measured it. Never trust this default.
    confidence_threshold: float = 7.5

    # --- Corrective retrieval (gate 2 middle band) -----------------------
    # Below confidence_threshold but at or above this floor, gate 2 returns
    # CORRECT instead of refusing: retrieval is retried once, wider and with a
    # different query representation, and then judged again against the SAME
    # confidence_threshold. See retrieval/corrective.py.
    corrective_enabled: bool = True
    # Floor of the band. Provisional, like confidence_threshold -- on the
    # reranker's scale 4-6 is "related topic, contains part of the answer",
    # which is the range worth a second attempt. Below it, retrieval is not in
    # the right neighbourhood and a retry only spends money to refuse later.
    corrective_threshold: float = 6.0
    # The retry goes deeper: chunks fused into ranks 21-40 were never scored by
    # the reranker at all, so this is recall the first pass could not have had.
    corrective_k: int = 40
    # ...and leans on the hypothetical, since the literal query is what already
    # failed. Ignored when hyde_enabled is false; the retry is then depth only.
    corrective_hyde_query_weight: float = 0.2

    # --- HyDE ------------------------------------------------------------
    # Search the dense half with a generated hypothetical answer blended into
    # the query vector. See retrieval/hyde.py for why it is blended rather
    # than substituted. Set HYDE_ENABLED=false to A/B it against plain dense
    # retrieval -- the fingerprint below covers these, so the two runs do not
    # share cached answers.
    hyde_enabled: bool = True
    # Passages per query, generated in one call. >1 averages several drafts to
    # damp a single unlucky generation, at one extra embedding call each.
    hyde_n: int = 1
    # Share of the blend kept by the real query. 1.0 disables HyDE in effect;
    # 0.0 is textbook HyDE, which this project deliberately does not do.
    hyde_query_weight: float = 0.5
    # None uses openai_guard_model. This is a cheap, high-volume call.
    hyde_model: str | None = None
    # Bumped by hand when the HyDE prompt changes, for the same reason as
    # guardrails_version: the fingerprint cannot see prompt text.
    hyde_version: str = "v1"

    # --- Guardrails ------------------------------------------------------
    # Bumped by hand whenever a gate prompt or gate rule changes. The
    # fingerprint below can only see declared configuration, never the gate
    # code itself, so without this a guardrail fix stays invisible to every
    # user holding a cached refusal until the TTL expires -- which is exactly
    # the case a fix is urgent for.
    guardrails_version: str = "v3"

    @property
    def mail_enabled(self) -> bool:
        """Whether this instance can send email at all.

        Everything email-gated keys off this, verification included. An
        instance that cannot send must not demand that people click a link it
        will never deliver -- that is not a stricter deployment, it is a
        deployment nobody can use.
        """
        return bool(self.ses_smtp_host and self.mail_from)

    @property
    def verification_required(self) -> bool:
        return self.mail_enabled

    @property
    def auth_required(self) -> bool:
        """Whether visitors must sign in.

        `app_password` counts even though it is no longer accepted as a
        credential: it means someone deliberately made this instance private,
        and an upgrade must not undo that decision on their behalf.
        """
        if self.auth_enabled is not None:
            return self.auth_enabled
        return bool(self.database_url or self.app_password)

    @property
    def table(self) -> str:
        return self.lancedb_medical_guidelines_table

    @property
    def is_remote(self) -> bool:
        return self.lancedb_uri.startswith(("http://", "https://"))

    @property
    def pipeline_fingerprint(self) -> str:
        """Identifies everything that can change an answer for a fixed query.

        Cached responses are namespaced by this, so switching model, rebuilding
        the index, or recalibrating gate 2 makes old entries unreachable rather
        than stale. It cannot see edits to *gate logic* by itself -- only
        declared configuration -- so `guardrails_version` is the hand-bumped
        stand-in for those, and the eval harness does not read the response
        cache at all (see evaluation.run.run_eval).
        """
        material = "|".join(
            str(x)
            for x in (
                self.openai_model,
                self.openai_guard_model,
                self.openai_embed_model,
                self.embed_dim,
                self.index_version,
                self.table,
                self.retrieve_k,
                self.rerank_top_n,
                self.confidence_threshold,
                self.guardrails_version,
                self.hyde_enabled,
                self.hyde_n,
                self.hyde_query_weight,
                self.hyde_model,
                self.hyde_version,
                self.corrective_enabled,
                self.corrective_threshold,
                self.corrective_k,
                self.corrective_hyde_query_weight,
            )
        )
        return hashlib.sha256(material.encode()).hexdigest()[:16]

    @property
    def raw_dir(self) -> Path:
        return ROOT / "data" / "raw"

    @property
    def chunks_dir(self) -> Path:
        return ROOT / "data" / "chunks"

    @property
    def eval_dir(self) -> Path:
        return ROOT / "data" / "eval"

    @property
    def manifest_path(self) -> Path:
        return ROOT / "data" / "corpus_manifest.yaml"

    @property
    def index_manifest_path(self) -> Path:
        return ROOT / "data" / "index_manifest.json"

    @property
    def calibration_path(self) -> Path:
        return ROOT / "data" / "calibration.json"

    def load_calibration(self) -> None:
        """Apply thresholds measured by `rag-eval calibrate`, if present."""
        if not self.calibration_path.exists():
            return
        data = json.loads(self.calibration_path.read_text())
        if "confidence_threshold" in data:
            self.confidence_threshold = float(data["confidence_threshold"])
        if "corrective_threshold" in data:
            self.corrective_threshold = float(data["corrective_threshold"])


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.load_calibration()
    return s

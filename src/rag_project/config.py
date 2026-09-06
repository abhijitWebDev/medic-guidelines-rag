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

    # --- Access control --------------------------------------------------
    # Empty disables the gate entirely, which is the default: local work and
    # the test suite must not need a password. Setting it turns on the login
    # page and locks every /api route.
    app_password: str = ""
    # How long a successful login stays valid.
    session_ttl_s: int = 7 * 24 * 3600
    # Questions per IP per window. 0 disables. This is a spend control, not a
    # security boundary -- the password is the boundary.
    rate_limit_per_window: int = 30
    rate_limit_window_s: int = 3600
    # Vercel sets x-forwarded-for; a client can forge it when nothing sits in
    # front of the app, so this is off unless the deployment really is proxied.
    trust_proxy_header: bool = True

    # --- Retrieval ------------------------------------------------------
    retrieve_k: int = 20
    rerank_top_n: int = 6
    # Gate 2. Provisional -- overwritten by data/calibration.json once the
    # eval harness has actually measured it. Never trust this default.
    confidence_threshold: float = 5.0

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
    corrective_threshold: float = 3.0
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

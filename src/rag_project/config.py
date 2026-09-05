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

    # --- Retrieval ------------------------------------------------------
    retrieve_k: int = 20
    rerank_top_n: int = 6
    # Gate 2. Provisional -- overwritten by data/calibration.json once the
    # eval harness has actually measured it. Never trust this default.
    confidence_threshold: float = 5.0

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
        than stale. It cannot see edits to *gate logic* -- only to declared
        configuration -- which is why the eval harness does not read the
        response cache (see evaluation.run.run_eval).
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


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.load_calibration()
    return s

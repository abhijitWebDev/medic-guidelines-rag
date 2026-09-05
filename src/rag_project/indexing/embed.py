"""OpenAI embeddings, cached in two places for two different jobs.

*Ingest* embeds thousands of chunks once, and wants a bulk store it can resume
from after a failed run: that is the npz on disk, keyed by
sha256(model | dim | text) rather than by chunk_id, so it stays valid when
chunk ids shift and correctly misses when chunk *text* changes.

*Queries* embed one short string with a user waiting, and want the opposite
thing. They go to Redis (see cache.py). Routing them through the npz instead
would make every novel query rewrite the entire ~89 MB file, and would make
every process that answers queries load 89 MB at startup to serve 12 KB
lookups. The npz is therefore loaded lazily, and only the batch path touches
it -- an API or UI process that only ever calls `embed_one` never reads it at
all.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from openai import OpenAI

from ..cache import get_cache, key_for
from ..config import ROOT, get_settings

# OpenAI accepts up to 2048 inputs per call; 128 keeps request bodies modest.
EMBED_BATCH = 128

_CACHE_DIR = ROOT / "data" / "embed_cache"


def _key(text: str, model: str, dim: int) -> str:
    return hashlib.sha256(f"{model}|{dim}|{text}".encode()).hexdigest()


class Embedder:
    def __init__(self, use_cache: bool = True) -> None:
        s = get_settings()
        if not s.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in .env")
        self.model = s.openai_embed_model
        self.dim = s.embed_dim
        self._client = OpenAI(api_key=s.openai_api_key)
        self._use_cache = use_cache
        self._cache_path = _CACHE_DIR / f"{self.model}-{self.dim}.npz"
        self._disk: dict[str, np.ndarray] | None = None

    # --- disk tier (ingest) ----------------------------------------------
    @property
    def _cache(self) -> dict[str, np.ndarray]:
        """The npz, loaded on first use. Lazy because it is large and the
        query path must never pay for it."""
        if self._disk is None:
            self._disk = {}
            if self._use_cache and self._cache_path.exists():
                with np.load(self._cache_path) as z:
                    self._disk = {k: z[k] for k in z.files}
        return self._disk

    # --- api ------------------------------------------------------------
    def embed(self, texts: list[str], progress: bool = False) -> np.ndarray:
        """Embed texts, hitting the cache where possible. Returns (n, dim)."""
        cache = self._cache
        keys = [_key(t, self.model, self.dim) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in cache]

        for start in range(0, len(missing), EMBED_BATCH):
            idxs = missing[start : start + EMBED_BATCH]
            batch = [texts[i] for i in idxs]
            vectors = self._call(batch)
            for i, vec in zip(idxs, vectors, strict=True):
                cache[keys[i]] = vec
            if progress:
                done = min(start + EMBED_BATCH, len(missing))
                print(f"  embedded {done}/{len(missing)} new chunks", flush=True)

        if missing and self._use_cache:
            self._save()

        return np.vstack([cache[k] for k in keys]) if keys else np.zeros((0, self.dim))

    def embed_one(self, text: str) -> list[float]:
        """Embed a single query, through Redis rather than the npz.

        Falls back to the in-process tier inside `Cache` when Redis is not
        configured, so repeat queries in one process are still free without
        any external service.
        """
        s = get_settings()
        cache = get_cache()
        ckey = f"emb:{key_for(self.model, self.dim, text)}"

        hit = cache.get_vector(ckey, self.dim)
        if hit is not None:
            return hit.tolist()

        vec = self._call([text])[0]
        cache.set_vector(ckey, vec, s.cache_embed_ttl_s)
        return vec.tolist()

    @property
    def cached_count(self) -> int:
        return len(self._cache)

    # --- internals ------------------------------------------------------
    def _call(self, batch: list[str]) -> list[np.ndarray]:
        kwargs = {"model": self.model, "input": batch}
        if self.model.startswith("text-embedding-3"):
            kwargs["dimensions"] = self.dim
        resp = self._client.embeddings.create(**kwargs)
        # The API may return items out of order; `index` is authoritative.
        ordered = sorted(resp.data, key=lambda d: d.index)
        return [np.asarray(d.embedding, dtype=np.float32) for d in ordered]

    def _save(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        # np.savez appends ".npz" to a path that lacks it, so write through an
        # open handle -- otherwise the temp file lands under the wrong name.
        tmp = self._cache_path.with_name(self._cache_path.name + ".tmp")
        with tmp.open("wb") as fh:
            np.savez(fh, **self._cache)
        tmp.replace(self._cache_path)

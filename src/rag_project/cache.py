"""Read-through cache over Upstash Redis.

Two rules govern this module, and they are the inverse of the rules in llm.py.

1. **The cache never fails closed.** llm.py refuses when a call errors, because
   every caller there is a safety gate. Nothing here is a safety gate: a miss,
   a timeout, or an unreachable Upstash must all degrade to "compute it
   normally". A cache that can refuse a query is a cache that has become a
   dependency, and this one is deliberately not one.

2. **The cache never *adds* latency.** Upstash is a remote TLS hop of tens of
   milliseconds, not a local socket, so a slow lookup is a real cost rather
   than a rounding error. Timeouts are short, and repeated failures trip a
   breaker that stops us paying the timeout on every subsequent query.

An in-process LRU sits in front of Redis. It exists for the eval harness and
for Streamlit reruns, which ask the same question many times in one process and
should not pay the network to be told the same thing.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any

import numpy as np

from .config import get_settings

# Consecutive failures before we stop trying, and how long we stay off.
_BREAKER_THRESHOLD = 3
_BREAKER_COOLDOWN_S = 60.0

# Bounded so a long-lived server process cannot grow one query at a time.
_LOCAL_MAX_ENTRIES = 256


def key_for(*parts: Any) -> str:
    """Stable short key from arbitrary parts. Text is hashed, not embedded, so
    a key can never carry a patient-identifying query string into Upstash."""
    joined = "|".join(str(p) for p in parts)
    return hashlib.sha256(joined.encode()).hexdigest()[:32]


class Cache:
    def __init__(self, url: str | None = None, timeout_ms: int | None = None) -> None:
        s = get_settings()
        self._url = s.redis_url if url is None else url
        self._timeout = (s.redis_timeout_ms if timeout_ms is None else timeout_ms) / 1000.0
        self._client: Any = None
        self._connected = False
        self._failures = 0
        self._off_until = 0.0
        self._local: OrderedDict[str, Any] = OrderedDict()
        self.hits = 0
        self.misses = 0

    # --- availability ----------------------------------------------------
    @property
    def configured(self) -> bool:
        return bool(self._url)

    def _redis(self) -> Any:
        """The client, or None if unconfigured or the breaker is open."""
        if not self._url or time.monotonic() < self._off_until:
            return None
        if not self._connected:
            self._connected = True
            try:
                import redis
                from redis.backoff import NoBackoff
                from redis.retry import Retry

                self._client = redis.Redis.from_url(
                    self._url,
                    socket_timeout=self._timeout,
                    socket_connect_timeout=self._timeout,
                    # Values are raw bytes (float32 vectors and UTF-8 JSON);
                    # decoding them as str would corrupt the vectors.
                    decode_responses=False,
                    # redis-py retries three times with backoff by default,
                    # which silently turns one timeout budget into several --
                    # measured at 0.73s against a 0.4s setting. Retrying is the
                    # right default for a database and the wrong one for a
                    # cache, where giving up costs nothing but a recomputation.
                    retry=Retry(NoBackoff(), 0),
                    retry_on_timeout=False,
                    # Upstash drops idle connections; without this the first
                    # query after a lull pays a reconnect.
                    health_check_interval=30,
                )
            except Exception:
                self._client = None
        return self._client

    def _trip(self) -> None:
        self._failures += 1
        if self._failures >= _BREAKER_THRESHOLD:
            self._off_until = time.monotonic() + _BREAKER_COOLDOWN_S
            self._failures = 0

    # --- local tier ------------------------------------------------------
    def _local_get(self, key: str) -> Any:
        if key in self._local:
            self._local.move_to_end(key)
            return self._local[key]
        return None

    def _local_put(self, key: str, value: Any) -> None:
        self._local[key] = value
        self._local.move_to_end(key)
        while len(self._local) > _LOCAL_MAX_ENTRIES:
            self._local.popitem(last=False)

    # --- bytes -----------------------------------------------------------
    def get_bytes(self, key: str) -> bytes | None:
        cached = self._local_get(key)
        if cached is not None:
            self.hits += 1
            return cached
        client = self._redis()
        if client is None:
            self.misses += 1
            return None
        try:
            raw = client.get(key)
        except Exception:
            self._trip()
            self.misses += 1
            return None
        if raw is None:
            self.misses += 1
            return None
        self._failures = 0
        self._local_put(key, raw)
        self.hits += 1
        return raw

    def set_bytes(self, key: str, value: bytes, ttl_s: int) -> None:
        self._local_put(key, value)
        client = self._redis()
        if client is None:
            return
        try:
            client.set(key, value, ex=ttl_s)
            self._failures = 0
        except Exception:
            self._trip()

    # --- typed helpers ---------------------------------------------------
    def get_vector(self, key: str, dim: int) -> np.ndarray | None:
        raw = self.get_bytes(key)
        if raw is None:
            return None
        vec = np.frombuffer(raw, dtype=np.float32)
        # A dimension mismatch means the key collided with an entry written
        # under a different embedding config. Treat it as a miss, never as a
        # vector -- a wrong-length vector would fail far from here.
        return vec if vec.shape == (dim,) else None

    def set_vector(self, key: str, vec: np.ndarray, ttl_s: int) -> None:
        self.set_bytes(key, np.asarray(vec, dtype=np.float32).tobytes(), ttl_s)

    def get_json(self, key: str) -> Any | None:
        raw = self.get_bytes(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return None

    def set_json(self, key: str, value: Any, ttl_s: int) -> None:
        try:
            payload = json.dumps(value).encode()
        except (TypeError, ValueError):
            return  # unserialisable: silently skip, this is only a cache
        self.set_bytes(key, payload, ttl_s)


_cache: Cache | None = None


def get_cache() -> Cache:
    global _cache
    if _cache is None:
        _cache = Cache()
    return _cache


def reset_cache() -> None:
    """Drop the singleton. For tests and for reconfiguring at runtime."""
    global _cache
    _cache = None

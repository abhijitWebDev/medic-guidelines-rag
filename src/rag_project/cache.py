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
for repeated asks within one server process, which should not pay the network
to be told the same thing. Entries in it expire, and callers holding state the
application itself rewrites can skip it entirely with `local=False` -- see the
comments on the local tier for why both matter on more than one instance.
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

# Ceiling on how long this process trusts its own copy of anything, however
# long the value is good for in Redis. Bounds how stale one instance can get
# after another deletes a key.
_LOCAL_TTL_CAP_S = 60


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
    #
    # Entries here expire. That is not symmetry with Redis for its own sake:
    # without it a value lives until LRU pressure evicts it, which on a quiet
    # instance is forever -- so a key deleted from Redis by *another* instance
    # stays readable here indefinitely. For content-addressed values (an
    # embedding, an answer under a pipeline fingerprint) that is harmless. For
    # anything this app itself rewrites, it is a stale read with no upper bound.
    def _local_get(self, key: str) -> Any:
        entry = self._local.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at is not None and time.monotonic() >= expires_at:
            del self._local[key]
            return None
        self._local.move_to_end(key)
        return value

    def _local_put(self, key: str, value: Any, ttl_s: int | None = None) -> None:
        self._local[key] = (value, None if ttl_s is None else time.monotonic() + ttl_s)
        self._local.move_to_end(key)
        while len(self._local) > _LOCAL_MAX_ENTRIES:
            self._local.popitem(last=False)

    # --- bytes -----------------------------------------------------------
    def get_bytes(self, key: str, local: bool = True) -> bytes | None:
        cached = self._local_get(key) if local else None
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
        if local:
            self._local_put(key, raw, _LOCAL_TTL_CAP_S)
        self.hits += 1
        return raw

    def set_bytes(self, key: str, value: bytes, ttl_s: int, local: bool = True) -> None:
        if local:
            # Capped, not the caller's TTL. A month-long embedding TTL is a
            # statement about Redis, where the key can be seen and deleted --
            # not licence for one process to trust its own copy that long.
            self._local_put(key, value, min(ttl_s, _LOCAL_TTL_CAP_S))
        client = self._redis()
        if client is None:
            return
        try:
            client.set(key, value, ex=ttl_s)
            self._failures = 0
        except Exception:
            self._trip()

    def forget(self, key: str) -> None:
        """Drop a key from both tiers.

        Only meaningful for the per-user keys written by db/users.py and
        db/history.py, which are caches of a database this app itself writes to.
        The answer and embedding caches have nothing to invalidate -- their
        keys already carry the pipeline fingerprint, so a changed answer is a
        different key rather than a stale one.
        """
        self._local.pop(key, None)
        client = self._redis()
        if client is None:
            return
        try:
            client.delete(key)
            self._failures = 0
        except Exception:
            # Deliberately survivable. The TTL on these keys is short, so a
            # failed invalidation is a brief window of stale history, not a
            # wrong answer -- and raising here would fail a question over a
            # cache blip, which rule 1 above forbids.
            self._trip()

    # --- counters --------------------------------------------------------
    def incr(self, key: str, ttl_s: int) -> int | None:
        """Increment a counter and return its new value, or None if Redis is
        unreachable.

        Deliberately bypasses the in-process tier: a counter shared across
        server instances is the entire point, and a local copy would let each
        instance grant its own quota. INCR and EXPIRE go in one pipeline so a
        rate-limit check costs one Upstash round-trip rather than two.
        """
        client = self._redis()
        if client is None:
            return None
        try:
            pipe = client.pipeline()
            pipe.incr(key)
            pipe.expire(key, ttl_s)
            count, _ = pipe.execute()
            self._failures = 0
            return int(count)
        except Exception:
            self._trip()
            return None

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

    def get_json(self, key: str, local: bool = True) -> Any | None:
        raw = self.get_bytes(key, local=local)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return None

    def set_json(self, key: str, value: Any, ttl_s: int, local: bool = True) -> None:
        """`local=False` keeps the value out of this process's memory.

        For state the application itself rewrites and must see change promptly
        across instances -- an account's verified flag, say. It is the same
        argument `incr` makes for rate-limit counters: a per-process copy is
        one each instance can disagree about, and here disagreement is a person
        who has just confirmed their email being told they have not.
        """
        try:
            payload = json.dumps(value).encode()
        except (TypeError, ValueError):
            return  # unserialisable: silently skip, this is only a cache
        self.set_bytes(key, payload, ttl_s, local=local)


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

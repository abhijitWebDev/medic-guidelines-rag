"""The cache's job is to be invisible: same answers, fewer calls, and no new
failure modes. These tests are mostly about the *absence* of behaviour -- that
an unreachable Upstash changes nothing, and that an outage is never persisted.
"""

from __future__ import annotations

import numpy as np
import pytest

from rag_project import llm
from rag_project.assistant import Assistant
from rag_project.cache import Cache, key_for, reset_cache
from rag_project.models import RefusalReason, Response


class FakeRedis:
    """Minimal stand-in. `fail` makes every command raise, as an unreachable
    Upstash endpoint does once the socket timeout expires."""

    def __init__(self, fail: bool = False) -> None:
        self.store: dict[str, bytes] = {}
        self.fail = fail
        self.gets = 0
        self.sets = 0

    def get(self, key):
        self.gets += 1
        if self.fail:
            raise ConnectionError("upstash unreachable")
        return self.store.get(key)

    def set(self, key, value, ex=None):
        self.sets += 1
        if self.fail:
            raise ConnectionError("upstash unreachable")
        self.store[key] = value


def make_cache(fake: FakeRedis | None) -> Cache:
    c = Cache(url="rediss://fake", timeout_ms=50)
    c._connected = True
    c._client = fake
    return c


@pytest.fixture(autouse=True)
def _isolate_cache():
    reset_cache()
    llm.clear_degradations()
    yield
    reset_cache()
    llm.clear_degradations()


# --- basic round-trips ---------------------------------------------------


def test_vector_round_trips_exactly():
    fake = FakeRedis()
    c = make_cache(fake)
    vec = np.random.rand(8).astype(np.float32)
    c.set_vector("k", vec, 60)
    # Force a real Redis read rather than the in-process tier.
    c._local.clear()
    assert np.array_equal(c.get_vector("k", 8), vec)


def test_vector_of_wrong_dim_is_a_miss_not_a_corrupt_read():
    """A key written under a different embed_dim must not come back as a
    truncated vector -- that would fail far away from here."""
    fake = FakeRedis()
    c = make_cache(fake)
    c.set_vector("k", np.random.rand(8).astype(np.float32), 60)
    c._local.clear()
    assert c.get_vector("k", 3072) is None


def test_json_round_trips():
    c = make_cache(FakeRedis())
    c.set_json("k", {"answered": True, "n": 3}, 60)
    c._local.clear()
    assert c.get_json("k") == {"answered": True, "n": 3}


def test_corrupt_json_reads_as_a_miss():
    fake = FakeRedis()
    c = make_cache(fake)
    fake.store["k"] = b"{not json"
    assert c.get_json("k") is None


# --- degradation ---------------------------------------------------------


def test_unreachable_redis_never_raises():
    """Rule 1: the cache never fails closed."""
    c = make_cache(FakeRedis(fail=True))
    assert c.get_vector("k", 8) is None
    c.set_vector("k", np.zeros(8, dtype=np.float32), 60)  # must not raise
    assert c.get_json("k") is None


def test_breaker_stops_paying_the_timeout_after_repeated_failures():
    """Rule 2: the cache never adds latency. Once Upstash is clearly down we
    stop issuing commands rather than burning the socket timeout per query."""
    fake = FakeRedis(fail=True)
    c = make_cache(fake)
    for _ in range(10):
        c.get_bytes(f"miss-{_}")
    # Three attempts trip it; the rest short-circuit without touching Redis.
    assert fake.gets == 3


def test_client_does_not_retry():
    """redis-py retries 3x with backoff by default, which multiplies the
    timeout budget -- 0.73s measured against a 0.4s setting. A cache must give
    up immediately; recomputing is cheaper than a second round-trip."""
    c = Cache(url="rediss://default:token@example.invalid:6379", timeout_ms=400)
    client = c._redis()
    assert client is not None
    kwargs = client.connection_pool.connection_kwargs
    assert kwargs["retry"].get_retries() == 0
    assert kwargs["socket_timeout"] == 0.4
    assert kwargs["socket_connect_timeout"] == 0.4
    assert kwargs["decode_responses"] is False


def test_unconfigured_cache_still_serves_within_the_process():
    """No REDIS_URL is the default. Repeat lookups in one process must still be
    free, so nothing regresses for someone who never sets up Upstash."""
    c = Cache(url="", timeout_ms=50)
    assert not c.configured
    c.set_json("k", {"a": 1}, 60)
    assert c.get_json("k") == {"a": 1}


def test_local_tier_is_bounded():
    c = Cache(url="", timeout_ms=50)
    for i in range(400):
        c.set_json(f"k{i}", {"i": i}, 60)
    assert len(c._local) <= 256
    assert c.get_json("k0") is None      # evicted
    assert c.get_json("k399") == {"i": 399}


# --- keys ----------------------------------------------------------------


def test_key_is_stable_and_discriminating():
    assert key_for("a", 1) == key_for("a", 1)
    assert key_for("a", 1) != key_for("a", 2)


def test_key_does_not_leak_query_text():
    """Queries can carry personal detail; only a digest may reach Upstash."""
    assert "chest pain" not in key_for("text-embedding-3-large", 3072, "chest pain")


def test_fingerprint_changes_when_the_pipeline_changes():
    from rag_project.config import Settings

    base = Settings(openai_api_key="x")
    assert base.pipeline_fingerprint == Settings(openai_api_key="x").pipeline_fingerprint
    assert base.pipeline_fingerprint != Settings(
        openai_api_key="x", index_version="v2"
    ).pipeline_fingerprint
    assert base.pipeline_fingerprint != Settings(
        openai_api_key="x", confidence_threshold=7.0
    ).pipeline_fingerprint


# --- the rule that matters: outages are not cached -----------------------


class StubAssistant(Assistant):
    """Real `Assistant.ask` -- so the caching logic under test is the shipped
    one -- over a stubbed pipeline. `degrade` makes the run report a fail-closed
    refusal the way a swallowed LLMError does."""

    def __init__(self, degrade: bool = False) -> None:
        self.runs = 0
        self.degrade = degrade  # no super(): retriever/reranker are unused here

    def _run(self, query, screen=True):
        self.runs += 1
        if self.degrade:
            llm.note_degraded("intent_classifier")
            return Response(
                query=query, answered=False, answer="unavailable",
                refusal_reason=RefusalReason.OUT_OF_DOMAIN, trace={"stages": []},
            )
        return Response(
            query=query, answered=True, answer=f"answer {self.runs}",
            trace={"stages": ["intent", "retrieve"]},
        )


def test_second_identical_query_skips_the_pipeline(monkeypatch):
    fake = FakeRedis()
    cache = make_cache(fake)
    monkeypatch.setattr("rag_project.assistant.get_cache", lambda: cache)

    a = StubAssistant()
    first = a.ask("What does the guideline say about TB?")
    second = a.ask("What does the guideline say about TB?")

    assert a.runs == 1, "pipeline re-ran on a cache hit"
    assert second.answer == first.answer
    assert second.trace["cache"] == "hit"


def test_cache_key_ignores_whitespace_and_case(monkeypatch):
    cache = make_cache(FakeRedis())
    monkeypatch.setattr("rag_project.assistant.get_cache", lambda: cache)

    a = StubAssistant()
    a.ask("TB   treatment duration")
    a.ask("tb treatment duration")
    assert a.runs == 1


def test_cached_response_reports_the_callers_own_wording(monkeypatch):
    cache = make_cache(FakeRedis())
    monkeypatch.setattr("rag_project.assistant.get_cache", lambda: cache)

    a = StubAssistant()
    a.ask("TB treatment duration")
    hit = a.ask("tb   treatment duration")
    assert hit.query == "tb   treatment duration"


def test_screening_off_does_not_reuse_a_screened_answer(monkeypatch):
    """screen=False skips gate 1's model pass. Different pipeline, so it must
    not collide with the screened result for the same text."""
    cache = make_cache(FakeRedis())
    monkeypatch.setattr("rag_project.assistant.get_cache", lambda: cache)

    a = StubAssistant()
    a.ask("dengue management", screen=True)
    a.ask("dengue management", screen=False)
    assert a.runs == 2


def test_fail_closed_refusal_is_never_cached(monkeypatch):
    """The rule this whole design turns on: a minute of OpenAI trouble must not
    become a day of a good question being refused."""
    fake = FakeRedis()
    cache = make_cache(fake)
    monkeypatch.setattr("rag_project.assistant.get_cache", lambda: cache)

    a = StubAssistant(degrade=True)
    r = a.ask("dengue management")
    assert not r.answered
    assert r.trace["degraded"] == ["intent_classifier"]
    assert fake.store == {}, "an outage was persisted to Redis"

    # And once the model recovers, the next ask genuinely re-runs.
    a.degrade = False
    llm.clear_degradations()
    cache._local.clear()
    assert a.ask("dengue management").answered


def test_use_cache_false_neither_reads_nor_writes(monkeypatch):
    """What the eval harness relies on."""
    fake = FakeRedis()
    cache = make_cache(fake)
    monkeypatch.setattr("rag_project.assistant.get_cache", lambda: cache)

    a = StubAssistant()
    a.ask("dengue management", use_cache=False)
    a.ask("dengue management", use_cache=False)
    assert a.runs == 2
    assert fake.store == {}


def test_degradations_do_not_leak_between_queries(monkeypatch):
    """One query's outage must not suppress caching for the next one."""
    fake = FakeRedis()
    cache = make_cache(fake)
    monkeypatch.setattr("rag_project.assistant.get_cache", lambda: cache)

    bad = StubAssistant(degrade=True)
    bad.ask("first query")
    good = StubAssistant()
    good.ask("second query")
    assert len(fake.store) == 1

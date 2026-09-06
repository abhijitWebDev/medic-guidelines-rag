"""Retrieval: HyDE blending, its fallbacks, and the halves it must not touch."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rag_project import llm
from rag_project.cache import reset_cache
from rag_project.config import get_settings
from rag_project.indexing.store import InMemoryStore
from rag_project.llm import LLMError
from rag_project.models import Chunk
from rag_project.retrieval import hyde
from rag_project.retrieval.search import HybridRetriever

DIM = 4


class FakeEmbedder:
    """Maps known strings to fixed vectors; anything else to a distinct axis.

    Deterministic on purpose -- the blending assertions below compare exact
    vectors, which is the only way to tell "HyDE moved the query" apart from
    "HyDE replaced the query".
    """

    def __init__(self, table: dict[str, list[float]] | None = None) -> None:
        self.table = table or {}
        self.calls: list[str] = []
        self.dim = DIM

    def embed_one(self, text: str) -> list[float]:
        self.calls.append(text)
        if text in self.table:
            return list(self.table[text])
        # Stable pseudo-vector so unknown text still embeds somewhere fixed.
        h = abs(hash(text)) % 1000
        return [float(h % 7), float(h % 5), float(h % 3), 1.0]


@pytest.fixture(autouse=True)
def _no_hyde_cache():
    """Each test starts with an empty hypothetical cache."""
    reset_cache()
    llm.clear_degradations()
    yield
    reset_cache()
    llm.clear_degradations()


def _settings(**overrides):
    get_settings.cache_clear()
    s = get_settings()
    for key, value in overrides.items():
        setattr(s, key, value)
    return s


def _unit(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32)
    return a / (np.linalg.norm(a) or 1.0)


# --- HyDE: the vector it produces ----------------------------------------

def test_disabled_returns_the_plain_query_vector(monkeypatch):
    _settings(hyde_enabled=False)
    monkeypatch.setattr(
        hyde, "generate", lambda *a, **k: pytest.fail("HyDE ran while disabled")
    )
    embedder = FakeEmbedder({"how is tb diagnosed": [1.0, 0.0, 0.0, 0.0]})

    vec, trace = hyde.query_vector("how is tb diagnosed", embedder)

    assert vec == [1.0, 0.0, 0.0, 0.0]
    assert trace == {"used": False, "reason": "disabled"}
    assert embedder.calls == ["how is tb diagnosed"]


def test_blend_keeps_the_query_in_the_mix(monkeypatch):
    """The defining property: the result is between the query and the
    hypothetical, never at either end. Textbook HyDE lands on the
    hypothetical alone; this project deliberately does not."""
    _settings(hyde_enabled=True, hyde_n=1, hyde_query_weight=0.5)
    monkeypatch.setattr(hyde, "generate", lambda q, n, model=None: ["HYPOTHETICAL"])

    query_v = [1.0, 0.0, 0.0, 0.0]
    hyp_v = [0.0, 1.0, 0.0, 0.0]
    embedder = FakeEmbedder({"q": query_v, "HYPOTHETICAL": hyp_v})

    vec, trace = hyde.query_vector("q", embedder)

    assert trace["used"] is True
    assert trace["passages"] == ["HYPOTHETICAL"]
    # Halfway between two orthogonal unit vectors, renormalised.
    expected = [1 / math.sqrt(2), 1 / math.sqrt(2), 0.0, 0.0]
    assert vec == pytest.approx(expected, abs=1e-6)
    # Unit length, so the store's cosine ranking is unaffected by the blend.
    assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("weight,expected_first", [(1.0, 1.0), (0.0, 0.0)])
def test_weight_spans_query_only_to_hypothetical_only(monkeypatch, weight, expected_first):
    _settings(hyde_enabled=True, hyde_n=1, hyde_query_weight=weight)
    monkeypatch.setattr(hyde, "generate", lambda q, n, model=None: ["H"])
    embedder = FakeEmbedder({"q": [1.0, 0.0, 0.0, 0.0], "H": [0.0, 1.0, 0.0, 0.0]})

    vec, _ = hyde.query_vector("q", embedder)

    assert vec[0] == pytest.approx(expected_first, abs=1e-6)


def test_out_of_range_weight_is_clamped_not_fatal(monkeypatch):
    """A bad HYDE_QUERY_WEIGHT degrades to a sane blend rather than failing
    every query -- config values arrive from the environment untyped."""
    _settings(hyde_enabled=True, hyde_n=1, hyde_query_weight=9.0)
    monkeypatch.setattr(hyde, "generate", lambda q, n, model=None: ["H"])
    embedder = FakeEmbedder({"q": [1.0, 0.0, 0.0, 0.0], "H": [0.0, 1.0, 0.0, 0.0]})

    vec, trace = hyde.query_vector("q", embedder)

    assert trace["query_weight"] == 1.0
    assert vec == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-6)


def test_multiple_passages_are_averaged(monkeypatch):
    _settings(hyde_enabled=True, hyde_n=2, hyde_query_weight=0.0)
    monkeypatch.setattr(hyde, "generate", lambda q, n, model=None: ["A", "B"])
    embedder = FakeEmbedder({
        "q": [1.0, 0.0, 0.0, 0.0],
        "A": [0.0, 1.0, 0.0, 0.0],
        "B": [0.0, 0.0, 1.0, 0.0],
    })

    vec, trace = hyde.query_vector("q", embedder)

    assert trace["n"] == 2
    assert vec == pytest.approx([0.0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0.0], abs=1e-6)


# --- HyDE: failure is open, unlike every gate ----------------------------

def test_unavailable_model_falls_back_instead_of_refusing(monkeypatch):
    """The one component here that does not fail closed. A gate that cannot
    run must refuse; a retrieval improvement that cannot run must step aside."""
    _settings(hyde_enabled=True, hyde_n=1, hyde_query_weight=0.5)

    def boom(*a, **k):
        raise LLMError("upstream is down")

    monkeypatch.setattr(hyde, "generate", boom)
    embedder = FakeEmbedder({"q": [1.0, 0.0, 0.0, 0.0]})

    vec, trace = hyde.query_vector("q", embedder)

    assert vec == [1.0, 0.0, 0.0, 0.0]
    assert trace["used"] is False
    assert "model unavailable" in trace["reason"]
    # Recorded, so assistant.ask will not cache this weaker retrieval path.
    assert "hyde" in llm.degradations()


def test_empty_generation_is_not_a_degradation(monkeypatch):
    """A well-formed response carrying no passage is not a service failure,
    so it must not suppress caching the way an outage does."""
    _settings(hyde_enabled=True, hyde_n=1, hyde_query_weight=0.5)
    monkeypatch.setattr(hyde, "generate", lambda q, n, model=None: [])
    embedder = FakeEmbedder({"q": [1.0, 0.0, 0.0, 0.0]})

    vec, trace = hyde.query_vector("q", embedder)

    assert vec == [1.0, 0.0, 0.0, 0.0]
    assert trace == {"used": False, "reason": "model returned no passage"}
    assert llm.degradations() == ()


def test_hypothetical_is_generated_once_per_query(monkeypatch):
    _settings(hyde_enabled=True, hyde_n=1, hyde_query_weight=0.5)
    calls: list[str] = []

    def counted(q, n, model=None):
        calls.append(q)
        return ["H"]

    monkeypatch.setattr(hyde, "generate", counted)
    embedder = FakeEmbedder({"q": [1.0, 0.0, 0.0, 0.0], "H": [0.0, 1.0, 0.0, 0.0]})

    first, _ = hyde.query_vector("q", embedder)
    second, _ = hyde.query_vector("q", embedder)

    assert calls == ["q"]
    assert first == second


# --- HyDE inside the hybrid retriever ------------------------------------

def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=cid, doc_id="d", title="Pulmonary Tuberculosis", source_url="u",
        publisher="MOHFW", section_path="Diagnosis", page_start=1, page_end=1,
        text=text, n_tokens=20, index_version="v1", ingested_at="now",
    )


def test_lexical_half_never_sees_the_hypothetical(monkeypatch):
    """BM25 is what preserves exact drug names and abbreviations. Handing it
    eighty words of generated prose is the fastest way to lose them."""
    _settings(hyde_enabled=True, hyde_n=1, hyde_query_weight=0.5)
    hypothetical = "Sputum smear microscopy is performed at peripheral facilities."
    monkeypatch.setattr(hyde, "generate", lambda q, n, model=None: [hypothetical])

    chunks = [
        _chunk("c1", "NAAT is recommended as the initial diagnostic test."),
        _chunk("c2", "Refer where drug resistance is suspected."),
    ]
    store = InMemoryStore("t")
    store.create([{**c.model_dump(), "vector": [1.0, 0.0, 0.0, 0.0]} for c in chunks])

    retriever = HybridRetriever(
        store=store, embedder=FakeEmbedder(), chunks=chunks
    )
    seen: list[str] = []
    original = retriever.bm25.search
    retriever.bm25.search = lambda q, limit=20: (seen.append(q), original(q, limit))[1]

    _, trace = retriever.retrieve("how is TB diagnosed", k=5)

    # The lexical half searched the literal rewritten query, expansions and all.
    assert seen == ["how is TB diagnosed tuberculosis"]
    assert hypothetical not in seen[0]
    # ...while the dense half did use HyDE.
    assert trace["hyde"]["used"] is True
    assert trace["hyde"]["passages"] == [hypothetical]


def test_retrieval_trace_records_the_hypothetical(monkeypatch):
    """A silent rewrite is untraceable when an answer later looks wrong, and
    that argument is stronger for a generated passage than for an expansion."""
    _settings(hyde_enabled=False)
    chunks = [_chunk("c1", "NAAT is recommended as the initial test.")]
    store = InMemoryStore("t")
    store.create([{**chunks[0].model_dump(), "vector": [1.0, 0.0, 0.0, 0.0]}])

    retriever = HybridRetriever(store=store, embedder=FakeEmbedder(), chunks=chunks)
    _, trace = retriever.retrieve("how is TB diagnosed", k=5)

    assert trace["hyde"] == {"used": False, "reason": "disabled"}

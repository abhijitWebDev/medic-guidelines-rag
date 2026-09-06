"""Gate 2's corrective band: the retry, and the invariants that keep it safe.

The band is the one place in this pipeline where a refusal is reconsidered, so
these tests are mostly about what it is *not* allowed to do.
"""

from __future__ import annotations

import pytest

from rag_project.config import get_settings
from rag_project.guardrails import confidence
from rag_project.models import Chunk, ConfidenceAction, Retrieved
from rag_project.retrieval import corrective
from rag_project.retrieval.search import RetrievalError


def _r(cid: str, score: float | None) -> Retrieved:
    return Retrieved(
        chunk=Chunk(
            chunk_id=cid, doc_id="d", title="T", source_url="u", publisher="MOHFW",
            section_path="S", page_start=1, page_end=1, text=f"text {cid}",
            n_tokens=10, index_version="v1", ingested_at="now",
        ),
        rerank_score=score,
    )


def _settings(**overrides):
    get_settings.cache_clear()
    s = get_settings()
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


class FakeRetriever:
    """Returns a fixed second-pass pool and records how it was called."""

    def __init__(self, pool=None, error=None):
        self.pool = pool or []
        self.error = error
        self.calls = []

    def retrieve(self, query, k=20, hyde_query_weight=None):
        self.calls.append({"k": k, "hyde_query_weight": hyde_query_weight})
        if self.error:
            raise self.error
        return [_r(c, None) for c in self.pool], {"hyde": {"used": True}}


class FakeReranker:
    """Assigns scores from a table; records exactly which passages it saw."""

    def __init__(self, scores: dict[str, float]):
        self.scores = scores
        self.seen: list[list[str]] = []

    def rerank(self, query, results):
        self.seen.append([r.chunk.chunk_id for r in results])
        for r in results:
            r.rerank_score = self.scores.get(r.chunk.chunk_id, 0.0)
        return results


# --- the three-way gate ---------------------------------------------------

@pytest.mark.parametrize(
    "top,expected",
    [
        (9.0, ConfidenceAction.PROCEED),   # clears the bar
        (4.0, ConfidenceAction.CORRECT),   # inside the band
        (3.0, ConfidenceAction.CORRECT),   # floor is inclusive
        (2.0, ConfidenceAction.REFUSE),    # below the floor: not worth a retry
    ],
)
def test_band_boundaries(top, expected):
    _settings(corrective_enabled=True)
    v = confidence.evaluate([_r("c1", top)], threshold=5.0, corrective_threshold=3.0)
    assert v.action is expected
    # Only PROCEED may ever pass the gate.
    assert v.passed is (expected is ConfidenceAction.PROCEED)


def test_correct_verdict_keeps_nothing():
    """A CORRECT verdict must not hand passages downstream: it is not a pass,
    and `kept` is what the generator would be grounded on."""
    v = confidence.evaluate([_r("c1", 4.0)], threshold=5.0, corrective_threshold=3.0)
    assert v.action is ConfidenceAction.CORRECT and v.kept == []


def test_second_look_cannot_ask_for_another_retry():
    """Bounded at one. `allow_correction=False` collapses the band, so the
    band can never become a loop."""
    v = confidence.evaluate(
        [_r("c1", 4.0)], threshold=5.0, corrective_threshold=3.0,
        allow_correction=False,
    )
    assert v.action is ConfidenceAction.REFUSE and not v.passed


def test_unscored_pool_refuses_rather_than_retrying():
    """An unavailable reranker is not a retrieval problem. Retrying would pay
    for a second pass to reach the same fail-closed refusal."""
    _settings(corrective_enabled=True)
    v = confidence.evaluate([_r("c1", None)], threshold=5.0, corrective_threshold=3.0)
    assert v.action is ConfidenceAction.REFUSE and not v.passed


def test_disabling_corrective_restores_the_binary_gate():
    _settings(corrective_enabled=False)
    v = confidence.evaluate([_r("c1", 4.0)], threshold=5.0, corrective_threshold=3.0)
    assert v.action is ConfidenceAction.REFUSE


# --- the corrective pass --------------------------------------------------

def test_retry_goes_wider_and_leans_on_the_hypothetical():
    _settings(corrective_k=40, corrective_hyde_query_weight=0.2)
    retriever = FakeRetriever(pool=["c1", "c2"])
    corrective.correct(retriever, FakeReranker({}), "q", [_r("c1", 4.0)])

    assert retriever.calls == [{"k": 40, "hyde_query_weight": 0.2}]


def test_only_unseen_passages_are_rescored():
    """Rescoring what the first pass already scored is waste. Sound because
    the reranker's scale is absolute, not relative (see rerank.py)."""
    _settings(corrective_k=40)
    retriever = FakeRetriever(pool=["c1", "c2", "c3"])
    reranker = FakeReranker({"c2": 9.0, "c3": 1.0})

    pool, trace = corrective.correct(retriever, reranker, "q", [_r("c1", 4.0)])

    assert reranker.seen == [["c2", "c3"]]      # c1 was not re-sent
    assert trace["new_candidates"] == 2
    assert {r.chunk.chunk_id: r.rerank_score for r in pool} == {
        "c1": 4.0, "c2": 9.0, "c3": 1.0
    }


def test_first_pass_passages_are_never_lost():
    """A retry that fuses differently must not drop a passage the first pass
    already scored. A correction can only add candidates."""
    _settings(corrective_k=40)
    retriever = FakeRetriever(pool=["c9"])       # c1 absent from the retry
    pool, _ = corrective.correct(
        retriever, FakeReranker({"c9": 2.0}), "q", [_r("c1", 4.0)]
    )

    assert {r.chunk.chunk_id for r in pool} == {"c1", "c9"}


def test_pool_is_sorted_best_first():
    _settings(corrective_k=40)
    retriever = FakeRetriever(pool=["c2"])
    pool, _ = corrective.correct(
        retriever, FakeReranker({"c2": 10.0}), "q", [_r("c1", 4.0)]
    )
    assert [r.chunk.chunk_id for r in pool] == ["c2", "c1"]


def test_failed_retry_leaves_the_original_verdict_standing():
    """A corrective pass that cannot run must not turn a clean refusal into
    an error."""
    _settings(corrective_k=40)
    retriever = FakeRetriever(error=RetrievalError("index unreachable"))
    previous = [_r("c1", 4.0)]

    pool, trace = corrective.correct(retriever, FakeReranker({}), "q", previous)

    assert pool is previous
    assert trace["ran"] is False and "index unreachable" in trace["reason"]


def test_trace_reports_whether_the_retry_helped():
    _settings(corrective_k=40)
    retriever = FakeRetriever(pool=["c2"])
    _, better = corrective.correct(
        retriever, FakeReranker({"c2": 9.0}), "q", [_r("c1", 4.0)]
    )
    _, worse = corrective.correct(
        FakeRetriever(pool=["c3"]), FakeReranker({"c3": 1.0}), "q", [_r("c1", 4.0)]
    )

    assert better["top_before"] == 4.0 and better["top_after"] == 9.0
    assert better["improved"] is True
    assert worse["improved"] is False


# --- the invariant --------------------------------------------------------

def test_correction_does_not_lower_the_bar():
    """The whole safety argument. A retry that fails to clear the ORIGINAL
    threshold still refuses -- the band buys another attempt at the bar, never
    a lower bar."""
    _settings(corrective_k=40, corrective_enabled=True)
    retriever = FakeRetriever(pool=["c2", "c3"])
    reranker = FakeReranker({"c2": 4.5, "c3": 4.9})   # better, still under 5.0

    pool, _ = corrective.correct(reranker=reranker, retriever=retriever,
                                 query="q", previous=[_r("c1", 4.0)])
    verdict = confidence.evaluate(
        pool, threshold=5.0, corrective_threshold=3.0, allow_correction=False
    )

    assert verdict.top_score == 4.9        # the retry genuinely improved things
    assert not verdict.passed              # and it still refuses
    assert verdict.kept == []

"""Gate 2: refuse when retrieval is not confident enough to ground an answer.

This is the gate that makes "I don't know" possible. Without it the generator is
handed the best five chunks no matter how poor they are, and a fluent model will
find something to say about them.

It reads the *absolute* rerank score, not a rank. See rerank.py: the top-1 of a
uniformly irrelevant set still ranks first, so ranking cannot express "nothing
here answers this".

The gate is three-way, not two-way. Between `corrective_threshold` and
`confidence_threshold` sits a band where retrieval found the right subject area
but not the answer -- 4-6 on the reranker's scale, "related topic, contains
part of the answer". Refusing there discards a recoverable query; answering
there grounds an answer in passages the evaluator just said were partial. So
that band returns CORRECT instead, and the caller retries retrieval before the
gate decides. See retrieval/corrective.py.

The bar itself never moves. A corrective pass earns a second attempt at
`confidence_threshold`, never a lower one -- the moment the band becomes a
cheaper way past gate 2, it has inverted the gate's purpose.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import get_settings
from ..models import ConfidenceAction, Retrieved


@dataclass
class ConfidenceVerdict:
    passed: bool
    top_score: float | None
    threshold: float
    reason: str
    kept: list[Retrieved]
    #: Defaults to REFUSE so a verdict built without one fails closed.
    action: ConfidenceAction = ConfidenceAction.REFUSE


def evaluate(
    results: list[Retrieved],
    threshold: float | None = None,
    top_n: int | None = None,
    corrective_threshold: float | None = None,
    allow_correction: bool = True,
) -> ConfidenceVerdict:
    """Judge whether retrieval can ground an answer.

    `allow_correction=False` collapses the band back to a binary decision. The
    caller passes it on the second look, after a corrective pass has already
    run: the retry is bounded at one, and a gate that could keep asking for
    another attempt is a loop, not a gate.
    """
    s = get_settings()
    threshold = s.confidence_threshold if threshold is None else threshold
    top_n = s.rerank_top_n if top_n is None else top_n
    floor = s.corrective_threshold if corrective_threshold is None else corrective_threshold
    correcting = allow_correction and s.corrective_enabled

    if not results:
        return ConfidenceVerdict(False, None, threshold, "retrieval returned nothing", [])

    scored = [r for r in results if r.rerank_score is not None]
    if not scored:
        # No retry here. An unscored pool means the reranker is unavailable, and
        # a corrective pass would re-retrieve, fail to score again, and refuse
        # a second time -- paying twice to reach the same fail-closed answer.
        return ConfidenceVerdict(
            False, None, threshold,
            "reranker produced no scores; refusing rather than guessing", [],
        )

    top = max(r.rerank_score or 0.0 for r in scored)
    if top < threshold:
        if correcting and top >= floor:
            return ConfidenceVerdict(
                False, top, threshold,
                f"best passage scored {top:.1f}, inside the corrective band "
                f"[{floor:.1f}, {threshold:.1f}) -- retrying retrieval",
                [], ConfidenceAction.CORRECT,
            )
        return ConfidenceVerdict(
            False, top, threshold,
            f"best passage scored {top:.1f} < {threshold:.1f}", [],
        )

    # Keep only passages that clear the bar. Padding the context with weak
    # passages invites the generator to cite them.
    kept = [r for r in scored if (r.rerank_score or 0.0) >= threshold][:top_n]
    for i, r in enumerate(kept, start=1):
        r.marker = f"C{i}"
    return ConfidenceVerdict(
        True, top, threshold, f"top passage scored {top:.1f}", kept,
        ConfidenceAction.PROCEED,
    )

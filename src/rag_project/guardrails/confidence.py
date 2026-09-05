"""Gate 2: refuse when retrieval is not confident enough to ground an answer.

This is the gate that makes "I don't know" possible. Without it the generator is
handed the best five chunks no matter how poor they are, and a fluent model will
find something to say about them.

It reads the *absolute* rerank score, not a rank. See rerank.py: the top-1 of a
uniformly irrelevant set still ranks first, so ranking cannot express "nothing
here answers this".
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import get_settings
from ..models import Retrieved


@dataclass
class ConfidenceVerdict:
    passed: bool
    top_score: float | None
    threshold: float
    reason: str
    kept: list[Retrieved]


def evaluate(
    results: list[Retrieved], threshold: float | None = None, top_n: int | None = None
) -> ConfidenceVerdict:
    s = get_settings()
    threshold = s.confidence_threshold if threshold is None else threshold
    top_n = s.rerank_top_n if top_n is None else top_n

    if not results:
        return ConfidenceVerdict(False, None, threshold, "retrieval returned nothing", [])

    scored = [r for r in results if r.rerank_score is not None]
    if not scored:
        return ConfidenceVerdict(
            False, None, threshold,
            "reranker produced no scores; refusing rather than guessing", [],
        )

    top = max(r.rerank_score or 0.0 for r in scored)
    if top < threshold:
        return ConfidenceVerdict(
            False, top, threshold,
            f"best passage scored {top:.1f} < {threshold:.1f}", [],
        )

    # Keep only passages that clear the bar. Padding the context with weak
    # passages invites the generator to cite them.
    kept = [r for r in scored if (r.rerank_score or 0.0) >= threshold][:top_n]
    for i, r in enumerate(kept, start=1):
        r.marker = f"C{i}"
    return ConfidenceVerdict(True, top, threshold, f"top passage scored {top:.1f}", kept)

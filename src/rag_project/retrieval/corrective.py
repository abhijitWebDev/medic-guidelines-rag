"""The corrective pass: one wider, differently-phrased retry before refusing.

This is CRAG (Yan et al., 2024) with its external-knowledge action removed.
CRAG scores retrieval, and on a poor score discards what it found and falls
back to web search. That fallback cannot exist here: every claim a user reads
must trace to one of the sha256-pinned MOHFW documents, which is what makes
the citations, the index manifest, and gate 3's numeric-provenance check mean
anything. Answers from the open web would break all three.

So the corrective action is directed at the *same corpus*, on the premise that
a middling score usually means the passage exists and the first query missed
it rather than that the corpus is silent. Two levers:

  * **Depth.** The first pass fuses to `retrieve_k` and reranks those. Chunks
    landing at ranks 21-40 were never scored at all, so widening to
    `corrective_k` is recall the reranker could not previously have seen.
  * **Phrasing.** The literal query has already demonstrably failed, so the
    retry leans onto the HyDE hypothetical (`corrective_hyde_query_weight`).
    This is where HyDE's latency is actually earned: paid on the queries that
    need it, rather than charged to every query that was already fine.

Three properties make this safe to run inside a refusal gate.

1. **The bar does not move.** The merged pool is re-judged against the same
   `confidence_threshold`. A corrective pass buys a second attempt at the bar,
   never a lower bar. Everything else here is an optimisation; this one is the
   invariant.

2. **Exactly one retry.** The second evaluation is made with
   `allow_correction=False`, so there is no path that loops.

3. **Nothing is discarded.** The merged pool is the union of both passes, so a
   passage the first pass scored well cannot be lost by a retry that happened
   to fuse differently. A correction can only ever add candidates.

Already-scored passages keep their scores and are not re-sent to the reranker.
That is sound precisely because the reranker scores on an absolute scale and is
told not to rank passages against each other (see rerank.py) -- a score means
the same thing whichever batch produced it.
"""

from __future__ import annotations

from ..config import get_settings
from ..indexing.store import StoreError
from ..llm import LLMError
from ..models import Retrieved
from .rerank import Reranker
from .search import RetrievalError


def correct(
    retriever,
    reranker: Reranker,
    query: str,
    previous: list[Retrieved],
) -> tuple[list[Retrieved], dict]:
    """Retry retrieval once, wider and HyDE-weighted, and score what is new.

    Returns the merged, fully-scored pool and a trace. On any failure it
    returns `previous` untouched: a corrective pass that cannot run must leave
    the original verdict standing, never turn a clean refusal into an error.
    """
    s = get_settings()
    scores = {
        r.chunk.chunk_id: r.rerank_score
        for r in previous
        if r.rerank_score is not None
    }
    before = max(scores.values(), default=0.0)

    try:
        retried, rtrace = retriever.retrieve(
            query, k=s.corrective_k, hyde_query_weight=s.corrective_hyde_query_weight
        )
    except (RetrievalError, StoreError, LLMError) as e:
        return previous, {"ran": False, "reason": f"retrieval failed: {e}"[:200]}

    # Union, first pass first: a chunk retrieved twice keeps the object that
    # already carries a score.
    merged: dict[str, Retrieved] = {r.chunk.chunk_id: r for r in previous}
    for r in retried:
        if r.chunk.chunk_id in merged:
            continue
        r.rerank_score = scores.get(r.chunk.chunk_id)
        merged[r.chunk.chunk_id] = r

    pool = list(merged.values())
    fresh = [r for r in pool if r.rerank_score is None]

    if fresh:
        # Only the passages the first pass never saw. Safe because the score is
        # absolute, not relative -- see this module's docstring.
        reranker.rerank(query, fresh)

    pool.sort(
        key=lambda r: (r.rerank_score is not None, r.rerank_score or 0.0), reverse=True
    )
    after = max((r.rerank_score or 0.0 for r in pool), default=0.0)

    return pool, {
        "ran": True,
        "k": s.corrective_k,
        "hyde_query_weight": s.corrective_hyde_query_weight,
        "new_candidates": len(fresh),
        "pool": len(pool),
        "top_before": before,
        "top_after": after,
        "improved": after > before,
        "hyde": rtrace.get("hyde"),
    }

"""Lexical half of hybrid retrieval.

The self-hosted LanceDB wrapper exposes only dense vector search, so BM25 runs
here, client-side, over the same chunks that were embedded. This is not a
consolation prize: in a corpus of drug names and abbreviations -- rifampicin vs
rifabutin, ORS, DOTS, NAAT -- exact lexical match is precisely where dense
embeddings are weakest, so the lexical half carries real weight.

The index is rebuilt from data/chunks/*.jsonl rather than persisted. At a few
thousand chunks that costs milliseconds, and it removes a whole class of bug
where the lexical and dense halves silently describe different corpora.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from ..models import Chunk

# Keep intra-word hyphens and digits: "co-trimoxazole", "B12", "25(OH)D".
_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        # Index the contextual form (title + section + body) so the lexical and
        # dense halves see identical text -- otherwise RRF fuses rankings over
        # two different documents.
        corpus = [
            tokenize(f"{c.title} {c.section_path} {c.text}") for c in chunks
        ]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        self._by_id = {c.chunk_id: i for i, c in enumerate(chunks)}

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: str, limit: int = 20) -> list[tuple[Chunk, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = [(self.chunks[i], float(scores[i])) for i in ranked[:limit]]
        return [(c, s) for c, s in out if s > 0.0]


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = 60, weights: list[float] | None = None
) -> dict[str, float]:
    """RRF over ranked chunk_id lists.

    Rank-based rather than score-based on purpose: BM25 scores are unbounded and
    corpus-relative while cosine similarity is bounded, so the two are not
    commensurable and normalising them invents a comparison that isn't there.
    RRF only ever compares a document's position within its own ranking.
    """
    weights = weights or [1.0] * len(rankings)
    fused: dict[str, float] = {}
    for ranking, w in zip(rankings, weights, strict=True):
        for rank, chunk_id in enumerate(ranking):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + w / (k + rank + 1)
    return fused

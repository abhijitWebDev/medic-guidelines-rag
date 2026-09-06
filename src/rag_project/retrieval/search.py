"""Hybrid retrieval: remote dense search + local BM25, fused with RRF.

Both halves must describe the same corpus or the fusion is meaningless. That is
enforced structurally: the dense half resolves its hits back to the *local*
chunk objects by chunk_id, so a chunk present remotely but absent locally is
reported rather than silently half-retrieved.

The two halves do *not* search with the same query, and that asymmetry is
deliberate: the dense half searches with a HyDE-blended vector, the lexical
half with the literal rewritten string. Same corpus, different query
representations -- which is the entire reason to fuse them. See hyde.py.
"""

from __future__ import annotations

from ..indexing.bm25 import BM25Index, reciprocal_rank_fusion
from ..indexing.embed import Embedder
from ..indexing.store import VectorStore, get_store
from ..ingest.pipeline import load_chunks
from ..models import Chunk, Retrieved
from . import hyde
from .rewrite import rewrite


class RetrievalError(RuntimeError):
    pass


class HybridRetriever:
    def __init__(
        self,
        store: VectorStore | None = None,
        embedder: Embedder | None = None,
        chunks: list[Chunk] | None = None,
    ) -> None:
        self.chunks = chunks if chunks is not None else load_chunks()
        if not self.chunks:
            raise RetrievalError(
                "No local chunks found. The lexical half of hybrid search reads "
                "data/chunks/*.jsonl — run `uv run rag ingest` first."
            )
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.bm25 = BM25Index(self.chunks)
        self.store = store if store is not None else get_store()
        self.embedder = embedder if embedder is not None else Embedder()

    def retrieve(self, query: str, k: int = 20) -> tuple[list[Retrieved], dict]:
        rewritten, applied = rewrite(query)

        # Dense half (remote). HyDE blends a generated hypothetical passage
        # into the query vector; it degrades to the plain query vector when
        # the model is unavailable, and never touches the lexical half below.
        qv, hyde_trace = hyde.query_vector(rewritten, self.embedder)
        raw_hits = self.store.search(qv, limit=k)
        dense: list[tuple[Chunk, float]] = []
        orphans: list[str] = []
        for hit in raw_hits:
            cid = hit.get("chunk_id")
            chunk = self.by_id.get(cid)
            if chunk is None:
                # Present remotely, absent locally: reconstruct so the answer is
                # still citable, but surface it -- the two halves have drifted.
                try:
                    chunk = Chunk(**{k2: v for k2, v in hit.items()
                                     if k2 in Chunk.model_fields})
                    orphans.append(cid or "?")
                except Exception:
                    continue
            dense.append((chunk, float(hit.get("_distance", 0.0))))

        # Lexical half (local).
        lexical = self.bm25.search(rewritten, limit=k)

        # Fuse on rank, never on score -- see reciprocal_rank_fusion's docstring.
        fused = reciprocal_rank_fusion(
            [[c.chunk_id for c, _ in dense], [c.chunk_id for c, _ in lexical]]
        )
        dense_scores = {c.chunk_id: s for c, s in dense}
        lex_scores = {c.chunk_id: s for c, s in lexical}
        pool = {c.chunk_id: c for c, _ in dense} | {c.chunk_id: c for c, _ in lexical}

        results = [
            Retrieved(
                chunk=pool[cid],
                vector_score=dense_scores.get(cid),
                fts_score=lex_scores.get(cid),
                fused_score=score,
            )
            for cid, score in sorted(fused.items(), key=lambda x: -x[1])
        ][:k]

        trace = {
            "query": query,
            "rewritten": rewritten,
            "expansions": applied,
            "hyde": hyde_trace,
            "dense_hits": len(dense),
            "lexical_hits": len(lexical),
            "fused_pool": len(fused),
            "orphan_chunks": orphans,
        }
        return results, trace

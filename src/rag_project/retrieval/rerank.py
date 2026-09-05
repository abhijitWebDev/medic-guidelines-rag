"""Cross-encoder style reranking, and the score the refusal gate depends on.

The deployment has no reranker endpoint, so this scores (query, chunk) pairs
with a cheap LLM in a single batched call. What matters more than the ordering
is that the score is *absolute*: gate 2 refuses on the top-1 value, so a
relative ranking would be useless -- the best of five irrelevant chunks still
ranks first. The prompt therefore asks how well a passage answers the question
on a fixed scale, never which passage is best.

`Reranker` is a Protocol so a real cross-encoder (Cohere Rerank, or a local
bge-reranker) can be dropped in without touching retrieval or the gate.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from ..llm import LLMError, note_degraded, parse
from ..models import Retrieved


class Reranker(Protocol):
    def rerank(self, query: str, results: list[Retrieved]) -> list[Retrieved]: ...


class _Score(BaseModel):
    index: int = Field(description="the passage number being scored")
    score: int = Field(ge=0, le=10, description="0 = irrelevant, 10 = fully answers")


class _Scores(BaseModel):
    scores: list[_Score]


_SYSTEM = """You score how well each passage answers a question, for a retrieval \
system over official medical guidelines.

Score each passage independently on this fixed scale. Do NOT rank them against \
each other, and do not spread scores out to differentiate — if every passage is \
irrelevant, every score should be low.

10 — directly and completely answers the question
7-9 — contains most of the answer
4-6 — related topic, contains part of the answer
1-3 — same general subject area, does not answer the question
0 — unrelated

Return one score for every passage, using the passage numbers given."""


class LLMReranker:
    def __init__(self, model: str | None = None) -> None:
        self.model = model

    def rerank(self, query: str, results: list[Retrieved]) -> list[Retrieved]:
        if not results:
            return []

        passages = "\n\n".join(
            f"[{i}] {r.chunk.title} — {r.chunk.section_path}\n{r.chunk.text}"
            for i, r in enumerate(results)
        )
        user = f"Question: {query}\n\nPassages:\n\n{passages}"

        try:
            out = parse(_Scores, _SYSTEM, user, model=self.model)
        except LLMError:
            note_degraded("reranker")
            # Fail closed: no scores means gate 2 sees no confidence and refuses,
            # which is the correct outcome when the reranker is unavailable.
            for r in results:
                r.rerank_score = None
            return results

        by_index = {s.index: s.score for s in out.scores}
        for i, r in enumerate(results):
            raw = by_index.get(i)
            r.rerank_score = float(raw) if raw is not None else None

        return sorted(
            results, key=lambda r: (r.rerank_score is not None, r.rerank_score or 0.0),
            reverse=True,
        )

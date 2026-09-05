"""End-to-end query pipeline.

    intent gate -> retrieve -> rerank -> confidence gate -> generate -> output gate

Every exit produces a Response. There is no path that returns raw model text,
and no path that skips a gate: a refusal at any stage returns the standard
message for that reason from guardrails.policy, with the trace explaining why.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import llm
from .cache import get_cache, key_for
from .config import get_settings
from .generation.generate import generate
from .guardrails import confidence, output_gate
from .guardrails.input_gate import classify
from .guardrails.policy import DISCLAIMER, refusal_text
from .models import Intent, RefusalReason, Response, Retrieved
from .retrieval.rerank import LLMReranker, Reranker
from .retrieval.rewrite import normalise
from .retrieval.search import HybridRetriever

_INTENT_TO_REFUSAL = {
    Intent.EMERGENCY: RefusalReason.EMERGENCY,
    Intent.PERSONALIZED_ADVICE: RefusalReason.PERSONALIZED_ADVICE,
    Intent.OUT_OF_DOMAIN: RefusalReason.OUT_OF_DOMAIN,
}


def _refuse(query: str, reason: RefusalReason, trace: dict, top: float | None = None) -> Response:
    return Response(
        query=query,
        answered=False,
        answer=refusal_text(reason),
        refusal_reason=reason,
        top_score=top,
        disclaimer=DISCLAIMER,
        trace=trace,
    )


def _citations(kept: list[Retrieved]) -> list[dict]:
    return [
        {
            "marker": r.marker,
            "chunk_id": r.chunk.chunk_id,
            "doc_id": r.chunk.doc_id,
            "title": r.chunk.title,
            "section": r.chunk.section_path,
            "pages": (
                "n/a"
                if not r.chunk.page_start
                else str(r.chunk.page_start)
                if r.chunk.page_start == r.chunk.page_end
                else f"{r.chunk.page_start}-{r.chunk.page_end}"
            ),
            "source_url": r.chunk.source_url,
            "rerank_score": r.rerank_score,
        }
        for r in kept
    ]


@dataclass
class Assistant:
    retriever: HybridRetriever
    reranker: Reranker

    @classmethod
    def build(cls) -> Assistant:
        return cls(retriever=HybridRetriever(), reranker=LLMReranker())

    def ask(self, query: str, screen: bool = True, use_cache: bool = True) -> Response:
        s = get_settings()
        cache = get_cache()
        # `screen` is part of the key: with it off, gate 1 never consults the
        # model, which is a different pipeline and may reach a different verdict.
        ckey = f"resp:{s.pipeline_fingerprint}:{key_for(normalise(query).casefold(), screen)}"

        llm.clear_degradations()

        if use_cache:
            hit = cache.get_json(ckey)
            if hit is not None:
                cached = Response.model_validate(hit)
                # Keep the caller's own wording; only the *answer* is reused.
                cached.query = query
                cached.trace = dict(cached.trace) | {"cache": "hit"}
                return cached

        response = self._run(query, screen)

        # Stamped here rather than at each return inside _run: one place that
        # cannot be forgotten when a new early-return is added.
        degraded = list(llm.degradations())
        if degraded:
            response.trace["degraded"] = degraded

        # Never cache a fail-closed refusal. Those are refusals produced by an
        # unavailable model rather than by a judgement about the query, and
        # writing one here would pin a transient outage into Redis for the
        # whole TTL -- turning a minute of API trouble into a day of a real
        # question being refused. See llm.note_degraded.
        if use_cache and not degraded:
            cache.set_json(ckey, response.model_dump(mode="json"), s.cache_response_ttl_s)
        return response

    def _run(self, query: str, screen: bool = True) -> Response:
        s = get_settings()
        trace: dict = {"stages": []}

        # --- Gate 1 ------------------------------------------------------
        verdict = classify(query, use_model=screen)
        trace["intent"] = {
            "label": verdict.intent.value,
            "reason": verdict.reason,
            "rule": verdict.matched_rule,
        }
        trace["stages"].append("intent")
        if verdict.intent is not Intent.IN_SCOPE:
            trace["short_circuited"] = "no retrieval performed"
            return _refuse(query, _INTENT_TO_REFUSAL[verdict.intent], trace)

        # --- Retrieve + rerank -------------------------------------------
        results, rtrace = self.retriever.retrieve(query, k=s.retrieve_k)
        trace["retrieval"] = rtrace
        trace["stages"].append("retrieve")

        results = self.reranker.rerank(query, results)
        trace["rerank"] = {
            "scored": sum(1 for r in results if r.rerank_score is not None),
            "top_scores": [r.rerank_score for r in results[:5]],
        }
        trace["stages"].append("rerank")

        # --- Gate 2 ------------------------------------------------------
        conf = confidence.evaluate(results)
        trace["confidence"] = {
            "passed": conf.passed,
            "top_score": conf.top_score,
            "threshold": conf.threshold,
            "reason": conf.reason,
            "kept": len(conf.kept),
        }
        trace["stages"].append("confidence")
        if not conf.passed:
            return _refuse(query, RefusalReason.LOW_CONFIDENCE, trace, conf.top_score)

        # --- Generate ----------------------------------------------------
        answer = generate(query, conf.kept)
        trace["generation"] = {
            "claims": len(answer.claims),
            "insufficient_context": answer.insufficient_context,
        }
        trace["stages"].append("generate")

        # --- Gate 3 ------------------------------------------------------
        checked = output_gate.validate(answer, conf.kept, query)
        trace["output_gate"] = {
            "passed": checked.passed,
            "detail": checked.detail,
            "invalid_citations": checked.invalid_citations,
            "unsupported_numbers": checked.unsupported_numbers,
            "unsupported_claims": checked.unsupported_claims,
            "safety_flags": checked.safety_flags,
        }
        trace["stages"].append("output_gate")
        if not checked.passed:
            assert checked.reason is not None
            return _refuse(query, checked.reason, trace, conf.top_score)

        # Gate 3 may have stripped unsupported claims; use what it verified.
        answer = checked.repaired or answer

        return Response(
            query=query,
            answered=True,
            answer=answer.answer,
            claims=answer.claims,
            citations=_citations(conf.kept),
            top_score=conf.top_score,
            disclaimer=DISCLAIMER,
            trace=trace,
        )

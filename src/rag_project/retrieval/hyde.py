"""HyDE: search with a hypothetical answer as well as with the question.

A question and the passage that answers it are written in different registers.
"How is drug-resistant TB confirmed?" shares almost no content words with
"Culture and drug susceptibility testing is performed on all presumptive
DR-TB cases...", and in embedding space that gap is real. HyDE (Gao et al.,
2022) closes it by asking a model to write the passage it *expects* to find,
and searching with that.

Asking a language model to write a plausible medical guideline is an alarming
thing to do in this application, so three rules bound it.

1. **The hypothetical document is never shown, never cited, never generated
   from.** It exists only to produce a query vector. It does not reach the
   reranker, the generator, or the citation list -- every sentence the user
   reads still comes from a retrieved chunk that gate 3 verified against the
   corpus. A fabricated passage can therefore change *which* real guidance is
   found; it can never change what is asserted.

2. **It is additive, exactly like the abbreviation expansion in rewrite.py.**
   The real query vector stays in the blend at `hyde_query_weight`, so a
   hypothetical that drifts to the wrong condition pulls retrieval partway
   rather than replacing the target outright. Textbook HyDE discards the query
   and searches with the generation alone; that hands the query to the model's
   priors, which is a poor trade when the corpus is the authority and the
   model is explicitly not.

3. **It touches the dense half only.** BM25 goes on searching the literal
   query. Same trade rewrite.py makes: the lexical half is what preserves
   exact drug names and abbreviations -- rifampicin vs rifabutin, NAAT, ORS --
   and burying those in eighty words of generated prose is the quickest way to
   lose them.

Failure is *open* here, which makes this the one component in the pipeline
that does not fail closed. Everything in llm.py's orbit is a safety gate, and
a gate that cannot run must refuse. HyDE is not a gate: it is a retrieval
improvement, and an unavailable model simply means falling back to embedding
the query -- which is precisely the retrieval this project had before HyDE
existed. The fallback is still reported through `note_degraded`, so a weaker
retrieval path is never pinned into the response cache for a day.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from ..cache import get_cache, key_for
from ..config import get_settings
from ..llm import LLMError, note_degraded, parse

# Roughly the mean chunk length in this corpus (214 tokens). A hypothetical
# that is much longer or much shorter than the things being searched lands in
# a different part of the embedding space for reasons that have nothing to do
# with its subject.
_TARGET_WORDS = 90


_SYSTEM = """You write short passages that imitate the Standard Treatment \
Guidelines published by the Ministry of Health and Family Welfare, Government \
of India.

Given a clinical question, write the guideline passage that would answer it. \
Write it as the guideline itself would: impersonal, declarative, clinical \
register, the vocabulary a guideline would use for this topic (drug names, \
investigations, criteria, thresholds).

Rules:
- About {words} words per passage. One paragraph, prose only.
- No headings, no markdown, no bullet points, no citations, no preamble.
- Do not address a reader, do not hedge, do not mention guidelines in general \
terms -- state the content.
- Write about the specific condition asked about and nothing else.
- If you are unsure of the exact contents, still write the passage you would \
expect: approximate specifics are useful here and are never presented as fact.

Return exactly {n} passage(s).

This text is retrieval scaffolding. It is never shown to anyone, never quoted, \
and never used to answer the question -- it is embedded and discarded, purely \
to find the real passages in the corpus. Write it directly."""


class _Hypothetical(BaseModel):
    passages: list[str] = Field(
        description="the hypothetical guideline passages, one per list entry"
    )


def generate(query: str, n: int, model: str | None = None) -> list[str]:
    """Write `n` hypothetical passages for `query`. Raises LLMError."""
    out = parse(
        _Hypothetical,
        _SYSTEM.format(n=n, words=_TARGET_WORDS),
        f"Question: {query}",
        model=model,
        # Non-zero only when more than one passage is wanted: identical
        # generations would average to a single point and buy nothing.
        temperature=0.0 if n == 1 else 0.7,
    )
    return [p.strip() for p in out.passages if p and p.strip()][:n]


def _cached_generate(query: str, n: int, model: str | None) -> list[str]:
    """`generate`, memoised. Raises LLMError like the function it wraps.

    Worth caching separately from the whole response: a query that later gets
    refused by gate 2, or asked again with different settings downstream, has
    already paid for this generation, and the passage is deterministic for a
    fixed (model, version, n, query).
    """
    s = get_settings()
    cache = get_cache()
    ckey = f"hyde:{key_for(model or s.openai_guard_model, s.hyde_version, n, query)}"

    hit = cache.get_json(ckey)
    if isinstance(hit, list) and hit and all(isinstance(p, str) for p in hit):
        return hit

    passages = generate(query, n, model=model)
    if passages:
        cache.set_json(ckey, passages, s.cache_hyde_ttl_s)
    return passages


def _unit(vector) -> np.ndarray:
    """L2-normalise. The blend is a sum of directions, so an un-normalised
    term would weight itself by its own magnitude rather than by the weight
    it was given."""
    v = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(v))
    return v if norm == 0.0 else v / norm


def query_vector(rewritten: str, embedder) -> tuple[list[float], dict]:
    """The vector the dense half searches with, plus what happened.

    `rewritten` is the output of retrieval.rewrite -- normalised, with the
    curated abbreviation expansions appended. That form is fed to the
    generator too, not just to the embedder: the glossary is this project's
    own disambiguation ("ARI" is acute respiratory infection here), and
    handing it over is cheaper and safer than letting the model guess which
    reading of an abbreviation the corpus meant.
    """
    s = get_settings()
    base = embedder.embed_one(rewritten)

    # hyde_n < 1 reads as "off" rather than as "ask for zero passages", which
    # would spend a call to be handed nothing.
    if not s.hyde_enabled or s.hyde_n < 1:
        return base, {"used": False, "reason": "disabled"}

    try:
        passages = _cached_generate(rewritten, s.hyde_n, s.hyde_model)
    except LLMError as e:
        note_degraded("hyde")
        return base, {"used": False, "reason": f"model unavailable: {e}"[:200]}

    if not passages:
        # Structurally valid response carrying nothing usable. Not a service
        # failure, so not a degradation -- just no expansion this time.
        return base, {"used": False, "reason": "model returned no passage"}

    # Clamped rather than validated at config time so a bad env var degrades to
    # a sane blend instead of failing every query at import.
    w = min(1.0, max(0.0, s.hyde_query_weight))
    hypothetical = np.mean([_unit(embedder.embed_one(p)) for p in passages], axis=0)
    blended = _unit(w * _unit(base) + (1.0 - w) * _unit(hypothetical))

    return blended.tolist(), {
        "used": True,
        "n": len(passages),
        "query_weight": w,
        # Kept in full. A silent rewrite is untraceable when an answer later
        # looks wrong (see rewrite.rewrite), and that argument is far stronger
        # for a whole generated passage than for a two-word expansion. The web
        # UI does not render this; it is here for `--trace` and the eval set.
        "passages": passages,
    }

"""Gate 3: validate what the model produced before a user ever sees it.

Four checks, ordered by cost, short-circuiting on the first failure:

  1. citation validity   -- deterministic, free
  2. numeric provenance  -- deterministic, free
  3. claim support       -- one LLM call
  4. framing / safety    -- one LLM call

The ordering is not just an optimisation. Checks 1 and 2 are *proofs*: a cited
id either exists or it does not; a number either appears in the cited text or it
does not. Checks 3 and 4 are judgements by the same class of system that
produced the output. Cheap proofs should never be gated behind expensive
opinions, and a failure that can be demonstrated is worth more than one that
must be argued.

Check 2 is the one that matters most in this corpus. A model that fabricates a
dose while correctly citing a real passage passes citation validation, reads
fluently, and is wrong in the most dangerous way available. Comparing the digits
in the answer against the digits in the cited passages catches exactly that, and
costs nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from ..llm import LLMError, note_degraded, parse
from ..models import GroundedAnswer, RefusalReason, Retrieved

# --- helpers -------------------------------------------------------------

_MARKER = re.compile(r"\[(?:C\d+)(?:\s*,\s*C\d+)*\]")
# Enumeration appears at a line start OR after a sentence end ("... [C1]. 2. ...").
_LIST_NUM = re.compile(r"(^|[.!?]\s+)\d+[.)]\s", re.MULTILINE)
# Clinical quantities: 10, 2.5, 10-15, 1/2. Excludes years like (2019) handled below.
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def strip_markers(text: str) -> str:
    return _LIST_NUM.sub(r"\1 ", _MARKER.sub(" ", text))


def numbers_in(text: str) -> set[str]:
    return {n.lstrip("0") or "0" for n in _NUMBER.findall(text)}


#: Above this share of unsupported claims, the answer is not salvaged but
#: refused: a couple of stray sentences is a drafting slip, but a majority means
#: the model was working from something other than the passages.
MAX_UNSUPPORTED_FRACTION = 0.4

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def repair(answer: GroundedAnswer, unsupported: list[str]) -> GroundedAnswer | None:
    """Drop unsupported claims and the sentences carrying them.

    The generator is explicitly told that partial answers are acceptable, so
    discarding six verified claims because a seventh was unsupported contradicts
    the instruction the answer was written under. Returns None when the answer
    cannot be salvaged, and the caller refuses.
    """
    kept_claims = [c for c in answer.claims if c.text not in set(unsupported)]
    if not kept_claims:
        return None

    sentences = _SENTENCE.split(answer.answer.strip())
    bad_words = [_words(u) for u in unsupported]
    kept: list[str] = []
    for sentence in sentences:
        sw = _words(sentence)
        if not sw:
            continue
        # Drop a sentence only when it clearly carries an unsupported claim.
        overlap = max(
            (len(sw & bw) / max(1, len(bw)) for bw in bad_words), default=0.0
        )
        if overlap < 0.6:
            kept.append(sentence)

    prose = " ".join(kept).strip()
    if not prose or len(_words(prose)) < 12:
        return None
    return GroundedAnswer(answer=prose, claims=kept_claims, insufficient_context=False)


@dataclass
class OutputVerdict:
    passed: bool
    reason: RefusalReason | None = None
    detail: str = ""
    #: Set when unsupported claims were stripped; callers must use this answer.
    repaired: "GroundedAnswer | None" = None
    invalid_citations: list[str] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    safety_flags: list[str] = field(default_factory=list)


# --- 1. citation validity ------------------------------------------------


def check_citations(answer: GroundedAnswer, results: list[Retrieved]) -> list[str]:
    """Cited ids must exist in what was actually retrieved."""
    valid = {r.chunk.chunk_id for r in results} | {
        (r.marker or "") for r in results if r.marker
    }
    bad: list[str] = []
    for claim in answer.claims:
        for cid in claim.chunk_ids:
            if cid not in valid:
                bad.append(cid)
    return sorted(set(bad))


def _cited_text(answer: GroundedAnswer, results: list[Retrieved]) -> str:
    """Text of every passage the answer cites, by id or by marker."""
    wanted: set[str] = set()
    for claim in answer.claims:
        wanted.update(claim.chunk_ids)
    parts = [
        r.chunk.text
        for r in results
        if r.chunk.chunk_id in wanted or (r.marker and r.marker in wanted)
    ]
    # An answer citing nothing is checked against nothing, and check 3 catches it.
    return "\n".join(parts)


# --- 2. numeric provenance ----------------------------------------------


def check_numbers(
    answer: GroundedAnswer, results: list[Retrieved], query: str = ""
) -> list[str]:
    """Every digit in the answer must appear in a cited passage.

    Markers, markdown list numbering, and numbers echoed from the user's own
    question are excluded -- those are not claims about the guidelines.
    """
    body = strip_markers(answer.answer)
    allowed = numbers_in(_cited_text(answer, results)) | numbers_in(query)
    return sorted(n for n in numbers_in(body) if n not in allowed)


# --- 3. claim support (LLM judge) ---------------------------------------


class _Support(BaseModel):
    index: int
    supported: bool = Field(description="does the passage state or directly entail this")
    why: str = ""


class _Supports(BaseModel):
    results: list[_Support]


_SUPPORT_SYSTEM = """You verify citations for a medical guideline assistant.

For each numbered claim you are given the passage(s) it cites. Decide whether
those passages actually state, or directly entail, the claim.

Be strict. Mark supported=false if the claim adds a detail the passage does not
contain, generalises beyond it, or changes a number, population, or condition.
Topical similarity is not support. If the passage is merely about the same
subject, that is false."""


def check_support(answer: GroundedAnswer, results: list[Retrieved]) -> list[str]:
    """Returns the text of claims the judge could not verify."""
    if not answer.claims:
        return []

    by_key: dict[str, Retrieved] = {}
    for r in results:
        by_key[r.chunk.chunk_id] = r
        if r.marker:
            by_key[r.marker] = r

    blocks = []
    for i, claim in enumerate(answer.claims):
        cited = [by_key[c] for c in claim.chunk_ids if c in by_key]
        passages = "\n".join(f"  - {r.chunk.text}" for r in cited) or "  (no passage)"
        blocks.append(f"[{i}] Claim: {claim.text}\nCited passages:\n{passages}")

    try:
        out = parse(_Supports, _SUPPORT_SYSTEM, "\n\n".join(blocks))
    except LLMError:
        note_degraded("support_judge")
        # Fail closed: an unverifiable answer is treated as unsupported.
        return [c.text for c in answer.claims]

    verdicts = {s.index: s.supported for s in out.results}
    return [
        c.text for i, c in enumerate(answer.claims) if not verdicts.get(i, False)
    ]


# --- 4. framing / safety -------------------------------------------------


class _Safety(BaseModel):
    personalized_advice: bool = Field(description="tells a specific person what to do")
    diagnostic: bool = Field(description="asserts what a person has")
    imperative_dosing: bool = Field(description="instructs the reader to take something")
    emergency_instruction: bool = Field(description="gives urgent care instructions")
    note: str = ""


_SAFETY_SYSTEM = """You check whether a drafted answer stays within its boundary.

The assistant may report what official medical guidelines say, in the third
person, attributed to a source. It may discuss drugs, doses, and treatments as
*guideline content*.

It may NOT address the reader as a patient, tell anyone what to do, assert what
someone has, or issue instructions.

The distinction is framing, not subject matter. In each pair the topic is
identical and only the framing differs:

  OK  - "The guidelines list rifampicin at 10 mg/kg daily for adults [C2]."
  NOT - "You should take rifampicin 10 mg/kg daily."

  OK  - "Lower chest indrawing identifies cases requiring urgent referral [C1]."
  NOT - "Take your child to hospital immediately."

  OK  - "The guidelines classify this presentation as severe dengue [C3]."
  NOT - "You have severe dengue."

Urgency, danger signs, and referral criteria are ordinary guideline CONTENT.
Reporting that a guideline calls something an emergency is not itself an
emergency instruction — flag emergency_instruction only when the text directs
the reader to act, not when it describes what a document says.

Flag only genuine boundary crossings. Descriptive, attributed, third-person
prose is not a violation however serious its subject matter."""


def check_framing(answer: GroundedAnswer) -> list[str]:
    if not answer.answer.strip():
        return []
    try:
        out = parse(_Safety, _SAFETY_SYSTEM, answer.answer)
    except LLMError:
        note_degraded("safety_classifier")
        return ["safety classifier unavailable"]
    flags = [
        name
        for name, hit in (
            ("personalized_advice", out.personalized_advice),
            ("diagnostic", out.diagnostic),
            ("imperative_dosing", out.imperative_dosing),
            ("emergency_instruction", out.emergency_instruction),
        )
        if hit
    ]
    return flags


# --- orchestration -------------------------------------------------------


def validate(
    answer: GroundedAnswer, results: list[Retrieved], query: str = ""
) -> OutputVerdict:
    if answer.insufficient_context or not answer.answer.strip():
        return OutputVerdict(
            False, RefusalReason.MODEL_DECLINED, "model reported insufficient context"
        )

    if not answer.claims:
        return OutputVerdict(
            False, RefusalReason.UNGROUNDED_OUTPUT, "answer carried no claims to verify"
        )

    invalid = check_citations(answer, results)
    if invalid:
        return OutputVerdict(
            False, RefusalReason.UNGROUNDED_OUTPUT,
            f"cited passages that were not retrieved: {', '.join(invalid)}",
            invalid_citations=invalid,
        )

    bad_numbers = check_numbers(answer, results, query)
    if bad_numbers:
        return OutputVerdict(
            False, RefusalReason.UNSAFE_OUTPUT,
            f"answer contains numbers absent from cited passages: {', '.join(bad_numbers)}",
            unsupported_numbers=bad_numbers,
        )

    unsupported = check_support(answer, results)
    repaired: GroundedAnswer | None = None
    if unsupported:
        share = len(unsupported) / len(answer.claims)
        repaired = repair(answer, unsupported) if share <= MAX_UNSUPPORTED_FRACTION else None
        if repaired is None:
            return OutputVerdict(
                False, RefusalReason.UNGROUNDED_OUTPUT,
                f"{len(unsupported)}/{len(answer.claims)} claim(s) unsupported; "
                "answer could not be salvaged",
                unsupported_claims=unsupported,
            )

    final = repaired or answer
    flags = check_framing(final)
    if flags:
        return OutputVerdict(
            False, RefusalReason.UNSAFE_OUTPUT,
            f"framing violations: {', '.join(flags)}", safety_flags=flags,
        )

    detail = (
        f"passed after stripping {len(unsupported)} unsupported claim(s)"
        if unsupported else "all output checks passed"
    )
    return OutputVerdict(True, None, detail, repaired=repaired,
                         unsupported_claims=unsupported)

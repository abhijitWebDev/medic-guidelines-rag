from __future__ import annotations

import pytest

from rag_project.guardrails import confidence, output_gate
from rag_project.guardrails.input_gate import rule_screen
from rag_project.guardrails.policy import REFUSALS
from rag_project.models import Chunk, Claim, GroundedAnswer, Intent, RefusalReason, Retrieved


def _retrieved(marker: str, text: str, score: float = 9.0) -> Retrieved:
    return Retrieved(
        chunk=Chunk(
            chunk_id=f"doc::{marker}", doc_id="doc", title="T", source_url="u",
            publisher="MOHFW", section_path="S", page_start=1, page_end=1,
            text=text, n_tokens=10, index_version="v1", ingested_at="now",
        ),
        marker=marker,
        rerank_score=score,
    )


# --- gate 1 ---------------------------------------------------------------

@pytest.mark.parametrize(
    "query,expected",
    [
        ("I have chest pain and cannot breathe", Intent.EMERGENCY),
        ("I want to kill myself", Intent.EMERGENCY),
        ("Should I take rifampicin?", Intent.PERSONALIZED_ADVICE),
        ("How much should I take?", Intent.PERSONALIZED_ADVICE),
        ("my lab results show high sugar", Intent.PERSONALIZED_ADVICE),
    ],
)
def test_rules_catch_unsafe_queries(query, expected):
    v = rule_screen(query)
    assert v is not None and v.intent is expected


@pytest.mark.parametrize(
    "query",
    [
        "What do the guidelines say about chest pain management?",
        "What dose of rifampicin do the guidelines recommend for adults?",
        "How is tuberculosis diagnosed?",
    ],
)
def test_rules_do_not_refuse_legitimate_guideline_questions(query):
    """Impersonal questions about drugs and doses must reach the model, not be
    blocked by keyword. Over-refusal makes the assistant useless."""
    assert rule_screen(query) is None


def test_rules_never_clear_a_query():
    """Rules may only escalate. If they could stamp IN_SCOPE, any phrasing they
    failed to anticipate would bypass the model check entirely."""
    for verdict in (rule_screen(q) for q in ["hello", "what is tb", "should i take x"]):
        assert verdict is None or verdict.intent is not Intent.IN_SCOPE


def test_every_refusal_reason_has_a_message():
    assert all(reason in REFUSALS for reason in RefusalReason)


# --- gate 2 ---------------------------------------------------------------

def test_confidence_refuses_when_all_passages_score_low():
    results = [_retrieved("C1", "x", 2.0), _retrieved("C2", "y", 1.0)]
    v = confidence.evaluate(results, threshold=5.0)
    assert not v.passed and v.kept == []


def test_confidence_refuses_when_reranker_returned_no_scores():
    """Fail closed: an unavailable reranker must not read as high confidence."""
    results = [_retrieved("C1", "x", 9.0)]
    results[0].rerank_score = None
    assert not confidence.evaluate(results, threshold=5.0).passed


def test_confidence_drops_weak_passages_it_does_not_need():
    results = [_retrieved("C1", "x", 9.0), _retrieved("C2", "y", 2.0)]
    v = confidence.evaluate(results, threshold=5.0)
    assert v.passed and [r.marker for r in v.kept] == ["C1"]


# --- gate 3 ---------------------------------------------------------------

def test_invalid_citation_is_caught():
    results = [_retrieved("C1", "text")]
    ans = GroundedAnswer(answer="claim [C9]", claims=[Claim(text="c", chunk_ids=["C9"])])
    assert output_gate.check_citations(ans, results) == ["C9"]


def test_fabricated_dose_is_caught():
    """The dangerous case: a real citation attached to an invented number."""
    results = [_retrieved("C1", "Isoniazid is given daily in the intensive phase.")]
    ans = GroundedAnswer(
        answer="The guidelines list isoniazid at 300 mg daily [C1].",
        claims=[Claim(text="isoniazid 300 mg daily", chunk_ids=["C1"])],
    )
    assert "300" in output_gate.check_numbers(ans, results)


def test_numbers_present_in_the_source_pass():
    results = [_retrieved("C1", "The intensive phase lasts eight weeks; 4 drugs are used.")]
    ans = GroundedAnswer(
        answer="The intensive phase uses 4 drugs [C1].",
        claims=[Claim(text="4 drugs", chunk_ids=["C1"])],
    )
    assert output_gate.check_numbers(ans, results) == []


def test_citation_markers_are_not_mistaken_for_clinical_numbers():
    results = [_retrieved("C1", "Zinc is advised."), _retrieved("C2", "ORS is advised.")]
    ans = GroundedAnswer(
        answer="1. Zinc is advised [C1]. 2. ORS is advised [C2].",
        claims=[Claim(text="zinc", chunk_ids=["C1"]), Claim(text="ors", chunk_ids=["C2"])],
    )
    assert output_gate.check_numbers(ans, results) == []


def test_answer_without_claims_is_rejected():
    results = [_retrieved("C1", "text")]
    ans = GroundedAnswer(answer="Some prose with no claims.", claims=[])
    v = output_gate.validate(ans, results)
    assert not v.passed and v.reason is RefusalReason.UNGROUNDED_OUTPUT


def test_insufficient_context_becomes_a_refusal():
    ans = GroundedAnswer(answer="", claims=[], insufficient_context=True)
    v = output_gate.validate(ans, [_retrieved("C1", "t")])
    assert not v.passed and v.reason is RefusalReason.MODEL_DECLINED


def test_repair_strips_unsupported_claim_and_its_sentence():
    from rag_project.guardrails.output_gate import repair

    ans = GroundedAnswer(
        answer=(
            "The guidelines describe screening at every visit [C1]. "
            "Patients should be educated about lifestyle modifications. "
            "Diagnosis requires readings on two occasions [C2]."
        ),
        claims=[
            Claim(text="screening at every visit", chunk_ids=["C1"]),
            Claim(text="Patients should be educated about lifestyle modifications", chunk_ids=["C1"]),
            Claim(text="diagnosis requires two readings", chunk_ids=["C2"]),
        ],
    )
    out = repair(ans, ["Patients should be educated about lifestyle modifications"])
    assert out is not None
    assert "educated about lifestyle" not in out.answer
    assert "screening at every visit" in out.answer
    assert "two occasions" in out.answer
    assert len(out.claims) == 2


def test_repair_gives_up_when_everything_is_unsupported():
    from rag_project.guardrails.output_gate import repair

    ans = GroundedAnswer(
        answer="One thing [C1]. Another thing [C1].",
        claims=[Claim(text="One thing", chunk_ids=["C1"]), Claim(text="Another thing", chunk_ids=["C1"])],
    )
    assert repair(ans, ["One thing", "Another thing"]) is None


def test_majority_unsupported_refuses_rather_than_salvages():
    results = [_retrieved("C1", "Some passage text about screening.")]
    ans = GroundedAnswer(
        answer="A [C1]. B [C1]. C [C1].",
        claims=[Claim(text=t, chunk_ids=["C1"]) for t in ("A", "B", "C")],
    )
    from rag_project.guardrails.output_gate import MAX_UNSUPPORTED_FRACTION
    assert MAX_UNSUPPORTED_FRACTION < 0.5, "a majority of bad claims must never be salvaged"

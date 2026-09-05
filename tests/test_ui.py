"""UI renderer tests. Pure functions only -- no Streamlit runtime, no network."""

from __future__ import annotations

from rag_project.models import RefusalReason, Response
from rag_project.ui.components import STAGES, _stage_state, highlight_citations


def test_citation_markers_are_highlighted():
    out = highlight_citations("The guidelines state X [C1] and Y [C2, C3].")
    assert out.count("<span") == 2
    assert ">C1<" in out and ">C2, C3<" in out


def test_prose_without_markers_is_untouched():
    text = "No citations here at all."
    assert highlight_citations(text) == text


def test_bracketed_non_citations_are_left_alone():
    text = "See [the guideline] and [Annexure 2] for detail."
    assert "<span" not in highlight_citations(text)


def _refused_trace() -> dict:
    return {
        "stages": ["intent"],
        "intent": {"label": "personalized_advice", "reason": "asks own dose", "rule": r"\bshould i\b"},
        "short_circuited": "no retrieval performed",
    }


def test_gate_one_refusal_marks_downstream_stages_unreached():
    """The point of the panel: you can see retrieval never ran."""
    trace = _refused_trace()
    icons = {label: _stage_state(key, trace)[0] for key, label in STAGES}
    assert icons["Gate 1 · intent"] == "✕"
    assert all(
        icon == "○"
        for label, icon in icons.items()
        if label != "Gate 1 · intent"
    )


def test_rule_match_is_distinguished_from_model_judgement():
    ruled = _stage_state("intent", _refused_trace())[1]
    assert "rule match" in ruled

    trace = {"stages": ["intent"], "intent": {"label": "in_scope", "reason": "asks what guidelines say", "rule": None}}
    assert "model" in _stage_state("intent", trace)[1]


def test_low_confidence_summary_shows_the_numbers_that_decided_it():
    trace = {
        "stages": ["intent", "retrieve", "rerank", "confidence"],
        "confidence": {"passed": False, "top_score": 0.0, "threshold": 5.5,
                       "reason": "best passage scored 0.0 < 5.5", "kept": 0},
    }
    icon, summary = _stage_state("confidence", trace)
    assert icon == "✕"
    assert "0.0" in summary and "5.5" in summary


def test_confidence_summary_survives_a_missing_score():
    trace = {"stages": ["confidence"],
             "confidence": {"passed": False, "top_score": None, "threshold": 5.5,
                            "reason": "reranker produced no scores", "kept": 0}}
    icon, summary = _stage_state("confidence", trace)
    assert icon == "✕" and "—" in summary


def test_stage_never_reached_when_absent_from_stages():
    assert _stage_state("generate", {"stages": []}) == ("○", "not reached")


def test_response_model_round_trips_through_the_ui_contract():
    r = Response(
        query="q", answered=False, answer="refused",
        refusal_reason=RefusalReason.EMERGENCY, trace=_refused_trace(),
    )
    assert r.refusal_reason.value == "emergency"
    assert r.trace["stages"] == ["intent"]

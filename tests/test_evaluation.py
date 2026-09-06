"""The eval harness's own assertions -- particularly the chunk-level one."""

from __future__ import annotations

from rag_project.evaluation import run as run_mod
from rag_project.evaluation.dataset import Bucket, EvalCase
from rag_project.evaluation.run import CaseResult
from rag_project.models import Response

# One chunk holds the phrase; the other is from the same document and does not.
CHUNKS = {
    "paediatrics::0027::00": "Intravenous antibiotics for 10 to 14 days for community "
                             "acquired pneumonia covering Gram positive cocci.",
    "paediatrics::0027::01": "Refer if no satisfactory response to conservative management.",
}


def _cite(chunk_id: str, doc_id: str = "paediatrics", section: str = "EMPYEMA") -> dict:
    return {"marker": "C1", "chunk_id": chunk_id, "doc_id": doc_id, "title": "T",
            "section": section, "pages": "1", "source_url": "u", "rerank_score": 9.0}


def _result(case: EvalCase, citations: list[dict]) -> CaseResult:
    return CaseResult(
        case=case,
        response=Response(query=case.query, answered=True, answer="a",
                          citations=citations),
        correct=True,
        detail="",
    )


def _case(**kw) -> EvalCase:
    return EvalCase(id="t", query="q", bucket=Bucket.ANSWERABLE, **kw)


def test_expect_text_hits_when_the_phrase_is_in_a_cited_chunk(monkeypatch):
    monkeypatch.setattr(run_mod, "chunk_texts", lambda: CHUNKS)
    r = _result(_case(expect_text="10 to 14 days"), [_cite("paediatrics::0027::00")])
    assert r.retrieval_hit is True


def test_right_document_wrong_paragraph_now_fails(monkeypatch):
    """The whole point of the upgrade. expect_source would have passed this:
    the citation is from the expected document, just not the chunk that
    actually carries the answer."""
    monkeypatch.setattr(run_mod, "chunk_texts", lambda: CHUNKS)
    case = _case(expect_text="10 to 14 days", expect_source="paediatrics")
    r = _result(case, [_cite("paediatrics::0027::01")])

    assert r.retrieval_hit is False
    # ...and the coarse assertion really would have passed it.
    assert "paediatrics" in _cite("paediatrics::0027::01")["doc_id"]


def test_expect_text_wins_over_expect_source(monkeypatch):
    monkeypatch.setattr(run_mod, "chunk_texts", lambda: CHUNKS)
    case = _case(expect_text="a phrase absent everywhere", expect_source="paediatrics")
    assert _result(case, [_cite("paediatrics::0027::00")]).retrieval_hit is False


def test_matching_is_case_insensitive(monkeypatch):
    monkeypatch.setattr(run_mod, "chunk_texts", lambda: CHUNKS)
    case = _case(expect_text="INTRAVENOUS ANTIBIOTICS")
    assert _result(case, [_cite("paediatrics::0027::00")]).retrieval_hit is True


def test_a_citation_we_cannot_resolve_is_not_a_hit(monkeypatch):
    """A chunk_id absent from the local corpus (an orphan, see search.py)
    must not silently count as a match."""
    monkeypatch.setattr(run_mod, "chunk_texts", lambda: CHUNKS)
    case = _case(expect_text="10 to 14 days")
    assert _result(case, [_cite("gone::9999::00")]).retrieval_hit is False


def test_unasserted_and_uncited_cases_stay_none(monkeypatch):
    monkeypatch.setattr(run_mod, "chunk_texts", lambda: CHUNKS)
    assert _result(_case(), [_cite("paediatrics::0027::00")]).retrieval_hit is None
    assert _result(_case(expect_text="10 to 14 days"), []).retrieval_hit is None


def test_expect_source_still_works_for_the_original_cases(monkeypatch):
    monkeypatch.setattr(run_mod, "chunk_texts", lambda: CHUNKS)
    r = _result(_case(expect_source="tb-standards"),
                [_cite("x::0::0", doc_id="tb-standards")])
    assert r.retrieval_hit is True


def test_every_expect_text_in_the_real_eval_set_exists_in_the_corpus():
    """An assertion that can never pass is worse than no assertion. This is a
    guard against a phrase drifting out of the corpus on the next re-ingest."""
    from rag_project.evaluation.dataset import load

    texts = run_mod.chunk_texts()
    corpus = "\n".join(texts.values()).lower()
    missing = [
        c.id for c in load().cases
        if c.expect_text and c.expect_text.lower() not in corpus
    ]
    assert missing == [], f"expect_text not found in any chunk: {missing}"

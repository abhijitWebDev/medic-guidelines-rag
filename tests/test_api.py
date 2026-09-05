"""API + served-UI tests. No network: every case either short-circuits at gate 1
or uses a stubbed assistant."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rag_project import api as api_mod
from rag_project.api import app
from rag_project.models import Claim, RefusalReason, Response


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_root_serves_the_ui(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "Medical Guideline Assistant" in r.text


def test_ui_is_self_contained():
    """No external scripts, styles or fonts: the page must work on a locked-down
    network and must not leak a query's existence to a CDN. A data: URI is not
    external, so the inline favicon is fine -- what must not appear is a fetch
    off-origin."""
    html = (api_mod.STATIC / "index.html").read_text()
    assert "<script src=" not in html
    assert 'href="http' not in html.split("<body")[0]
    assert "@import" not in html
    for bad in ("cdn.", "googleapis", "gstatic", "unpkg", "jsdelivr"):
        assert bad not in html


def test_info_reports_models_and_threshold(client):
    body = client.get("/api/info").json()
    for key in ("table", "embedding_model", "generation_model", "guard_model",
                "confidence_threshold", "index_version"):
        assert key in body, f"/api/info is missing {key}, which the UI renders"


def test_refusal_is_200_not_an_error(client):
    """A refusal is a correct outcome, not a fault -- encoding it as 4xx would
    push callers toward retrying around it."""
    r = client.post("/api/ask", json={"query": "Should I take rifampicin for my cough?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answered"] is False
    assert body["refusal_reason"] == RefusalReason.PERSONALIZED_ADVICE.value


def test_trace_is_withheld_unless_requested(client):
    q = {"query": "Should I take rifampicin for my cough?"}
    assert client.post("/api/ask", json=q).json()["trace"] == {}
    assert client.post("/api/ask", json={**q, "trace": True}).json()["trace"]["stages"]


def test_empty_query_is_rejected(client):
    assert client.post("/api/ask", json={"query": ""}).status_code == 422


def test_overlong_query_is_rejected(client):
    assert client.post("/api/ask", json={"query": "x" * 2001}).status_code == 422


def test_answer_shape_matches_what_the_ui_renders(client, monkeypatch):
    """The frontend reads marker/title/section/pages/rerank_score off each
    citation; a rename here breaks the Sources panel silently."""

    class Stub:
        def ask(self, query, **kw):
            return Response(
                query=query, answered=True, answer="Two months [C1].",
                claims=[Claim(text="two months", chunk_ids=["c::1"])],
                citations=[{"marker": "C1", "chunk_id": "c::1", "doc_id": "stg-tb",
                            "title": "TB", "section": "Treatment", "pages": "12",
                            "source_url": "https://x", "rerank_score": 9.0}],
                top_score=9.0, disclaimer="Not medical advice.",
                trace={"stages": ["intent", "retrieve"]},
            )

    monkeypatch.setattr(api_mod, "_assistant", lambda: Stub())
    body = client.post("/api/ask", json={"query": "TB duration?"}).json()
    assert body["answered"] and body["disclaimer"]
    for key in ("marker", "title", "section", "pages", "rerank_score", "source_url"):
        assert key in body["citations"][0]


def test_unreachable_index_refuses_rather_than_500(client, monkeypatch):
    from rag_project.retrieval.search import RetrievalError

    class Broken:
        def ask(self, query, **kw):
            raise RetrievalError("index unreachable")

    monkeypatch.setattr(api_mod, "_assistant", lambda: Broken())
    r = client.post("/api/ask", json={"query": "How is tuberculosis diagnosed?"})
    assert r.status_code == 200
    assert r.json()["refusal_reason"] == RefusalReason.LOW_CONFIDENCE.value

from __future__ import annotations

from rag_project.indexing.bm25 import BM25Index, reciprocal_rank_fusion, tokenize
from rag_project.indexing.build import to_record
from rag_project.indexing.store import InMemoryStore
from rag_project.ingest.chunk import chunk_document
from rag_project.ingest.parse import parse_pdf


def _chunks(stg_pdf, stg_doc):
    return chunk_document(parse_pdf(stg_pdf, "stg-tb", stg_doc.title), stg_doc)


def test_tokenizer_preserves_clinical_terms():
    """Dense retrieval is weakest exactly here, so the lexical half must not
    shred hyphenated drug names or alphanumeric vitamin codes."""
    toks = tokenize("Co-trimoxazole and B12 with NAAT-based testing")
    assert "co-trimoxazole" in toks
    assert "b12" in toks
    assert "naat-based" in toks


def test_bm25_finds_exact_rare_term(stg_pdf, stg_doc):
    chunks = _chunks(stg_pdf, stg_doc)
    hits = BM25Index(chunks).search("rifampicin", limit=3)
    assert hits, "exact drug name should retrieve lexically"
    assert "Treatment" in hits[0][0].section_path


def test_rrf_rewards_agreement_between_rankings():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a", "d"]])
    # b and d each appear in one list only; a and c appear in both.
    assert fused["a"] > fused["b"]
    assert fused["c"] > fused["d"]


def test_records_have_no_nulls_in_schema_defining_columns(stg_pdf, stg_doc):
    """Regression guard: the remote wrapper infers schema from the first batch,
    and an all-null column infers as null type and rejects later strings."""
    chunks = _chunks(stg_pdf, stg_doc)
    for c in chunks:
        c.specialty = None
        c.doc_version = None
        rec = to_record(c, [0.0] * 4)
        assert rec["specialty"] == ""
        assert rec["doc_version"] == ""


def test_store_contract_roundtrip(stg_pdf, stg_doc):
    chunks = _chunks(stg_pdf, stg_doc)
    store = InMemoryStore()
    recs = [to_record(c, [float(i), 1.0, 0.0]) for i, c in enumerate(chunks)]
    store.create(recs[:1])
    store.insert(recs[1:])
    assert store.count() == len(chunks)
    hits = store.search([0.0, 1.0, 0.0], limit=2)
    assert len(hits) == 2
    assert "vector" not in hits[0], "search results must not echo raw vectors"
    assert "_distance" in hits[0]

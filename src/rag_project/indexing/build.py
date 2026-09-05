"""chunks.jsonl -> embeddings -> remote table, plus an index manifest.

The manifest is the governance artifact: it records exactly which documents,
which chunk parameters, and which embedding model produced the table now being
queried. Without it, "which version of the guidelines did this answer come
from?" has no auditable answer -- which for a medical assistant is the whole
ballgame.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ..config import get_settings
from ..corpus.manifest import load_manifest
from ..indexing.embed import Embedder
from ..indexing.store import VectorStore, get_store
from ..ingest.chunk import embed_text
from ..ingest.pipeline import load_chunks
from ..models import Chunk

# Columns whose None must become "" -- the wrapper infers the table schema from
# the first batch, and an all-null column infers as null type and then rejects
# every later string written into it.
_NULLABLE_STR = ("specialty", "doc_version", "source_url")


@dataclass
class BuildReport:
    table: str
    chunks: int
    inserted: int
    embed_dim: int
    created: bool
    cached_vectors: int


def to_record(chunk: Chunk, vector: list[float]) -> dict[str, Any]:
    rec = chunk.model_dump()
    for field in _NULLABLE_STR:
        if rec.get(field) is None:
            rec[field] = ""
    rec["vector"] = vector
    return rec


def run(recreate: bool = False, store: VectorStore | None = None) -> BuildReport:
    settings = get_settings()
    chunks = load_chunks()
    if not chunks:
        raise RuntimeError(
            "No chunks found. Run `uv run rag ingest` first."
        )

    embedder = Embedder()
    texts = [embed_text(c.title, c.section_path, c.text) for c in chunks]
    vectors = embedder.embed(texts, progress=True)

    store = store or get_store()
    created = False
    if recreate and store.exists():
        store.drop()
    if not store.exists():
        created = True

    records = [to_record(c, v.tolist()) for c, v in zip(chunks, vectors, strict=True)]

    if created:
        # First batch defines the schema, so it must carry every column.
        store.create(records[:1])
        inserted = 1 + store.insert(records[1:], progress=True)
    else:
        inserted = store.insert(records, progress=True)

    write_index_manifest(chunks, embedder, inserted)
    return BuildReport(
        table=settings.table,
        chunks=len(chunks),
        inserted=inserted,
        embed_dim=embedder.dim,
        created=created,
        cached_vectors=embedder.cached_count,
    )


def write_index_manifest(chunks: list[Chunk], embedder: Embedder, inserted: int) -> None:
    settings = get_settings()
    # Only documents that actually contributed chunks are "in the index". A
    # corpus entry that was skipped (no text layer) must not be counted as
    # indexed -- reporting 21 documents when 20 are searchable is exactly the
    # kind of quiet inaccuracy the manifest exists to prevent.
    indexed_ids = {c.doc_id for c in chunks}
    try:
        docs = load_manifest()
        doc_rows = [
            {"doc_id": d.doc_id, "title": d.title, "sha256": d.sha256,
             "url": d.url, "version": d.version}
            for d in docs
            if d.doc_id in indexed_ids
        ]
        skipped_rows = [
            {"doc_id": d.doc_id, "title": d.title, "filename": d.filename}
            for d in docs
            if d.doc_id not in indexed_ids
        ]
    except FileNotFoundError:
        doc_rows, skipped_rows = [], []

    payload = {
        "index_version": settings.index_version,
        "table": settings.table,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "embedding": {"model": embedder.model, "dim": embedder.dim},
        "chunking": {
            "target_tokens": settings.chunk_target_tokens,
            "overlap_tokens": settings.chunk_overlap_tokens,
            "drop_below_tokens": settings.chunk_drop_below_tokens,
        },
        "retrieval": {
            "hybrid": "dense (remote) + BM25 (client-side), fused with RRF",
            "note": "the deployed LanceDB wrapper exposes vector search only",
        },
        "counts": {
            "chunks": len(chunks),
            "inserted": inserted,
            "documents": len(doc_rows),
            "skipped_documents": len(skipped_rows),
        },
        "documents": doc_rows,
        "skipped_documents": skipped_rows,
    }
    settings.index_manifest_path.write_text(json.dumps(payload, indent=2))

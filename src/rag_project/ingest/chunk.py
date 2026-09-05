"""Sections -> token-bounded chunks that never straddle a section boundary.

Two decisions worth naming:

1. A chunk never spans two sections. Slightly smaller chunks are a cheap price
   for never producing a citation that points at the wrong section heading --
   which, in a medical corpus, is the difference between a correct citation and
   a dangerous one.

2. What gets embedded is not what gets displayed. The embedded text is prefixed
   with the document title and section path, so a chunk reading "two months of
   isoniazid..." still matches the query "TB treatment intensive phase" even
   though neither "TB" nor "treatment" appears in the body. The stored `text`
   stays clean for display and citation checking.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

import tiktoken

from ..config import get_settings
from ..models import Chunk, SourceDoc
from .parse import ParsedDoc, Section

_ENC = tiktoken.get_encoding("cl100k_base")

_SENT_SPLIT = re.compile(r"(?<=[.!?:])\s+(?=[A-Z(])|\n+")


def n_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def embed_text(title: str, section_path: str, body: str) -> str:
    """Contextual prefix -- see module docstring, decision 2."""
    return f"{title} — {section_path}\n\n{body}"


def _sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p and p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _pack(sentences: list[str], target: int, overlap: int) -> list[str]:
    """Greedy pack to `target` tokens, carrying `overlap` tokens of tail."""
    chunks: list[str] = []
    cur: list[str] = []
    cur_tok = 0

    for sent in sentences:
        st = n_tokens(sent)
        # A single oversized sentence becomes its own chunk rather than being
        # silently truncated.
        if st > target and not cur:
            chunks.append(sent)
            continue
        if cur_tok + st > target and cur:
            chunks.append(" ".join(cur))
            tail: list[str] = []
            tail_tok = 0
            for prev in reversed(cur):
                pt = n_tokens(prev)
                if tail_tok + pt > overlap:
                    break
                tail.insert(0, prev)
                tail_tok += pt
            cur, cur_tok = tail, tail_tok
        cur.append(sent)
        cur_tok += st

    if cur:
        chunks.append(" ".join(cur))
    return chunks


def chunk_section(
    section: Section, doc: SourceDoc, ingested_at: str, seq: int
) -> list[Chunk]:
    settings = get_settings()
    pieces = _pack(
        _sentences(section.text),
        settings.chunk_target_tokens,
        settings.chunk_overlap_tokens,
    )

    # Merge a runt tail into its predecessor rather than indexing a fragment.
    if len(pieces) > 1 and n_tokens(pieces[-1]) < settings.chunk_min_tokens:
        # Pop first. Writing `pieces[-2] = ... pieces.pop()` resolves the target
        # index against the already-shortened list, which is out of range when
        # there are exactly two pieces.
        tail = pieces.pop()
        pieces[-1] = f"{pieces[-1]} {tail}"

    out: list[Chunk] = []
    for i, body in enumerate(pieces):
        if n_tokens(body) < settings.chunk_drop_below_tokens:
            continue  # stray heading echo, not content
        out.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::{seq:04d}::{i:02d}",
                doc_id=doc.doc_id,
                title=doc.title,
                source_url=doc.url,
                publisher=doc.publisher,
                specialty=doc.specialty,
                section_path=section.path,
                page_start=section.page_start,
                page_end=section.page_end,
                text=body,
                n_tokens=n_tokens(body),
                doc_version=doc.version,
                index_version=settings.index_version,
                ingested_at=ingested_at,
            )
        )
    return out


def chunk_document(parsed: ParsedDoc, doc: SourceDoc) -> list[Chunk]:
    ingested_at = datetime.now(UTC).isoformat(timespec="seconds")
    chunks: list[Chunk] = []
    for seq, section in enumerate(parsed.sections):
        chunks.extend(chunk_section(section, doc, ingested_at, seq))
    return chunks

"""Manifest -> parsed sections -> chunks on disk.

Chunks land in data/chunks/*.jsonl *before* anything is embedded. That split is
deliberate: chunking is the parameter you will tune most, it is the cheapest
stage to inspect, and a bad chunk boundary is far easier to see as text than as
a vector. Embedding a corpus you have not eyeballed is how silent quality loss
gets baked into an index.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..config import get_settings
from ..corpus.manifest import load_manifest, verify
from ..models import Chunk
from .chunk import chunk_document
from .parse import parse_pdf
from .text import parse_text


class CorpusNotClean(RuntimeError):
    """Raised when data/raw/ disagrees with the manifest."""


@dataclass
class IngestReport:
    documents: int
    sections: int
    chunks: int
    tokens: int
    per_doc: dict[str, int]
    #: doc_id -> why it produced nothing. Reported loudly: a document that
    #: silently contributes no chunks is a corpus gap nobody notices.
    skipped: dict[str, str] = field(default_factory=dict)

    @property
    def mean_tokens(self) -> float:
        return self.tokens / self.chunks if self.chunks else 0.0


def run(strict: bool = True) -> IngestReport:
    settings = get_settings()
    diff = verify()

    if not diff.clean and strict:
        problems = []
        if diff.unlisted:
            problems.append(
                f"{len(diff.unlisted)} PDF(s) in data/raw/ are not in the manifest "
                f"and will NOT be ingested: {', '.join(diff.unlisted)}"
            )
        if diff.changed:
            problems.append(
                f"{len(diff.changed)} PDF(s) no longer match their recorded "
                f"sha256: {', '.join(diff.changed)}"
            )
        if diff.missing:
            problems.append(
                f"{len(diff.missing)} manifest entr(ies) have no file on disk: "
                f"{', '.join(diff.missing)}"
            )
        raise CorpusNotClean(
            "\n".join(problems)
            + "\n\nRe-run `uv run rag corpus scan` to regenerate the manifest, "
            "or pass --no-strict to ingest only the entries that do match."
        )

    settings.chunks_dir.mkdir(parents=True, exist_ok=True)
    ok = set(diff.listed_ok)

    n_sections = n_chunks = n_tokens = 0
    per_doc: dict[str, int] = {}
    skipped: dict[str, str] = {}

    for doc in load_manifest():
        if doc.filename not in ok:
            continue
        source = settings.raw_dir / doc.filename
        parse = parse_text if source.suffix.lower() == ".txt" else parse_pdf
        parsed = parse(source, doc.doc_id, doc.title)

        if not parsed.has_text_layer:
            skipped[doc.doc_id] = (
                f"no extractable text layer ({parsed.chars_per_page:.0f} chars/page "
                f"over {parsed.n_pages} pages) - scanned images, needs OCR"
            )
            continue

        chunks: list[Chunk] = chunk_document(parsed, doc)
        if not chunks:
            skipped[doc.doc_id] = "parsed but produced no chunks"
            continue

        out = settings.chunks_dir / f"{doc.doc_id}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c.model_dump(), ensure_ascii=False) + "\n")

        n_sections += len(parsed.sections)
        n_chunks += len(chunks)
        n_tokens += sum(c.n_tokens for c in chunks)
        per_doc[doc.doc_id] = len(chunks)

    return IngestReport(len(per_doc), n_sections, n_chunks, n_tokens, per_doc, skipped)


def load_chunks() -> list[Chunk]:
    """Read back what `run()` wrote -- the input to the indexing stage."""
    settings = get_settings()
    chunks: list[Chunk] = []
    for path in sorted(settings.chunks_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            chunks.extend(Chunk(**json.loads(line)) for line in f if line.strip())
    return chunks

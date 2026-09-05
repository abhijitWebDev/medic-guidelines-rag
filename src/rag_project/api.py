"""HTTP API, and the web UI it serves.

The response shape is identical to the CLI's, because both return the same
`Response` object. A refusal is a 200 with `answered: false` and a
`refusal_reason` -- not an HTTP error. Refusing is a correct, expected outcome
of this system, and encoding it as a 4xx would push callers toward treating it
as a fault to be retried or worked around.

The browser UI is a single static page under web/static, served at `/` and
talking to the same `/api/*` endpoints any other client would use. It is
deliberately not a template: nothing about the page depends on server-side
state, so there is no reason for the server to render it, and keeping it static
means the whole app is `fastapi` plus stdlib at runtime.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .assistant import Assistant
from .config import get_settings
from .indexing.store import StoreError
from .models import Response as AnswerResponse
from .retrieval.search import RetrievalError

STATIC = Path(__file__).parent / "web" / "static"

app = FastAPI(
    title="Medical Guideline Assistant",
    description=(
        "Answers strictly from official government health guidelines, with "
        "citations. Refuses personalized medical advice, emergencies, and "
        "questions the corpus does not cover."
    ),
    version="0.1.0",
)

api = APIRouter(prefix="/api", tags=["assistant"])


@lru_cache
def _assistant() -> Assistant:
    """Built once: it loads the chunk corpus and the BM25 index."""
    return Assistant.build()


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    trace: bool = Field(default=False, description="include the per-gate trace")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@api.get("/info")
def info() -> dict:
    s = get_settings()
    payload: dict = {
        "table": s.table,
        "index_version": s.index_version,
        "embedding_model": s.openai_embed_model,
        "generation_model": s.openai_model,
        "guard_model": s.openai_guard_model,
        "confidence_threshold": s.confidence_threshold,
    }
    if s.index_manifest_path.exists():
        manifest = json.loads(s.index_manifest_path.read_text())
        payload["built_at"] = manifest.get("built_at")
        payload["counts"] = manifest.get("counts")
        payload["documents"] = [
            {"doc_id": d["doc_id"], "title": d["title"], "url": d.get("url")}
            for d in manifest.get("documents", [])
        ]
        # Shown in the UI: a document in the corpus that produced no chunks
        # answers nothing, and a user who cannot see that reads a refusal as a
        # failure of the assistant rather than a gap in the index.
        payload["skipped_documents"] = [
            {"doc_id": d["doc_id"], "title": d.get("title"),
             "filename": d.get("filename")}
            for d in manifest.get("skipped_documents", [])
        ]
    return payload


@api.post("/ask", response_model=AnswerResponse)
def ask(req: AskRequest) -> AnswerResponse:
    try:
        response = _assistant().ask(req.query)
    except (RetrievalError, StoreError) as e:
        # The index is missing or unreachable. Refuse rather than 500: from the
        # caller's side the correct behaviour is still "no grounded answer".
        from .guardrails.policy import DISCLAIMER, refusal_text
        from .models import RefusalReason

        return AnswerResponse(
            query=req.query,
            answered=False,
            answer=refusal_text(RefusalReason.LOW_CONFIDENCE),
            refusal_reason=RefusalReason.LOW_CONFIDENCE,
            disclaimer=DISCLAIMER,
            trace={"error": str(e)},
        )

    if not req.trace:
        response.trace = {}
    return response


app.include_router(api)


def main() -> None:
    import os

    import uvicorn

    # Bind from the environment so a container can be reached from outside it;
    # localhost stays the default so running it bare does not expose the box.
    uvicorn.run(
        "rag_project.api:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )

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

from fastapi import APIRouter, Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from pydantic import BaseModel, Field

from . import security
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


def _authed(request: Request) -> bool:
    return security.request_is_authenticated(request.cookies.get(security.COOKIE_NAME))


def require_auth(request: Request) -> None:
    """Fails closed: a configured password with no valid session means 401."""
    if not _authed(request):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "authentication required")


def _rate_limit(request: Request, scope: str) -> None:
    ip = security.client_ip(request.headers, request.client.host if request.client else None)
    verdict = security.check_rate_limit(ip, scope)
    if not verdict.allowed:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "rate limit reached; each question costs real model calls",
            headers={"Retry-After": str(verdict.retry_after_s)},
        )


def enforce_rate_limit(request: Request) -> None:
    """Meters questions, which cost model calls."""
    _rate_limit(request, "ask")


# Deliberately outside the gate: uptime checks must not need a password, and it
# reveals nothing but liveness.
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def index(request: Request) -> Response:
    if not _authed(request):
        return _login_page()
    return FileResponse(STATIC / "index.html")


def _login_page(error: str = "", code: int = 200) -> HTMLResponse:
    html = (STATIC / "login.html").read_text()
    if error:
        html = html.replace("<!--ERROR-->", f'<div class="error">{error}</div>')
    return HTMLResponse(html, status_code=code)


@app.post("/login", include_in_schema=False)
def login(request: Request, password: str = Form("")) -> Response:
    # A brute-forcer costs nothing to run, so login is metered too -- but on
    # its own counter, so failed logins never consume the question quota.
    _rate_limit(request, "login")
    if not security.password_matches(password):
        return _login_page("Incorrect password.", code=401)

    # Secure only where the connection actually is HTTPS. Hard-coding it would
    # make the cookie silently undeliverable on http://localhost -- the login
    # would appear to succeed and then bounce straight back to the form.
    https = (
        request.url.scheme == "https"
        or request.headers.get("x-forwarded-proto", "").split(",")[0].strip() == "https"
    )
    resp = RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)
    resp.set_cookie(
        security.COOKIE_NAME,
        security.issue_token(),
        max_age=get_settings().session_ttl_s,
        httponly=True,    # unreadable from JavaScript
        secure=https,
        samesite="lax",   # not sent on cross-site POSTs
    )
    return resp


@app.post("/logout", include_in_schema=False)
def logout() -> Response:
    resp = RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)
    resp.delete_cookie(security.COOKIE_NAME)
    return resp


@api.get("/info", dependencies=[Depends(require_auth)])
def info() -> dict:
    s = get_settings()
    payload: dict = {
        "table": s.table,
        "index_version": s.index_version,
        "embedding_model": s.openai_embed_model,
        "generation_model": s.openai_model,
        "guard_model": s.openai_guard_model,
        "confidence_threshold": s.confidence_threshold,
        "auth_enabled": security.auth_required(),
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


@api.post(
    "/ask",
    response_model=AnswerResponse,
    dependencies=[Depends(require_auth), Depends(enforce_rate_limit)],
)
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

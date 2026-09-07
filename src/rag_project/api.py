"""HTTP API, and the web UI it serves.

The response shape is identical to the CLI's, because both return the same
`Response` object. A refusal is a 200 with `answered: false` and a
`refusal_reason` -- not an HTTP error. Refusing is a correct, expected outcome
of this system, and encoding it as a 4xx would push callers toward treating it
as a fault to be retried or worked around.

Access is per account. A visitor signs up or signs in, gets a signed cookie
naming them, and every question they ask is written to their own history in
Postgres. Two things follow from that split, and they are the reason the stores
are separate:

* The **account store fails closed.** A question that cannot be attributed to
  a user is a question we decline to answer (503), never one we answer
  anonymously.
* The **answer cache stays shared.** Its keys are a pipeline fingerprint plus
  a hash of the question, and the corpus is the same for everyone, so two
  users asking the same thing should not both pay for the model. Nothing
  user-authored is stored in it and nothing user-specific comes out.

The browser UI is a static page under web/static, served at `/` and talking to
the same `/api/*` endpoints any other client would use. The sign-in page is the
one exception to "static": it is read from disk and has an error message and a
mode marker substituted into it, so that authentication works with JavaScript
switched off.
"""

from __future__ import annotations

import html
import json
import sys
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from pydantic import BaseModel, Field

from . import mailer, security
from .assistant import Assistant
from .config import get_settings
from .db import AccountError, StorageError, User, history, users
from .db.models import to_epoch
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
    version="0.2.0",
)

api = APIRouter(prefix="/api", tags=["assistant"])


@lru_cache
def _assistant() -> Assistant:
    """Built once: it loads the chunk corpus and the BM25 index."""
    return Assistant.build()


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    trace: bool = Field(default=False, description="include the per-gate trace")


# --- who is asking -------------------------------------------------------


def current_user(request: Request) -> User | None:
    """The signed-in user, or None. FastAPI caches this per request.

    Returns None rather than raising when there is no session: several callers
    treat "nobody" as a legitimate answer (an open instance, the health check,
    an anonymous rate-limit subject). The routes that need a user say so.
    """
    if not security.auth_required():
        return None
    session = security.read_token(request.cookies.get(security.COOKIE_NAME))
    if session is None:
        return None
    try:
        user = users.get(session.user_id)
    except StorageError as e:
        # Fails closed, unlike every cache in this project. We cannot tell
        # whether this cookie names a real account, so we decline to guess.
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "the account store is unavailable"
        ) from e
    if user is None:
        return None
    if _predates_password_change(session, user):
        return None
    return user


def _predates_password_change(session: security.Session, user: User) -> bool:
    """Whether this session was issued before the password last changed.

    This is what makes a reset sign the account out everywhere, and it needs no
    revocation list: the cookie says when it was issued, the row says when the
    password moved, and anything older than the change is simply not honoured.

    Both sides are milliseconds. `set_password` stamps the row and then issues
    a fresh session microseconds later, so at coarser resolution the reset
    would reject the very session it just created.
    """
    changed = to_epoch(user.password_changed_at)
    return changed is not None and session.issued_at_ms < int(changed * 1000)


def require_auth(user: User | None = Depends(current_user)) -> None:
    """Fails closed: accounts enabled with no valid session means 401."""
    if security.auth_required() and user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "authentication required")


def require_user(user: User | None = Depends(current_user)) -> User:
    """For routes that are meaningless without an identity, like history."""
    if user is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "sign in to use per-account history"
        )
    return user


def _quota_headers(verdict: security.RateVerdict) -> dict[str, str]:
    """The `X-RateLimit-*` convention, so a client can see the wall coming.

    Worth the three headers: the quota is small enough that people reach it in
    normal use, and a limit you only learn about by hitting it reads as the app
    breaking rather than as a budget.
    """
    return {
        "X-RateLimit-Limit": str(get_settings().rate_limit_per_window),
        "X-RateLimit-Remaining": str(max(verdict.remaining, 0)),
        "X-RateLimit-Reset": str(verdict.retry_after_s),
    }


def require_verified(user: User | None = Depends(current_user)) -> None:
    """Asking costs money, so it is the thing verification actually gates.

    403 rather than 401: the session is fine and signing in again will not
    help. Everything else stays open -- history, sign-out, the corpus panel --
    because an account nobody can look at is harder to finish setting up, not
    safer.
    """
    if not get_settings().verification_required or user is None or user.is_verified:
        return
    raise HTTPException(
        status.HTTP_403_FORBIDDEN,
        "Confirm your email address to start asking questions. "
        "Check your inbox for the link, or request a new one.",
    )


def _rate_limit(subject: str, scope: str) -> security.RateVerdict:
    verdict = security.check_rate_limit(subject, scope)
    if not verdict.allowed:
        minutes = max(1, round(verdict.retry_after_s / 60))
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            # Says what the limit is and when it lifts. "Rate limit reached" on
            # its own leaves someone refreshing, with no way to tell a quota
            # from an outage.
            f"You have used all {get_settings().rate_limit_per_window} questions "
            f"for this hour. Each one costs real model calls. Try again in "
            f"{minutes} minute{'s' if minutes != 1 else ''}.",
            headers={"Retry-After": str(verdict.retry_after_s), **_quota_headers(verdict)},
        )
    return verdict


def _ip(request: Request) -> str:
    return security.client_ip(
        request.headers, request.client.host if request.client else None
    )


def enforce_rate_limit(
    request: Request,
    response: Response,
    user: User | None = Depends(current_user),
) -> None:
    """Meters questions, which cost model calls.

    Per account where there is one. An IP quota is the wrong unit once people
    sign in: a clinic behind one NAT would share a single allowance.

    The verdict is reported on every answer, not only on the one that gets
    refused -- `response` here is the real outgoing response, so the headers
    set on it survive to the client.
    """
    subject = f"user:{user.id}" if user else f"ip:{_ip(request)}"
    verdict = _rate_limit(subject, "ask")
    response.headers.update(_quota_headers(verdict))


# Deliberately outside the gate: uptime checks must not need an account, and it
# reveals nothing but liveness.
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# --- sign in / sign up ---------------------------------------------------

_STORE_DOWN = "The account store is unavailable. Try again in a moment."


def _accounts_or_404() -> None:
    """Sign-in and sign-up do not exist on an instance without accounts.

    Without this, /signup on an open instance still writes a user row and hands
    back a cookie that nothing then reads -- an anonymous visitor able to insert
    rows into a database the operator did not ask to use.
    """
    if not security.auth_required():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "this instance has no accounts")


@app.get("/", include_in_schema=False)
def index(user: User | None = Depends(current_user)) -> Response:
    if security.auth_required() and user is None:
        return _auth_page()
    return FileResponse(STATIC / "index.html")


def _auth_page(
    mode: str = "signin",
    error: str = "",
    notice: str = "",
    email: str = "",
    token: str = "",
    code: int = 200,
) -> HTMLResponse:
    """The sign-in / sign-up / reset page, with the server's half filled in.

    Substitution rather than a template engine: there are a handful of holes,
    and a template dependency to fill a handful of holes is a dependency to
    keep patched forever. Everything interpolated is escaped -- `email` and
    `token` are whatever arrived in the request, and both go back into
    attributes.
    """
    page = (STATIC / "login.html").read_text()
    if error:
        page = page.replace("<!--ERROR-->", f'<div class="error">{html.escape(error)}</div>')
    if notice:
        page = page.replace("<!--NOTICE-->", f'<div class="notice">{html.escape(notice)}</div>')
    page = page.replace('value="<!--EMAIL-->"', f'value="{html.escape(email)}"')
    page = page.replace('value="<!--TOKEN-->"', f'value="{html.escape(token)}"')
    # One replacement, anchored on a string that appears nowhere in the
    # stylesheet. Both hazards here are real and were both hit:
    #
    #   * Replacing a bare `data-mode="signin"` also rewrites the CSS rules,
    #     which select on that attribute -- and the rule hiding the sign-in
    #     panel stops matching, so two forms render at once.
    #   * Using a placeholder like <!--MODE--> instead makes the *unsubstituted*
    #     file match no panel rule, so anything serving the page without this
    #     substitution renders a blank card. The defaults in the file are real,
    #     working values precisely so that failure mode does not exist.
    mail = "on" if get_settings().mail_enabled else "off"
    page = page.replace(
        'class="card" data-mode="signin" data-mail="on"',
        f'class="card" data-mode="{html.escape(mode)}" data-mail="{mail}"',
    )
    if not get_settings().database_url:
        # An account stored on an ephemeral filesystem disappears at the next
        # cold start. Saying so on the page is the only warning the person
        # signing up will ever get.
        page = page.replace(
            "<!--NOTICE-->",
            '<div class="notice">No <code>DATABASE_URL</code> is configured, so '
            "accounts and history are kept in a local file. Fine for trying "
            "this out; on a serverless deployment they will not survive a "
            "restart.</div>",
        )
    return HTMLResponse(page, status_code=code)



def _start_session(request: Request, user_id: str) -> Response:
    resp = RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)
    # Secure only where the connection actually is HTTPS. Hard-coding it would
    # make the cookie silently undeliverable on http://localhost -- the login
    # would appear to succeed and then bounce straight back to the form.
    https = (
        request.url.scheme == "https"
        or request.headers.get("x-forwarded-proto", "").split(",")[0].strip() == "https"
    )
    resp.set_cookie(
        security.COOKIE_NAME,
        security.issue_token(user_id),
        max_age=get_settings().session_ttl_s,
        httponly=True,    # unreadable from JavaScript
        secure=https,
        samesite="lax",   # not sent on cross-site POSTs
    )
    return resp


@app.post("/login", include_in_schema=False)
def login(request: Request, email: str = Form(""), password: str = Form("")) -> Response:
    _accounts_or_404()
    # A brute-forcer costs nothing to run, so signing in is metered too -- but
    # on its own counter, so failed logins never consume the question quota.
    # Keyed by IP because the whole point is that we do not yet know who this
    # is; a per-account key would let an attacker exhaust *someone else's*.
    _rate_limit(f"ip:{_ip(request)}", "login")
    try:
        user = users.authenticate(email, password)
    except StorageError:
        return _auth_page(error=_STORE_DOWN, email=email, code=503)
    if user is None:
        # One message for both halves. "No such account" would turn the form
        # into an oracle for which addresses are registered.
        return _auth_page(error="Incorrect email or password.", email=email, code=401)
    return _start_session(request, user.id)


@app.get("/signup", include_in_schema=False)
def signup_page(user: User | None = Depends(current_user)) -> Response:
    _accounts_or_404()
    if user is not None:
        return RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)
    return _auth_page(mode="signup")


def _base_url(request: Request) -> str:
    """Absolute origin for a link in an email.

    Configured value wins. Deriving from the request is right locally and
    behind a proxy that sets the forwarded headers honestly, but Host is
    attacker-controllable in general -- and a verification link is exactly the
    thing you do not want pointed somewhere else. PUBLIC_BASE_URL is the
    answer wherever that matters.
    """
    configured = get_settings().public_base_url
    if configured:
        return configured.rstrip("/")
    proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip()
    host = request.headers.get("x-forwarded-host", "").split(",")[0].strip()
    scheme = proto or request.url.scheme
    netloc = host or request.url.netloc
    return f"{scheme}://{netloc}"


def _send_verification(request: Request, user: User) -> bool:
    """True if SES took it. False is reported to the user, never swallowed."""
    token = security.issue_link_token(
        security.PURPOSE_VERIFY, user.id, get_settings().verify_token_ttl_s
    )
    link = f"{_base_url(request)}/verify?token={token}"
    try:
        mailer.send_verification(user.email, link)
        return True
    except mailer.MailError:
        return False


@app.post("/signup", include_in_schema=False)
def signup(request: Request, email: str = Form(""), password: str = Form("")) -> Response:
    _accounts_or_404()
    # Metered harder than it looks like it needs to be: signup is the one
    # endpoint an anonymous visitor can use to write rows to our database.
    _rate_limit(f"ip:{_ip(request)}", "signup")
    try:
        user = users.create(email, password)
    except AccountError as e:
        return _auth_page(mode="signup", error=str(e), email=email, code=400)
    except StorageError:
        return _auth_page(mode="signup", error=_STORE_DOWN, email=email, code=503)

    # The session starts either way. An account whose verification mail bounced
    # is still an account, and signing them in is what lets them press "resend"
    # -- turning them away would leave the address taken and unreachable.
    if get_settings().verification_required:
        _send_verification(request, user)
    return _start_session(request, user.id)


@app.get("/verify", include_in_schema=False)
def verify(request: Request, token: str = "") -> Response:
    """Confirm an address. Idempotent, and safe to open while signed out."""
    session = security.read_link_token(security.PURPOSE_VERIFY, token)
    if session is None:
        return _auth_page(
            error="That confirmation link is invalid or has expired. "
            "Sign in and request a new one.",
            code=400,
        )
    try:
        user = users.mark_verified(session.user_id)
    except StorageError:
        return _auth_page(error=_STORE_DOWN, code=503)
    if user is None:
        return _auth_page(error="That account no longer exists.", code=404)
    # Signed in on the spot: the click proves both the address and, since the
    # token names the account, which one it belongs to. Making someone verify
    # and then sign in is a step that establishes nothing further.
    return _start_session(request, user.id)


@api.post("/resend-verification")
def resend_verification(request: Request, user: User = Depends(require_user)) -> dict:
    if not get_settings().verification_required or user.is_verified:
        return {"sent": False, "detail": "This address is already confirmed."}
    # Its own counter. Resending is cheap for us and free for an abuser to
    # trigger, and it must not eat the question quota of an account that cannot
    # ask questions yet anyway.
    _rate_limit(f"user:{user.id}", "verify")
    if not _send_verification(request, user):
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            "We could not send the email just now. Try again in a moment.",
        )
    return {"sent": True, "detail": f"Sent to {user.email}. Check your inbox."}


# --- forgotten passwords -------------------------------------------------

_RESET_SENT = (
    "If that address has an account, a reset link is on its way. "
    "The link is good for one hour."
)


@app.get("/forgot", include_in_schema=False)
def forgot_form() -> Response:
    _accounts_or_404()
    if not get_settings().mail_enabled:
        return _auth_page(
            error="Password reset is unavailable on this instance: it has no "
            "email configured. Ask an administrator to reset it for you.",
            code=503,
        )
    return _auth_page(mode="forgot")


@app.post("/forgot", include_in_schema=False)
def forgot(request: Request, email: str = Form("")) -> Response:
    """Send a reset link, and say the same thing either way.

    The response never reveals whether the address is registered. That matters
    less here than it would elsewhere -- signup already answers the same
    question -- but a reset form that confirms accounts is a free enumeration
    endpoint, and matching the signup form's leak is not a reason to add a
    second one.
    """
    _accounts_or_404()
    if not get_settings().mail_enabled:
        return _auth_page(
            error="Password reset is unavailable on this instance: it has no "
            "email configured. Ask an administrator to reset it for you.",
            code=503,
        )
    _rate_limit(f"ip:{_ip(request)}", "forgot")

    try:
        user = users.get_by_email(email)
    except StorageError:
        return _auth_page(error=_STORE_DOWN, email=email, code=503)

    if user is not None:
        token = security.issue_link_token(
            security.PURPOSE_RESET, user.id, get_settings().reset_token_ttl_s
        )
        try:
            mailer.send_password_reset(
                user.email, f"{_base_url(request)}/reset?token={token}"
            )
        except mailer.MailError:
            # Still not disclosed. A send failure and an unknown address must
            # look identical, or the difference becomes the oracle.
            pass
    return _auth_page(notice=_RESET_SENT)


@app.get("/reset", include_in_schema=False)
def reset_form(token: str = "") -> Response:
    if _reset_target(token) is None:
        return _auth_page(
            error="That reset link is invalid, expired, or has already been used.",
            code=400,
        )
    return _auth_page(mode="reset", token=token)


@app.post("/reset", include_in_schema=False)
def reset(request: Request, token: str = Form(""), password: str = Form("")) -> Response:
    _accounts_or_404()
    _rate_limit(f"ip:{_ip(request)}", "reset")
    user = _reset_target(token)
    if user is None:
        return _auth_page(
            error="That reset link is invalid, expired, or has already been used.",
            code=400,
        )
    try:
        users.set_password(user.id, password)
    except AccountError as e:
        return _auth_page(mode="reset", token=token, error=str(e), code=400)
    except StorageError:
        return _auth_page(mode="reset", token=token, error=_STORE_DOWN, code=503)

    # Reaching a mailbox proves the address as surely as a verification link
    # does, so an account that resets this way is confirmed by the same act.
    if not user.is_verified:
        users.mark_verified(user.id)
    # Issued after set_password stamped the row, so this session outlives the
    # invalidation it just triggered -- every *other* session does not.
    return _start_session(request, user.id)


def _reset_target(token: str) -> User | None:
    """The account a reset token names, if the token is still good.

    Single use falls out of the timestamp comparison rather than a stored
    list: `set_password` stamps `password_changed_at`, so a token issued before
    that stamp -- including the one that caused it -- no longer resolves.
    """
    session = security.read_link_token(security.PURPOSE_RESET, token)
    if session is None:
        return None
    try:
        user = users.get(session.user_id)
    except StorageError:
        return None
    if user is None:
        return None
    changed = to_epoch(user.password_changed_at)
    if changed is not None and session.issued_at_ms < int(changed * 1000):
        return None
    return user


@app.post("/logout", include_in_schema=False)
def logout() -> Response:
    resp = RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)
    resp.delete_cookie(security.COOKIE_NAME)
    return resp


# --- the assistant -------------------------------------------------------


@api.get("/info", dependencies=[Depends(require_auth)])
def info(user: User | None = Depends(current_user)) -> dict:
    s = get_settings()
    payload: dict = {
        "table": s.table,
        "index_version": s.index_version,
        "embedding_model": s.openai_embed_model,
        "generation_model": s.openai_model,
        "guard_model": s.openai_guard_model,
        "confidence_threshold": s.confidence_threshold,
        "auth_enabled": security.auth_required(),
        # The UI shows the allowance before the first question, so that a small
        # quota reads as a stated budget rather than as a surprise refusal.
        "verification_required": s.verification_required,
        "verified": user.is_verified if user else True,
        "rate_limit": s.rate_limit_per_window,
        "rate_limit_window_s": s.rate_limit_window_s,
        # The UI hides the history panel on this, rather than on auth alone:
        # with no account there is nobody to file a turn under.
        "history_enabled": user is not None,
        "user": user.public() if user else None,
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
    # Order matters. Verification is checked before the limiter, so a question
    # that was never going to run does not spend a slot from the allowance.
    dependencies=[
        Depends(require_auth),
        Depends(require_verified),
        Depends(enforce_rate_limit),
    ],
)
def ask(req: AskRequest, user: User | None = Depends(current_user)) -> AnswerResponse:
    try:
        response = _assistant().ask(req.query)
    except (RetrievalError, StoreError) as e:
        # The index is missing or unreachable. Refuse rather than 500: from the
        # caller's side the correct behaviour is still "no grounded answer".
        from .guardrails.policy import DISCLAIMER, refusal_text
        from .models import RefusalReason

        response = AnswerResponse(
            query=req.query,
            answered=False,
            answer=refusal_text(RefusalReason.LOW_CONFIDENCE),
            refusal_reason=RefusalReason.LOW_CONFIDENCE,
            disclaimer=DISCLAIMER,
            trace={"error": str(e)},
        )

    if user is not None:
        _remember(user, response)

    if not req.trace:
        response.trace = {}
    return response


def _remember(user: User, response: AnswerResponse) -> None:
    """File a turn under its user, without letting that fail the answer.

    History is a convenience; the answer on screen is the product. A database
    that has gone away must not turn a question the assistant already answered
    into an error the person has to retry -- so this swallows, logs, and marks
    the trace instead.
    """
    try:
        history.save(user.id, response)
    except StorageError as e:
        print(f"history: could not save a turn for {user.id}: {e}", file=sys.stderr)
        response.trace["history"] = "unsaved"


# --- history -------------------------------------------------------------


@api.get("/history")
def list_history(
    limit: int = 50,
    before: str | None = None,
    user: User = Depends(require_user),
) -> dict:
    limit = max(1, min(limit, 100))
    try:
        turns = history.list_for(user.id, limit=limit, before=before)
    except StorageError as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e)) from e
    return {
        "turns": [t.public() for t in turns],
        # The cursor for the next page, or null at the end. Handing it back
        # beats making the client know that pagination keys on created_at.
        "next_before": turns[-1].created_at if len(turns) == limit else None,
    }


@api.get("/history/{turn_id}")
def get_history_item(turn_id: str, user: User = Depends(require_user)) -> dict:
    try:
        turn = history.get(user.id, turn_id)
    except StorageError as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e)) from e
    if turn is None:
        # 404 whether it never existed or belongs to someone else. Telling
        # those apart would confirm the existence of other people's turns.
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such history item")
    return turn.public()


@api.delete("/history/{turn_id}")
def delete_history_item(turn_id: str, user: User = Depends(require_user)) -> dict:
    try:
        deleted = history.delete(user.id, turn_id)
    except StorageError as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e)) from e
    if not deleted:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such history item")
    return {"deleted": 1}


@api.delete("/history")
def clear_history(user: User = Depends(require_user)) -> dict:
    try:
        return {"deleted": history.clear(user.id)}
    except StorageError as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e)) from e


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

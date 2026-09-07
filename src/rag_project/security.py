"""Sessions and the spend cap.

These protect two different things and fail in opposite directions.

The **session** is a security boundary. It fails closed: if accounts are on and
a request cannot prove which user it belongs to, the request is refused.

The **rate limit** is a cost control, not a boundary. It fails *open*: if
Upstash is unreachable we serve the request rather than refusing it, for the
same reason cache.py degrades to computing normally. A limiter that can take
the whole app down when Redis blips is a worse outcome than a brief gap in
quota enforcement -- and the real backstop against a runaway bill is a spend
limit on a project-scoped OpenAI key, which no code here can undo.

Neither is a substitute for that spend limit.

The cookie is a signed statement of *identity*, not of authorisation: it says
"this is user X", and every route decides for itself what X may do. It carries
no privileges and no history, so a stolen cookie is worth exactly one account
until it expires, and revoking an account is a database delete rather than a
key rotation.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sys
import time
from dataclasses import dataclass

from .cache import get_cache
from .config import get_settings

COOKIE_NAME = "rag_session"

# User ids are uuid4 hex. Pinning the shape means a '.' can never appear in the
# first field of a token, so splitting one is unambiguous.
_USER_ID_RE = re.compile(r"^[0-9a-f]{32}$")


# --- session cookie ------------------------------------------------------

# Used only when no SESSION_SECRET is configured. Regenerated every process,
# which is what makes the misconfiguration visible: sessions stop working
# across restarts and across instances instead of being signed with something
# guessable.
_ephemeral_secret: str | None = None
_warned = False


def _secret() -> str:
    global _ephemeral_secret, _warned
    configured = get_settings().session_secret
    if configured:
        return configured
    if _ephemeral_secret is None:
        _ephemeral_secret = secrets.token_hex(32)
    if not _warned:
        _warned = True
        print(
            "warning: SESSION_SECRET is not set. Sessions are signed with a "
            "random per-process key, so every restart signs everyone out and "
            "multiple instances will not share logins. Set SESSION_SECRET to "
            "a long random string for any real deployment.",
            file=sys.stderr,
        )
    return _ephemeral_secret


def _sign(*parts: object) -> str:
    payload = ".".join(str(p) for p in parts).encode()
    return hmac.new(_secret().encode(), payload, hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class Session:
    """What a valid cookie attests to."""

    user_id: str
    # When it was issued, in **milliseconds**, so that a password change can
    # invalidate everything older than itself.
    #
    # Milliseconds rather than seconds because the two events this has to order
    # happen microseconds apart: `set_password` stamps the row and then issues
    # a fresh session, and at one-second resolution those are simultaneous --
    # so either the new session is rejected by the change that created it, or
    # a session from the same second survives a reset. Neither is acceptable,
    # and the fix is resolution, not slack.
    #
    # It is also why the field is an integer count rather than a float: the
    # token is split on ".", and "1788765476.123" is not one field.
    issued_at_ms: int


def issue_token(user_id: str, now: float | None = None) -> str:
    s = get_settings()
    now = time.time() if now is None else now
    issued_ms = int(now * 1000)
    expires_at = int(now) + s.session_ttl_s
    return f"{user_id}.{issued_ms}.{expires_at}.{_sign(user_id, issued_ms, expires_at)}"


def read_token(token: str | None, now: float | None = None) -> Session | None:
    """The session a cookie attests to, or None if it attests to nothing.

    Everything about this function is a rejection: a malformed token, a forged
    signature and an expired session all return None, and no caller can tell
    them apart. That is deliberate -- there is nothing a client can do about
    any of them except sign in again.
    """
    if not token:
        return None
    parts = token.split(".")
    if len(parts) != 4:
        return None
    user_id, raw_issued, raw_expiry, signature = parts
    if not _USER_ID_RE.match(user_id):
        return None
    try:
        issued_ms, expires_at = int(raw_issued), int(raw_expiry)
    except ValueError:
        return None
    # Signature first, then expiry: an unsigned token is not merely stale.
    if not hmac.compare_digest(signature, _sign(user_id, issued_ms, expires_at)):
        return None
    if (time.time() if now is None else now) >= expires_at:
        return None
    return Session(user_id=user_id, issued_at_ms=issued_ms)


# --- links sent by email -------------------------------------------------
#
# Verification and reset links carry the same kind of signed statement as the
# session cookie, with one addition: a purpose. Without it the two are the same
# string shape signed by the same key, and a verification link -- the one we
# hand out freely, to an address we have not yet confirmed -- would be usable
# to reset a password. The purpose goes *inside* the signature, so it cannot be
# edited in transit.
#
# Nothing is stored server-side for these. A reset link is made single-use by
# `users.set_password` stamping `password_changed_at`: the API refuses a token
# issued before that moment, so using one invalidates it, and so does any other
# password change. That is a revocation list you do not have to keep.


def issue_link_token(purpose: str, user_id: str, ttl_s: int, now: float | None = None) -> str:
    now = time.time() if now is None else now
    issued_ms = int(now * 1000)
    expires_at = int(now) + ttl_s
    return (
        f"{user_id}.{issued_ms}.{expires_at}."
        f"{_sign(purpose, user_id, issued_ms, expires_at)}"
    )


def read_link_token(purpose: str, token: str | None, now: float | None = None) -> Session | None:
    """Same rejections as `read_token`, plus: signed for a different purpose."""
    if not token:
        return None
    parts = token.split(".")
    if len(parts) != 4:
        return None
    user_id, raw_issued, raw_expiry, signature = parts
    if not _USER_ID_RE.match(user_id):
        return None
    try:
        issued_ms, expires_at = int(raw_issued), int(raw_expiry)
    except ValueError:
        return None
    if not hmac.compare_digest(
        signature, _sign(purpose, user_id, issued_ms, expires_at)
    ):
        return None
    if (time.time() if now is None else now) >= expires_at:
        return None
    return Session(user_id=user_id, issued_at_ms=issued_ms)


PURPOSE_VERIFY = "verify-email"
PURPOSE_RESET = "reset-password"


def auth_required() -> bool:
    return get_settings().auth_required


def reset_sessions() -> None:
    """Forget the per-process signing key. For tests."""
    global _ephemeral_secret, _warned
    _ephemeral_secret = None
    _warned = False


# --- rate limiting -------------------------------------------------------

# Per-process fallback for when Upstash is unreachable. Weaker than the shared
# counter (each instance gets its own allowance) but better than no cap at all.
_local_counts: dict[str, tuple[int, int]] = {}


@dataclass
class RateVerdict:
    allowed: bool
    remaining: int
    retry_after_s: int


def client_ip(headers, fallback: str | None) -> str:
    """The caller's address, preferring the proxy header where one is trusted.

    x-forwarded-for is a comma-separated chain and the *first* entry is the
    original client; taking the last would rate-limit the proxy itself.
    """
    if get_settings().trust_proxy_header:
        forwarded = headers.get("x-forwarded-for") or headers.get("x-real-ip")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return fallback or "unknown"


def check_rate_limit(subject: str, scope: str = "ask") -> RateVerdict:
    """Count one request against `subject`'s quota for `scope`.

    The subject is the signed-in user where there is one, and the client IP
    otherwise. Metering by user is the accurate unit now that accounts exist:
    an IP is shared by everyone behind a clinic's NAT, so an IP quota either
    starves a whole building or is generous enough to be no quota at all.
    Sign-up and sign-in stay on the IP, since they are what you use *before*
    you have an identity to meter.

    Scopes get separate counters on purpose. Asking questions is metered
    because it costs money; signing in is metered because guessing a password
    is free. Sharing one bucket would let a few failed logins eat the day's
    questions -- two unrelated concerns punishing each other.
    """
    s = get_settings()
    limit = s.rate_limit_per_window
    if limit <= 0:
        return RateVerdict(True, -1, 0)

    window = s.rate_limit_window_s
    now = int(time.time())
    bucket = now // window
    reset_in = (bucket + 1) * window - now
    # The subject is hashed, not stored: a rate-limit key should not turn
    # Upstash into a log of who asked medical questions.
    digest = hashlib.sha256(f"{subject}|{s.pipeline_fingerprint}".encode()).hexdigest()[:24]
    key = f"rl:{scope}:{bucket}:{digest}"

    count = get_cache().incr(key, window + 60)
    if count is None:
        count = _local_incr(key, bucket)

    return RateVerdict(count <= limit, max(0, limit - count), reset_in)


def _local_incr(key: str, bucket: int) -> int:
    seen_bucket, count = _local_counts.get(key, (bucket, 0))
    count = count + 1 if seen_bucket == bucket else 1
    _local_counts[key] = (bucket, count)
    if len(_local_counts) > 4096:  # bound it; stale buckets are worthless
        for k, (b, _) in list(_local_counts.items()):
            if b != bucket:
                _local_counts.pop(k, None)
    return count


def reset_rate_limits() -> None:
    """For tests."""
    _local_counts.clear()

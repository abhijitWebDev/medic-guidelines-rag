"""Access control for the deployed app: a shared password, and a spend cap.

These protect two different things and fail in opposite directions.

The **password** is a security boundary. It fails closed: if it is configured
and the request cannot prove it passed, the request is refused.

The **rate limit** is a cost control, not a boundary. It fails *open*: if
Upstash is unreachable we serve the request rather than refusing it, for the
same reason cache.py degrades to computing normally. A limiter that can take
the whole app down when Redis blips is a worse outcome than a brief gap in
quota enforcement -- and the real backstop against a runaway bill is a
spend limit on a project-scoped OpenAI key, which no code here can undo.

Neither is a substitute for that spend limit.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass

from .cache import get_cache
from .config import get_settings

COOKIE_NAME = "rag_session"


# --- session cookie ------------------------------------------------------


def _secret() -> str:
    """Signing key, derived from the password itself.

    Deriving rather than configuring a second value means changing the
    password invalidates every outstanding session for free -- which is the
    behaviour you want the moment you change it because someone shared it.
    """
    return hashlib.sha256(f"rag-session-v1:{get_settings().app_password}".encode()).hexdigest()


def _sign(expires_at: int) -> str:
    return hmac.new(_secret().encode(), str(expires_at).encode(), hashlib.sha256).hexdigest()


def issue_token(now: float | None = None) -> str:
    s = get_settings()
    expires_at = int((time.time() if now is None else now) + s.session_ttl_s)
    return f"{expires_at}.{_sign(expires_at)}"


def token_is_valid(token: str | None, now: float | None = None) -> bool:
    if not token or "." not in token:
        return False
    raw_expiry, _, signature = token.partition(".")
    try:
        expires_at = int(raw_expiry)
    except ValueError:
        return False
    # Signature first, then expiry: an unsigned token is not merely stale.
    if not hmac.compare_digest(signature, _sign(expires_at)):
        return False
    return (time.time() if now is None else now) < expires_at


def password_matches(candidate: str) -> bool:
    """Constant-time compare, so a wrong guess leaks nothing by timing."""
    expected = get_settings().app_password
    if not expected:
        return True  # gate disabled
    return hmac.compare_digest(candidate or "", expected)


def auth_required() -> bool:
    return bool(get_settings().app_password)


def request_is_authenticated(cookie_value: str | None) -> bool:
    return not auth_required() or token_is_valid(cookie_value)


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


def check_rate_limit(ip: str, scope: str = "ask") -> RateVerdict:
    """Count one request against `ip`'s quota for `scope`.

    Scopes get separate counters on purpose. Asking questions is metered
    because it costs money; logging in is metered because guessing a shared
    password is free. Sharing one bucket would let a couple of failed logins
    eat the day's questions -- two unrelated concerns punishing each other.
    """
    s = get_settings()
    limit = s.rate_limit_per_window
    if limit <= 0:
        return RateVerdict(True, -1, 0)

    window = s.rate_limit_window_s
    now = int(time.time())
    bucket = now // window
    reset_in = (bucket + 1) * window - now
    # The IP is hashed, not stored: a rate-limit key should not turn Upstash
    # into a log of who asked medical questions.
    digest = hashlib.sha256(f"{ip}|{s.pipeline_fingerprint}".encode()).hexdigest()[:24]
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

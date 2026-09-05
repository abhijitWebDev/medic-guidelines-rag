"""Access control: the password is a boundary, the rate limit is a spend cap.

They are tested for opposite failure behaviour on purpose. The gate must fail
closed; the limiter must fail open.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from rag_project import security
from rag_project.api import app
from rag_project.cache import reset_cache
from rag_project.config import get_settings

PASSWORD = "correct-horse-battery-staple"


@pytest.fixture(autouse=True)
def _clean():
    get_settings.cache_clear()
    reset_cache()
    security.reset_rate_limits()
    yield
    get_settings.cache_clear()
    reset_cache()
    security.reset_rate_limits()


@pytest.fixture
def locked(monkeypatch) -> TestClient:
    """An instance with the gate switched on."""
    monkeypatch.setenv("APP_PASSWORD", PASSWORD)
    monkeypatch.setenv("RATE_LIMIT_PER_WINDOW", "0")  # isolate auth from spend
    get_settings.cache_clear()
    return TestClient(app)


@pytest.fixture
def open_instance(monkeypatch) -> TestClient:
    monkeypatch.setenv("APP_PASSWORD", "")
    get_settings.cache_clear()
    return TestClient(app)


# --- the gate is off by default ------------------------------------------


def test_no_password_configured_means_no_gate(open_instance):
    """Local work and the test suite must not need a password."""
    assert open_instance.get("/").status_code == 200
    assert open_instance.get("/api/info").status_code == 200


# --- the gate fails closed -----------------------------------------------


def test_api_is_locked_without_a_session(locked):
    assert locked.post("/api/ask", json={"query": "How is TB diagnosed?"}).status_code == 401
    assert locked.get("/api/info").status_code == 401


def test_root_shows_the_login_page_not_the_app(locked):
    r = locked.get("/")
    assert r.status_code == 200
    assert "access password" in r.text.lower()
    assert "Try one" not in r.text, "the app shell leaked to an anonymous visitor"


def test_wrong_password_is_rejected(locked):
    r = locked.post("/login", data={"password": "hunter2"}, follow_redirects=False)
    assert r.status_code == 401
    assert security.COOKIE_NAME not in r.cookies


def test_correct_password_opens_the_app(locked):
    r = locked.post("/login", data={"password": PASSWORD}, follow_redirects=False)
    assert r.status_code == 303
    assert r.cookies.get(security.COOKIE_NAME)
    assert locked.get("/api/info").status_code == 200


def test_session_cookie_is_not_readable_by_javascript(locked):
    r = locked.post("/login", data={"password": PASSWORD}, follow_redirects=False)
    header = r.headers["set-cookie"].lower()
    assert "httponly" in header, "an XSS could otherwise lift the session"
    assert "samesite=lax" in header


def test_cookie_is_not_secure_over_plain_http(locked):
    """secure=True on http:// makes the browser drop the cookie silently, so
    login would appear to work and then bounce back to the form."""
    r = locked.post("/login", data={"password": PASSWORD}, follow_redirects=False)
    assert "secure" not in r.headers["set-cookie"].lower()


def test_cookie_is_secure_behind_an_https_proxy(locked):
    r = locked.post("/login", data={"password": PASSWORD},
                    headers={"x-forwarded-proto": "https"}, follow_redirects=False)
    assert "secure" in r.headers["set-cookie"].lower()


def test_logout_clears_the_session(locked):
    locked.post("/login", data={"password": PASSWORD})
    assert locked.get("/api/info").status_code == 200
    locked.post("/logout")
    assert locked.get("/api/info").status_code == 401


def test_health_stays_open(locked):
    """Uptime checks must not need a password."""
    assert locked.get("/health").status_code == 200


# --- token forgery -------------------------------------------------------


def test_forged_and_expired_tokens_are_rejected(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", PASSWORD)
    get_settings.cache_clear()

    good = security.issue_token()
    assert security.token_is_valid(good)

    expiry = good.split(".")[0]
    assert not security.token_is_valid(f"{expiry}.{'0' * 64}"), "bad signature accepted"
    assert not security.token_is_valid(f"{int(expiry) + 99999}.{good.split('.')[1]}"), \
        "expiry was extended without re-signing"
    assert not security.token_is_valid("garbage")
    assert not security.token_is_valid(None)
    assert not security.token_is_valid(security.issue_token(now=time.time() - 10**7)), \
        "an expired token was accepted"


def test_changing_the_password_invalidates_old_sessions(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", PASSWORD)
    get_settings.cache_clear()
    token = security.issue_token()
    assert security.token_is_valid(token)

    monkeypatch.setenv("APP_PASSWORD", "a-new-password")
    get_settings.cache_clear()
    assert not security.token_is_valid(token), \
        "rotating the password must log everyone out"


# --- rate limiting -------------------------------------------------------


def test_rate_limit_blocks_after_the_quota(monkeypatch, open_instance):
    monkeypatch.setenv("RATE_LIMIT_PER_WINDOW", "3")
    get_settings.cache_clear()
    security.reset_rate_limits()

    verdicts = [security.check_rate_limit("1.2.3.4") for _ in range(5)]
    assert [v.allowed for v in verdicts] == [True, True, True, False, False]
    assert verdicts[-1].retry_after_s > 0


def test_rate_limit_is_per_client(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_WINDOW", "1")
    get_settings.cache_clear()
    security.reset_rate_limits()
    assert security.check_rate_limit("1.1.1.1").allowed
    assert not security.check_rate_limit("1.1.1.1").allowed
    assert security.check_rate_limit("2.2.2.2").allowed, "one caller exhausted another's quota"


def test_rate_limit_of_zero_disables_it(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_WINDOW", "0")
    get_settings.cache_clear()
    assert all(security.check_rate_limit("9.9.9.9").allowed for _ in range(50))


def test_limiter_fails_open_when_redis_is_unreachable(monkeypatch):
    """Opposite of the auth gate. A limiter that 500s on a Redis blip is worse
    than a brief gap in enforcement -- the OpenAI spend cap is the real
    backstop. It still falls back to a per-process counter."""
    monkeypatch.setenv("RATE_LIMIT_PER_WINDOW", "2")
    get_settings.cache_clear()
    security.reset_rate_limits()

    from rag_project import cache as cache_mod

    class DeadCache:
        def incr(self, key, ttl_s):
            return None  # Redis unreachable

    monkeypatch.setattr(cache_mod, "get_cache", lambda: DeadCache())
    monkeypatch.setattr("rag_project.security.get_cache", lambda: DeadCache())

    results = [security.check_rate_limit("5.5.5.5").allowed for _ in range(4)]
    assert results[0] is True, "an unreachable cache must not refuse the request"
    assert results[-1] is False, "the in-process fallback should still cap it"


def test_ip_is_hashed_not_stored(monkeypatch):
    """A rate-limit key must not turn Upstash into a log of who asked what."""
    monkeypatch.setenv("RATE_LIMIT_PER_WINDOW", "5")
    get_settings.cache_clear()
    seen = {}

    class Recorder:
        def incr(self, key, ttl_s):
            seen["key"] = key
            return 1

    monkeypatch.setattr("rag_project.security.get_cache", lambda: Recorder())
    security.check_rate_limit("203.0.113.7")
    assert "203.0.113.7" not in seen["key"]


def test_forwarded_header_identifies_the_original_client():
    headers = {"x-forwarded-for": "203.0.113.7, 70.41.3.18, 150.172.238.178"}
    assert security.client_ip(headers, "10.0.0.1") == "203.0.113.7", \
        "must take the first hop, not the proxy"


def test_falls_back_to_socket_address_without_the_header():
    assert security.client_ip({}, "10.0.0.1") == "10.0.0.1"


def test_ask_returns_429_when_limited(monkeypatch, open_instance):
    monkeypatch.setenv("RATE_LIMIT_PER_WINDOW", "1")
    get_settings.cache_clear()
    security.reset_rate_limits()

    q = {"query": "Should I take rifampicin for my cough?"}
    assert open_instance.post("/api/ask", json=q).status_code == 200
    blocked = open_instance.post("/api/ask", json=q)
    assert blocked.status_code == 429
    assert "Retry-After" in blocked.headers


def test_login_is_rate_limited_too(monkeypatch):
    """Guessing a shared password is free otherwise."""
    monkeypatch.setenv("APP_PASSWORD", PASSWORD)
    monkeypatch.setenv("RATE_LIMIT_PER_WINDOW", "2")
    get_settings.cache_clear()
    security.reset_rate_limits()

    c = TestClient(app)
    for _ in range(2):
        c.post("/login", data={"password": "guess"}, follow_redirects=False)
    assert c.post("/login", data={"password": "guess"},
                  follow_redirects=False).status_code == 429


def test_login_attempts_do_not_consume_the_question_quota(monkeypatch):
    """Brute-force protection and spend control are separate concerns; sharing
    one counter let two failed logins eat the day's questions."""
    monkeypatch.setenv("RATE_LIMIT_PER_WINDOW", "3")
    get_settings.cache_clear()
    security.reset_rate_limits()

    for _ in range(3):
        security.check_rate_limit("4.4.4.4", "login")
    assert not security.check_rate_limit("4.4.4.4", "login").allowed, "login not limited"
    assert security.check_rate_limit("4.4.4.4", "ask").allowed, \
        "failed logins consumed the question quota"

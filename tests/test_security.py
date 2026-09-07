"""Access control: the session is a boundary, the rate limit is a spend cap.

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

EMAIL = "clinician@example.in"
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
def accounts_on(monkeypatch) -> TestClient:
    """An instance with accounts switched on and nobody registered yet."""
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("RATE_LIMIT_PER_WINDOW", "0")  # isolate auth from spend
    # The suite hashes a lot of passwords; the algorithm under test is the same
    # at any cost. test_default_password_cost_is_not_cheap pins the real one.
    monkeypatch.setenv("SCRYPT_N", "1024")
    get_settings.cache_clear()
    return TestClient(app)


@pytest.fixture
def signed_in(accounts_on) -> TestClient:
    r = accounts_on.post(
        "/signup", data={"email": EMAIL, "password": PASSWORD}, follow_redirects=False
    )
    assert r.status_code == 303, r.text
    return accounts_on


@pytest.fixture
def open_instance(monkeypatch) -> TestClient:
    monkeypatch.setenv("AUTH_ENABLED", "false")
    get_settings.cache_clear()
    return TestClient(app)


# --- the gate is off by default ------------------------------------------


def test_no_database_configured_means_no_gate(open_instance):
    """Local work and the test suite must not need an account."""
    assert open_instance.get("/").status_code == 200
    assert open_instance.get("/api/info").status_code == 200


def test_configuring_a_database_turns_the_gate_on(monkeypatch):
    """The one setting that matters is DATABASE_URL: provisioning a user store
    is the act of deciding this instance has users."""
    monkeypatch.delenv("AUTH_ENABLED", raising=False)  # conftest pins it off
    monkeypatch.setenv("DATABASE_URL", "postgresql://example/db")
    get_settings.cache_clear()
    assert get_settings().auth_required


def test_a_legacy_password_still_forces_the_gate(monkeypatch):
    """APP_PASSWORD is no longer a credential, but it did mean "private".
    Upgrading must not quietly publish an instance someone locked."""
    monkeypatch.delenv("AUTH_ENABLED", raising=False)  # conftest pins it off
    monkeypatch.setenv("APP_PASSWORD", "the-old-shared-secret")
    get_settings.cache_clear()
    assert get_settings().auth_required


# --- the gate fails closed -----------------------------------------------


def test_api_is_locked_without_a_session(accounts_on):
    assert accounts_on.post("/api/ask", json={"query": "How is TB diagnosed?"}).status_code == 401
    assert accounts_on.get("/api/info").status_code == 401
    assert accounts_on.get("/api/history").status_code == 401


def test_root_shows_the_sign_in_page_not_the_app(accounts_on):
    r = accounts_on.get("/")
    assert r.status_code == 200
    assert "Sign in" in r.text
    assert "Try one" not in r.text, "the app shell leaked to an anonymous visitor"


def test_wrong_password_is_rejected(signed_in):
    signed_in.cookies.clear()
    r = signed_in.post(
        "/login", data={"email": EMAIL, "password": "hunter2"}, follow_redirects=False
    )
    assert r.status_code == 401
    assert security.COOKIE_NAME not in r.cookies


def test_unknown_email_is_rejected_indistinguishably(signed_in):
    """The two failures must read the same, or the form becomes an oracle for
    which addresses have accounts."""
    signed_in.cookies.clear()
    unknown = signed_in.post(
        "/login", data={"email": "nobody@example.in", "password": PASSWORD},
        follow_redirects=False,
    )
    wrong_pw = signed_in.post(
        "/login", data={"email": EMAIL, "password": "hunter2"}, follow_redirects=False
    )
    assert unknown.status_code == wrong_pw.status_code == 401
    assert "Incorrect email or password." in unknown.text
    assert "Incorrect email or password." in wrong_pw.text


def test_signing_in_opens_the_app(signed_in):
    signed_in.cookies.clear()
    r = signed_in.post(
        "/login", data={"email": EMAIL, "password": PASSWORD}, follow_redirects=False
    )
    assert r.status_code == 303
    assert r.cookies.get(security.COOKIE_NAME)
    assert signed_in.get("/api/info").status_code == 200


def test_email_case_does_not_make_a_second_account(signed_in):
    signed_in.cookies.clear()
    r = signed_in.post(
        "/login", data={"email": "Clinician@Example.IN ", "password": PASSWORD},
        follow_redirects=False,
    )
    assert r.status_code == 303, "the same person typed their address differently"


# --- signing up ----------------------------------------------------------


def test_signup_creates_an_account_and_signs_in(accounts_on):
    r = accounts_on.post(
        "/signup", data={"email": "new@example.in", "password": PASSWORD},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert accounts_on.get("/api/info").json()["user"]["email"] == "new@example.in"


def test_duplicate_email_is_refused(signed_in):
    r = signed_in.post(
        "/signup", data={"email": EMAIL, "password": PASSWORD}, follow_redirects=False
    )
    assert r.status_code == 400
    assert "already has an account" in r.text


def test_a_short_password_is_refused(accounts_on):
    r = accounts_on.post(
        "/signup", data={"email": "weak@example.in", "password": "short"},
        follow_redirects=False,
    )
    assert r.status_code == 400
    assert security.COOKIE_NAME not in r.cookies


def test_a_malformed_email_is_refused(accounts_on):
    r = accounts_on.post(
        "/signup", data={"email": "not-an-address", "password": PASSWORD},
        follow_redirects=False,
    )
    assert r.status_code == 400


def test_the_form_is_returned_with_the_email_still_in_it(accounts_on):
    """Retyping an address after a rejected password is pure friction."""
    r = accounts_on.post(
        "/signup", data={"email": "typo@example.in", "password": "short"},
        follow_redirects=False,
    )
    assert 'value="typo@example.in"' in r.text


def test_the_page_escapes_what_it_echoes_back(accounts_on):
    """The email field is reflected into an attribute; it is user input."""
    r = accounts_on.post(
        "/signup", data={"email": '"><script>alert(1)</script>', "password": "short"},
        follow_redirects=False,
    )
    assert "<script>alert(1)</script>" not in r.text


# --- the cookie ----------------------------------------------------------


def test_session_cookie_is_not_readable_by_javascript(signed_in):
    signed_in.cookies.clear()
    r = signed_in.post(
        "/login", data={"email": EMAIL, "password": PASSWORD}, follow_redirects=False
    )
    header = r.headers["set-cookie"].lower()
    assert "httponly" in header, "an XSS could otherwise lift the session"
    assert "samesite=lax" in header


def test_cookie_is_not_secure_over_plain_http(signed_in):
    signed_in.cookies.clear()
    r = signed_in.post(
        "/login", data={"email": EMAIL, "password": PASSWORD}, follow_redirects=False
    )
    assert "secure" not in r.headers["set-cookie"].lower()


def test_cookie_is_secure_behind_an_https_proxy(signed_in):
    signed_in.cookies.clear()
    r = signed_in.post(
        "/login", data={"email": EMAIL, "password": PASSWORD},
        headers={"x-forwarded-proto": "https"}, follow_redirects=False,
    )
    assert "secure" in r.headers["set-cookie"].lower()


def test_logout_clears_the_session(signed_in):
    assert signed_in.get("/api/info").status_code == 200
    signed_in.post("/logout")
    assert signed_in.get("/api/info").status_code == 401


def test_health_stays_open(accounts_on):
    """Uptime checks must not need an account."""
    assert accounts_on.get("/health").status_code == 200


# --- token forgery -------------------------------------------------------

UID = "0" * 32


def test_forged_and_expired_tokens_are_rejected():
    good = security.issue_token(UID)
    assert security.read_token(good).user_id == UID

    _, issued_ms, expiry, signature = good.split(".")
    assert security.read_token(f"{UID}.{issued_ms}.{expiry}.{'0' * 64}") is None, \
        "bad signature accepted"
    assert security.read_token(f"{UID}.{issued_ms}.{int(expiry) + 99999}.{signature}") is None, \
        "expiry was extended without re-signing"
    assert security.read_token(f"{UID}.{int(issued_ms) - 99999}.{expiry}.{signature}") is None, \
        "issue time was backdated without re-signing"
    assert security.read_token("garbage") is None
    assert security.read_token(None) is None
    assert security.read_token(security.issue_token(UID, now=time.time() - 10**7)) is None, \
        "an expired token was accepted"


def test_a_token_cannot_be_repointed_at_another_user():
    """The user id is inside the signature, not merely beside it."""
    mine = security.issue_token(UID)
    _, issued_ms, expiry, signature = mine.split(".")
    theirs = "f" * 32
    assert security.read_token(f"{theirs}.{issued_ms}.{expiry}.{signature}") is None


def test_a_session_token_is_not_a_link_token():
    """A verification link is handed out freely, to an address nobody has
    confirmed yet. If the two were interchangeable it would be a password
    reset."""
    session = security.issue_token(UID)
    verify = security.issue_link_token(security.PURPOSE_VERIFY, UID, 3600)
    reset = security.issue_link_token(security.PURPOSE_RESET, UID, 3600)

    assert security.read_link_token(security.PURPOSE_VERIFY, session) is None
    assert security.read_token(verify) is None
    assert security.read_link_token(security.PURPOSE_RESET, verify) is None, \
        "a confirmation link doubled as a password reset"
    assert security.read_link_token(security.PURPOSE_VERIFY, reset) is None
    assert security.read_link_token(security.PURPOSE_VERIFY, verify).user_id == UID
    assert security.read_link_token(security.PURPOSE_RESET, reset).user_id == UID


def test_link_tokens_expire():
    stale = security.issue_link_token(
        security.PURPOSE_RESET, UID, 3600, now=time.time() - 10**5
    )
    assert security.read_link_token(security.PURPOSE_RESET, stale) is None


def test_a_token_with_a_bogus_user_id_is_rejected():
    """The id shape is pinned so the token can never be split ambiguously."""
    assert security.read_token(f"not.a.uuid.{int(time.time()) + 999}.x") is None


def test_rotating_the_session_secret_invalidates_old_sessions(monkeypatch):
    token = security.issue_token(UID)
    assert security.read_token(token).user_id == UID

    monkeypatch.setenv("SESSION_SECRET", "a-different-secret")
    get_settings.cache_clear()
    assert security.read_token(token) is None, \
        "rotating the signing key must log everyone out"


def test_an_unset_secret_does_not_fall_back_to_something_guessable(monkeypatch):
    """A missing key must break sessions loudly, never sign them predictably."""
    monkeypatch.setenv("SESSION_SECRET", "")
    get_settings.cache_clear()
    security.reset_sessions()
    token = security.issue_token(UID)

    security.reset_sessions()  # stands in for a restart / another instance
    assert security.read_token(token) is None


def test_default_password_cost_is_not_cheap():
    """Pinned deliberately: the fixtures lower it for speed, and a default that
    quietly drifted down would make every stored hash cheaper to crack."""
    s = get_settings()
    assert s.scrypt_n >= 2**14
    assert s.scrypt_n & (s.scrypt_n - 1) == 0, "scrypt n must be a power of two"


# --- rate limiting -------------------------------------------------------


def test_rate_limit_blocks_after_the_quota(monkeypatch, open_instance):
    monkeypatch.setenv("RATE_LIMIT_PER_WINDOW", "3")
    get_settings.cache_clear()
    security.reset_rate_limits()

    verdicts = [security.check_rate_limit("ip:1.2.3.4") for _ in range(5)]
    assert [v.allowed for v in verdicts] == [True, True, True, False, False]
    assert verdicts[-1].retry_after_s > 0


def test_rate_limit_is_per_subject(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_WINDOW", "1")
    get_settings.cache_clear()
    security.reset_rate_limits()
    assert security.check_rate_limit("user:aaa").allowed
    assert not security.check_rate_limit("user:aaa").allowed
    assert security.check_rate_limit("user:bbb").allowed, \
        "one account exhausted another's quota"


def test_rate_limit_of_zero_disables_it(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_WINDOW", "0")
    get_settings.cache_clear()
    assert all(security.check_rate_limit("ip:9.9.9.9").allowed for _ in range(50))


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

    results = [security.check_rate_limit("ip:5.5.5.5").allowed for _ in range(4)]
    assert results[0] is True, "an unreachable cache must not refuse the request"
    assert results[-1] is False, "the in-process fallback should still cap it"


def test_the_subject_is_hashed_not_stored(monkeypatch):
    """A rate-limit key must not turn Upstash into a log of who asked what."""
    monkeypatch.setenv("RATE_LIMIT_PER_WINDOW", "5")
    get_settings.cache_clear()
    seen = {}

    class Recorder:
        def incr(self, key, ttl_s):
            seen["key"] = key
            return 1

    monkeypatch.setattr("rag_project.security.get_cache", lambda: Recorder())
    security.check_rate_limit("ip:203.0.113.7")
    assert "203.0.113.7" not in seen["key"]

    security.check_rate_limit("user:4f9c2b1a")
    assert "4f9c2b1a" not in seen["key"], "a key must not name the account either"


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


def test_questions_are_metered_per_account(monkeypatch, signed_in):
    """Two clinicians behind one hospital NAT must not share an allowance."""
    monkeypatch.setenv("RATE_LIMIT_PER_WINDOW", "1")
    get_settings.cache_clear()
    security.reset_rate_limits()

    q = {"query": "Should I take rifampicin for my cough?"}
    assert signed_in.post("/api/ask", json=q).status_code == 200
    assert signed_in.post("/api/ask", json=q).status_code == 429

    # Same client, same IP, different account.
    signed_in.cookies.clear()
    signed_in.post(
        "/signup", data={"email": "second@example.in", "password": PASSWORD},
        follow_redirects=False,
    )
    assert signed_in.post("/api/ask", json=q).status_code == 200, \
        "one account's quota was charged to another"


def test_login_is_rate_limited_too(monkeypatch):
    """Guessing a password is free otherwise."""
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("RATE_LIMIT_PER_WINDOW", "2")
    monkeypatch.setenv("SCRYPT_N", "1024")
    get_settings.cache_clear()
    security.reset_rate_limits()

    c = TestClient(app)
    for _ in range(2):
        c.post("/login", data={"email": EMAIL, "password": "guess"},
               follow_redirects=False)
    assert c.post("/login", data={"email": EMAIL, "password": "guess"},
                  follow_redirects=False).status_code == 429


def test_signup_is_rate_limited_too(monkeypatch):
    """It is the one endpoint an anonymous visitor can use to write rows."""
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("RATE_LIMIT_PER_WINDOW", "2")
    monkeypatch.setenv("SCRYPT_N", "1024")
    get_settings.cache_clear()
    security.reset_rate_limits()

    c = TestClient(app)
    for i in range(2):
        c.post("/signup", data={"email": f"a{i}@example.in", "password": PASSWORD},
               follow_redirects=False)
    assert c.post("/signup", data={"email": "a9@example.in", "password": PASSWORD},
                  follow_redirects=False).status_code == 429


def test_login_attempts_do_not_consume_the_question_quota(monkeypatch):
    """Brute-force protection and spend control are separate concerns; sharing
    one counter let two failed logins eat the day's questions."""
    monkeypatch.setenv("RATE_LIMIT_PER_WINDOW", "3")
    get_settings.cache_clear()
    security.reset_rate_limits()

    for _ in range(3):
        security.check_rate_limit("ip:4.4.4.4", "login")
    assert not security.check_rate_limit("ip:4.4.4.4", "login").allowed, "login not limited"
    assert security.check_rate_limit("ip:4.4.4.4", "ask").allowed, \
        "failed logins consumed the question quota"


def test_signup_does_not_exist_on_an_open_instance(open_instance):
    """Otherwise an anonymous visitor can write user rows into a database the
    operator never turned on."""
    assert open_instance.get("/signup").status_code == 404
    assert open_instance.post(
        "/signup", data={"email": "sneaky@example.in", "password": PASSWORD},
        follow_redirects=False,
    ).status_code == 404
    assert open_instance.post(
        "/login", data={"email": EMAIL, "password": PASSWORD}, follow_redirects=False
    ).status_code == 404


# --- the configured allowance --------------------------------------------


def test_the_default_allowance_is_five_per_hour():
    """Pinned because it is a spend decision, not an implementation detail: a
    question costs an intent call, an embedding, HyDE, up to 40 rerank calls,
    an answer and two output-gate checks."""
    s = get_settings()
    assert s.rate_limit_per_window == 5
    assert s.rate_limit_window_s == 3600


def test_a_user_gets_five_questions_then_is_held(signed_in, monkeypatch):
    monkeypatch.delenv("RATE_LIMIT_PER_WINDOW", raising=False)  # the real default
    get_settings.cache_clear()
    security.reset_rate_limits()

    q = {"query": "Should I take rifampicin for my cough?"}
    codes = [signed_in.post("/api/ask", json=q).status_code for _ in range(6)]
    assert codes == [200, 200, 200, 200, 200, 429]


def test_an_answer_reports_what_is_left(signed_in, monkeypatch):
    """The allowance is small enough to reach in ordinary use, so the UI shows
    it before it bites. That needs the count on every answer, not just on the
    one that gets refused."""
    monkeypatch.delenv("RATE_LIMIT_PER_WINDOW", raising=False)
    get_settings.cache_clear()
    security.reset_rate_limits()

    first = signed_in.post("/api/ask", json={"query": "Should I take rifampicin?"})
    assert first.headers["X-RateLimit-Limit"] == "5"
    assert first.headers["X-RateLimit-Remaining"] == "4"

    second = signed_in.post("/api/ask", json={"query": "Should I take rifampicin?"})
    assert second.headers["X-RateLimit-Remaining"] == "3"


def test_the_refusal_says_what_the_limit_is_and_when_it_lifts(signed_in, monkeypatch):
    """"Rate limit reached" alone leaves someone refreshing, unable to tell a
    quota from an outage."""
    monkeypatch.setenv("RATE_LIMIT_PER_WINDOW", "1")
    get_settings.cache_clear()
    security.reset_rate_limits()

    q = {"query": "Should I take rifampicin for my cough?"}
    signed_in.post("/api/ask", json=q)
    blocked = signed_in.post("/api/ask", json=q)

    assert blocked.status_code == 429
    detail = blocked.json()["detail"]
    assert "1 question" in detail
    assert "minute" in detail
    assert int(blocked.headers["Retry-After"]) > 0
    assert blocked.headers["X-RateLimit-Remaining"] == "0"


# --- the auth page renders exactly one panel -----------------------------


def _style_block(html: str) -> str:
    return html.split("<style>", 1)[1].split("</style>", 1)[0]


def _card_tag(html: str) -> str:
    return [line for line in html.splitlines() if '<div class="card"' in line][0].strip()


def test_each_mode_marks_exactly_one_panel(accounts_on):
    """Which panel shows is decided by one attribute, so it is the one thing
    worth pinning."""
    assert 'data-mode="signin"' in _card_tag(accounts_on.get("/").text)
    assert 'data-mode="signup"' in _card_tag(accounts_on.get("/signup").text)


def test_substitution_never_rewrites_the_stylesheet(accounts_on):
    """The bug this exists to catch: the page is rendered by string
    substitution, and the stylesheet selects on the very attributes being
    substituted. Replacing a literal `data-mode="signin"` rewrote the CSS rules
    along with the markup, so `.card:not([data-mode="signin"]) #panel-signin`
    became `:not([data-mode="signup"])` on the sign-up page -- and both forms
    rendered at once. Placeholders keep the two apart."""
    baseline = _style_block(accounts_on.get("/").text)

    pages = [
        accounts_on.get("/signup").text,
        accounts_on.post(
            "/signup", data={"email": "x@example.in", "password": "short"},
            follow_redirects=False,
        ).text,
        accounts_on.post(
            "/login", data={"email": "x@example.in", "password": "nope"},
            follow_redirects=False,
        ).text,
    ]
    for page in pages:
        assert _style_block(page) == baseline, "server substitution edited the CSS"


def test_every_panel_has_a_rule_that_hides_it(accounts_on):
    """Each panel must be named in the hiding rule, or it renders in every
    mode -- which is what "both forms at once" looks like."""
    css = _style_block(accounts_on.get("/").text)
    for panel in ("signin", "signup", "forgot", "reset"):
        assert f'.card:not([data-mode="{panel}"]) #panel-{panel}' in css


def test_the_page_renders_a_form_with_no_substitution_at_all():
    """The static file must stand on its own.

    Both failures this pins were real. Replacing a bare `data-mode="signin"`
    also rewrote the CSS selectors, so two forms rendered at once. Switching to
    a `<!--MODE-->` placeholder fixed that and introduced the opposite: the
    unsubstituted file matched no panel rule, so anything serving it without
    substitution -- an older process reading this newer file -- rendered a
    blank card. The defaults in the file are real values so neither can recur.
    """
    from rag_project.api import STATIC

    raw = (STATIC / "login.html").read_text()
    css = _style_block(raw)
    assert 'class="card" data-mode="signin" data-mail="on"' in raw
    # The mode the file ships with must be one the rules actually show.
    assert '.card:not([data-mode="signin"]) #panel-signin' in css
    assert "<!--MODE-->" not in raw, "a placeholder here means a blank card if unsubstituted"


def test_the_substitution_anchor_cannot_match_the_stylesheet():
    """Why the anchor is the whole `class="card" ...` string: the stylesheet
    selects on these attributes, and a shorter anchor matches selectors too."""
    from rag_project.api import STATIC

    css = _style_block((STATIC / "login.html").read_text())
    assert 'class="card"' not in css

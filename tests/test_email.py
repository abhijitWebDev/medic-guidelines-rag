"""Email verification and password reset.

No SMTP server is involved. Two different things are checked, and separating
them is the point:

* **The mailer's choreography** -- STARTTLS on 587, implicit TLS on 465, login
  with the SES credentials, one plain-text message -- is tested against a fake
  `smtplib` class. That is where the protocol mistakes would be.
* **The flows** are tested with `mailer.send` replaced by a recorder, so a test
  can read the link out of the message body and follow it, exactly as a person
  would from their inbox.
"""

from __future__ import annotations

import re
import time

import pytest
from fastapi.testclient import TestClient

from rag_project import mailer, security
from rag_project.api import app
from rag_project.config import get_settings
from rag_project.db import users

EMAIL = "clinician@example.in"
PASSWORD = "correct-horse-battery-staple"
NEW_PASSWORD = "an-entirely-different-password"


class Sent(list):
    """Captured messages. Each is (to, subject, body)."""

    @property
    def last_link(self) -> str:
        match = re.search(r"https?://\S+", self[-1][2])
        assert match, f"no link in the message body: {self[-1][2]!r}"
        return match.group(0)

    @property
    def last_token(self) -> str:
        return self.last_link.split("token=", 1)[1]


@pytest.fixture
def sent(monkeypatch) -> Sent:
    """SES configured, but nothing leaves the process."""
    box = Sent()
    monkeypatch.setenv("SES_SMTP_HOST", "email-smtp.eu-west-1.amazonaws.com")
    monkeypatch.setenv("MAIL_FROM", "noreply@example.in")
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("SCRYPT_N", "1024")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://assistant.example.in")
    get_settings.cache_clear()
    monkeypatch.setattr(mailer, "send", lambda to, subject, body: box.append((to, subject, body)))
    return box


@pytest.fixture
def client(sent) -> TestClient:
    return TestClient(app)


@pytest.fixture
def signed_up(client, sent) -> TestClient:
    r = client.post(
        "/signup", data={"email": EMAIL, "password": PASSWORD}, follow_redirects=False
    )
    assert r.status_code == 303, r.text
    return client


# --- the mailer ----------------------------------------------------------


class FakeSMTP:
    """Records the protocol steps smtplib would have performed."""

    instances: list[FakeSMTP] = []

    def __init__(self, host, port, timeout=None, context=None):
        self.host, self.port, self.timeout = host, port, timeout
        self.implicit_tls = context is not None
        self.started_tls = False
        self.login_args = None
        self.messages = []
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self, context=None):
        self.started_tls = True

    def login(self, user, password):
        self.login_args = (user, password)

    def send_message(self, msg):
        self.messages.append(msg)


@pytest.fixture
def smtp(monkeypatch):
    FakeSMTP.instances = []
    monkeypatch.setenv("SES_SMTP_HOST", "email-smtp.eu-west-1.amazonaws.com")
    monkeypatch.setenv("SES_SMTP_USER", "AKIAEXAMPLE")
    monkeypatch.setenv("SES_SMTP_PASSWORD", "ses-smtp-secret")
    monkeypatch.setenv("MAIL_FROM", "noreply@example.in")
    get_settings.cache_clear()
    monkeypatch.setattr(mailer.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(mailer.smtplib, "SMTP_SSL", FakeSMTP)
    return FakeSMTP


def test_port_587_upgrades_with_starttls(smtp, monkeypatch):
    monkeypatch.setenv("SES_SMTP_PORT", "587")
    get_settings.cache_clear()
    mailer.send("someone@example.in", "Subject", "Body")

    [conn] = smtp.instances
    assert conn.port == 587
    assert conn.started_tls, "credentials would have gone over a plaintext socket"
    assert conn.login_args == ("AKIAEXAMPLE", "ses-smtp-secret")


def test_port_465_uses_implicit_tls(smtp, monkeypatch):
    """The port picks the handshake; a separate flag could disagree with it and
    hang rather than fail."""
    monkeypatch.setenv("SES_SMTP_PORT", "465")
    get_settings.cache_clear()
    mailer.send("someone@example.in", "Subject", "Body")

    [conn] = smtp.instances
    assert conn.implicit_tls
    assert not conn.started_tls, "STARTTLS on an already-encrypted connection"


def test_the_message_is_plain_text_and_addressed_from_the_verified_sender(smtp):
    mailer.send("someone@example.in", "Confirm your email", "Body text")
    [msg] = smtp.instances[0].messages
    assert msg["To"] == "someone@example.in"
    assert msg["Subject"] == "Confirm your email"
    assert "noreply@example.in" in msg["From"]
    assert msg.get_content_type() == "text/plain"


def test_a_refused_send_raises_rather_than_being_swallowed(monkeypatch, smtp):
    import smtplib as real_smtplib

    def refuse(*a, **kw):
        # What SES returns in the sandbox for an unverified recipient.
        raise real_smtplib.SMTPRecipientsRefused({"x@example.in": (554, b"not verified")})

    monkeypatch.setattr(FakeSMTP, "send_message", refuse)
    with pytest.raises(mailer.MailError):
        mailer.send("x@example.in", "Subject", "Body")


def test_no_configuration_means_no_mailer(monkeypatch):
    monkeypatch.setenv("SES_SMTP_HOST", "")
    get_settings.cache_clear()
    assert not mailer.enabled()
    assert not get_settings().verification_required
    with pytest.raises(mailer.MailError):
        mailer.send("x@example.in", "Subject", "Body")


# --- verification --------------------------------------------------------


def test_signup_sends_a_confirmation_link(signed_up, sent):
    [(to, subject, _body)] = sent
    assert to == EMAIL
    assert "confirm" in subject.lower()
    assert sent.last_link.startswith("https://assistant.example.in/verify?token=")


def test_an_unverified_account_cannot_ask(signed_up):
    r = signed_up.post("/api/ask", json={"query": "How is TB diagnosed?"})
    assert r.status_code == 403
    assert "confirm" in r.json()["detail"].lower()


def test_an_unverified_account_can_still_use_everything_else(signed_up):
    """Locking the account out entirely makes it harder to finish setting up,
    not safer."""
    assert signed_up.get("/").status_code == 200
    assert signed_up.get("/api/info").status_code == 200
    assert signed_up.get("/api/history").status_code == 200


def test_a_blocked_question_does_not_spend_quota(signed_up, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_WINDOW", "2")
    get_settings.cache_clear()
    security.reset_rate_limits()

    for _ in range(5):
        assert signed_up.post("/api/ask", json={"query": "How is TB diagnosed?"}).status_code == 403

    # Now verify, and check the allowance was never touched by the refusals.
    users.mark_verified(signed_up.get("/api/info").json()["user"]["id"])
    r = signed_up.post("/api/ask", json={"query": "Should I take rifampicin for my cough?"})
    assert r.status_code == 200
    assert r.headers["X-RateLimit-Remaining"] == "1", "refused questions ate the quota"


def test_following_the_link_verifies_and_signs_in(client, sent):
    client.post("/signup", data={"email": EMAIL, "password": PASSWORD},
                follow_redirects=False)
    client.cookies.clear()  # open the link in a fresh browser, as people do

    r = client.get(f"/verify?token={sent.last_token}", follow_redirects=False)
    assert r.status_code == 303
    assert r.cookies.get(security.COOKIE_NAME), "verifying should not require signing in again"

    assert client.get("/api/info").json()["verified"] is True
    assert client.post(
        "/api/ask", json={"query": "Should I take rifampicin for my cough?"}
    ).status_code == 200


def test_verifying_twice_is_harmless(signed_up, sent):
    token = sent.last_token
    assert signed_up.get(f"/verify?token={token}", follow_redirects=False).status_code == 303
    first = signed_up.get("/api/info").json()["user"]["verified_at"]
    assert signed_up.get(f"/verify?token={token}", follow_redirects=False).status_code == 303
    assert signed_up.get("/api/info").json()["user"]["verified_at"] == first, \
        "a forwarded link rewrote when the account was confirmed"


def test_a_forged_or_expired_link_is_refused(client):
    assert client.get("/verify?token=garbage", follow_redirects=False).status_code == 400
    stale = security.issue_link_token(
        security.PURPOSE_VERIFY, "a" * 32, 3600, now=time.time() - 10**6
    )
    assert client.get(f"/verify?token={stale}", follow_redirects=False).status_code == 400


def test_resending_produces_a_working_link(signed_up, sent):
    assert signed_up.post("/api/resend-verification").json()["sent"] is True
    assert len(sent) == 2
    assert signed_up.get(
        f"/verify?token={sent.last_token}", follow_redirects=False
    ).status_code == 303


def test_resending_is_metered_on_its_own_counter(signed_up, sent, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_WINDOW", "2")
    get_settings.cache_clear()
    security.reset_rate_limits()

    for _ in range(2):
        signed_up.post("/api/resend-verification")
    assert signed_up.post("/api/resend-verification").status_code == 429


def test_a_failed_send_still_leaves_a_usable_account(client, monkeypatch):
    """SES rejects unverified recipients while an account is in the sandbox.
    Turning the signup away would leave the address taken and unreachable."""
    def refuse(*a, **kw):
        raise mailer.MailError("554 recipient not verified")

    monkeypatch.setattr(mailer, "send", refuse)
    r = client.post("/signup", data={"email": EMAIL, "password": PASSWORD},
                    follow_redirects=False)
    assert r.status_code == 303, "the account was lost along with the email"
    assert client.get("/api/info").json()["verified"] is False
    assert client.post("/api/resend-verification").status_code == 502


def test_verification_is_off_where_mail_is_not_configured(monkeypatch):
    """An instance that cannot send must not demand a link it will never
    deliver -- that is a deployment nobody can sign into."""
    monkeypatch.setenv("SES_SMTP_HOST", "")
    monkeypatch.setenv("MAIL_FROM", "")
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("SCRYPT_N", "1024")
    get_settings.cache_clear()

    c = TestClient(app)
    c.post("/signup", data={"email": EMAIL, "password": PASSWORD}, follow_redirects=False)
    assert c.get("/api/info").json()["verification_required"] is False
    assert c.post(
        "/api/ask", json={"query": "Should I take rifampicin for my cough?"}
    ).status_code == 200


# --- password reset ------------------------------------------------------


def test_forgot_sends_a_reset_link(signed_up, sent):
    sent.clear()
    r = signed_up.post("/forgot", data={"email": EMAIL}, follow_redirects=False)
    assert r.status_code == 200
    assert sent.last_link.startswith("https://assistant.example.in/reset?token=")


def test_forgot_says_the_same_thing_for_an_unknown_address(signed_up, sent):
    """A reset form that confirms which addresses exist is a free enumeration
    endpoint."""
    known = signed_up.post("/forgot", data={"email": EMAIL}, follow_redirects=False)
    sent.clear()
    unknown = signed_up.post(
        "/forgot", data={"email": "nobody@example.in"}, follow_redirects=False
    )
    assert known.status_code == unknown.status_code
    assert "on its way" in unknown.text
    assert sent == [], "an email went to an address with no account"


def test_the_link_sets_a_new_password_and_signs_in(signed_up, sent):
    signed_up.post("/forgot", data={"email": EMAIL}, follow_redirects=False)
    token = sent.last_token

    assert signed_up.get(f"/reset?token={token}").status_code == 200
    r = signed_up.post(
        "/reset", data={"token": token, "password": NEW_PASSWORD}, follow_redirects=False
    )
    assert r.status_code == 303

    signed_up.cookies.clear()
    assert signed_up.post(
        "/login", data={"email": EMAIL, "password": NEW_PASSWORD}, follow_redirects=False
    ).status_code == 303
    assert signed_up.post(
        "/login", data={"email": EMAIL, "password": PASSWORD}, follow_redirects=False
    ).status_code == 401, "the old password still works"


def test_a_reset_link_works_once(signed_up, sent):
    signed_up.post("/forgot", data={"email": EMAIL}, follow_redirects=False)
    token = sent.last_token
    signed_up.post("/reset", data={"token": token, "password": NEW_PASSWORD},
                   follow_redirects=False)

    again = signed_up.post(
        "/reset", data={"token": token, "password": "yet-another-password"},
        follow_redirects=False,
    )
    assert again.status_code == 400
    assert "already been used" in again.text


def test_resetting_signs_out_every_other_session(client, sent):
    """What someone resetting because they think they were compromised is
    actually asking for."""
    client.post("/signup", data={"email": EMAIL, "password": PASSWORD},
                follow_redirects=False)
    stolen = client.cookies.get(security.COOKIE_NAME)

    other = TestClient(app)
    other.post("/login", data={"email": EMAIL, "password": PASSWORD},
               follow_redirects=False)
    other.post("/forgot", data={"email": EMAIL}, follow_redirects=False)
    other.post("/reset", data={"token": sent.last_token, "password": NEW_PASSWORD},
               follow_redirects=False)

    assert other.get("/api/info").status_code == 200, "the resetting session was logged out too"

    held = TestClient(app)
    held.cookies.set(security.COOKIE_NAME, stolen)
    assert held.get("/api/info").status_code == 401, "a session from before the reset survived"


def test_a_reset_confirms_the_address(client, sent):
    """Reaching the mailbox proves the address as surely as a confirmation link
    does."""
    client.post("/signup", data={"email": EMAIL, "password": PASSWORD},
                follow_redirects=False)
    assert client.get("/api/info").json()["verified"] is False

    client.post("/forgot", data={"email": EMAIL}, follow_redirects=False)
    client.post("/reset", data={"token": sent.last_token, "password": NEW_PASSWORD},
                follow_redirects=False)
    assert client.get("/api/info").json()["verified"] is True


def test_a_short_new_password_is_refused(signed_up, sent):
    signed_up.post("/forgot", data={"email": EMAIL}, follow_redirects=False)
    r = signed_up.post(
        "/reset", data={"token": sent.last_token, "password": "short"},
        follow_redirects=False,
    )
    assert r.status_code == 400
    assert "at least 8" in r.text


def test_a_verification_link_cannot_reset_a_password(signed_up, sent):
    """The purpose is inside the signature. Confirmation links are handed out
    to addresses nobody has proven yet."""
    verify_token = sent.last_token
    assert signed_up.get(f"/reset?token={verify_token}").status_code == 400
    assert signed_up.post(
        "/reset", data={"token": verify_token, "password": NEW_PASSWORD},
        follow_redirects=False,
    ).status_code == 400


def test_reset_is_unavailable_without_a_mailer(monkeypatch):
    monkeypatch.setenv("SES_SMTP_HOST", "")
    monkeypatch.setenv("MAIL_FROM", "")
    monkeypatch.setenv("AUTH_ENABLED", "true")
    get_settings.cache_clear()

    c = TestClient(app)
    r = c.get("/forgot")
    assert r.status_code == 503
    assert "no email configured" in r.text
    assert 'data-mail="off"' in c.get("/").text, "the page still offered a dead link"


def test_the_recovery_panels_render_alone_too(signed_up, sent):
    """Same substitution hazard as sign-in/sign-up: the reset page is reached
    from a link, so nobody would notice a second form on it quickly."""
    def card(html):
        return [line for line in html.splitlines() if '<div class="card"' in line][0]

    assert 'data-mode="forgot"' in card(signed_up.get("/forgot").text)

    signed_up.post("/forgot", data={"email": EMAIL}, follow_redirects=False)
    reset_page = signed_up.get(f"/reset?token={sent.last_token}").text
    assert 'data-mode="reset"' in card(reset_page)
    # The token is carried in a hidden field; the page is useless without it.
    assert f'value="{sent.last_token}"' in reset_page


def test_the_forgot_link_is_offered_where_mail_works(client):
    """The signed-out page is the one that carries it; test_reset_is_unavailable
    _without_a_mailer covers the other direction."""
    assert 'data-mail="on"' in client.get("/").text


def test_the_suite_cannot_reach_a_real_ses_endpoint():
    """conftest blanks the SES settings for every test.

    Two failures this prevents, both of which happened. `.env` carries live
    credentials once someone sets SES up, so a signup test opened a real SMTP
    connection to Amazon; and because configuring mail is what turns
    verification on, a suite that passed on a laptop without MAIL_FROM started
    returning 403 on one that had it.
    """
    s = get_settings()
    assert not s.ses_smtp_host
    assert not s.mail_from
    assert not s.mail_enabled
    assert not s.verification_required

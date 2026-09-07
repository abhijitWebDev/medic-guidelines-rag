"""Accounts and per-user history.

Everything here runs against the SQLite backend, which conftest points at a
fresh file per test. That is deliberate rather than a compromise: the schema
and every statement are shared with Postgres, so exercising them here catches
the SQL bugs, and doing it without a server means these run on a laptop with
no network. What SQLite cannot prove is dialect behaviour -- see
test_placeholder_translation for the one seam that differs.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rag_project import passwords
from rag_project.api import app
from rag_project.config import get_settings
from rag_project.db import AccountError, get_db, history, users
from rag_project.db.engine import PostgresDatabase
from rag_project.models import Claim, RefusalReason, Response

PASSWORD = "correct-horse-battery-staple"


@pytest.fixture(autouse=True)
def _cheap_hashing(monkeypatch):
    """The KDF's cost is the point of it, and irrelevant to these tests."""
    monkeypatch.setenv("SCRYPT_N", "1024")
    get_settings.cache_clear()


# --- passwords -----------------------------------------------------------


def test_a_password_verifies_against_its_own_hash():
    encoded = passwords.hash_password(PASSWORD)
    assert passwords.verify_password(PASSWORD, encoded)
    assert not passwords.verify_password(PASSWORD + "x", encoded)


def test_the_hash_is_not_the_password():
    assert PASSWORD not in passwords.hash_password(PASSWORD)


def test_the_same_password_hashes_differently_every_time():
    """Salted. Otherwise identical passwords are visibly identical in a dump,
    and one cracked hash cracks every account that shares it."""
    assert passwords.hash_password(PASSWORD) != passwords.hash_password(PASSWORD)


def test_the_hash_carries_its_own_parameters(monkeypatch):
    """So the cost can be raised later without invalidating everyone."""
    monkeypatch.setenv("SCRYPT_N", "1024")
    get_settings.cache_clear()
    old = passwords.hash_password(PASSWORD)

    monkeypatch.setenv("SCRYPT_N", "2048")
    get_settings.cache_clear()
    assert passwords.verify_password(PASSWORD, old), \
        "raising the cost must not lock existing users out"


def test_a_malformed_hash_is_rejected_not_crashed_on():
    for junk in ("", "x", "scrypt$$$$", "bcrypt$1$2$3$4$5", "scrypt$a$b$c$d$e"):
        assert passwords.verify_password(PASSWORD, junk) is False


# --- accounts ------------------------------------------------------------


def test_create_and_authenticate():
    user = users.create("doc@example.in", PASSWORD)
    assert users.authenticate("doc@example.in", PASSWORD).id == user.id
    assert users.authenticate("doc@example.in", "wrong") is None
    assert users.authenticate("nobody@example.in", PASSWORD) is None


def test_email_is_normalised():
    user = users.create("  Doc@Example.IN  ", PASSWORD)
    assert user.email == "doc@example.in"
    assert users.authenticate("DOC@EXAMPLE.IN", PASSWORD).id == user.id


def test_duplicate_email_raises():
    users.create("doc@example.in", PASSWORD)
    with pytest.raises(AccountError):
        users.create("DOC@example.in", PASSWORD)


@pytest.mark.parametrize(
    "email", ["", "no-at-sign", "no@domain", "two@@at.in", "spaces in@mail.in"]
)
def test_bad_emails_are_refused(email):
    with pytest.raises(AccountError):
        users.create(email, PASSWORD)


def test_short_passwords_are_refused():
    with pytest.raises(AccountError):
        users.create("doc@example.in", "1234567")


def test_absurdly_long_passwords_are_refused():
    """scrypt hashes the whole input; without a cap it is a free memory burn."""
    with pytest.raises(AccountError):
        users.create("doc@example.in", "x" * 100_000)


def test_a_user_record_never_carries_the_hash():
    """`public()` feeds /api/info, which is JSON in a browser."""
    user = users.create("doc@example.in", PASSWORD)
    assert "password" not in str(user.public()).lower()


def test_deleting_a_user_deletes_their_history():
    user = users.create("doc@example.in", PASSWORD)
    history.save(user.id, _response("How is TB diagnosed?"))
    assert history.list_for(user.id)

    users.delete(user.id)
    assert users.get(user.id) is None
    assert history.list_for(user.id) == []


# --- history -------------------------------------------------------------


def _response(query: str, answered: bool = True) -> Response:
    return Response(
        query=query,
        answered=answered,
        answer="Sputum smear microscopy is the primary modality [C1].",
        claims=[Claim(text="Sputum smear microscopy.", chunk_ids=["c1"])],
        citations=[{"marker": "C1", "chunk_id": "c1", "title": "TB", "pages": "12"}],
        refusal_reason=None if answered else RefusalReason.OUT_OF_DOMAIN,
        top_score=7.5,
        trace={"stages": ["intent", "retrieve"], "confidence": {"top_score": 7.5}},
    )


@pytest.fixture
def user():
    return users.create("doc@example.in", PASSWORD)


def test_a_saved_turn_round_trips(user):
    saved = history.save(user.id, _response("How is TB diagnosed?"))
    [turn] = history.list_for(user.id)
    assert turn.id == saved.id
    assert turn.query == "How is TB diagnosed?"
    assert turn.answered is True
    assert turn.citations[0]["marker"] == "C1"
    assert turn.top_score == 7.5


def test_a_refusal_is_kept_with_its_reason(user):
    history.save(user.id, _response("What is the capital of France?", answered=False))
    [turn] = history.list_for(user.id)
    assert turn.answered is False
    assert turn.refusal_reason == "out_of_domain"


def test_the_trace_is_not_stored(user):
    """It is debug data, it is the largest part of a Response, and nothing in
    the history UI reads it."""
    history.save(user.id, _response("How is TB diagnosed?"))
    row = get_db().query_one("SELECT * FROM history")
    assert "stages" not in str(row)


def test_history_is_newest_first(user):
    for i in range(3):
        history.save(user.id, _response(f"question {i}"))
    assert [t.query for t in history.list_for(user.id)] == [
        "question 2", "question 1", "question 0",
    ]


def test_history_is_private_to_its_owner(user):
    other = users.create("other@example.in", PASSWORD)
    saved = history.save(user.id, _response("mine"))
    history.save(other.id, _response("theirs"))

    assert [t.query for t in history.list_for(user.id)] == ["mine"]
    assert history.get(other.id, saved.id) is None, \
        "a turn id must not be an authorisation"
    assert history.delete(other.id, saved.id) is False
    assert history.get(user.id, saved.id) is not None, "deleted someone else's turn"


def test_deleting_one_turn(user):
    a = history.save(user.id, _response("first"))
    history.save(user.id, _response("second"))
    assert history.delete(user.id, a.id) is True
    assert [t.query for t in history.list_for(user.id)] == ["second"]


def test_clearing_history(user):
    for i in range(3):
        history.save(user.id, _response(f"q{i}"))
    assert history.clear(user.id) == 3
    assert history.list_for(user.id) == []


def test_history_is_capped_per_user(monkeypatch, user):
    """An unbounded per-user log is a slow leak nobody notices."""
    monkeypatch.setenv("HISTORY_MAX_ITEMS", "3")
    get_settings.cache_clear()
    for i in range(6):
        history.save(user.id, _response(f"q{i}"))
    turns = history.list_for(user.id)
    assert len(turns) == 3
    assert [t.query for t in turns] == ["q5", "q4", "q3"], "kept the wrong end"


def test_pagination_walks_backwards(user):
    for i in range(5):
        history.save(user.id, _response(f"q{i}"))
    first = history.list_for(user.id, limit=2)
    assert [t.query for t in first] == ["q4", "q3"]
    second = history.list_for(user.id, limit=2, before=first[-1].created_at)
    assert [t.query for t in second] == ["q2", "q1"]


# --- the HTTP surface ----------------------------------------------------


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.setenv("AUTH_ENABLED", "true")
    get_settings.cache_clear()
    c = TestClient(app)
    r = c.post(
        "/signup", data={"email": "doc@example.in", "password": PASSWORD},
        follow_redirects=False,
    )
    assert r.status_code == 303, r.text
    return c


def test_asking_a_question_files_it_under_the_asker(client):
    q = "Should I take rifampicin for my cough?"  # refused at gate 1, no network
    assert client.post("/api/ask", json={"query": q}).status_code == 200

    turns = client.get("/api/history").json()["turns"]
    assert [t["query"] for t in turns] == [q]
    assert turns[0]["answered"] is False


def test_history_endpoints_reject_another_account(client):
    client.post("/api/ask", json={"query": "Should I take rifampicin for my cough?"})
    turn_id = client.get("/api/history").json()["turns"][0]["id"]

    client.cookies.clear()
    client.post(
        "/signup", data={"email": "intruder@example.in", "password": PASSWORD},
        follow_redirects=False,
    )
    assert client.get("/api/history").json()["turns"] == []
    assert client.get(f"/api/history/{turn_id}").status_code == 404
    assert client.delete(f"/api/history/{turn_id}").status_code == 404


def test_a_stored_turn_is_reopenable_with_its_sources(client):
    client.post("/api/ask", json={"query": "Should I take rifampicin for my cough?"})
    turn_id = client.get("/api/history").json()["turns"][0]["id"]

    turn = client.get(f"/api/history/{turn_id}").json()
    assert turn["answer"]
    assert "citations" in turn, "an answer without its sources is not reusable here"
    assert "trace" not in turn


def test_deleting_through_the_api(client):
    client.post("/api/ask", json={"query": "Should I take rifampicin for my cough?"})
    turn_id = client.get("/api/history").json()["turns"][0]["id"]
    assert client.delete(f"/api/history/{turn_id}").json() == {"deleted": 1}
    assert client.get("/api/history").json()["turns"] == []


def test_clearing_through_the_api(client):
    for q in ("Should I take rifampicin?", "Is my chest pain serious?"):
        client.post("/api/ask", json={"query": q})
    assert client.delete("/api/history").json()["deleted"] == 2
    assert client.get("/api/history").json()["turns"] == []


def test_pagination_hands_back_a_cursor(client):
    for i in range(3):
        client.post("/api/ask", json={"query": f"Should I take drug {i} for my cough?"})
    page = client.get("/api/history?limit=2").json()
    assert len(page["turns"]) == 2
    assert page["next_before"], "a full page must say how to ask for the next one"
    assert client.get("/api/history?limit=2&before=" + page["next_before"]).json()[
        "turns"
    ][0]["query"].endswith("drug 0 for my cough?")


def test_an_unavailable_store_does_not_lose_the_answer(client, monkeypatch):
    """History is a convenience; the answer is the product. A database that has
    gone away must not turn an answered question into an error."""
    from rag_project.db import StorageError, history as history_mod

    def dead(*a, **kw):
        raise StorageError("connection refused")

    monkeypatch.setattr(history_mod, "save", dead)
    r = client.post(
        "/api/ask",
        json={"query": "Should I take rifampicin for my cough?", "trace": True},
    )
    assert r.status_code == 200
    assert r.json()["trace"]["history"] == "unsaved", \
        "a dropped turn must be visible, not silent"


def test_info_reports_who_is_signed_in(client):
    body = client.get("/api/info").json()
    assert body["user"]["email"] == "doc@example.in"
    assert body["history_enabled"] is True
    assert "id" in body["user"]


def test_an_open_instance_has_no_history(monkeypatch):
    """With no accounts there is nobody to file a turn under, and the UI hides
    the panel on exactly this flag."""
    monkeypatch.setenv("AUTH_ENABLED", "false")
    get_settings.cache_clear()
    c = TestClient(app)
    assert c.get("/api/info").json()["history_enabled"] is False
    assert c.get("/api/history").status_code == 401


# --- the dialect seam ----------------------------------------------------


def test_placeholder_translation():
    """The one thing SQLite cannot exercise: every statement is written in
    SQLite's '?' style and rewritten for psycopg. A statement containing a
    literal '?' would be corrupted by this, which is why none may."""
    assert PostgresDatabase._translate("SELECT * FROM t WHERE a = ? AND b = ?") == (
        "SELECT * FROM t WHERE a = %s AND b = %s"
    )


def test_prepared_statements_are_disabled_for_the_pooler():
    """Pinned because the failure it prevents only appears in production.

    psycopg3 promotes a repeated statement to a server-side prepared one. Neon
    and Supabase both put PgBouncer in front of Postgres, where the next
    execution can land on a different backend connection and the prepared
    statement is not there -- so this breaks after the app has been up a while,
    under load, and never in a test.
    """
    import inspect

    src = inspect.getsource(PostgresDatabase.__init__)
    assert '"prepare_threshold": None' in src
    assert "min_size=0" in src, "a serverless cold start must not open connections eagerly"


# --- migrating a database that already exists ----------------------------


def _rebuild_as_pre_migration() -> None:
    """Recreate the schema as it was before verification existed.

    Runs through the active backend, so this exercises whichever of Postgres
    or SQLite the suite was pointed at -- the ALTER-already-applied detection
    is the one part of the migration that behaves differently between them.
    """
    from rag_project.db import reset_db

    db = get_db()
    db.execute("DROP TABLE IF EXISTS history")
    db.execute("DROP TABLE IF EXISTS users")
    db.execute(
        "CREATE TABLE users (id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE,"
        " password_hash TEXT NOT NULL, created_at TEXT NOT NULL)"
    )
    db.execute(
        "INSERT INTO users (id, email, password_hash, created_at)"
        " VALUES ('a0000000000000000000000000000000', 'old@example.in', 'scrypt$x',"
        " '2026-09-01T10:00:00.000+00:00')"
    )
    reset_db()  # the next get_db() creates the schema and migrates


def test_an_existing_database_gains_the_new_columns():
    _rebuild_as_pre_migration()
    row = get_db().query_one("SELECT * FROM users")
    assert "verified_at" in row
    assert "password_changed_at" in row


def test_accounts_that_predate_verification_are_not_locked_out():
    """They signed up before the rule existed. Enforcing it retroactively
    would lock out real people to satisfy a check they never saw."""
    _rebuild_as_pre_migration()
    [existing] = users.all_users()
    assert existing.is_verified
    assert existing.verified_at == existing.created_at


def test_the_backfill_does_not_run_again_on_later_starts():
    """The subtle one. This list runs on every cold start, so a backfill that
    is not conditional on the column being new would verify every account that
    had signed up since -- which is exactly the check it enforces."""
    from rag_project.db import reset_db

    _rebuild_as_pre_migration()
    get_db()  # apply the migration

    fresh = users.create("after@example.in", PASSWORD)
    assert not fresh.is_verified

    for _ in range(2):  # two more cold starts
        reset_db()
        get_db()

    assert not users.get(fresh.id).is_verified, "a restart verified an account for free"
    assert users.get("a0000000000000000000000000000000").is_verified

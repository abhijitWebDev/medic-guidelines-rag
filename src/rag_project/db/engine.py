"""Connections, and the two backends behind them.

This is the only store in the project that fails **closed**. cache.py degrades
to "compute it normally" when Upstash blips, because nothing it holds is a
source of truth. Here the opposite is required: if we cannot reach the database
we cannot tell whether a session belongs to a real user, and serving the app
anyway would mean serving it to nobody in particular. Every failure raises
`StorageError`, and the API turns that into a 503 rather than a guess.

Two backends, one schema:

* **Postgres** (`DATABASE_URL`) is the deployment target -- Neon, Supabase, or
  any plain Postgres. Setting it is also what turns accounts on.
* **SQLite** is the fallback, so a fresh clone, the test suite, and offline
  work all run without provisioning anything. It is not a deployment option:
  on Vercel the filesystem is ephemeral, so accounts written to it vanish at
  the next cold start. That is loud (everyone is logged out) rather than
  silent, which is the right way for a missing DATABASE_URL to fail.

Callers never see either class. They ask for `get_db()` and get something with
`execute`, `query` and `query_one` -- which is the whole interface the
repositories in this package are written against.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..config import ROOT, get_settings
from . import schema


class StorageError(RuntimeError):
    """The store is unreachable, or rejected the statement."""


class DuplicateKey(StorageError):
    """A unique constraint refused the row -- an email that already exists."""


class Database:
    """A thin query interface. Subclasses supply the connection handling."""

    dialect: str

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        """Run a statement; returns the number of rows it touched."""
        raise NotImplementedError

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict]:
        raise NotImplementedError

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> dict | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def _create_schema(self) -> None:
        for stmt in schema.for_dialect(self.dialect):
            try:
                self.execute(stmt)
            except StorageError:
                # Two instances starting at once can collide inside CREATE ...
                # IF NOT EXISTS -- Postgres raises a duplicate-key error from
                # its own catalogue rather than treating it as "already there".
                # The table exists either way, and the next real statement will
                # fail honestly if it does not.
                pass
        self._migrate()

    def _migrate(self) -> None:
        """Apply the column additions in schema.MIGRATIONS.

        The failure of the ALTER is the test for whether the column was already
        there -- neither backend gives a portable "add if missing" -- and it is
        also what decides whether the backfill runs. That coupling is the
        point: a backfill must touch rows exactly once, on the deployment that
        introduces the column, and never again on the cold starts after it.
        """
        for _name, alter, backfill in schema.MIGRATIONS:
            try:
                self.execute(alter)
            except StorageError:
                continue  # already applied
            if backfill:
                try:
                    self.execute(backfill)
                except StorageError:
                    # The column exists but the backfill did not run. Loud
                    # rather than silent: the rows it would have touched are
                    # the ones about to behave unexpectedly.
                    print(
                        f"warning: schema backfill for {_name} failed; "
                        "existing rows may need it applied by hand",
                        file=sys.stderr,
                    )


class SqliteDatabase(Database):
    """One connection, guarded by a lock.

    Connecting per call would be simpler, but it cannot hold a `:memory:`
    database open across calls, and it would reopen the file on every query.
    """

    dialect = "sqlite"

    def __init__(self, path: str) -> None:
        self._lock = threading.Lock()
        try:
            self._conn = sqlite3.connect(path, check_same_thread=False)
        except sqlite3.Error as e:  # an unwritable directory, mostly
            raise StorageError(f"cannot open the SQLite store at {path}: {e}") from e
        self._conn.row_factory = sqlite3.Row
        # Off by default in SQLite, and history rows must not outlive the user
        # they belong to.
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._create_schema()

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        with self._lock:
            try:
                cur = self._conn.execute(sql, tuple(params))
                self._conn.commit()
                return cur.rowcount
            except sqlite3.IntegrityError as e:
                self._conn.rollback()
                raise DuplicateKey(str(e)) from e
            except sqlite3.Error as e:
                self._conn.rollback()
                raise StorageError(str(e)) from e

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict]:
        with self._lock:
            try:
                return [dict(r) for r in self._conn.execute(sql, tuple(params)).fetchall()]
            except sqlite3.Error as e:
                raise StorageError(str(e)) from e


class PostgresDatabase(Database):
    """Pooled psycopg connections, tuned for a pooler in front of the database.

    Both managed options this project targets put PgBouncer between the app and
    Postgres -- Neon's `-pooler` host, Supabase's port 6543 -- and two settings
    below exist because of it.

    `min_size=0`: on a serverless instance the pool is created during a cold
    start and may never be used again. Opening connections eagerly would spend
    a handshake per instance against a database that counts them. Warm
    instances still reuse whatever the pool holds, which is the case that
    actually repeats.

    `prepare_threshold=None`: psycopg3 promotes a statement to a server-side
    prepared statement after it has run a few times. Under a pooler in
    transaction mode the next execution can land on a different backend
    connection, where that prepared statement does not exist -- a failure that
    only appears under load, after the app has been running a while, which is
    the worst possible time to discover it. These queries are trivial and the
    saving is noise, so preparation is switched off rather than gambled on.
    """

    dialect = "postgres"

    def __init__(self, url: str) -> None:
        try:
            from psycopg import errors as pg_errors
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool
        except ImportError as e:  # pragma: no cover - depends on the install
            raise StorageError(
                "DATABASE_URL is set but psycopg is not installed; "
                "run `uv sync` (psycopg is a project dependency)"
            ) from e

        s = get_settings()
        self._unique_violation = pg_errors.UniqueViolation
        try:
            self._pool = ConnectionPool(
                url,
                min_size=0,
                max_size=s.db_max_connections,
                timeout=s.db_timeout_s,
                kwargs={"row_factory": dict_row, "prepare_threshold": None},
                open=False,
            )
            self._pool.open()
        except Exception as e:
            raise StorageError(f"cannot reach the database: {e}") from e

        self._create_schema()

    @staticmethod
    def _translate(sql: str) -> str:
        """SQLite placeholders to psycopg ones.

        Safe only because no statement in this package contains a literal '?'.
        Keep it that way: parameterise, never interpolate.
        """
        return sql.replace("?", "%s")

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        try:
            with self._pool.connection() as conn:
                cur = conn.execute(self._translate(sql), tuple(params))
                return cur.rowcount
        except self._unique_violation as e:
            raise DuplicateKey(str(e)) from e
        except Exception as e:
            raise StorageError(str(e)) from e

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict]:
        try:
            with self._pool.connection() as conn:
                return list(conn.execute(self._translate(sql), tuple(params)).fetchall())
        except Exception as e:
            raise StorageError(str(e)) from e


def _sqlite_path() -> str:
    """Where the fallback store lives.

    data/ is the natural home and where a developer expects to find it. A
    read-only deployment filesystem makes that impossible, so fall back to the
    temp directory rather than failing to start -- an ephemeral store that logs
    everyone out is still better than an app that will not boot while someone
    works out why DATABASE_URL was missing.
    """
    configured = get_settings().sqlite_path
    if configured:
        return configured
    preferred = ROOT / "data" / "app.db"
    if os.access(preferred.parent, os.W_OK):
        return str(preferred)
    return str(Path(os.environ.get("TMPDIR", "/tmp")) / "rag-project-app.db")


_db: Database | None = None
_lock = threading.Lock()


def get_db() -> Database:
    global _db
    with _lock:
        if _db is None:
            url = get_settings().database_url
            _db = PostgresDatabase(url) if url else SqliteDatabase(_sqlite_path())
        return _db


def reset_db() -> None:
    """Drop the singleton. For tests and for reconfiguring at runtime."""
    global _db
    with _lock:
        _db = None

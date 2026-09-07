"""The database schema, in one place.

Two tables, created on first connection. `CREATE TABLE IF NOT EXISTS` is
idempotent, runs in milliseconds, and cannot drift from the code because it
*is* the code.

Columns added after the fact are the case that breaks, and `MIGRATIONS` below
is the answer to it -- a list of statements run in order, each safe to run
against a database that already has them. Two rules keep that honest:

1. **Never edit a statement that has shipped.** A deployed database has
   already run it; changing it only changes what a *fresh* database gets, and
   the two silently diverge. Append a new statement instead.
2. **Backfills are conditional on the column actually being new**, because
   this list runs on every cold start. `UPDATE users SET verified_at = ...`
   run unconditionally would verify every account that had signed up since --
   which is precisely the check it was added to enforce.

The dialect seam lives here too. Every statement in this package is written in
SQLite's `?` placeholder style and rewritten for psycopg (see engine.py); the
only other difference between the two backends is the handful of type names in
`TYPES`.
"""

from __future__ import annotations

# Types that spell differently in the two dialects.
#
# `created_at` is deliberately not among them: it is an ISO-8601 UTC string in
# both. Text sorts correctly in that format, needs no parsing on the way out,
# survives a dump/restore between the two backends unchanged, and stays
# readable in whatever table viewer someone opens the database with.
#
# `citations` is TEXT rather than JSONB for a related reason -- one code path,
# no adapters, and nothing in this application queries *into* the JSON. Move it
# to JSONB the day something needs to filter on a citation's contents.
TYPES = {
    "postgres": {"BOOL": "BOOLEAN", "FLOAT": "DOUBLE PRECISION"},
    "sqlite": {"BOOL": "INTEGER", "FLOAT": "REAL"},
}

USERS = """
CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    email         TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL
)
"""

HISTORY = """
CREATE TABLE IF NOT EXISTS history (
    id             TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    query          TEXT NOT NULL,
    answer         TEXT NOT NULL,
    answered       {BOOL} NOT NULL,
    refusal_reason TEXT,
    top_score      {FLOAT},
    citations      TEXT NOT NULL DEFAULT '[]',
    created_at     TEXT NOT NULL
)
"""

# Every history read is "this user's turns, newest first". Without this index
# that is a full table scan the moment the table stops being small.
HISTORY_INDEX = """
CREATE INDEX IF NOT EXISTS history_user_time
    ON history (user_id, created_at DESC)
"""

STATEMENTS = (USERS, HISTORY, HISTORY_INDEX)


def for_dialect(dialect: str) -> list[str]:
    return [stmt.format(**TYPES[dialect]) for stmt in STATEMENTS]


# --- migrations ----------------------------------------------------------
#
# Each entry is (name, ALTER statement, backfill or None). The backfill runs
# only when the ALTER actually added something -- see rule 2 above.
#
# Postgres has ADD COLUMN IF NOT EXISTS and SQLite does not, so the caller
# detects "already there" from the error either way rather than branching here.

MIGRATIONS = (
    (
        "users.verified_at",
        "ALTER TABLE users ADD COLUMN verified_at TEXT",
        # Accounts that existed before verification was introduced predate the
        # requirement. Blocking them would lock out real people to enforce a
        # rule that did not exist when they signed up.
        "UPDATE users SET verified_at = created_at WHERE verified_at IS NULL",
    ),
    (
        "users.password_changed_at",
        "ALTER TABLE users ADD COLUMN password_changed_at TEXT",
        # Left NULL on purpose. It means "never changed since signup", which is
        # exactly right for existing rows, and the session check reads a NULL
        # as "nothing to invalidate".
        None,
    ),
)

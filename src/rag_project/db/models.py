"""The record types, and how a database row becomes one.

Deliberately separate from the modules that query for them. A repository
(`users.py`, `history.py`) owns SQL; this file owns shape. Anything that needs
to know what a user *is* -- the API, the CLI, a test -- imports from here and
never has to import a query.

These are frozen dataclasses rather than Pydantic models, unlike models.py at
the top level. That file describes the pipeline's wire format, where validation
and JSON schema generation earn their keep. These describe rows this
application itself wrote and has already validated; there is nothing left to
parse, and a dataclass says so.

Note what `User` does *not* carry: the password hash. It is read inside
`users.authenticate` and never leaves it, so no route, log line, or template
can accidentally hand one to a browser.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone


def now_iso() -> str:
    """The timestamp format every row in this schema uses. See schema.py."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def to_epoch(iso: str | None) -> float | None:
    """An ISO timestamp from this schema as seconds, or None.

    Needed because signed tokens carry epoch seconds (they have to be short)
    while rows carry ISO strings (they have to be readable), and one place has
    to compare the two.
    """
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso).timestamp()
    except ValueError:
        return None


@dataclass(frozen=True)
class User:
    id: str
    email: str
    created_at: str
    # None means unverified. Only meaningful where mail is configured -- an
    # instance that cannot send verification links does not demand them.
    verified_at: str | None = None
    # None means "never changed since signup". Sessions and reset links issued
    # before this moment stop being valid, which is what makes a password reset
    # sign the account out everywhere and makes a reset link single-use.
    password_changed_at: str | None = None

    @classmethod
    def from_row(cls, row: dict) -> User:
        return cls(
            id=row["id"],
            email=row["email"],
            created_at=row["created_at"],
            # .get, not [], so a row selected before these columns existed --
            # or by a narrower SELECT -- still builds.
            verified_at=row.get("verified_at"),
            password_changed_at=row.get("password_changed_at"),
        )

    @property
    def is_verified(self) -> bool:
        return bool(self.verified_at)

    def public(self) -> dict:
        """What may cross the wire.

        Narrower than the record on purpose. `password_changed_at` is internal
        machinery for expiring sessions and reset links; a browser has no use
        for it, and the smallest payload that answers the UI's questions is the
        right one. The cache stores the whole record separately -- see
        `users.get`.
        """
        return {
            "id": self.id,
            "email": self.email,
            "created_at": self.created_at,
            "verified_at": self.verified_at,
        }


@dataclass(frozen=True)
class Turn:
    """One question and what the assistant did with it."""

    id: str
    query: str
    answer: str
    answered: bool
    refusal_reason: str | None
    top_score: float | None
    citations: list[dict]
    created_at: str

    @classmethod
    def from_row(cls, row: dict) -> Turn:
        try:
            citations = json.loads(row.get("citations") or "[]")
        except (ValueError, TypeError):
            # A row we cannot parse is still a real question someone asked.
            # Show it without its sources rather than dropping it from their
            # history, which would look like the question was never saved.
            citations = []
        return cls(
            id=row["id"],
            query=row["query"],
            answer=row["answer"],
            # SQLite has no boolean type, so this arrives as 0/1 there and as a
            # real bool from psycopg.
            answered=bool(row["answered"]),
            refusal_reason=row["refusal_reason"],
            top_score=row["top_score"],
            citations=citations,
            created_at=row["created_at"],
        )

    def public(self) -> dict:
        return {
            "id": self.id,
            "query": self.query,
            "answer": self.answer,
            "answered": self.answered,
            "refusal_reason": self.refusal_reason,
            "top_score": self.top_score,
            "citations": self.citations,
            "created_at": self.created_at,
        }

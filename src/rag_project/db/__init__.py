"""Accounts and per-user history: the durable half of the application.

Layered on purpose, one concern per file:

    schema.py    the DDL, and the only place a column is described
    models.py    User and Turn -- shape, with no SQL
    engine.py    connections, pooling, and the Postgres/SQLite seam
    users.py     every query against `users`
    history.py   every query against `history`

Callers import from this package rather than reaching into it:

    from .db import StorageError, User, history, users

    user = users.authenticate(email, password)
    history.save(user.id, response)

Password hashing is deliberately *not* here -- see passwords.py. Storing a hash
and computing one are different jobs, and only one of them is about a database.
"""

from __future__ import annotations

from . import history, users
from .engine import Database, DuplicateKey, StorageError, get_db, reset_db
from .models import Turn, User
from .users import AccountError

__all__ = [
    "AccountError",
    "Database",
    "DuplicateKey",
    "StorageError",
    "Turn",
    "User",
    "get_db",
    "history",
    "reset_db",
    "users",
]

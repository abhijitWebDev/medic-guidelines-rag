"""The users table: every query that touches it, and nothing else.

Password hashing lives in passwords.py and the record shape in models.py, so
what is left here is exactly the data access -- which is the point of the
split. A caller that wants a user calls a function in this file; a caller that
wants to know what a user *is* imports the model.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import asdict

from ..cache import get_cache
from ..passwords import MAX_LENGTH, MIN_LENGTH, burn_time, hash_password, verify_password
from .engine import DuplicateKey, get_db
from .models import User, now_iso

# Deliberately permissive. This is a sanity check on an input box, not an
# assertion about which addresses exist -- rejecting a valid address is a worse
# failure than accepting an unreachable one, and nothing here mails anybody.
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s.]+(\.[^@\s.]+)+$")
EMAIL_MAX = 254

# Every authenticated request resolves a session cookie to a user, and that
# must not be a database round trip each time. Short, because the only thing it
# delays is an account deletion taking effect.
_CACHE_TTL_S = 300


# Every column that makes a User. Named once so that adding one cannot leave a
# query behind selecting the old set -- which fails as a silently absent field
# rather than as an error.
_FIELDS = "id, email, created_at, verified_at, password_changed_at"


class AccountError(ValueError):
    """Something the person filling in the form can fix. The message is shown."""


def normalise_email(email: str) -> str:
    """Lowercased and trimmed. Two people typing the same address in different
    cases are the same person, and the UNIQUE constraint has to agree."""
    return (email or "").strip().casefold()


def _validate(email: str, password: str) -> str:
    email = normalise_email(email)
    if not email or len(email) > EMAIL_MAX or not EMAIL_RE.match(email):
        raise AccountError("Enter a valid email address.")
    if len(password) < MIN_LENGTH:
        raise AccountError(f"Password must be at least {MIN_LENGTH} characters.")
    if len(password) > MAX_LENGTH:
        raise AccountError("Password is too long.")
    return email


def create(email: str, password: str) -> User:
    """Register an account. Raises AccountError for anything the form can fix."""
    email = _validate(email, password)
    user = User(id=uuid.uuid4().hex, email=email, created_at=now_iso())
    try:
        get_db().execute(
            "INSERT INTO users (id, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (user.id, user.email, hash_password(password), user.created_at),
        )
    except DuplicateKey:
        # Checking first and inserting second would still race; letting the
        # UNIQUE constraint be the arbiter means there is exactly one place the
        # rule is enforced, and it is the database.
        raise AccountError("That email already has an account. Sign in instead.") from None
    return user


def authenticate(email: str, password: str) -> User | None:
    """The credential check. None means "no", without saying which half failed.

    This is the only function that reads `password_hash`, and it does not put
    it in the User it returns -- so a hash cannot reach a response by accident.
    """
    row = get_db().query_one(
        f"SELECT password_hash, {_FIELDS} FROM users WHERE email = ?",
        (normalise_email(email),),
    )
    if row is None:
        burn_time()  # an unknown address must not answer faster than a wrong password
        return None
    if not verify_password(password, row["password_hash"]):
        return None
    return User.from_row(row)


def get(user_id: str) -> User | None:
    """Resolve a session's user id, through the cache.

    A cache miss here is free (we read the database) and a cache *outage* is
    equally free. Nothing about authorisation is decided here -- the session
    signature already did that -- so this is purely about not paying for a
    query on every request.

    `local=False` keeps it out of the process's own memory, which costs a Redis
    round trip and buys the thing that matters more: `invalidate` takes effect
    everywhere at once. A per-instance copy would mean somebody who has just
    confirmed their address, or just changed their password, keeps being told
    otherwise by whichever instance still holds the old row.
    """
    if not user_id:
        return None
    cache = get_cache()
    key = _cache_key(user_id)
    hit = cache.get_json(key, local=False)
    if hit is not None:
        return User(**hit)

    row = get_db().query_one(
        f"SELECT {_FIELDS} FROM users WHERE id = ?", (user_id,)
    )
    if row is None:
        return None
    user = User.from_row(row)
    # The full record, not `public()`: a session check needs
    # `password_changed_at`, which public() deliberately withholds.
    cache.set_json(key, asdict(user), _CACHE_TTL_S, local=False)
    return user


def get_by_email(email: str) -> User | None:
    """For the password-reset flow, which knows an address and nothing else."""
    row = get_db().query_one(
        f"SELECT {_FIELDS} FROM users WHERE email = ?", (normalise_email(email),)
    )
    return User.from_row(row) if row else None


def mark_verified(user_id: str) -> User | None:
    """Idempotent: clicking a verification link twice is not an error.

    The WHERE clause keeps the first timestamp rather than overwriting it, so a
    forwarded link cannot quietly rewrite when the account was confirmed.
    """
    get_db().execute(
        "UPDATE users SET verified_at = ? WHERE id = ? AND verified_at IS NULL",
        (now_iso(), user_id),
    )
    invalidate(user_id)
    return get(user_id)


def set_password(user_id: str, password: str) -> None:
    """Replace the password, and stamp when it happened.

    That stamp is load-bearing in two places, both of which fall out of it for
    free rather than needing a revocation list:

    * every session issued before now stops validating, so a reset signs the
      account out everywhere -- which is what someone resetting because they
      think they were compromised is actually asking for;
    * the reset link that got them here stops working, making it single-use.
    """
    if len(password) < MIN_LENGTH:
        raise AccountError(f"Password must be at least {MIN_LENGTH} characters.")
    if len(password) > MAX_LENGTH:
        raise AccountError("Password is too long.")
    get_db().execute(
        "UPDATE users SET password_hash = ?, password_changed_at = ? WHERE id = ?",
        (hash_password(password), now_iso(), user_id),
    )
    invalidate(user_id)


def all_users() -> list[User]:
    rows = get_db().query(f"SELECT {_FIELDS} FROM users ORDER BY created_at")
    return [User.from_row(r) for r in rows]


def delete(user_id: str) -> bool:
    """Removes the account and everything it ever asked."""
    # Two things are going on here. The history is deleted explicitly rather
    # than left to ON DELETE CASCADE, because SQLite only enforces that with
    # the pragma engine.py sets and a store restored from a dump elsewhere may
    # not have it. And it goes through history.clear rather than a raw DELETE
    # so that the deletion also drops the cached copy of the list -- otherwise
    # the rows are gone and a stale page of them survives in Redis.
    from . import history

    history.clear(user_id)
    deleted = get_db().execute("DELETE FROM users WHERE id = ?", (user_id,))
    invalidate(user_id)
    return deleted > 0


def _cache_key(user_id: str) -> str:
    return f"usr:{user_id}"


def invalidate(user_id: str) -> None:
    get_cache().forget(_cache_key(user_id))

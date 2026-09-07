"""Password hashing.

Its own module because it is not a database concern. `db/users.py` stores
whatever string this produces and never inspects it; this file never touches a
row. The seam matters the day the algorithm changes -- that is an edit here and
nowhere else.

`hashlib.scrypt` is used rather than argon2 or bcrypt for one reason: it is in
the standard library. An authentication dependency is a thing you must then
keep patched forever, and scrypt is a memory-hard KDF that ships with CPython
and needs no such care. The cost parameters are configurable because "expensive
enough" is a property of the hardware, not of the algorithm.

A stored hash carries the parameters it was made with (`scrypt$n$r$p$salt$hash`),
so raising the cost later re-hashes people as they sign in rather than locking
everyone out.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

from .config import get_settings

MIN_LENGTH = 8
# scrypt hashes the whole input; without a cap a multi-megabyte "password" is a
# free way to make the server allocate.
MAX_LENGTH = 1024


def hash_password(password: str) -> str:
    s = get_settings()
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(
        password.encode(), salt=salt, n=s.scrypt_n, r=s.scrypt_r, p=s.scrypt_p, dklen=32
    )
    return "$".join(
        ["scrypt", str(s.scrypt_n), str(s.scrypt_r), str(s.scrypt_p), _b64(salt), _b64(dk)]
    )


def verify_password(password: str, encoded: str) -> bool:
    """Constant-time check against a stored hash. False on anything malformed.

    Malformed is not an error case worth raising on: a hash that does not parse
    is a hash this password does not match, and a caller that had to catch an
    exception would be one refactor away from catching it as success.
    """
    try:
        scheme, n, r, p, salt_b64, hash_b64 = encoded.split("$")
        if scheme != "scrypt":
            return False
        salt, expected = _b64d(salt_b64), _b64d(hash_b64)
        dk = hashlib.scrypt(
            password.encode(), salt=salt, n=int(n), r=int(r), p=int(p), dklen=len(expected)
        )
    except (ValueError, TypeError, MemoryError):
        return False
    return hmac.compare_digest(dk, expected)


# One hash of a throwaway value, verified when an email does not exist so that
# a wrong address and a wrong password take the same time. Built lazily and
# reused: the point is to spend the time, not to spend it twice.
_DUMMY: str | None = None


def burn_time() -> None:
    global _DUMMY
    if _DUMMY is None:
        _DUMMY = hash_password(secrets.token_hex(16))
    verify_password("x", _DUMMY)


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def _b64d(text: str) -> bytes:
    return base64.b64decode(text.encode())

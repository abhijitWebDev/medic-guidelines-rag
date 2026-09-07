"""The history table: one row per question a signed-in user asked.

What is kept is the whole *turn* -- the question, the answer, whether it was
answered or refused, and the citations that supported it. Not the trace. The
trace is debugging data about which gate fired, it is by far the largest part
of a Response, and a saved answer whose citations have gone missing is worse
than useless in a system whose entire claim is that every statement is sourced.
So citations are stored and the trace is dropped.

Writes here are never allowed to fail a question. A saved turn is a
convenience; the answer on screen is the product. The API calls `save` inside a
try for that reason.
"""

from __future__ import annotations

import json
import uuid

from ..cache import get_cache
from ..config import get_settings
from ..models import Response
from .engine import get_db
from .models import Turn, now_iso

# Short. The list changes only when its owner asks something or deletes
# something, and both invalidate it explicitly -- this TTL exists to bound the
# damage from a write that happened on another instance.
_LIST_TTL_S = 120

# Only these page sizes are cached. Invalidation has to name the keys it drops,
# so a limit that cannot be invalidated must never be written: a caller asking
# for an unusual page size gets an uncached read, correct but slower.
_CACHED_LIMITS = (20, 50, 100)


def save(user_id: str, response: Response) -> Turn:
    turn = Turn(
        id=uuid.uuid4().hex,
        query=response.query,
        answer=response.answer,
        answered=response.answered,
        refusal_reason=response.refusal_reason.value if response.refusal_reason else None,
        top_score=response.top_score,
        citations=response.citations,
        created_at=now_iso(),
    )
    get_db().execute(
        "INSERT INTO history (id, user_id, query, answer, answered, refusal_reason,"
        " top_score, citations, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            turn.id,
            user_id,
            turn.query,
            turn.answer,
            turn.answered,
            turn.refusal_reason,
            turn.top_score,
            json.dumps(turn.citations),
            turn.created_at,
        ),
    )
    _prune(user_id)
    _invalidate(user_id)
    return turn


def _prune(user_id: str) -> None:
    """Keep only the newest `history_max_items` turns for this user.

    An unbounded per-user log is a slow leak that nobody notices until the
    database bill arrives, and nobody scrolls back a thousand questions. The
    cap is applied on write so there is no sweeper to schedule.
    """
    keep = get_settings().history_max_items
    if keep <= 0:
        return
    get_db().execute(
        "DELETE FROM history WHERE user_id = ? AND id NOT IN ("
        "  SELECT id FROM history WHERE user_id = ?"
        "  ORDER BY created_at DESC, id DESC LIMIT ?)",
        (user_id, user_id, keep),
    )


def list_for(user_id: str, limit: int = 50, before: str | None = None) -> list[Turn]:
    """Newest first. `before` is a `created_at` from the last row of a page.

    Only the first page is cached: it is the one the panel opens on and the one
    every ask invalidates. Caching deeper pages would trade a rare query for a
    key per user per scroll position.
    """
    cacheable = before is None and limit in _CACHED_LIMITS
    key = _list_key(user_id, limit)
    if cacheable:
        hit = get_cache().get_json(key)
        if hit is not None:
            return [Turn(**t) for t in hit]

    if before:
        rows = get_db().query(
            "SELECT * FROM history WHERE user_id = ? AND created_at < ?"
            " ORDER BY created_at DESC, id DESC LIMIT ?",
            (user_id, before, limit),
        )
    else:
        rows = get_db().query(
            "SELECT * FROM history WHERE user_id = ?"
            " ORDER BY created_at DESC, id DESC LIMIT ?",
            (user_id, limit),
        )

    turns = [Turn.from_row(r) for r in rows]
    if cacheable:
        get_cache().set_json(key, [t.public() for t in turns], _LIST_TTL_S)
    return turns


def get(user_id: str, turn_id: str) -> Turn | None:
    """Scoped by user id on purpose: a turn id is not an authorisation."""
    row = get_db().query_one(
        "SELECT * FROM history WHERE id = ? AND user_id = ?", (turn_id, user_id)
    )
    return Turn.from_row(row) if row else None


def delete(user_id: str, turn_id: str) -> bool:
    deleted = get_db().execute(
        "DELETE FROM history WHERE id = ? AND user_id = ?", (turn_id, user_id)
    )
    _invalidate(user_id)
    return deleted > 0


def clear(user_id: str) -> int:
    deleted = get_db().execute("DELETE FROM history WHERE user_id = ?", (user_id,))
    _invalidate(user_id)
    return max(deleted, 0)


# --- per-user cache keys -------------------------------------------------


def _list_key(user_id: str, limit: int) -> str:
    return f"hist:{user_id}:{limit}"


def _invalidate(user_id: str) -> None:
    """Drop every cached first page for this user.

    The limit is part of the key because a 20-row page and a 50-row page are
    different answers and the UI is free to ask for either, so invalidation has
    to cover the handful of limits anything actually requests.
    """
    cache = get_cache()
    for limit in _CACHED_LIMITS:
        cache.forget(_list_key(user_id, limit))

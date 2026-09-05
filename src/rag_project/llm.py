"""Thin wrapper over the OpenAI client.

Centralised so that model choice, retries, and — most importantly — the
*failure* behaviour are decided once. In this application an LLM call that
errors must never degrade into "answer anyway": every caller here is either a
safety gate or the generator itself, so the safe default on failure is to
refuse, and that decision belongs at the call site, not buried in a retry loop.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import TypeVar

from openai import OpenAI, OpenAIError
from pydantic import BaseModel

from .config import get_settings

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


# --- degradation tracking -------------------------------------------------
# Callers that swallow an LLMError produce a refusal that looks exactly like a
# considered one. That ambiguity is fine for the user -- both are "we won't
# answer" -- but it is not fine for the response cache, which would otherwise
# pin a transient API outage into Redis for the whole TTL. Each fail-closed
# path records itself here so the caller can tell the two apart.
#
# A ContextVar rather than a global: the API serves concurrent requests in one
# process, and one request's outage must not suppress caching for another's
# perfectly good answer.
_degraded: ContextVar[tuple[str, ...]] = ContextVar("llm_degraded", default=())


def note_degraded(stage: str) -> None:
    """Record that `stage` fell back because the model was unavailable."""
    _degraded.set(_degraded.get() + (stage,))


def degradations() -> tuple[str, ...]:
    return _degraded.get()


def clear_degradations() -> None:
    _degraded.set(())


_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        s = get_settings()
        if not s.openai_api_key:
            raise LLMError("OPENAI_API_KEY is not set in .env")
        _client = OpenAI(api_key=s.openai_api_key, max_retries=3)
    return _client


def parse(
    schema: type[T],
    system: str,
    user: str,
    model: str | None = None,
    temperature: float | None = 0.0,
) -> T:
    """Structured call. Raises LLMError rather than returning something partial."""
    s = get_settings()
    kwargs = {
        "model": model or s.openai_guard_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": schema,
    }
    # Reasoning models reject `temperature`; non-reasoning ones want 0 here
    # because every caller is a classifier or a grounded generator.
    if temperature is not None:
        kwargs["temperature"] = temperature

    try:
        resp = client().chat.completions.parse(**kwargs)
    except OpenAIError as e:
        if temperature is not None and "temperature" in str(e).lower():
            kwargs.pop("temperature")
            try:
                resp = client().chat.completions.parse(**kwargs)
            except OpenAIError as e2:
                raise LLMError(str(e2)) from e2
        else:
            raise LLMError(str(e)) from e

    parsed = resp.choices[0].message.parsed
    if parsed is None:
        raise LLMError("model returned no parseable structured output")
    return parsed

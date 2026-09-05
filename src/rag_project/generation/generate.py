"""Grounded generation with structured, citation-bearing output."""

from __future__ import annotations

from ..config import get_settings
from ..llm import LLMError, note_degraded, parse
from ..models import GroundedAnswer, Retrieved
from .prompt import SYSTEM, build_user_prompt


def generate(query: str, results: list[Retrieved]) -> GroundedAnswer:
    """Returns a GroundedAnswer. On LLM failure, returns insufficient_context
    rather than raising -- the caller turns that into a standard refusal."""
    s = get_settings()
    try:
        return parse(
            GroundedAnswer,
            SYSTEM,
            build_user_prompt(query, results),
            model=s.openai_model,
        )
    except LLMError:
        note_degraded("generator")
        return GroundedAnswer(answer="", claims=[], insufficient_context=True)

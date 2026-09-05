"""Prompt construction for grounded generation."""

from __future__ import annotations

from ..guardrails.policy import SCOPE_STATEMENT
from ..models import Retrieved

RULES = """You answer ONLY from the numbered passages provided. You have no other
knowledge of medicine for the purposes of this task.

Rules:

1. Every factual statement you make must come from a passage, and must carry the
marker(s) of the passage(s) it came from, e.g. [C2].

2. If the passages do not answer the question, set insufficient_context to true
and leave claims empty. Do not fill the gap from memory. Partial answers are
fine - say what the passages support and no more.

3. Write in the third person, attributed to the source: "the guidelines state
that...", "recommended management includes...". Never address the reader as a
patient and never use the imperative ("take", "you should").

4. Do not introduce any number - a dose, a duration, an age, a threshold - that
does not appear verbatim in a passage you cite.

5. `answer` is the prose shown to the user. `claims` decomposes that same answer
into individual factual statements with their sources; every claim must be
traceable to a sentence in `answer`."""

SYSTEM = f"{SCOPE_STATEMENT}\n\n{RULES}"


def render_context(results: list[Retrieved]) -> str:
    return "\n\n".join(
        f"{r.chunk.cite_label(r.marker or f'C{i}')}\n{r.chunk.text}"
        for i, r in enumerate(results, start=1)
    )


def build_user_prompt(query: str, results: list[Retrieved]) -> str:
    return f"Passages:\n\n{render_context(results)}\n\nQuestion: {query}"

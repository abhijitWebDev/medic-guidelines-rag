"""Gate 1: classify intent before any retrieval happens.

Ordering matters and is not arbitrary. Rules run first and may only ever
*escalate* to a refusal; they are never allowed to stamp a query IN_SCOPE. If
rules could clear a query, then any phrasing the regexes failed to anticipate
would bypass the model check entirely -- the classic pattern where a blocklist
becomes an allowlist by accident. So: rules catch the obvious and stop; anything
they do not catch goes to the model, which decides.

Retrieval never runs for a refused query. That is the point of gating here
rather than after retrieval: an out-of-scope query should not consume an
embedding call, and a personalised one should not have guideline text pulled up
next to it where a later stage might be tempted to answer from it.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from ..llm import LLMError, note_degraded, parse
from ..models import Intent, IntentVerdict
from .policy import SCOPE_STATEMENT

# --- Rule pass -----------------------------------------------------------
# Each pattern needs a *personal* framing to fire. "chest pain management" is a
# legitimate guideline question; "I have chest pain" is an emergency. Matching
# the symptom alone would refuse most of the corpus.

_FIRST_PERSON = r"(?:\bi\b|\bmy\b|\bme\b|\bmine\b|i'm|i've|im\b)"

EMERGENCY_PATTERNS: list[tuple[str, str]] = [
    (rf"{_FIRST_PERSON}[^.?!]{{0,40}}\b(chest pain|can'?t breathe|cannot breathe|"
     r"trouble breathing|unconscious|passed out|severe bleeding|bleeding heavily|"
     r"overdosed?|poisoned)\b", "first-person acute symptom"),
    (r"\b(kill myself|suicidal|suicide|end my life|harm myself|self[- ]harm)\b",
     "self-harm"),
    (r"\b(emergency|urgent|right now|immediately)\b[^.?!]{0,30}\b(help|what do i do|"
     r"what should i do)\b", "explicit urgency"),
    (rf"\b(is|are)\s+(this|these|it|my)\b[^.?!]{{0,30}}\b(life[- ]threatening|"
     r"an emergency|dangerous)\b", "asks if own situation is an emergency"),
]

ADVICE_PATTERNS: list[tuple[str, str]] = [
    (r"\bshould i\b", "asks what the user should do"),
    (r"\bcan i (take|use|stop|start|have)\b", "asks permission for own action"),
    (r"\b(do|does) (i|my \w+) have\b", "asks for a diagnosis"),
    (r"\bwhat('?s| is) wrong with me\b", "asks for a diagnosis"),
    (r"\bdiagnos(e|is) me\b", "asks for a diagnosis"),
    (rf"\bhow (much|many)\b[^.?!]{{0,30}}\b(should|do)\s+{_FIRST_PERSON}", "asks own dose"),
    (rf"{_FIRST_PERSON}\s+(have|had|has|feel|felt|am feeling|experience)\b[^.?!]{{0,40}}"
     r"\b(symptom|pain|fever|rash|cough|lump|bleeding)\b", "describes own symptoms"),
    (r"\bis it safe for me\b", "asks personal safety"),
    (r"\bwhat should i (take|do|use)\b", "asks what the user should do"),
    (r"\bmy (test|blood|lab|scan|report|result)s?\b", "asks to interpret own results"),
]

_COMPILED = {
    Intent.EMERGENCY: [(re.compile(p, re.IGNORECASE), why) for p, why in EMERGENCY_PATTERNS],
    Intent.PERSONALIZED_ADVICE: [(re.compile(p, re.IGNORECASE), why) for p, why in ADVICE_PATTERNS],
}


def rule_screen(query: str) -> IntentVerdict | None:
    """Return a refusal verdict, or None meaning 'undecided -- ask the model'."""
    # Emergency is checked first: a query can be both personal and urgent, and
    # the urgent reading is the one that must win.
    for intent in (Intent.EMERGENCY, Intent.PERSONALIZED_ADVICE):
        for pattern, why in _COMPILED[intent]:
            if pattern.search(query):
                return IntentVerdict(
                    intent=intent, reason=why, matched_rule=pattern.pattern[:60]
                )
    return None


# --- Model pass ----------------------------------------------------------


class _IntentOut(BaseModel):
    intent: Intent
    reason: str = Field(description="one short clause justifying the label")


_SYSTEM = f"""{SCOPE_STATEMENT}

Classify the user's query into exactly one label:

- "in_scope": a general question about health, disease or medicine -- what a \
condition is, its causes, symptoms, complications, risk factors, prevention, \
diagnosis, management, drugs, doses or referral. Third-person or impersonal \
framing. The user does not have to mention the guidelines: a bare "what is X" \
or "what are the side effects of X" is a request for what the guidelines say \
about X.
- "personalized_advice": asks you to advise, diagnose, dose, or interpret \
findings for a specific person (usually the user or a named relative).
- "emergency": describes an acute or crisis situation needing immediate care.
- "out_of_domain": not about health or medicine at all (sport, code, travel, \
politics), or a question about you rather than about a health topic.

Asking about a drug, a dose, or a treatment is IN SCOPE when the question is \
about what the guideline states. It is PERSONALIZED_ADVICE only when it asks \
what a particular person should do.

Whether the corpus actually covers the topic is NOT yours to decide -- a later \
retrieval stage refuses when it finds nothing. "out_of_domain" means the wrong \
subject, never a subject you suspect is thinly covered. When a health question \
is neither personal nor urgent, it is in_scope.

Examples:
"what is diabetes" -> in_scope
"what are the side effects of metformin" -> in_scope
"how is drug-resistant tuberculosis treated" -> in_scope
"should i stop taking my metformin" -> personalized_advice
"my father is unconscious, what do i do" -> emergency
"who won the match last night" -> out_of_domain"""


def classify(query: str, use_model: bool = True) -> IntentVerdict:
    ruled = rule_screen(query)
    if ruled is not None:
        return ruled
    if not use_model:
        return IntentVerdict(intent=Intent.IN_SCOPE, reason="rules only; not screened by model")
    try:
        out = parse(_IntentOut, _SYSTEM, query)
    except LLMError as e:
        # Fail closed. An unavailable classifier must not mean "assume safe".
        note_degraded("intent_classifier")
        return IntentVerdict(
            intent=Intent.OUT_OF_DOMAIN,
            reason=f"intent classifier unavailable ({e}); refusing rather than assuming safe",
            matched_rule="fail-closed",
        )
    return IntentVerdict(intent=out.intent, reason=out.reason)

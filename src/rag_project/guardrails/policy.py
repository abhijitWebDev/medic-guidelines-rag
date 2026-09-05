"""The scope boundary, stated once, in one place.

Every refusal path in the system resolves to a message defined here so the
assistant's boundary is auditable in a single file rather than scattered
across prompts.
"""

from __future__ import annotations

from ..models import RefusalReason

DISCLAIMER = (
    "This assistant provides information from official government health "
    "guidelines only and does not give personalized medical advice. "
    "Consult a qualified clinician for anything concerning your own health."
)

EMERGENCY_MESSAGE = (
    "This sounds like it may be a medical emergency. I can't help with urgent "
    "or personal medical situations.\n\n"
    "**Please contact emergency services now — dial 112 (national emergency) "
    "or 108 (ambulance) in India, or go to the nearest emergency department.**\n\n"
    "For mental health crisis support in India, Tele-MANAS is available "
    "24x7 at 14416."
)

REFUSALS: dict[RefusalReason, str] = {
    RefusalReason.EMERGENCY: EMERGENCY_MESSAGE,
    RefusalReason.PERSONALIZED_ADVICE: (
        "I can't give personalized medical advice, diagnose a condition, or "
        "recommend a treatment or dose for a specific person — that requires a "
        "qualified clinician who can examine you.\n\n"
        "I can tell you what the official guidelines say on a topic in general "
        "terms. Try asking something like \"what do the guidelines say about "
        "how tuberculosis is diagnosed?\" instead."
    ),
    RefusalReason.OUT_OF_DOMAIN: (
        "I only answer questions about the official government health "
        "guidelines in my knowledge base. That question falls outside them."
    ),
    RefusalReason.LOW_CONFIDENCE: (
        "I don't have enough information in the guidelines to answer this."
    ),
    RefusalReason.MODEL_DECLINED: (
        "I don't have enough information in the guidelines to answer this."
    ),
    RefusalReason.UNGROUNDED_OUTPUT: (
        "I don't have enough information in the guidelines to answer this."
    ),
    RefusalReason.UNSAFE_OUTPUT: (
        "I can't answer that in a way that stays within my boundaries — the "
        "response would have read as personalized medical advice rather than a "
        "summary of what the guidelines say."
    ),
}


def refusal_text(reason: RefusalReason) -> str:
    return REFUSALS[reason]


# --- The distinction that makes this project work ------------------------
#
# The MOHFW Standard Treatment Guidelines are largely *made of* dosages and
# treatment algorithms. So a naive "block anything containing a dose" filter
# would refuse nearly every correctly-retrieved answer, and the assistant
# would be useless.
#
# The real line is attribution and addressee, not subject matter:
#
#   ALLOWED    "The guidelines list rifampicin at 10 mg/kg daily for adults
#               in the intensive phase [C2]."   -> reports what a document says
#   FORBIDDEN  "You should take rifampicin 10 mg/kg daily."
#                                               -> instructs a specific person
#
# Same drug, same number. Gate 3 enforces the framing, and separately checks
# that any number in the answer actually appears in a cited chunk.

SCOPE_STATEMENT = """\
You are a Medical Guideline Assistant. You explain what official government \
health guidelines (MOHFW Standard Treatment Guidelines) say. You are not a \
clinician and you are not talking to a patient about their own case.

ALLOWED: summarising, quoting, and explaining guideline content in the third \
person, always attributed to a cited source.

FORBIDDEN: telling a specific person what to do; diagnosing; recommending a \
treatment or dose for an individual; interpreting someone's symptoms, test \
results, or history; emergency or crisis instructions.

If a question asks for something forbidden, refuse — do not answer a nearby \
allowed question instead unless the user asks."""

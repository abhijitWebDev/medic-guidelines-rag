"""Shared data contracts. Written before the pipeline so every stage agrees."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class SourceDoc(BaseModel):
    """One manifest entry. A PDF not described here is never ingested."""

    doc_id: str
    title: str
    filename: str
    url: str
    publisher: str = "MOHFW, Government of India"
    specialty: str | None = None
    version: str | None = None
    published: str | None = None
    license: str = "Government of India public health advisory"
    sha256: str


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    source_url: str
    publisher: str
    specialty: str | None = None
    section_path: str
    page_start: int
    page_end: int
    text: str
    n_tokens: int
    doc_version: str | None = None
    index_version: str
    ingested_at: str

    def cite_label(self, marker: str) -> str:
        """How this chunk is shown to the model in the prompt."""
        if not self.page_start:
            # 0 means the source carried no page evidence (plain text without
            # page breaks). Saying "p.1" for a 55-page document would be
            # confidently wrong, which is worse than admitting the gap.
            pages = "page n/a"
        elif self.page_start == self.page_end:
            pages = f"p.{self.page_start}"
        else:
            pages = f"pp.{self.page_start}-{self.page_end}"
        return f"[{marker} | {self.doc_id} | {self.title} | §{self.section_path} | {pages}]"


class Retrieved(BaseModel):
    chunk: Chunk
    marker: str = ""
    vector_score: float | None = None
    fts_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None


class Intent(StrEnum):
    """Gate 1 outcomes. Only IN_SCOPE proceeds to retrieval."""

    IN_SCOPE = "in_scope"
    PERSONALIZED_ADVICE = "personalized_advice"
    EMERGENCY = "emergency"
    OUT_OF_DOMAIN = "out_of_domain"


class IntentVerdict(BaseModel):
    intent: Intent
    reason: str
    matched_rule: str | None = None


class ConfidenceAction(StrEnum):
    """Gate 2 outcomes. The middle one is what makes the gate corrective.

    A binary gate can only answer or refuse, which throws away the case it is
    worst at judging: retrieval that found the right neighbourhood but not the
    answer. PROCEED and REFUSE are the confident ends; CORRECT is "try again,
    differently, before deciding".
    """

    PROCEED = "proceed"
    CORRECT = "correct"
    REFUSE = "refuse"


class Claim(BaseModel):
    """One factual statement. chunk_ids may not be empty -- enforced by schema."""

    text: str
    chunk_ids: list[str] = Field(min_length=1)


class GroundedAnswer(BaseModel):
    answer: str
    claims: list[Claim] = Field(default_factory=list)
    insufficient_context: bool = False


class RefusalReason(StrEnum):
    PERSONALIZED_ADVICE = "personalized_advice"
    EMERGENCY = "emergency"
    OUT_OF_DOMAIN = "out_of_domain"
    LOW_CONFIDENCE = "low_confidence"
    MODEL_DECLINED = "model_declined"
    UNGROUNDED_OUTPUT = "ungrounded_output"
    UNSAFE_OUTPUT = "unsafe_output"


class Response(BaseModel):
    """Final object returned by CLI and API alike."""

    query: str
    answered: bool
    answer: str
    claims: list[Claim] = Field(default_factory=list)
    citations: list[dict] = Field(default_factory=list)
    refusal_reason: RefusalReason | None = None
    top_score: float | None = None
    disclaimer: str = ""
    trace: dict = Field(default_factory=dict)

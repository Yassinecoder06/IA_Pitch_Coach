"""Structured-output schemas for AI deck generation and critique."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

SlideKind = Literal[
    "title", "hook", "problem", "solution", "product", "market", "business_model",
    "traction", "competition", "team", "financials", "ask", "closing",
]


class DeckBrief(BaseModel):
    company: str = Field(min_length=1, max_length=120)
    one_liner: str = Field(min_length=3, max_length=400, description="What the company does, in one sentence")
    audience: str = Field(default="seed investors", max_length=120)
    stage: str = Field(default="pre-seed", max_length=60)
    minutes: float = Field(default=3, ge=0.5, le=30)
    details: str = Field(default="", max_length=4000, description="Facts the deck may use: metrics, customers, team, ask")


class GeneratedSlide(BaseModel):
    kind: SlideKind
    title: str = Field(description="Slide headline, at most 8 words")
    bullets: List[str] = Field(default_factory=list, description="0-4 bullets, each at most 12 words")
    notes: str = Field(description="What the speaker says on this slide, 40-100 words, conversational")
    time_budget_sec: int = Field(ge=5, le=600)


class GeneratedDeck(BaseModel):
    title: str
    slides: List[GeneratedSlide] = Field(min_length=3, max_length=16)


class SlideCritique(BaseModel):
    slide_number: int = Field(description="1-based")
    density: Literal["ok", "too_dense", "too_sparse"]
    issues: List[str] = Field(default_factory=list)
    suggestion: str = ""


class DeckCritique(BaseModel):
    overall: str = Field(description="2-3 sentence assessment of the deck's story and design")
    missing_sections: List[str] = Field(default_factory=list, description="Pitch sections the deck lacks, e.g. 'ask'")
    slides: List[SlideCritique] = Field(default_factory=list)


class SlideIn(BaseModel):
    """A slide sent by the client (outline editor, PDF/PPTX import)."""

    title: str = Field(default="", max_length=300)
    bullets: List[str] = Field(default_factory=list, max_length=20)
    notes: str = Field(default="", max_length=6000)
    text: str = Field(default="", max_length=8000)
    kind: Optional[str] = Field(default=None, max_length=24)
    time_budget_sec: Optional[int] = Field(default=None, ge=1, le=3600)

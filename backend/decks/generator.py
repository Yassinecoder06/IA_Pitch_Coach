"""AI deck generation and slide critique through the multi-provider router (task: deck_gen)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence

from backend.decks.schemas import DeckBrief, DeckCritique, GeneratedDeck, SlideIn
from backend.llm.router import llm_router
from backend.llm.types import LLMMessage

GENERATE_SYSTEM = """You are a top pitch-deck strategist (think YC demo day).
Write a deck the founder will deliver out loud. Rules:
- One idea per slide; headline-style titles; at most 4 short bullets; plenty of white space.
- Classic arc: hook, problem, solution, product, market, business model, traction, competition, team, ask.
  Drop sections the time budget cannot afford; never drop problem, solution or ask.
- Speaker notes are what the founder SAYS: conversational, concrete, with numbers from the brief.
- Use only facts from the brief. Where a number is unknown, write a placeholder like [X customers].
- time_budget_sec across slides must add up to the requested length."""

CRITIQUE_SYSTEM = """You review pitch decks like a design-minded investor.
Judge each slide for one-idea-per-slide, text density, headline clarity and whether it
moves the story forward. Note pitch sections that are missing. Be specific and brief."""


def _fit_budgets(slides: List[SlideIn], total_seconds: float) -> List[SlideIn]:
    """Scale per-slide budgets so they add up to the requested length (models rarely nail the sum)."""
    current = sum(s.time_budget_sec or 0 for s in slides)
    if current <= 0:
        each = max(5, int(total_seconds / max(1, len(slides))))
        return [s.model_copy(update={"time_budget_sec": each}) for s in slides]
    factor = total_seconds / current
    return [s.model_copy(update={"time_budget_sec": max(5, round((s.time_budget_sec or 0) * factor))}) for s in slides]


async def generate_deck(brief: DeckBrief) -> Dict[str, Any]:
    seconds = brief.minutes * 60
    suggested = max(4, min(12, round(brief.minutes * 2.5)))
    prompt = (
        f"Company: {brief.company}\nOne-liner: {brief.one_liner}\nAudience: {brief.audience}\n"
        f"Stage: {brief.stage}\nTalk length: {seconds:.0f} seconds (about {suggested} slides)\n"
        f"Facts you may use:\n{brief.details or '(none given: use placeholders)'}"
    )
    deck, model_ref = await llm_router.generate_structured(
        messages=[LLMMessage(role="user", content=prompt)],
        schema=GeneratedDeck,
        system_prompt=GENERATE_SYSTEM,
        task="deck_gen",
        temperature=0.6,
        max_tokens=8000,
    )
    slides = [
        SlideIn(kind=s.kind, title=s.title, bullets=s.bullets[:6], notes=s.notes, time_budget_sec=s.time_budget_sec)
        for s in deck.slides
    ]
    return {"title": deck.title, "slides": _fit_budgets(slides, seconds), "model": model_ref}


def density(slide: Dict[str, Any]) -> Dict[str, Any]:
    text = " ".join([slide.get("title") or "", *(slide.get("bullets") or []), slide.get("text") or ""])
    words = len(text.split())
    return {"words": words, "bullets": len(slide.get("bullets") or []), "dense": words > 60 or len(slide.get("bullets") or []) > 5}


async def critique_deck(deck: Dict[str, Any]) -> Dict[str, Any]:
    slides: Sequence[Dict[str, Any]] = deck["slides"]
    outline = [
        {
            "slide_number": s["idx"] + 1,
            "title": s["title"],
            "bullets": s["bullets"],
            "visible_text": (s["text"] or "")[:600],
            "word_count": density(s)["words"],
        }
        for s in slides
    ]
    result, model_ref = await llm_router.generate_structured(
        messages=[LLMMessage(role="user", content=f"Deck: {deck['title']}\n{json.dumps(outline)}")],
        schema=DeckCritique,
        system_prompt=CRITIQUE_SYSTEM,
        task="deck_gen",
    )
    valid = {s["idx"] + 1 for s in slides}
    critique = result.model_dump()
    critique["slides"] = [c for c in critique["slides"] if c["slide_number"] in valid]
    critique["density"] = [{"slide_number": s["idx"] + 1, **density(s)} for s in slides]
    critique["model"] = model_ref
    return critique

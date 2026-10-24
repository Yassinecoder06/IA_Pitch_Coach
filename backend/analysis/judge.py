"""
LLM judge for a rehearsal: content and delivery feedback as structured output.
Every priority must cite a cue id or a quote; citations that do not exist are dropped.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from backend.coaching.rubric import Scorecard
from backend.llm.router import llm_router
from backend.llm.types import LLMMessage

logger = logging.getLogger(__name__)


class Priority(BaseModel):
    title: str = Field(description="Short imperative, e.g. 'Lead with the 30% cost saving'")
    detail: str = Field(description="What to change and how, in 1-3 sentences")
    evidence: List[str] = Field(default_factory=list, description="Cue ids from the list, or exact short quotes from the transcript")


class SlideFeedback(BaseModel):
    slide_number: int = Field(description="1-based slide number as shown in the transcript headers")
    feedback: str
    suggested_line: Optional[str] = Field(default=None, description="A stronger line the speaker could say on this slide")


class Rewrite(BaseModel):
    original: str = Field(description="The weakest passage, quoted exactly from the transcript")
    improved: str


class RehearsalJudgement(BaseModel):
    scorecard: Scorecard
    summary: str = Field(description="2-3 sentence verdict a coach would give")
    strengths: List[str] = Field(default_factory=list)
    priorities: List[Priority] = Field(default_factory=list, description="The 3 most impactful fixes, most important first")
    slide_feedback: List[SlideFeedback] = Field(default_factory=list)
    rewrite: Optional[Rewrite] = None


SYSTEM = """You are an elite pitch coach reviewing one rehearsal of a talk.
You receive the transcript split by slide, measured delivery metrics and a list of
measured coaching cues with ids. Judge the content (story, clarity, persuasion,
structure: hook, problem, solution, market, traction, ask) and the delivery.
Rules:
- Ground every claim in the data. In each priority's evidence list, cite cue ids exactly
  as given (e.g. "pace_fast@2") and/or short exact quotes from the transcript.
- Do not invent numbers, customers or facts that are not in the transcript or slides.
- If the camera was off, do not comment on eye contact or posture.
- Give exactly 3 priorities, most impactful first. Quote the weakest passage exactly in rewrite.original."""


def _prompt(analysis: Dict[str, Any], slides: Sequence[dict], sensors: Dict[str, Any]) -> str:
    by_idx = {s["idx"]: s for s in slides}
    parts = []
    for seg in analysis["segments"]:
        slide = by_idx.get(seg["slide_idx"], {})
        head = f"[Slide {seg['slide_idx'] + 1}" + (f": {slide.get('title')}" if slide.get("title") else "") + "]"
        notes = f"\n(speaker notes: {slide['notes'][:400]})" if slide.get("notes") else ""
        st = seg["stats"]
        parts.append(f"{head} {st['duration']:.0f}s, {st['wpm']:.0f} wpm{notes}\n{st['text'] or '(nothing said)'}")
    metrics = {
        "duration_s": analysis["duration"],
        "target_s": analysis.get("target_seconds"),
        "wpm": analysis["pace"]["wpm"],
        "fillers_per_min": analysis["fillers"]["per_minute"],
        "pitch_range_semitones": analysis["prosody"]["range_st"],
        "upspeak_share": analysis["upspeak"]["share"],
        "pauses": {k: analysis["pauses"][k] for k in ("count", "strategic", "hesitation", "long")},
        "hedges_per_min": analysis["language"]["hedges_per_minute"],
        "camera": bool(sensors.get("camera")),
        "eye_contact_pct": sensors.get("eye_contact_pct"),
        "posture_avg": sensors.get("posture_avg"),
        "delivery_scores_0_100": analysis["scores"],
    }
    cues = [{"id": c["id"], "message": c["message"]} for c in analysis["cues"]]
    return (
        "TRANSCRIPT BY SLIDE:\n" + "\n\n".join(parts)
        + "\n\nMEASURED METRICS:\n" + json.dumps(metrics)
        + "\n\nCOACHING CUES:\n" + json.dumps(cues)
    )


def _verify(j: RehearsalJudgement, analysis: Dict[str, Any]) -> RehearsalJudgement:
    """Drop citations that are neither real cue ids nor text actually said."""
    cue_ids = {c["id"] for c in analysis["cues"]}
    spoken = " ".join(analysis["transcript"].lower().split())
    norm = lambda s: " ".join(s.lower().strip().strip("\"'“”").split())  # noqa: E731
    for p in j.priorities:
        p.evidence = [e for e in p.evidence if e in cue_ids or (len(norm(e)) >= 3 and norm(e) in spoken)]
    if j.rewrite and norm(j.rewrite.original) not in spoken:
        logger.info("Judge rewrite quote not found in transcript; keeping it but flagging.")
        j.rewrite.original = j.rewrite.original.strip()
    valid_numbers = {s["slide_idx"] + 1 for s in analysis["segments"]}
    j.slide_feedback = [f for f in j.slide_feedback if f.slide_number in valid_numbers]
    return j


async def judge_rehearsal(
    analysis: Dict[str, Any], slides: Sequence[dict], sensors: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    if not analysis["word_count"]:
        return None
    judgement, model_ref = await llm_router.generate_structured(
        messages=[LLMMessage(role="user", content=_prompt(analysis, slides, sensors))],
        schema=RehearsalJudgement,
        system_prompt=SYSTEM,
        task="judge",
        max_tokens=6000,
    )
    result = _verify(judgement, analysis).model_dump()
    for f in result["slide_feedback"]:
        f["slide_idx"] = f.pop("slide_number") - 1
    result["model"] = model_ref
    return result

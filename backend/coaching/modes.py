"""
Coaching modes definitions and catalog.
"""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel


class CoachingModeInfo(BaseModel):
    id: str
    name: str
    description: str
    icon: str
    produces_scorecard: bool = True
    default_tts: bool = False
    skill_file: str


COACHING_MODES: Dict[str, CoachingModeInfo] = {
    "pitch_analysis": CoachingModeInfo(
        id="pitch_analysis",
        name="Pitch Analysis",
        description="Full pitch assessment with radar scores, structure breakdown, and top fixes.",
        icon="Target",
        produces_scorecard=True,
        default_tts=False,
        skill_file="pitch_analysis.md"
    ),
    "interactive_coaching": CoachingModeInfo(
        id="interactive_coaching",
        name="Interactive Coaching",
        description="Step-by-step conversational drill. Gives one high-impact tip per turn.",
        icon="MessagesSquare",
        produces_scorecard=False,
        default_tts=True,
        skill_file="interactive_coaching.md"
    ),
    "investor_qa": CoachingModeInfo(
        id="investor_qa",
        name="Investor Q&A",
        description="Simulates a seasoned VC partner probing market size, moats, and traction.",
        icon="Briefcase",
        produces_scorecard=True,
        default_tts=True,
        skill_file="investor_qa.md"
    ),
    "delivery_coaching": CoachingModeInfo(
        id="delivery_coaching",
        name="Delivery & Vocal Coaching",
        description="Focuses on speaking cadence, strategic pauses, vocal confidence, and filler reduction.",
        icon="Activity",
        produces_scorecard=True,
        default_tts=True,
        skill_file="delivery_coaching.md"
    ),
    "objection_handling": CoachingModeInfo(
        id="objection_handling",
        name="Objection Handling",
        description="Practice framing confident, factual answers to skepticism and pushback.",
        icon="ShieldAlert",
        produces_scorecard=False,
        default_tts=True,
        skill_file="objection_handling.md"
    ),
    "pitch_rewrite": CoachingModeInfo(
        id="pitch_rewrite",
        name="Pitch Polish & Rewrite",
        description="Rewrites your pitch to lead with urgency, pain points, and memorable metrics.",
        icon="Sparkles",
        produces_scorecard=False,
        default_tts=False,
        skill_file="pitch_rewrite.md"
    ),
    "web_research": CoachingModeInfo(
        id="web_research",
        name="Market & Competitor Intel",
        description="Injects live market intelligence, competitor comparisons, and industry benchmarks.",
        icon="Globe",
        produces_scorecard=False,
        default_tts=False,
        skill_file="web_research.md"
    ),
    "conversation": CoachingModeInfo(
        id="conversation",
        name="Freeform Conversation",
        description="Supportive sparring partner for open public speaking and warm-ups.",
        icon="Smile",
        produces_scorecard=False,
        default_tts=True,
        skill_file="conversation.md"
    ),
}


def get_mode_catalog() -> List[Dict[str, Any]]:
    return [m.model_dump() for m in COACHING_MODES.values()]


def normalize_mode(mode_str: str) -> str:
    raw = (mode_str or "").strip().lower().replace("-", "_")
    if raw in COACHING_MODES:
        return raw
    aliases = {
        "interactive": "interactive_coaching",
        "interactive_coach": "interactive_coaching",
        "qa": "investor_qa",
        "investor": "investor_qa",
        "delivery": "delivery_coaching",
        "rewrite": "pitch_rewrite",
        "research": "web_research",
    }
    return aliases.get(raw, "pitch_analysis")

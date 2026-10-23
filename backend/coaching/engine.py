"""
Coaching Engine: Prompt composition, streaming generation,
and scorecard calculation.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from backend.coaching.modes import COACHING_MODES, normalize_mode
from backend.coaching.rubric import Scorecard
from backend.config.settings import ROOT_DIR
from backend.llm.router import llm_router
from backend.llm.types import LLMMessage

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are a rigorous startup pitch judge.
Score the speaker's latest pitch turn on the 6-dimension rubric: clarity, structure, persuasion,
delivery, confidence and relevance (0-10 each) plus an overall 0-100 score.
Base every score on evidence in the transcript and the measured delivery metrics; never invent facts.
Typed turns have no audio: score delivery and confidence from wording only.
checklist marks which pitch components are clearly present (hook, problem, solution, market, traction, ask)."""

SKILLS_DIR = ROOT_DIR / "backend" / "skills"


def load_skill_content(filename: str) -> str:
    path = SKILLS_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


class CoachingEngine:
    def build_system_prompt(
        self,
        mode_id: str,
        speech_metrics: Optional[Dict[str, Any]] = None,
        context_md: Optional[str] = None
    ) -> str:
        mode_info = COACHING_MODES.get(normalize_mode(mode_id), COACHING_MODES["pitch_analysis"])
        skill_text = load_skill_content(mode_info.skill_file)

        base_rules = [
            f"You are the world's best Startup Pitch Coach operating in mode: {mode_info.name}.",
            "Give direct, highly actionable, encouraging, and specific feedback.",
            "Do not use filler phrases or generic praise.",
        ]

        if speech_metrics:
            observations = []
            if "words_per_minute" in speech_metrics:
                observations.append(f"Pace: {speech_metrics['words_per_minute']:.1f} WPM (target: 130-160 WPM)")
            if "filler_count" in speech_metrics:
                observations.append(f"Filler words detected: {speech_metrics['filler_count']}")
            if "pause_frequency" in speech_metrics:
                observations.append(f"Pause frequency: {speech_metrics['pause_frequency']:.1f}/min")
            if speech_metrics.get("source") == "text":
                observations.append("This turn was typed, not spoken: do not comment on pace, pauses or voice")
            base_rules.append(
                "SPEECH METRICS FOR THIS TURN: " + ", ".join(observations) + ". "
                "Incorporate these delivery observations directly into your coaching."
            )

        prompt_parts = ["\n".join(base_rules)]
        if skill_text:
            prompt_parts.append("\n=== SPECIFIC MODE INSTRUCTIONS ===\n" + skill_text)
        if context_md:
            prompt_parts.append("\n=== SESSION CONTEXT ===\n" + context_md)

        return "\n\n".join(prompt_parts)

    async def stream_coaching(
        self,
        transcript: str,
        mode: str,
        conversation_history: List[Dict[str, str]],
        speech_metrics: Optional[Dict[str, Any]] = None,
        context_md: Optional[str] = None,
        model_ref: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        system_prompt = self.build_system_prompt(mode, speech_metrics, context_md)

        messages: List[LLMMessage] = []
        for turn in conversation_history[-8:]:
            messages.append(LLMMessage(role=turn.get("role", "user"), content=turn.get("content", "")))

        messages.append(LLMMessage(role="user", content=transcript))

        async for chunk in llm_router.stream_with_fallbacks(
            messages=messages,
            system_prompt=system_prompt,
            model_ref=model_ref,
            task="qa" if normalize_mode(mode) == "investor_qa" else "chat",
            temperature=0.7,
            max_tokens=1500
        ):
            yield chunk

    async def score_turn(
        self,
        transcript: str,
        coach_feedback: str,
        speech_metrics: Optional[Dict[str, Any]] = None,
        model_ref: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Structured 6-D scorecard for one pitch turn, or None if no model could produce one."""
        payload = (
            f"PITCH TRANSCRIPT:\n{transcript}\n\n"
            f"MEASURED DELIVERY METRICS: {json.dumps(speech_metrics or {})}\n\n"
            f"COACH'S WRITTEN FEEDBACK (for context):\n{coach_feedback[:4000]}"
        )
        try:
            card, _ = await llm_router.generate_structured(
                messages=[LLMMessage(role="user", content=payload)],
                schema=Scorecard,
                system_prompt=JUDGE_SYSTEM_PROMPT,
                model_ref=model_ref,
                task="judge",
            )
            return card.model_dump()
        except Exception as e:
            logger.warning(f"Scorecard generation failed: {e}")
            return None


coaching_engine = CoachingEngine()

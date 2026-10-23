"""
Coaching Engine: Prompt composition, streaming generation,
and scorecard calculation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from backend.config.settings import ROOT_DIR
from backend.llm.types import LLMMessage
from backend.llm.router import llm_router
from backend.coaching.modes import COACHING_MODES, normalize_mode
from backend.coaching.rubric import parse_scorecard_from_text

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
            wpm = speech_metrics.get("words_per_minute", 0)
            fillers = speech_metrics.get("filler_count", 0)
            pauses = speech_metrics.get("pause_frequency", 0)
            base_rules.append(
                f"SPEECH METRICS FOR THIS TURN: Pace: {wpm:.1f} WPM (target: 130-160 WPM), "
                f"Filler words detected: {fillers}, Pause frequency: {pauses:.1f}/min. "
                "Incorporate these delivery observations directly into your coaching."
            )

        if mode_info.produces_scorecard:
            base_rules.append(
                "At the very end of your response, output a JSON scorecard block enclosed in ```json ``` with these exact keys:\n"
                "{\n"
                '  "overall": <0-100>,\n'
                '  "clarity": {"score": <0-10>, "feedback": "<one sentence>"},\n'
                '  "structure": {"score": <0-10>, "feedback": "<one sentence>"},\n'
                '  "persuasion": {"score": <0-10>, "feedback": "<one sentence>"},\n'
                '  "delivery": {"score": <0-10>, "feedback": "<one sentence>"},\n'
                '  "confidence": {"score": <0-10>, "feedback": "<one sentence>"},\n'
                '  "relevance": {"score": <0-10>, "feedback": "<one sentence>"},\n'
                '  "checklist": {"hook": bool, "problem": bool, "solution": bool, "market": bool, "traction": bool, "ask": bool},\n'
                '  "strengths": ["<strength 1>", "<strength 2>"],\n'
                '  "top_fixes": ["<fix 1>", "<fix 2>"],\n'
                '  "rewrite": "<one sentence punchier rewrite of the weakest sentence>"\n'
                "}"
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
        user_id: str = "00000000-0000-0000-0000-000000000001"
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
            user_id=user_id,
            temperature=0.7,
            max_tokens=1500
        ):
            yield chunk


coaching_engine = CoachingEngine()

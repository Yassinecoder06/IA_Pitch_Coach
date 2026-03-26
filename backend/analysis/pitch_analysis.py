"""
Pitch Analysis Module
=====================
Provides structured pitch analysis with multiple coaching modes.
"""

import re
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, AsyncGenerator

from backend.llm import get_provider, Message


class CoachingMode(str, Enum):
    """Available coaching modes."""
    PITCH_ANALYSIS = "pitch_analysis"
    INTERACTIVE = "interactive"
    INVESTOR_QA = "investor_qa"


@dataclass
class PitchScores:
    """Structured pitch scores."""
    clarity: int = 0
    language: int = 0
    confidence: int = 0
    topic_relevance: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "clarity": self.clarity,
            "language": self.language,
            "confidence": self.confidence,
            "topic_relevance": self.topic_relevance
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PitchScores":
        return cls(
            clarity=data.get("clarity", 0),
            language=data.get("language", 0),
            confidence=data.get("confidence", 0),
            topic_relevance=data.get("topic_relevance", 0)
        )


# System prompts for different modes
PITCH_COACH_SYSTEM_PROMPT = """You are an expert pitch coach and communication specialist.
Your role is to analyze speeches and pitches, providing structured feedback.

When analyzing a pitch, you MUST return your response in the following exact format:

SCORES:
Clarity: X/10
Language: X/10
Confidence: X/10
Topic Relevance: X/10

ANALYSIS:
[Brief 2-3 sentence overall assessment]

ADVICE:
- [First specific improvement suggestion]
- [Second specific improvement suggestion]
- [Third specific improvement suggestion]

Keep your feedback concise, actionable, and encouraging.
Focus on the most impactful improvements the speaker can make."""


INTERACTIVE_COACH_PROMPT = """You are an interactive pitch coach helping someone refine their startup pitch.
Your role is to have a conversation that helps them improve.

Guidelines:
- Start by acknowledging what they said
- Give one specific piece of feedback
- Ask a follow-up question to help them think deeper
- Be encouraging but direct
- Keep responses under 100 words
- If they've improved from a previous attempt, acknowledge the improvement

Remember to maintain context from previous turns in the conversation."""


INVESTOR_QA_PROMPT = """You are a seasoned startup investor conducting a Q&A session.
Your role is to ask challenging but fair questions about the startup pitch.

Guidelines:
- Ask ONE focused question at a time
- Questions should probe:
  - Problem/solution fit
  - Market size and competition
  - Business model and unit economics
  - Team and execution capability
  - Traction and milestones
- After they answer, briefly evaluate (1-2 sentences) then ask the next question
- Be professional but direct
- Don't be unnecessarily harsh, but don't accept vague answers

Start with a greeting and your first question about the problem they're solving."""


class PitchAnalyzer:
    """
    Analyzes pitches using configured LLM provider.
    Supports multiple coaching modes.
    """

    def __init__(
        self,
        provider_name: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize the analyzer.

        Args:
            provider_name: LLM provider to use (default from settings)
            model: Model to use (default from provider)
        """
        self.provider_name = provider_name
        self.model = model
        self._provider = None

    def _get_provider(self):
        """Get the configured LLM provider."""
        if self._provider is None:
            self._provider = get_provider(self.provider_name)
        return self._provider

    def _get_model(self) -> str:
        """Get the model to use."""
        if self.model:
            return self.model

        # Get default model from settings or provider
        from backend.config import settings

        if settings.default_model:
            return settings.default_model

        # Use provider's first model
        provider = self._get_provider()
        if provider:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                models = loop.run_until_complete(provider.list_models())
                if models:
                    return models[0]
            except RuntimeError:
                pass

        # Fallback
        return "qwen3:0.6b"

    def _get_system_prompt(self, mode: CoachingMode) -> str:
        """Get system prompt for coaching mode."""
        prompts = {
            CoachingMode.PITCH_ANALYSIS: PITCH_COACH_SYSTEM_PROMPT,
            CoachingMode.INTERACTIVE: INTERACTIVE_COACH_PROMPT,
            CoachingMode.INVESTOR_QA: INVESTOR_QA_PROMPT
        }
        return prompts.get(mode, PITCH_COACH_SYSTEM_PROMPT)

    async def analyze(
        self,
        transcript: str,
        filler_count: int = 0,
        word_count: int = 0,
        mode: CoachingMode = CoachingMode.PITCH_ANALYSIS,
        conversation_history: Optional[List[Dict]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Analyze a pitch transcript and stream feedback.

        Args:
            transcript: The transcribed speech text
            filler_count: Number of filler words detected
            word_count: Total word count
            mode: Coaching mode to use
            conversation_history: Previous conversation turns (for interactive modes)

        Yields:
            Text chunks as they're generated
        """
        provider = self._get_provider()
        if provider is None:
            yield "Error: No LLM provider available"
            return

        model = self._get_model()
        system_prompt = self._get_system_prompt(mode)

        # Build messages
        messages = [Message.system(system_prompt)]

        # Add conversation history if provided
        if conversation_history:
            for turn in conversation_history:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role == "user":
                    messages.append(Message.user(content))
                else:
                    messages.append(Message.assistant(content))

        # Build user prompt based on mode
        if mode == CoachingMode.PITCH_ANALYSIS:
            user_prompt = f"""Analyze this pitch transcript and provide structured feedback.

TRANSCRIPT:
{transcript}

METRICS:
- Word count: {word_count}
- Filler words detected: {filler_count}

Please analyze this pitch and provide your feedback in the required format."""

        elif mode == CoachingMode.INTERACTIVE:
            user_prompt = transcript

        elif mode == CoachingMode.INVESTOR_QA:
            if not conversation_history:
                # First message - introduce yourself
                user_prompt = f"Here's my pitch: {transcript}"
            else:
                user_prompt = transcript

        else:
            user_prompt = transcript

        messages.append(Message.user(user_prompt))

        # Stream the response
        async for chunk in provider.stream(messages, model):
            yield chunk

    @staticmethod
    def parse_scores(response: str) -> PitchScores:
        """
        Parse structured scores from LLM response.

        Args:
            response: Full LLM response text

        Returns:
            PitchScores with parsed values
        """
        scores = PitchScores()

        patterns = {
            "clarity": r"Clarity:\s*(\d+)/10",
            "language": r"Language:\s*(\d+)/10",
            "confidence": r"Confidence:\s*(\d+)/10",
            "topic_relevance": r"Topic Relevance:\s*(\d+)/10"
        }

        for attr, pattern in patterns.items():
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                setattr(scores, attr, int(match.group(1)))

        return scores

    @staticmethod
    def extract_tts_summary(response: str) -> str:
        """
        Extract a short summary from LLM response for TTS.
        Keeps the spoken response concise.
        """
        if not response or not response.strip():
            return ""

        text = response.strip()

        # Remove reasoning tags
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"<\/?analysis>", "", text, flags=re.IGNORECASE)

        # Prefer ANALYSIS section
        analysis_match = re.search(
            r"ANALYSIS:\s*\n?(.+?)(?=\n\s*ADVICE:|\n\s*SCORES:|$)",
            text,
            re.IGNORECASE | re.DOTALL
        )
        if analysis_match:
            analysis_text = " ".join(analysis_match.group(1).split())
            if analysis_text:
                return analysis_text

        # Collect meaningful lines
        lines = text.split("\n")
        candidate_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Strip markdown formatting
            line = re.sub(r"^[-*]\s+", "", line)
            line = re.sub(r"^\d+[.)]\s+", "", line)

            # Skip headers and score lines
            if re.match(r"^(SCORES|ANALYSIS|ADVICE):?$", line, re.IGNORECASE):
                continue
            if re.match(
                r"^(Clarity|Language|Confidence|Topic Relevance|Filler\s*Words?)\s*:\s*\d+\s*/\s*10\s*$",
                line,
                re.IGNORECASE
            ):
                continue
            if re.match(r"^[#`*_\-]+$", line):
                continue

            if len(line) >= 8:
                candidate_lines.append(line)
            if len(candidate_lines) >= 2:
                break

        if candidate_lines:
            return " ".join(candidate_lines)

        # Fallback
        cleaned = " ".join(text.split())
        return cleaned[:220].strip()


# Convenience functions for backward compatibility
def parse_scores_from_response(response: str) -> Dict[str, int]:
    """Parse scores from response (backward compatible)."""
    scores = PitchAnalyzer.parse_scores(response)
    return scores.to_dict()

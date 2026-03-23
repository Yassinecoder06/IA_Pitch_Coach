"""
LLM Module using Ollama
=======================
Provides pitch analysis and coaching feedback using local LLM.
Uses qwen3.5:2b model optimized for low-resource machines.
"""

import os
import json
import httpx
from typing import AsyncGenerator, Dict, Any, Optional

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")

# System prompt for pitch coaching
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


async def check_ollama_status() -> Dict[str, Any]:
    """
    Check if Ollama is running and model is available.

    Returns:
        Dict with status information
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{OLLAMA_BASE_URL}/api/tags",
                timeout=5.0
            )

            if response.status_code == 200:
                data = response.json()
                models = [m.get("name", "") for m in data.get("models", [])]

                # Check if our model is available
                model_available = any(
                    OLLAMA_MODEL in model or model.startswith(OLLAMA_MODEL.split(":")[0])
                    for model in models
                )

                return {
                    "status": "ok",
                    "model_available": model_available,
                    "model": OLLAMA_MODEL,
                    "available_models": models
                }

            return {"status": "error", "message": "Unexpected response from Ollama"}

    except httpx.ConnectError:
        return {
            "status": "error",
            "message": "Cannot connect to Ollama. Run 'ollama serve' first."
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def analyze_pitch(
    transcript: str,
    filler_count: int,
    word_count: int
) -> AsyncGenerator[str, None]:
    """
    Analyze a pitch transcript and stream feedback.

    Args:
        transcript: The transcribed speech text
        filler_count: Number of filler words detected
        word_count: Total word count

    Yields:
        Text chunks as they're generated
    """
    # Build the analysis prompt
    prompt = f"""Analyze this pitch transcript and provide structured feedback.

TRANSCRIPT:
{transcript}

METRICS:
- Word count: {word_count}
- Filler words detected: {filler_count}

Please analyze this pitch and provide your feedback in the required format."""

    messages = [
        {"role": "system", "content": PITCH_COACH_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    async for chunk in stream_llm_response(messages):
        yield chunk


async def stream_llm_response(
    messages: list,
    temperature: float = 0.7
) -> AsyncGenerator[str, None]:
    """
    Stream response from Ollama API.

    Args:
        messages: List of message dicts with 'role' and 'content'
        temperature: Sampling temperature (0-1)

    Yields:
        Text chunks as they arrive from the model
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": temperature,
            # Limit context for memory efficiency
            "num_ctx": 2048,
            # Limit response length
            "num_predict": 512
        }
    }

    try:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                url,
                json=payload,
                timeout=120.0
            ) as response:
                if response.status_code == 404:
                    yield f"Model '{OLLAMA_MODEL}' not found. Run: ollama pull {OLLAMA_MODEL}"
                    return

                response.raise_for_status()

                # Process streaming NDJSON response
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)

                            # Extract content from message
                            if "message" in data and "content" in data["message"]:
                                content = data["message"]["content"]
                                if content:
                                    yield content

                            # Check if generation is complete
                            if data.get("done", False):
                                break

                        except json.JSONDecodeError:
                            continue

    except httpx.ConnectError:
        yield "Error: Cannot connect to Ollama. Ensure 'ollama serve' is running."
    except httpx.TimeoutException:
        yield "Error: Request timed out. The model may be overloaded."
    except Exception as e:
        yield f"Error: {str(e)}"


async def get_llm_response(
    messages: list,
    temperature: float = 0.7
) -> str:
    """
    Get a complete (non-streaming) response from the LLM.

    Args:
        messages: List of message dicts
        temperature: Sampling temperature

    Returns:
        Complete response text
    """
    full_response = []

    async for chunk in stream_llm_response(messages, temperature):
        full_response.append(chunk)

    return "".join(full_response)


def parse_scores_from_response(response: str) -> Dict[str, Any]:
    """
    Parse structured scores from LLM response.

    Args:
        response: Full LLM response text

    Returns:
        Dict with parsed scores
    """
    import re

    scores = {
        "clarity": 0,
        "language": 0,
        "confidence": 0,
        "topic_relevance": 0
    }

    # Extract scores using regex
    patterns = {
        "clarity": r"Clarity:\s*(\d+)/10",
        "language": r"Language:\s*(\d+)/10",
        "confidence": r"Confidence:\s*(\d+)/10",
        "topic_relevance": r"Topic Relevance:\s*(\d+)/10"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            scores[key] = int(match.group(1))

    return scores


# Filler words to detect
FILLER_WORDS = [
    "um", "uh", "like", "you know", "basically",
    "actually", "literally", "so", "well", "right"
]


def count_filler_words(text: str) -> Dict[str, int]:
    """
    Count filler words in text.

    Args:
        text: Text to analyze

    Returns:
        Dict mapping filler words to their counts
    """
    import re

    text_lower = text.lower()
    filler_counts = {}

    for filler in FILLER_WORDS:
        # Use word boundaries to avoid partial matches
        pattern = r"\b" + re.escape(filler) + r"\b"
        count = len(re.findall(pattern, text_lower))
        if count > 0:
            filler_counts[filler] = count

    return filler_counts


def get_total_filler_count(text: str) -> int:
    """
    Get total count of all filler words.

    Args:
        text: Text to analyze

    Returns:
        Total filler word count
    """
    filler_counts = count_filler_words(text)
    return sum(filler_counts.values())

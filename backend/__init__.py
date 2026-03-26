"""
AI Pitch Coach Backend Package
==============================
Modular voice-based pitch coaching system.

Modules:
- config: Environment-based configuration
- llm: Multi-provider LLM integration
- voice: STT, TTS, and voice loop
- analysis: Pitch analysis and filler detection
"""

__version__ = "2.0.0"

__all__ = [
    "config",
    "llm",
    "voice",
    "analysis"
]

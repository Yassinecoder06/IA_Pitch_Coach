"""Voice module for AI Pitch Coach."""

from .stt import STTEngine, async_transcribe_audio, load_model, transcribe_audio
from .tts import (
    TTSEngine,
    async_synthesize_speech,
    check_piper_available,
    preload_tts_model,
    split_into_sentences,
    synthesize_speech,
)

__all__ = [
    "STTEngine",
    "transcribe_audio",
    "async_transcribe_audio",
    "load_model",
    "TTSEngine",
    "synthesize_speech",
    "async_synthesize_speech",
    "split_into_sentences",
    "check_piper_available",
    "preload_tts_model"
]

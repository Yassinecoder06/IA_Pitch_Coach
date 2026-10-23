"""Voice module for AI Pitch Coach."""

from .stt import STTEngine, transcribe_audio, async_transcribe_audio, load_model
from .tts import TTSEngine, synthesize_speech, async_synthesize_speech, split_into_sentences, check_piper_available, preload_tts_model

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

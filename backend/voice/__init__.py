"""Voice module for AI Pitch Coach."""
from .stt import STTEngine, transcribe_audio, load_model
from .tts import TTSEngine, synthesize_speech, split_into_sentences
from .voice_loop import VoiceLoop, VoiceLoopState

__all__ = [
    "STTEngine",
    "transcribe_audio",
    "load_model",
    "TTSEngine",
    "synthesize_speech",
    "split_into_sentences",
    "VoiceLoop",
    "VoiceLoopState"
]

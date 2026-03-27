"""
Speech-to-Text Engine
=====================
Provides real-time speech transcription using faster-whisper.
Optimized for low-resource machines with configurable model sizes.
"""

import os
import tempfile
from typing import Optional, Tuple, List
from dataclasses import dataclass
from pathlib import Path

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None


@dataclass
class TranscriptionResult:
    """Result of speech transcription."""
    text: str
    confidence: float
    language: str
    segments: List[dict]


class STTEngine:
    """
    Speech-to-Text engine using faster-whisper.
    Singleton pattern ensures model loads only once.
    """

    _instance: Optional["STTEngine"] = None
    _model: Optional["WhisperModel"] = None

    def __new__(cls) -> "STTEngine":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True

        # Load configuration
        from backend.config import settings

        self.model_size = settings.stt.model_size
        self.device = settings.stt.device
        self.compute_type = settings.stt.compute_type
        self.language = settings.stt.language
        self.models_dir = settings.stt.models_dir

    @staticmethod
    def _cpu_safe_compute_type(compute_type: str) -> str:
        """Return a compute type that is typically supported on CPU."""
        normalized = (compute_type or "").strip().lower()
        if normalized in {"int8", "int16", "float32"}:
            return normalized
        return "int8"

    def load_model(self) -> bool:
        """
        Load the faster-whisper model.

        Returns:
            True if model loaded successfully, False otherwise
        """
        if WhisperModel is None:
            print("[STT] faster-whisper not installed")
            return False

        if STTEngine._model is not None:
            return True

        requested_device = (self.device or "cpu").strip().lower()
        primary_device = "cuda" if requested_device in {"gpu", "nvidia", "cuda"} else "cpu"

        attempts = [(primary_device, self.compute_type)]
        if primary_device != "cpu":
            attempts.append(("cpu", self._cpu_safe_compute_type(self.compute_type)))

        print(f"[STT] Loading faster-whisper model: {self.model_size}")

        last_error: Optional[Exception] = None
        for device, compute_type in attempts:
            try:
                print(f"[STT] Device: {device}, Compute type: {compute_type}")
                STTEngine._model = WhisperModel(
                    self.model_size,
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=4,
                    download_root=str(self.models_dir) if self.models_dir else None
                )
                if device != primary_device:
                    print(f"[STT] Falling back to {device} mode")
                print("[STT] Model loaded successfully")
                return True
            except Exception as e:
                last_error = e
                print(f"[STT] Failed to load model on {device}: {e}")

        if last_error:
            print(f"[STT] Failed to load model: {last_error}")
        return False

    def transcribe(
        self,
        audio_data: bytes,
        language: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Transcribe audio data to text.

        Args:
            audio_data: Raw audio bytes (WAV format)
            language: Language code (e.g., "en"). If None, uses default.

        Returns:
            TranscriptionResult with text and metadata
        """
        if STTEngine._model is None:
            if not self.load_model():
                return TranscriptionResult(
                    text="",
                    confidence=0.0,
                    language="",
                    segments=[]
                )

        # Write audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_data)
            tmp_path = tmp_file.name

        try:
            segments, info = STTEngine._model.transcribe(
                tmp_path,
                beam_size=5,
                language=language or self.language,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=200
                )
            )

            text_parts = []
            segment_data = []
            total_probability = 0.0
            segment_count = 0

            for segment in segments:
                text_parts.append(segment.text.strip())
                segment_data.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "avg_logprob": segment.avg_logprob
                })
                total_probability += segment.avg_logprob
                segment_count += 1

            full_text = " ".join(text_parts)

            # Calculate average confidence
            avg_confidence = 0.0
            if segment_count > 0:
                avg_logprob = total_probability / segment_count
                avg_confidence = min(1.0, max(0.0, (avg_logprob + 1.0)))

            return TranscriptionResult(
                text=full_text,
                confidence=avg_confidence,
                language=info.language,
                segments=segment_data
            )

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return STTEngine._model is not None


# Global instance and convenience functions
_engine: Optional[STTEngine] = None


def get_engine() -> STTEngine:
    """Get the STT engine instance."""
    global _engine
    if _engine is None:
        _engine = STTEngine()
    return _engine


def load_model() -> bool:
    """Load the STT model."""
    return get_engine().load_model()


def transcribe_audio(audio_data: bytes, sample_rate: int = 16000) -> Tuple[str, float]:
    """
    Transcribe audio data to text.

    Args:
        audio_data: Raw audio bytes (WAV format)
        sample_rate: Audio sample rate (default 16kHz)

    Returns:
        Tuple of (transcribed_text, confidence_score)
    """
    result = get_engine().transcribe(audio_data)
    return result.text, result.confidence

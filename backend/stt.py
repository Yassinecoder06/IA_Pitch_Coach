"""
Speech-to-Text Module using faster-whisper
==========================================
Provides real-time speech transcription optimized for low-resource machines.
faster-whisper uses CTranslate2 backend which is 4x faster than original Whisper.
"""

import os
import tempfile
from typing import Optional, Tuple
from faster_whisper import WhisperModel

# Global model instance - loaded once at startup
_model: Optional[WhisperModel] = None

# Model configuration optimized for 8GB RAM
# "tiny" or "base" recommended for low-resource machines
MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "tiny")
# Use "cpu" for machines without GPU, "cuda" for NVIDIA GPU
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")


def load_model() -> WhisperModel:
    """
    Load the faster-whisper model.
    Uses singleton pattern to ensure model loads only once.

    Model sizes:
    - tiny: ~39M params, fastest, good for real-time
    - base: ~74M params, balanced speed/accuracy
    - small: ~244M params, better accuracy, slower

    Returns:
        WhisperModel instance
    """
    global _model

    if _model is None:
        print(f"[STT] Loading faster-whisper model: {MODEL_SIZE}")
        print(f"[STT] Device: {DEVICE}, Compute type: {COMPUTE_TYPE}")

        _model = WhisperModel(
            MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            # Limit CPU threads for low-resource machines
            cpu_threads=4,
            # Download to local models directory
            download_root=os.path.join(os.path.dirname(__file__), "..", "models", "whisper")
        )
        print("[STT] Model loaded successfully")

    return _model


def transcribe_audio(audio_data: bytes, sample_rate: int = 16000) -> Tuple[str, float]:
    """
    Transcribe audio data to text.

    Args:
        audio_data: Raw audio bytes (WAV format)
        sample_rate: Audio sample rate (default 16kHz)

    Returns:
        Tuple of (transcribed_text, confidence_score)
    """
    model = load_model()

    # Write audio to temporary file (faster-whisper requires file path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(audio_data)
        tmp_path = tmp_file.name

    try:
        # Transcribe with beam search for better accuracy
        segments, info = model.transcribe(
            tmp_path,
            beam_size=5,
            language="en",  # Force English for pitch coaching
            vad_filter=True,  # Filter out non-speech segments
            vad_parameters=dict(
                min_silence_duration_ms=500,  # Minimum silence to split
                speech_pad_ms=200  # Padding around speech
            )
        )

        # Collect all segments
        text_parts = []
        total_probability = 0.0
        segment_count = 0

        for segment in segments:
            text_parts.append(segment.text.strip())
            total_probability += segment.avg_logprob
            segment_count += 1

        full_text = " ".join(text_parts)

        # Calculate average confidence (convert log probability to 0-1 scale)
        avg_confidence = 0.0
        if segment_count > 0:
            # Log probability is typically -1 to 0, higher is better
            avg_logprob = total_probability / segment_count
            # Convert to 0-1 scale (rough approximation)
            avg_confidence = min(1.0, max(0.0, (avg_logprob + 1.0)))

        return full_text, avg_confidence

    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def transcribe_streaming(audio_chunks: list) -> str:
    """
    Transcribe a list of audio chunks.
    Combines chunks and processes as a single audio segment.

    Args:
        audio_chunks: List of audio byte chunks

    Returns:
        Transcribed text
    """
    if not audio_chunks:
        return ""

    # Combine all chunks
    combined_audio = b"".join(audio_chunks)

    text, _ = transcribe_audio(combined_audio)
    return text


# Pre-load model on module import if environment variable is set
if os.getenv("PRELOAD_STT_MODEL", "false").lower() == "true":
    load_model()

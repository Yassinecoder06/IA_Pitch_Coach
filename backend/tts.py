"""
Text-to-Speech Module using Piper
=================================
Provides fast, local text-to-speech synthesis.
Piper is optimized for low-resource machines and runs entirely offline.
"""

import os
import io
import re
import wave
import tempfile
import sys
from typing import Optional, List, Tuple, AsyncGenerator
from pathlib import Path

# Piper configuration
MODELS_DIR = Path(os.getenv("PIPER_MODELS_DIR", os.path.join(
    os.path.dirname(__file__), "..", "models", "piper"
)))

# Default voice model (en_US-lessac-medium is good quality/speed balance)
DEFAULT_VOICE = os.getenv("PIPER_VOICE", "en_US-lessac-medium")

# Cache for piper availability and voice
_piper_available: Optional[bool] = None
_piper_voice = None


def check_piper_available() -> Tuple[bool, str]:
    """
    Check if Piper TTS is available.

    Returns:
        Tuple of (is_available, message)
    """
    global _piper_available

    if _piper_available is not None:
        return _piper_available, "Piper TTS is available" if _piper_available else "Piper TTS not found"

    try:
        # Try to import piper
        import piper

        # Check if voice model exists
        model_path, config_path = get_voice_model_path()
        if model_path is None:
            _piper_available = False
            return False, f"Piper installed but voice model not found in {MODELS_DIR}"

        _piper_available = True
        return True, "Piper TTS is available"

    except ImportError:
        _piper_available = False
        return False, "Piper not installed. Run: pip install piper-tts"
    except Exception as e:
        _piper_available = False
        return False, f"Error checking Piper: {e}"


def get_voice_model_path(voice_name: str = DEFAULT_VOICE) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Get paths to voice model files.

    Args:
        voice_name: Name of the voice model

    Returns:
        Tuple of (model_path, config_path) or (None, None) if not found
    """
    # Expected files: voice_name.onnx and voice_name.onnx.json
    model_path = MODELS_DIR / f"{voice_name}.onnx"
    config_path = MODELS_DIR / f"{voice_name}.onnx.json"

    if model_path.exists() and config_path.exists():
        return model_path, config_path

    # Try without full name (just the .onnx file)
    for onnx_file in MODELS_DIR.glob("*.onnx"):
        if voice_name in onnx_file.stem:
            json_file = onnx_file.with_suffix(".onnx.json")
            if json_file.exists():
                return onnx_file, json_file

    return None, None


def load_voice(voice_name: str = DEFAULT_VOICE):
    """
    Load Piper voice model.

    Args:
        voice_name: Name of the voice model

    Returns:
        Piper voice object or None
    """
    global _piper_voice

    if _piper_voice is not None:
        return _piper_voice

    try:
        from piper import PiperVoice

        model_path, config_path = get_voice_model_path(voice_name)
        if model_path is None:
            print(f"[TTS] Voice model '{voice_name}' not found in {MODELS_DIR}")
            return None

        print(f"[TTS] Loading voice model: {model_path}")
        _piper_voice = PiperVoice.load(str(model_path), str(config_path))
        print("[TTS] Voice model loaded successfully")
        return _piper_voice

    except Exception as e:
        print(f"[TTS] Failed to load voice: {e}")
        return None


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences for progressive TTS.

    Args:
        text: Input text

    Returns:
        List of sentences
    """
    if not text.strip():
        return []

    # Split on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Filter and clean
    return [s.strip() for s in sentences if s.strip()]


def synthesize_speech(
    text: str,
    voice: str = DEFAULT_VOICE,
    output_path: Optional[str] = None
) -> Optional[bytes]:
    """
    Synthesize speech from text using Piper.

    Args:
        text: Text to synthesize
        voice: Voice model name
        output_path: Optional path to save WAV file

    Returns:
        WAV audio bytes, or None on failure
    """
    if not text.strip():
        return None

    voice_model = load_voice(voice)
    if voice_model is None:
        return None

    try:
        # Create temporary output file
        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            cleanup = True
        else:
            cleanup = False

        # Get audio parameters from voice config
        sample_rate = voice_model.config.sample_rate

        # Synthesize to WAV file with proper parameters
        with wave.open(output_path, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)

            # synthesize() yields AudioChunk objects - extract audio bytes from each
            for audio_chunk in voice_model.synthesize(text):
                wav_file.writeframes(audio_chunk.audio_int16_bytes)

        # Read the generated audio
        with open(output_path, "rb") as f:
            audio_bytes = f.read()

        # Cleanup temp file
        if cleanup and os.path.exists(output_path):
            os.unlink(output_path)

        return audio_bytes

    except Exception as e:
        print(f"[TTS] Synthesis error: {e}")
        import traceback
        traceback.print_exc()
        return None


async def synthesize_sentence_stream(
    sentences: List[str],
    voice: str = DEFAULT_VOICE
) -> AsyncGenerator[bytes, None]:
    """
    Synthesize sentences one by one and yield audio chunks.
    This allows progressive playback - speaking each sentence as it's ready.

    Args:
        sentences: List of sentences to synthesize
        voice: Voice model name

    Yields:
        WAV audio bytes for each sentence
    """
    for sentence in sentences:
        if sentence.strip():
            audio = synthesize_speech(sentence, voice)
            if audio:
                yield audio


def ensure_models_directory():
    """Create models directory if it doesn't exist."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


# Create models directory on module import
ensure_models_directory()

"""
Text-to-Speech Engine
=====================
Provides fast, local text-to-speech synthesis using Piper.
Optimized for low-resource machines with offline operation.
"""

import os
import re
import wave
import tempfile
import subprocess
import sys
from typing import Any, Optional, List, Tuple, AsyncGenerator
from pathlib import Path
from dataclasses import dataclass
from urllib.request import urlretrieve

try:
    from piper import PiperVoice
except ImportError:
    PiperVoice = None


@dataclass
class TTSResult:
    """Result of speech synthesis."""
    audio: bytes
    sample_rate: int
    duration_seconds: float


class TTSEngine:
    """
    Text-to-Speech engine using Piper.
    Singleton pattern ensures voice model loads only once.
    """

    _instance: Optional["TTSEngine"] = None
    _voice: Optional[Any] = None
    _available: Optional[bool] = None

    def __new__(cls) -> "TTSEngine":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True

        # Load configuration
        from backend.config import settings

        self.voice_name = settings.tts.voice
        self.models_dir = settings.tts.models_dir

        # Ensure models directory exists
        if self.models_dir:
            self.models_dir.mkdir(parents=True, exist_ok=True)

        self.auto_install = os.getenv("AUTO_INSTALL_TTS", "true").lower() == "true"

    def _install_piper_package(self) -> bool:
        """Install piper-tts dynamically when enabled."""
        global PiperVoice

        if PiperVoice is not None:
            return True

        if not self.auto_install:
            return False

        print("[TTS] Piper package missing. Installing piper-tts...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "piper-tts"],
                check=True,
                capture_output=True,
                text=True,
            )
            from piper import PiperVoice as LoadedPiperVoice  # type: ignore

            PiperVoice = LoadedPiperVoice
            print("[TTS] piper-tts installed successfully")
            return True
        except Exception as e:
            print(f"[TTS] Failed to install piper-tts dynamically: {e}")
            return False

    def _download_voice_model(self, voice_name: Optional[str] = None) -> bool:
        """Download voice model files dynamically when enabled."""
        voice_name = voice_name or self.voice_name

        if not self.auto_install or not self.models_dir:
            return False

        model_path = self.models_dir / f"{voice_name}.onnx"
        config_path = self.models_dir / f"{voice_name}.onnx.json"

        if model_path.exists() and config_path.exists():
            return True

        # Currently optimized for the default lessac model.
        if voice_name != "en_US-lessac-medium":
            print(f"[TTS] Auto-download is currently configured for en_US-lessac-medium only. Requested: {voice_name}")
            return False

        model_url = (
            "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"
            "en/en_US/lessac/medium/en_US-lessac-medium.onnx"
        )
        config_url = (
            "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"
            "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
        )

        try:
            print(f"[TTS] Downloading Piper model to {self.models_dir}...")
            urlretrieve(model_url, model_path)
            urlretrieve(config_url, config_path)
            print("[TTS] Piper voice model downloaded successfully")
            return True
        except Exception as e:
            print(f"[TTS] Failed to download voice model: {e}")
            return False

    def _bootstrap_tts(self) -> None:
        """Attempt to make TTS ready by installing package and voice model."""
        if not self.auto_install:
            return

        self._install_piper_package()
        model_path, _ = self._get_voice_paths()
        if model_path is None:
            self._download_voice_model()

    def check_available(self) -> Tuple[bool, str]:
        """
        Check if Piper TTS is available.

        Returns:
            Tuple of (is_available, message)
        """
        if TTSEngine._available is not None:
            msg = "Piper TTS is available" if TTSEngine._available else "Piper TTS not available"
            return TTSEngine._available, msg

        if PiperVoice is None or self._get_voice_paths()[0] is None:
            self._bootstrap_tts()

        if PiperVoice is None:
            TTSEngine._available = False
            return False, "Piper not installed. Run: pip install piper-tts"

        model_path, config_path = self._get_voice_paths()
        if model_path is None:
            TTSEngine._available = False
            return False, f"Voice model not found in {self.models_dir}"

        TTSEngine._available = True
        return True, "Piper TTS is available"

    def _get_voice_paths(self, voice_name: Optional[str] = None) -> Tuple[Optional[Path], Optional[Path]]:
        """Get paths to voice model files."""
        voice_name = voice_name or self.voice_name

        if not self.models_dir:
            return None, None

        # Expected files: voice_name.onnx and voice_name.onnx.json
        model_path = self.models_dir / f"{voice_name}.onnx"
        config_path = self.models_dir / f"{voice_name}.onnx.json"

        if model_path.exists() and config_path.exists():
            return model_path, config_path

        # Try finding any matching onnx file
        for onnx_file in self.models_dir.glob("*.onnx"):
            if voice_name in onnx_file.stem:
                json_file = onnx_file.with_suffix(".onnx.json")
                if json_file.exists():
                    return onnx_file, json_file

        return None, None

    def load_voice(self, voice_name: Optional[str] = None) -> bool:
        """
        Load Piper voice model.

        Args:
            voice_name: Name of the voice model

        Returns:
            True if loaded successfully
        """
        if PiperVoice is None:
            return False

        if TTSEngine._voice is not None:
            return True

        try:
            model_path, config_path = self._get_voice_paths(voice_name)
            if model_path is None:
                print(f"[TTS] Voice model not found")
                return False

            print(f"[TTS] Loading voice model: {model_path}")
            TTSEngine._voice = PiperVoice.load(str(model_path), str(config_path))
            print("[TTS] Voice model loaded successfully")
            return True

        except Exception as e:
            print(f"[TTS] Failed to load voice: {e}")
            return False

    def synthesize(self, text: str, voice_name: Optional[str] = None) -> Optional[bytes]:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            voice_name: Optional voice model name

        Returns:
            WAV audio bytes, or None on failure
        """
        if not text.strip():
            return None

        if TTSEngine._voice is None:
            if not self.load_voice(voice_name):
                return None

        try:
            # Create temporary output file
            fd, output_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

            # Get audio parameters
            sample_rate = TTSEngine._voice.config.sample_rate

            # Synthesize to WAV file
            with wave.open(output_path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)

                for audio_chunk in TTSEngine._voice.synthesize(text):
                    wav_file.writeframes(audio_chunk.audio_int16_bytes)

            # Read the generated audio
            with open(output_path, "rb") as f:
                audio_bytes = f.read()

            # Cleanup
            if os.path.exists(output_path):
                os.unlink(output_path)

            return audio_bytes

        except Exception as e:
            print(f"[TTS] Synthesis error: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def synthesize_stream(
        self,
        sentences: List[str],
        voice_name: Optional[str] = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Synthesize sentences one by one for progressive playback.

        Args:
            sentences: List of sentences to synthesize
            voice_name: Optional voice model name

        Yields:
            WAV audio bytes for each sentence
        """
        for sentence in sentences:
            if sentence.strip():
                audio = self.synthesize(sentence, voice_name)
                if audio:
                    yield audio

    def is_loaded(self) -> bool:
        """Check if voice model is loaded."""
        return TTSEngine._voice is not None


# Global instance and convenience functions
_engine: Optional[TTSEngine] = None


def get_engine() -> TTSEngine:
    """Get the TTS engine instance."""
    global _engine
    if _engine is None:
        _engine = TTSEngine()
    return _engine


def check_piper_available() -> Tuple[bool, str]:
    """Check if Piper TTS is available."""
    return get_engine().check_available()


def get_voice_model_path(voice_name: Optional[str] = None) -> Tuple[Optional[Path], Optional[Path]]:
    """Get paths to voice model files."""
    return get_engine()._get_voice_paths(voice_name)


def synthesize_speech(
    text: str,
    voice: Optional[str] = None,
    output_path: Optional[str] = None
) -> Optional[bytes]:
    """
    Synthesize speech from text.

    Args:
        text: Text to synthesize
        voice: Voice model name
        output_path: Optional path to save WAV file (not used, kept for compatibility)

    Returns:
        WAV audio bytes, or None on failure
    """
    return get_engine().synthesize(text, voice)


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

    # Remove markdown emphasis/list asterisks so TTS doesn't speak "asterisk".
    text = text.replace("*", "")

    # Split on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Filter and clean
    return [s.strip() for s in sentences if s.strip()]

"""Speech metrics extraction for coaching feedback."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import librosa
import numpy as np
import soundfile as sf

try:
    from pyAudioAnalysis import audioBasicIO, ShortTermFeatures
except Exception:  # pragma: no cover
    audioBasicIO = None  # type: ignore
    ShortTermFeatures = None  # type: ignore


@dataclass
class SpeechMetrics:
    words_per_minute: float
    pause_frequency: float
    pause_duration: float
    long_pause_count: int
    energy_variation: float
    rhythm_score: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "words_per_minute": float(self.words_per_minute),
            "pause_frequency": float(self.pause_frequency),
            "pause_duration": float(self.pause_duration),
            "long_pause_count": float(self.long_pause_count),
            "energy_variation": float(self.energy_variation),
            "rhythm_score": float(self.rhythm_score),
        }


def _decode_audio(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    wav_io = io.BytesIO(audio_bytes)
    y, sr = sf.read(wav_io, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
    return y, sr


def _contiguous_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
    regions: List[Tuple[int, int]] = []
    start = None

    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            regions.append((start, idx - 1))
            start = None

    if start is not None:
        regions.append((start, len(mask) - 1))
    return regions


def _rhythm_score(speech_durations: List[float], pause_durations: List[float]) -> float:
    if len(speech_durations) < 2:
        return 6.0

    speech_cv = float(np.std(speech_durations) / (np.mean(speech_durations) + 1e-6))
    pause_cv = float(np.std(pause_durations) / (np.mean(pause_durations) + 1e-6)) if pause_durations else 0.0

    penalty = min(1.0, (speech_cv * 0.7) + (pause_cv * 0.3))
    return float(max(0.0, min(10.0, 10.0 * (1.0 - penalty))))


def analyze_speech_metrics(audio_bytes: bytes, transcript: str, filler_count: int = 0) -> Dict[str, Any]:
    """Compute timing, pause, rhythm, and energy metrics from an utterance."""
    if not audio_bytes:
        return SpeechMetrics(0.0, 0.0, 0.0, 0, 0.0, 0.0).to_dict()

    y, sr = _decode_audio(audio_bytes)

    duration_seconds = float(librosa.get_duration(y=y, sr=sr))
    duration_minutes = max(duration_seconds / 60.0, 1e-6)

    words = len((transcript or "").split())
    words_per_minute = words / duration_minutes

    frame_length = int(0.03 * sr)
    hop_length = int(0.01 * sr)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    silence_threshold = float(np.percentile(rms, 25) * 0.75)
    is_silence = rms <= silence_threshold
    silence_regions = _contiguous_regions(is_silence)
    speech_regions = _contiguous_regions(~is_silence)

    frame_to_sec = hop_length / sr
    pause_durations = [
        (end - start + 1) * frame_to_sec
        for start, end in silence_regions
        if (end - start + 1) * frame_to_sec >= 0.2
    ]

    avg_pause_duration = float(np.mean(pause_durations)) if pause_durations else 0.0
    pause_frequency = float(len(pause_durations) / duration_minutes)
    long_pause_count = int(sum(1 for d in pause_durations if d > 1.5))

    speech_durations = [(end - start + 1) * frame_to_sec for start, end in speech_regions]
    rhythm_score = _rhythm_score(speech_durations, pause_durations)

    if audioBasicIO is not None and ShortTermFeatures is not None:
        try:
            signal = (y * 32767.0).astype(np.int16)
            st_features, _ = ShortTermFeatures.feature_extraction(
                signal,
                sr,
                0.050 * sr,
                0.025 * sr,
            )
            energy_variation = float(np.var(st_features[1]))
        except Exception:
            energy_variation = float(np.var(rms))
    else:
        energy_variation = float(np.var(rms))

    metrics = SpeechMetrics(
        words_per_minute=float(words_per_minute),
        pause_frequency=pause_frequency,
        pause_duration=avg_pause_duration,
        long_pause_count=long_pause_count,
        energy_variation=energy_variation,
        rhythm_score=rhythm_score,
    ).to_dict()

    metrics["filler_count"] = int(filler_count)
    metrics["duration_seconds"] = duration_seconds
    return metrics

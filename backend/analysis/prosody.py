"""Intonation and vocal energy from the recording (librosa pYIN)."""

from __future__ import annotations

import io
from typing import Any, Dict, List, Sequence

import numpy as np
import soundfile as sf

SR = 16000
HOP = 320  # 20 ms
MONOTONE_SEMITONES = 3.0  # p10..p90 spread under this reads as monotone (threshold from iamspeaker)
UPSPEAK_SEMITONES = 2.0
MIN_VOICED_FRAMES = 25


def load_wav(wav: bytes) -> np.ndarray:
    y, sr = sf.read(io.BytesIO(wav), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y


def pitch_track(y: np.ndarray) -> np.ndarray:
    """f0 in Hz per 20 ms frame, NaN where unvoiced."""
    import librosa
    if len(y) < SR // 2:
        return np.full(0, np.nan)
    f0, _, _ = librosa.pyin(y, fmin=65, fmax=400, sr=SR, frame_length=1024, hop_length=HOP)
    return f0


def _semitone_range(f0: np.ndarray) -> float:
    voiced = f0[~np.isnan(f0)]
    if len(voiced) < MIN_VOICED_FRAMES:
        return 0.0
    p10, p90 = np.percentile(voiced, [10, 90])
    return float(12 * np.log2(p90 / p10))


def prosody(y: np.ndarray, f0: np.ndarray, windows_s: float = 8.0, step_s: float = 2.0) -> Dict[str, Any]:
    voiced = f0[~np.isnan(f0)]
    rng = _semitone_range(f0)

    frames = len(y) // HOP
    rms = np.array([np.sqrt(np.mean(y[i * HOP:(i + 1) * HOP] ** 2)) for i in range(frames)]) if frames else np.zeros(0)
    db = 20 * np.log10(np.maximum(rms, 1e-6))
    speech = db[db > max(-50.0, float(np.percentile(db, 20)) + 6)] if len(db) else db
    dynamics = float(np.percentile(speech, 90) - np.percentile(speech, 25)) if len(speech) > 10 else 0.0

    # Sliding windows (8 s wide, every 2 s); "t" is the window centre.
    per = int(windows_s * SR / HOP)
    step = int(step_s * SR / HOP)
    window_ranges: List[Dict[str, float]] = []
    for start in range(0, max(1, len(f0) - per + 1), step):
        r = _semitone_range(f0[start:start + per])
        if r > 0:
            window_ranges.append({"start": round(start * HOP / SR, 1), "t": round((start + per / 2) * HOP / SR, 1),
                                  "range_st": round(r, 1)})

    return {
        "median_hz": round(float(np.median(voiced)), 1) if len(voiced) else 0.0,
        "range_st": round(rng, 1),
        "monotone": len(voiced) >= MIN_VOICED_FRAMES and rng < MONOTONE_SEMITONES,
        "voiced_ratio": round(len(voiced) / max(1, len(f0)), 2),
        "dynamics_db": round(dynamics, 1),
        "windows": window_ranges,
        # Non-overlapping count, so overlapping windows do not inflate it.
        "monotone_windows": sum(1 for w in window_ranges[::max(1, int(windows_s / step_s))] if w["range_st"] < MONOTONE_SEMITONES),
    }


def upspeak(words: Sequence[dict], f0: np.ndarray) -> Dict[str, Any]:
    """Rising pitch at the end of declarative sentences ("uptalk"), which sounds unsure."""
    def median_st(t0: float, t1: float) -> float:
        a, b = int(t0 * SR / HOP), int(t1 * SR / HOP)
        seg = f0[max(0, a):max(0, b)]
        seg = seg[~np.isnan(seg)]
        return float(np.median(seg)) if len(seg) >= 3 else float("nan")

    checked, rising = 0, []
    for i, w in enumerate(words):
        text = w["w"].strip()
        if not text.endswith("."):
            continue  # questions are allowed to rise
        end = w["end"]
        tail = median_st(end - 0.3, end)
        body = median_st(end - 1.0, end - 0.3)
        if np.isnan(tail) or np.isnan(body):
            continue
        checked += 1
        rise = 12 * np.log2(tail / body)
        if rise >= UPSPEAK_SEMITONES:
            rising.append({"word_index": i, "t": round(end, 2), "rise_st": round(float(rise), 1)})
    return {
        "sentences_checked": checked,
        "rising": len(rising),
        "share": round(len(rising) / checked, 2) if checked else 0.0,
        "items": rising,
    }

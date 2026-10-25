"""Turns one recorded take into the full delivery analysis (everything except the LLM judge)."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from backend.analysis import cues as cue_mod
from backend.analysis import language, pronunciation, prosody, scores, timing
from backend.analysis.fillers import find_fillers_in_words


def analyse_take(
    words: Sequence[dict],
    wav: Optional[bytes],
    duration: float,
    slide_events: Sequence[dict],
    slides: Sequence[dict],
    sensors: Dict[str, Any],
    target_seconds: Optional[float],
) -> Dict[str, Any]:
    """
    words: Whisper words [{w, start, end, prob}] with take-relative times
    slides: deck slides [{idx, title, notes, text, time_budget_sec}] (may be empty)
    sensors: browser summary (eye contact, posture, timeline...); may be empty
    """
    words = list(words)
    fillers = find_fillers_in_words(words)
    minutes = max(duration / 60, 1 / 60)

    if wav:
        y = prosody.load_wav(wav)
        f0 = prosody.pitch_track(y)
    else:
        y, f0 = np.zeros(0, dtype=np.float32), np.full(0, np.nan)
    pro = prosody.prosody(y, f0) if len(f0) else {
        "median_hz": 0.0, "range_st": 0.0, "monotone": False, "voiced_ratio": 0.0,
        "dynamics_db": 0.0, "windows": [], "monotone_windows": 0,
    }

    by_slide = {s["idx"]: s for s in slides}
    deck_budget = (target_seconds / len(slides)) if target_seconds and slides else None
    segments: List[Dict[str, Any]] = []
    for seg in timing.slide_segments(slide_events, duration):
        slide = by_slide.get(seg["slide_idx"], {})
        budget = slide.get("time_budget_sec") or deck_budget
        seg["stats"] = timing.segment_stats(seg, words, fillers, sensors.get("timeline") or [], budget)
        seg["title"] = slide.get("title")
        segments.append(seg)

    analysis: Dict[str, Any] = {
        "duration": round(duration, 1),
        "word_count": len(words),
        "transcript": timing.join_words(words),
        "pace": timing.pace(words, duration),
        "pauses": timing.pauses(words, [f["word_index"] for f in fillers]),
        "fillers": {
            "count": len(fillers),
            "per_minute": round(len(fillers) / minutes, 1),
            "by_word": dict(Counter(f["filler"] for f in fillers)),
            "items": fillers,
        },
        "prosody": pro,
        "upspeak": prosody.upspeak(words, f0) if len(f0) else {"sentences_checked": 0, "rising": 0, "share": 0.0, "items": []},
        "language": language.analyse_language(words),
        "clarity": pronunciation.clarity(words, [
            "\n".join([s.get("title") or "", *(s.get("bullets") or [])]) for s in slides
        ]),
        "segments": segments,
        "target_seconds": target_seconds,
    }
    analysis["scores"] = scores.compute_scores(analysis, sensors, target_seconds)
    analysis["delivery_score"] = scores.delivery_score(analysis["scores"])
    analysis["cues"] = cue_mod.generate_cues(analysis, sensors, {s["idx"]: s.get("title") or "" for s in slides})
    return analysis

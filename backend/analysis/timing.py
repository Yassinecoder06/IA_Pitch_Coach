"""Pace, pauses and per-slide timing from Whisper word timestamps."""

from __future__ import annotations

import re
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

PAUSE_MIN_S = 0.3
LONG_PAUSE_S = 3.0
STRATEGIC_MAX_S = 2.5
WINDOW_S = 12.0
WINDOW_STEP_S = 3.0
SENTENCE_END = (".", "?", "!")


@dataclass
class Pause:
    start: float
    end: float
    duration: float
    kind: str  # strategic | hesitation | long
    after_word: int  # index of the word before the pause


def wpm(word_count: int, seconds: float) -> float:
    return round(word_count / (seconds / 60), 1) if seconds > 0 else 0.0


def pace(words: Sequence[dict], duration: float) -> Dict[str, Any]:
    """Overall pace, articulation rate (excluding long pauses) and 15 s windows."""
    if not words:
        return {"wpm": 0.0, "articulation_wpm": 0.0, "windows": [], "variability": 0.0}
    span = max(duration, words[-1]["end"])
    long_pause_time = sum(
        max(0.0, b["start"] - a["end"]) for a, b in zip(words, words[1:], strict=False) if b["start"] - a["end"] >= LONG_PAUSE_S
    )
    speaking_time = max(1.0, words[-1]["end"] - words[0]["start"] - long_pause_time)

    # Sliding windows (12 s wide, every 3 s) so short takes still get a curve.
    windows = []
    t = 0.0
    width = min(WINDOW_S, max(span, 1.0))
    while t + width <= span + 1e-6:
        end = t + width
        n = sum(1 for w in words if t <= w["start"] < end)
        windows.append({"start": round(t, 1), "end": round(end, 1), "wpm": wpm(n, width)})
        t += WINDOW_STEP_S
    speaking_windows = [w["wpm"] for w in windows if w["wpm"] > 0]
    return {
        "wpm": wpm(len(words), span),
        "articulation_wpm": wpm(len(words), speaking_time),
        "windows": windows,
        "variability": round(statistics.pstdev(speaking_windows), 1) if len(speaking_windows) > 1 else 0.0,
    }


def pauses(words: Sequence[dict], filler_word_indices: Sequence[int] = ()) -> Dict[str, Any]:
    """Pauses between words, classified as strategic (after a sentence), hesitation or long."""
    fillers = set(filler_word_indices)
    found: List[Pause] = []
    for i, (a, b) in enumerate(zip(words, words[1:], strict=False)):
        gap = b["start"] - a["end"]
        if gap < PAUSE_MIN_S:
            continue
        if gap >= LONG_PAUSE_S:
            kind = "long"
        elif a["w"].rstrip().endswith(SENTENCE_END + (",", ":", ";")) and gap <= STRATEGIC_MAX_S \
                and i not in fillers and (i + 1) not in fillers:
            kind = "strategic"
        else:
            kind = "hesitation"
        found.append(Pause(round(a["end"], 2), round(b["start"], 2), round(gap, 2), kind, i))

    minutes = max((words[-1]["end"] - words[0]["start"]) / 60, 1 / 60) if words else 1
    count = len(found)
    return {
        "count": count,
        "per_minute": round(count / minutes, 1),
        "avg_duration": round(statistics.mean(p.duration for p in found), 2) if found else 0.0,
        "longest": max((p.duration for p in found), default=0.0),
        "strategic": sum(1 for p in found if p.kind == "strategic"),
        "hesitation": sum(1 for p in found if p.kind == "hesitation"),
        "long": sum(1 for p in found if p.kind == "long"),
        "strategic_share": round(sum(1 for p in found if p.kind == "strategic") / count, 2) if count else 0.0,
        "items": [asdict(p) for p in found],
    }


def slide_segments(slide_events: Sequence[dict], duration: float) -> List[Dict[str, Any]]:
    """[{slide_idx, start, end}] from slide-change events; consecutive repeats are merged."""
    events = sorted(({"index": int(e["index"]), "t": float(e["t"])} for e in slide_events), key=lambda e: e["t"])
    if not events or events[0]["t"] > 0:
        events.insert(0, {"index": events[0]["index"] if events else 0, "t": 0.0})
    segments: List[Dict[str, Any]] = []
    for i, e in enumerate(events):
        end = events[i + 1]["t"] if i + 1 < len(events) else duration
        if end - e["t"] <= 0:
            continue
        if segments and segments[-1]["slide_idx"] == e["index"]:
            segments[-1]["end"] = round(end, 2)
        else:
            segments.append({"slide_idx": e["index"], "start": round(e["t"], 2), "end": round(end, 2)})
    return segments


_ATTACH = re.compile(r"^(?:[.,!?;:%)\]]|'[a-z]|\.\d|n't)", re.IGNORECASE)


def join_words(words: Sequence[dict]) -> str:
    """Whisper word tokens back to text: no space before ".5", "%", "'s", punctuation."""
    out = ""
    for w in words:
        tok = w["w"].strip()
        if not tok:
            continue
        out += tok if (not out or _ATTACH.match(tok)) else " " + tok
    return out


def words_in(words: Sequence[dict], start: float, end: float) -> List[dict]:
    return [w for w in words if start <= (w["start"] + w["end"]) / 2 < end]


def segment_stats(
    segment: dict,
    words: Sequence[dict],
    fillers: Sequence[dict],
    timeline: Sequence[dict],
    budget_s: Optional[float],
) -> Dict[str, Any]:
    start, end = segment["start"], segment["end"]
    seg_words = words_in(words, start, end)
    seg_fillers = [f for f in fillers if start <= f["start"] < end]
    samples = [s for s in timeline if start <= s.get("t", -1) < end]
    gaze = [s["gaze"] for s in samples if s.get("gaze") is not None]
    good = [s for s in samples if s.get("gazeState") is not None]
    duration = end - start
    return {
        "duration": round(duration, 1),
        "budget": budget_s,
        "words": len(seg_words),
        "wpm": wpm(len(seg_words), duration),
        "fillers": len(seg_fillers),
        "eye_contact_pct": round(100 * sum(1 for s in good if s["gazeState"] == "good") / len(good), 1) if good else None,
        "gaze_avg": round(statistics.mean(gaze), 2) if gaze else None,
        "text": join_words(seg_words),
    }

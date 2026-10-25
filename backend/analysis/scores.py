"""
Delivery metric scores, each 0-100 where higher is always better.
A fixed direction makes attempts directly comparable (no "is a higher WPM better?").
"""

from __future__ import annotations

from statistics import mean
from typing import Any, Dict, Optional

TARGET_WPM = (130, 160)
METRICS = ("pace", "fillers", "pauses", "vocal_variety", "energy", "eye_contact", "posture", "composure",
           "language", "clarity", "timing")


def _lerp(x: float, x0: float, y0: float, x1: float, y1: float) -> float:
    if x1 == x0:
        return y1
    t = min(1.0, max(0.0, (x - x0) / (x1 - x0)))
    return y0 + t * (y1 - y0)


def _clamp(v: float) -> float:
    return round(max(0.0, min(100.0, v)), 1)


def pace_score(wpm: float) -> Optional[float]:
    if wpm <= 0:
        return None
    lo, hi = TARGET_WPM
    if lo <= wpm <= hi:
        return 100.0
    return _clamp(_lerp(wpm, lo, 100, 70, 0) if wpm < lo else _lerp(wpm, hi, 100, 230, 0))


def timing_score(duration: float, target: Optional[float]) -> Optional[float]:
    if not target:
        return None
    r = duration / target
    if 0.9 <= r <= 1.05:
        return 100.0
    return _clamp(_lerp(r, 0.9, 100, 0.5, 0) if r < 0.9 else _lerp(r, 1.05, 100, 1.5, 0))


def compute_scores(a: Dict[str, Any], sensors: Dict[str, Any], target_seconds: Optional[float]) -> Dict[str, Optional[float]]:
    pauses = a["pauses"]
    minutes = max(a["duration"] / 60, 1 / 60)
    hesitation_pm = pauses["hesitation"] / minutes
    if pauses["count"] == 0 and a["duration"] > 60:
        pause_score = 50.0  # a minute without a single pause: no room to breathe
    else:
        pause_score = _clamp(60 + 40 * pauses["strategic_share"] - 8 * hesitation_pm - 10 * pauses["long"])

    prosody = a["prosody"]
    variety = None
    if prosody["range_st"] > 0:
        variety = _clamp(_lerp(prosody["range_st"], 2, 20, 8, 100) - 30 * a["upspeak"]["share"])
    energy = _clamp(_lerp(prosody["dynamics_db"], 4, 30, 15, 100)) if prosody["dynamics_db"] > 0 else None

    lang = a["language"]["counts"]
    language = _clamp(100 - 12 * a["language"]["hedges_per_minute"] - 3 * lang.get("jargon", 0)
                      - 2 * lang.get("passive", 0) - 2 * lang.get("weak_verb", 0) - 4 * lang.get("repetition", 0))

    camera = bool(sensors.get("camera"))
    fidget, sway = sensors.get("fidget_avg"), sensors.get("sway_avg")
    composure = None
    if camera and fidget is not None and sway is not None:
        composure = _clamp(100 - 60 * fidget - 40 * sway)

    return {
        "pace": pace_score(a["pace"]["wpm"]),
        "fillers": _clamp(100 * (1 - a["fillers"]["per_minute"] / 12)) if a["word_count"] else None,
        "pauses": pause_score if a["word_count"] else None,
        "vocal_variety": variety,
        "energy": energy,
        "eye_contact": sensors.get("eye_contact_pct") if camera else None,
        "posture": sensors.get("posture_avg") if camera else None,
        "composure": composure,
        "language": language if a["word_count"] else None,
        "clarity": _clamp(_lerp(a["clarity"]["avg_confidence"], 0.6, 40, 0.92, 100)) if a["word_count"] else None,
        "timing": timing_score(a["duration"], target_seconds),
    }


def delivery_score(scores: Dict[str, Optional[float]]) -> Optional[float]:
    values = [v for v in scores.values() if v is not None]
    return round(mean(values), 1) if values else None

"""
Take-to-take comparison (model adapted from iamspeaker, MIT): metric scores are 0-100 and
always higher-is-better, so a delta's sign is unambiguous; coaching cues are matched by
their stable id and classified as resolved, persisting or new.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.analysis.scores import METRICS

STATUS_ORDER = {"resolved": 0, "persisting": 1, "new": 2}


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return round(b - a, 1) if a is not None and b is not None else None


def compare_scores(a: Dict[str, Any], b: Dict[str, Any]) -> List[Dict[str, Any]]:
    keys = ["overall", "delivery", "content", *METRICS]
    out = []
    for k in keys:
        av = a.get(k)
        bv = b.get(k)
        if av is None and bv is None:
            continue
        out.append({"metric": k, "a": av, "b": bv, "delta": _delta(av, bv)})
    return out


def compare_cues(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    a_ids = {c["id"]: c for c in a}
    b_ids = {c["id"]: c for c in b}
    out = [{"id": i, "status": "resolved", "message": c["message"]} for i, c in a_ids.items() if i not in b_ids]
    out += [{"id": i, "status": "persisting" if i in a_ids else "new", "message": c["message"]} for i, c in b_ids.items()]
    return sorted(out, key=lambda x: STATUS_ORDER[x["status"]])


def raw_facts(r: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Plain measurements behind the scores (units differ, so these are shown, not scored)."""
    a = r.get("analysis") or {}
    s = r.get("sensors") or {}
    return {
        "duration_s": r.get("duration_sec"),
        "wpm": (a.get("pace") or {}).get("wpm"),
        "fillers_per_min": (a.get("fillers") or {}).get("per_minute"),
        "pitch_range_st": (a.get("prosody") or {}).get("range_st"),
        "eye_contact_pct": s.get("eye_contact_pct"),
        "posture_avg": s.get("posture_avg"),
        "hedges_per_min": (a.get("language") or {}).get("hedges_per_minute"),
    }


def flat_scores(r: Dict[str, Any]) -> Dict[str, Any]:
    return {**(r.get("scores") or {}), "overall": r.get("overall_score")}


def compare(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    fa, fb = raw_facts(a), raw_facts(b)
    return {
        "a": {k: a[k] for k in ("id", "title", "created_at", "deck_id", "overall_score")},
        "b": {k: b[k] for k in ("id", "title", "created_at", "deck_id", "overall_score")},
        "scores": compare_scores(flat_scores(a), flat_scores(b)),
        "facts": [{"metric": k, "a": fa[k], "b": fb[k], "delta": _delta(fa[k], fb[k])} for k in fa],
        "cues": compare_cues(a.get("cues") or [], b.get("cues") or []),
        "priorities": {
            "a": [p["title"] for p in ((a.get("judge") or {}).get("priorities") or [])],
            "b": [p["title"] for p in ((b.get("judge") or {}).get("priorities") or [])],
        },
    }

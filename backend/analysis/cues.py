"""
Prescriptive coaching notes: "where and what", each backed by a measurement.

Cue ids are stable ("<kind>" or "<kind>@<slide>") so two attempts can be compared:
a cue present before and gone after is resolved, present in both is persisting.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

SEVERITY_ORDER = {"critical": 0, "warn": 1, "info": 2}
MAX_CUES = 14


def _cue(kind: str, severity: str, message: str, slide: Optional[int] = None, t: Optional[float] = None,
         evidence: Optional[str] = None, value: Any = None) -> Dict[str, Any]:
    return {
        "id": f"{kind}@{slide}" if slide is not None else kind,
        "kind": kind,
        "severity": severity,
        "slide_idx": slide,
        "t": t,
        "message": message,
        "evidence": evidence,
        "value": value,
    }


def generate_cues(a: Dict[str, Any], sensors: Dict[str, Any], slide_titles: Dict[int, str]) -> List[Dict[str, Any]]:
    cues: List[Dict[str, Any]] = []

    def cap(text: str) -> str:
        return text[:1].upper() + text[1:]

    def label(idx: int) -> str:
        title = slide_titles.get(idx)
        return f'slide {idx + 1}{f" ({title})" if title else ""}'

    for seg in a["segments"]:
        idx, st = seg["slide_idx"], seg["stats"]
        if st["words"] >= 15 and st["wpm"] > 175:
            cues.append(_cue("pace_fast", "warn", f"You rushed {label(idx)} at {st['wpm']:.0f} wpm. Slow down to 130-160.",
                             idx, seg["start"], f"{st['words']} words in {st['duration']:.0f}s", st["wpm"]))
        elif st["words"] >= 15 and st["wpm"] < 110:
            cues.append(_cue("pace_slow", "info", f"{cap(label(idx))} dragged at {st['wpm']:.0f} wpm.",
                             idx, seg["start"], f"{st['words']} words in {st['duration']:.0f}s", st["wpm"]))
        budget = st.get("budget")
        if budget and st["duration"] > budget * 1.5:
            cues.append(_cue("time_over", "warn", f"{cap(label(idx))} took {st['duration']:.0f}s against a {budget:.0f}s budget.",
                             idx, seg["start"], None, st["duration"]))
        elif budget and st["duration"] < budget * 0.5 and st["duration"] > 0:
            cues.append(_cue("time_under", "info", f"{cap(label(idx))} got only {st['duration']:.0f}s of a {budget:.0f}s budget: is it doing its job?",
                             idx, seg["start"], None, st["duration"]))
        if st["fillers"] >= 3 and st["fillers"] / max(st["duration"] / 60, 1 / 60) > 6:
            cues.append(_cue("filler_cluster", "warn", f"{st['fillers']} fillers on {label(idx)}: pause silently instead.",
                             idx, seg["start"], None, st["fillers"]))
        if st.get("eye_contact_pct") is not None and st["eye_contact_pct"] < 50 and st["duration"] >= 8:
            cues.append(_cue("low_eye_contact", "warn", f"Eye contact dropped to {st['eye_contact_pct']:.0f}% on {label(idx)}: look at the lens, not the slide.",
                             idx, seg["start"], None, st["eye_contact_pct"]))

    pro, up = a["prosody"], a["upspeak"]
    if pro["monotone"]:
        cues.append(_cue("monotone", "critical", f"Your pitch stayed within {pro['range_st']} semitones: vary your voice on key numbers.",
                         evidence=f"pitch range {pro['range_st']} st", value=pro["range_st"]))
    if up["rising"] >= 2 and up["share"] >= 0.3:
        first = up["items"][0]
        cues.append(_cue("upspeak", "warn", f"{up['rising']} statements ended with a rising pitch, which sounds unsure. Land your sentences down.",
                         t=first["t"], value=up["share"]))

    pauses = a["pauses"]
    for p in [p for p in pauses["items"] if p["kind"] == "long"][:2]:
        cues.append(_cue("long_pause", "info", f"A {p['duration']:.1f}s silence at {p['start']:.0f}s: did you lose your thread?",
                         t=p["start"], value=p["duration"]))
    minutes = max(a["duration"] / 60, 1 / 60)
    if pauses["hesitation"] / minutes > 4:
        cues.append(_cue("hesitation", "warn", f"{pauses['hesitation']} mid-sentence hesitations. Rehearse the transitions until they flow.",
                         value=pauses["hesitation"]))
    if a["fillers"]["per_minute"] > 6 and not any(c["kind"] == "filler_cluster" for c in cues):
        top = ", ".join(f'"{k}" x{v}' for k, v in sorted(a["fillers"]["by_word"].items(), key=lambda kv: -kv[1])[:3])
        cues.append(_cue("fillers", "warn", f"{a['fillers']['per_minute']} fillers per minute ({top}).", value=a["fillers"]["per_minute"]))

    lang = a["language"]
    hedges = [i for i in lang["items"] if i["kind"] in ("hedge", "apology")]
    if len(hedges) >= 3:
        examples = ", ".join(sorted({f'"{h["text"]}"' for h in hedges})[:3])
        cues.append(_cue("hedging", "warn", f"{len(hedges)} hedges ({examples}) weaken your claims. State them with evidence.",
                         t=hedges[0]["t"], value=len(hedges)))
    for kind, minimum, text in (("passive", 3, "passive sentences: say who did what"),
                                ("jargon", 2, "jargon words: use plain English"),
                                ("weak_verb", 4, "weak verbs (get, make, do): pick stronger ones")):
        found = [i for i in lang["items"] if i["kind"] == kind]
        if len(found) >= minimum:
            cues.append(_cue(kind, "info", f"{len(found)} {text}.", t=found[0]["t"], value=len(found)))

    clar = a["clarity"]
    if clar["unclear_count"] >= 5:
        sample = ", ".join(f'"{u["text"]}"' for u in clar["unclear"][:3])
        cues.append(_cue("unclear_words", "info", f"{clar['unclear_count']} words were hard to make out ({sample}). Articulate the ends of words.",
                         t=clar["unclear"][0]["t"], value=clar["unclear_count"]))
    if clar["key_terms_missing"]:
        cues.append(_cue("missing_key_terms", "info",
                         f"Key terms from your slides never came up: {', '.join(clar['key_terms_missing'][:4])}.",
                         value=len(clar["key_terms_missing"])))

    if sensors.get("camera"):
        duration = max(sensors.get("duration_s") or a["duration"], 1)
        for issue, secs in (sensors.get("posture_issue_seconds") or {}).items():
            if secs / duration >= 0.2:
                text = {"arms_crossed": "Arms crossed", "slouching": "Slouching", "hand_on_face": "Hand on your face",
                        "shoulders_tilted": "Shoulders tilted", "hands_hidden": "Hands out of view"}.get(issue, issue)
                cues.append(_cue(f"posture_{issue}", "warn" if issue in ("arms_crossed", "slouching") else "info",
                                 f"{text} for {secs:.0f}s ({100 * secs / duration:.0f}% of the talk).", value=secs))
        if (sensors.get("fidget_avg") or 0) > 0.5:
            cues.append(_cue("fidget", "warn", "Frequent fidgeting. Rest your hands between gestures.", value=sensors["fidget_avg"]))
        if (sensors.get("sway_avg") or 0) > 0.5:
            cues.append(_cue("sway", "info", "You swayed side to side. Plant your feet.", value=sensors["sway_avg"]))
        if sensors.get("face_present_pct") is not None and sensors["face_present_pct"] < 60:
            cues.append(_cue("off_camera", "warn", f"Your face was in frame only {sensors['face_present_pct']:.0f}% of the time.",
                             value=sensors["face_present_pct"]))

    cues.sort(key=lambda c: (SEVERITY_ORDER[c["severity"]], c["t"] if c["t"] is not None else 1e9))
    return cues[:MAX_CUES]

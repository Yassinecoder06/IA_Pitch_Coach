import io
import wave

import numpy as np
import pytest

from backend.analysis import language, pronunciation, prosody, scores, timing
from backend.analysis.cues import generate_cues
from backend.analysis.report import analyse_take

SR = 16000


def words_from(text: str, start=0.0, step=0.4, gap_after=None, prob=0.95):
    """Evenly timed words; gap_after = {word_index: extra_seconds}."""
    out, t = [], start
    for i, w in enumerate(text.split()):
        out.append({"w": w, "start": round(t, 3), "end": round(t + step * 0.8, 3), "prob": prob})
        t += step + (gap_after or {}).get(i, 0.0)
    return out


def tone_wav(seconds: float, hz) -> bytes:
    t = np.arange(int(seconds * SR)) / SR
    freq = hz(t) if callable(hz) else np.full_like(t, hz)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    y = 0.3 * (np.sin(phase) + 0.5 * np.sin(2 * phase)) / 1.5
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((y * 32767).astype("<i2").tobytes())
    return buf.getvalue()


# ---------- timing ----------
def test_pace_counts_words_per_minute():
    w = words_from(" ".join(["word"] * 100), step=0.4)  # 150 wpm
    p = timing.pace(w, 40.0)
    assert p["wpm"] == pytest.approx(150, abs=2)
    assert len(p["windows"]) >= 2


def test_pauses_are_classified():
    text = "We grew fast. Then we stalled and um we recovered later on."
    w = words_from(text, gap_after={2: 0.8, 6: 0.6, 9: 4.0})
    p = timing.pauses(w, filler_word_indices=[7])
    kinds = [i["kind"] for i in p["items"]]
    assert kinds == ["strategic", "hesitation", "long"]
    assert p["strategic_share"] == pytest.approx(1 / 3, abs=0.01)


def test_slide_segments_merge_and_cover_the_take():
    segs = timing.slide_segments([{"index": 0, "t": 0}, {"index": 1, "t": 10}, {"index": 1, "t": 12}, {"index": 2, "t": 20}], 30)
    assert segs == [
        {"slide_idx": 0, "start": 0.0, "end": 10.0},
        {"slide_idx": 1, "start": 10.0, "end": 20.0},
        {"slide_idx": 2, "start": 20.0, "end": 30.0},
    ]


# ---------- prosody ----------
def test_flat_voice_is_monotone_and_glide_is_not():
    flat = prosody.load_wav(tone_wav(4, 180))
    rec = prosody.prosody(flat, prosody.pitch_track(flat))
    assert rec["monotone"] is True and rec["range_st"] < 1

    glide = prosody.load_wav(tone_wav(4, lambda t: 140 * 2 ** (t / 4)))  # one octave up
    rec = prosody.prosody(glide, prosody.pitch_track(glide))
    assert rec["monotone"] is False and 8 < rec["range_st"] < 13


def test_upspeak_detects_rising_statement_endings():
    frames = int(6 * SR / prosody.HOP)
    f0 = np.full(frames, 150.0)
    t = np.arange(frames) * prosody.HOP / SR
    f0[(t > 2.7) & (t <= 3.0)] = 190.0  # rise at the end of "growth."
    words = [
        {"w": "Revenue", "start": 1.5, "end": 2.0, "prob": 1},
        {"w": "growth.", "start": 2.2, "end": 3.0, "prob": 1},
        {"w": "Margins", "start": 4.0, "end": 4.6, "prob": 1},
        {"w": "improved.", "start": 4.7, "end": 5.5, "prob": 1},
        {"w": "Right?", "start": 5.6, "end": 5.9, "prob": 1},
    ]
    up = prosody.upspeak(words, f0)
    assert up["sentences_checked"] == 2
    assert up["rising"] == 1 and up["items"][0]["t"] == 3.0


# ---------- language ----------
def test_language_flags_hedges_jargon_passive_and_repetition():
    text = ("I think our platform is kind of unique. The app was built by us to leverage data. "
            "We get customers. Pricing pricing pricing pricing matters. Sorry, if that makes sense.")
    res = language.analyse_language(words_from(text))
    kinds = {i["kind"] for i in res["items"]}
    assert {"hedge", "jargon", "passive", "weak_verb", "repetition", "apology"} <= kinds
    texts = {i["text"] for i in res["items"]}
    assert "i think" in texts and "kind of" in texts and "was built" in texts and "if that makes sense" in texts


def test_clean_language_has_no_flags():
    res = language.analyse_language(words_from("Retailers save thirty percent on every shipment we book."))
    assert res["items"] == []


# ---------- clarity ----------
def test_clarity_flags_low_confidence_and_missing_terms():
    w = words_from("FreightPilot pools volume for retailers", prob=0.95)
    w[2]["prob"] = 0.2
    res = pronunciation.clarity(w, ["Meet FreightPilot and Carrierhub\nSave 30% with marketplace automation"])
    assert res["unclear_count"] == 1 and res["unclear"][0]["text"] == "volume"
    assert "carrierhub" in res["key_terms_missing"] and "marketplace" in res["key_terms_missing"]
    assert "freightpilot" not in res["key_terms_missing"]  # said clearly
    assert "meet" not in res["key_terms_missing"] and "save" not in res["key_terms_missing"]  # sentence starts


def test_figures_and_sentence_starts_are_not_key_terms():
    assert pronunciation.key_terms(["Raising $1.5M seed. Open with the pain"]) == []


# ---------- scores & cues ----------
def test_scores_are_bounded_and_directional():
    assert scores.pace_score(145) == 100
    assert scores.pace_score(200) < scores.pace_score(170) < 100
    assert scores.timing_score(60, 60) == 100 and scores.timing_score(90, 60) == 0
    assert scores.timing_score(60, None) is None


def test_full_analysis_produces_scores_segments_and_stable_cue_ids():
    rushed = " ".join(["fast"] * 60) + "."  # 60 words in 15 s on slide 2 = 240 wpm
    w = words_from("I think um we basically help retailers ship.", step=0.5)
    w += words_from(rushed, start=10.0, step=0.25)
    w += words_from("So, uh, we are raising one million.", start=26.0, step=0.5)
    slides = [
        {"idx": 0, "title": "Problem", "notes": "", "time_budget_sec": 10},
        {"idx": 1, "title": "Solution", "notes": "", "time_budget_sec": 10},
        {"idx": 2, "title": "Ask", "notes": "", "time_budget_sec": 10},
    ]
    sensors = {"camera": True, "eye_contact_pct": 40.0, "posture_avg": 70.0, "fidget_avg": 0.6, "sway_avg": 0.1,
               "posture_issue_seconds": {"arms_crossed": 12}, "duration_s": 30, "face_present_pct": 95,
               "timeline": [{"t": t, "gaze": 0.2, "gazeState": "lost"} for t in range(10, 26)]}
    a = analyse_take(w, tone_wav(30, lambda t: 150 + 30 * np.sin(t)), 30.0,
                     [{"index": 0, "t": 0}, {"index": 1, "t": 10}, {"index": 2, "t": 25.5}], slides, sensors, 30)

    assert [s["slide_idx"] for s in a["segments"]] == [0, 1, 2]
    assert a["segments"][1]["stats"]["wpm"] > 200
    ids = {c["id"] for c in a["cues"]}
    assert "pace_fast@1" in ids and "low_eye_contact@1" in ids and "fidget" in ids and "posture_arms_crossed" in ids
    assert a["fillers"]["count"] >= 4
    assert set(a["scores"]) == set(scores.METRICS)
    assert all(v is None or 0 <= v <= 100 for v in a["scores"].values())
    assert a["scores"]["eye_contact"] == 40.0 and 0 < a["delivery_score"] < 100


def test_analysis_without_camera_or_deck():
    a = analyse_take(words_from("We help retailers ship for less."), None, 4.0, [], [], {}, None)
    assert a["scores"]["eye_contact"] is None and a["scores"]["timing"] is None
    assert a["segments"] == [{"slide_idx": 0, "start": 0.0, "end": 4.0, "stats": a["segments"][0]["stats"], "title": None}]
    assert generate_cues(a, {}, {}) == a["cues"]


def test_join_words_reattaches_split_tokens():
    w = [{"w": t, "start": 0, "end": 0} for t in ["raising", "$1", ".5", "million,", "it", "'s", "30", "%", "growth."]]
    assert timing.join_words(w) == "raising $1.5 million, it's 30% growth."


def test_short_take_still_gets_a_pace_curve():
    p = timing.pace(words_from(" ".join(["w"] * 40), step=0.45), 19.0)
    assert len(p["windows"]) >= 3


def test_split_brand_name_counts_as_said():
    w = words_from("Freight Pilot books carriers")
    assert "freightpilot" not in pronunciation.clarity(w, ["We are FreightPilot"])["key_terms_missing"]


def test_cue_labels_keep_title_case():
    a = analyse_take(words_from(" ".join(["fast"] * 40) + ".", step=0.2), None, 8.0, [], [{"idx": 0, "title": "FreightPilot Seed"}], {}, None)
    assert any("FreightPilot Seed" in c["message"] for c in a["cues"])

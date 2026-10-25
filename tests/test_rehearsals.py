import numpy as np
import pytest

from backend.rehearsals import service as service_mod
from backend.voice import streaming
from backend.voice.stt import TranscriptionResult

SR = 16000
SENTENCES = [
    "Retailers overpay for shipping by thirty percent.",
    "We pool their volume and book carriers automatically.",
    "We are raising one point five million dollars.",
]


def voiced(seconds, hz=150.0):
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * (hz + 20 * np.sin(2 * np.pi * 0.5 * t)) * t) * 32767).astype("<i2").tobytes()


def silence(seconds):
    return np.zeros(int(seconds * SR), dtype="<i2").tobytes()


@pytest.fixture
def fake_ai(monkeypatch):
    calls = {"n": 0}

    async def transcribe(wav, context):
        text = SENTENCES[calls["n"] % len(SENTENCES)]
        calls["n"] += 1
        words = [{"w": w, "start": 0.1 + 0.3 * i, "end": 0.35 + 0.3 * i, "prob": 0.9} for i, w in enumerate(text.split())]
        return TranscriptionResult(text=text, confidence=0.9, language="en", segments=[], words=words)

    async def judge(analysis, slides, sensors):
        return {
            "scorecard": {"overall": 72, **{k: {"score": 7, "feedback": "ok"} for k in
                          ("clarity", "structure", "persuasion", "delivery", "confidence", "relevance")},
                          "checklist": {}, "strengths": [], "top_fixes": [], "rewrite": None},
            "summary": "Solid.", "strengths": ["Clear ask"], "priorities": [], "slide_feedback": [], "rewrite": None,
            "model": "fake:judge",
        }

    monkeypatch.setattr(streaming, "default_transcriber", transcribe)
    monkeypatch.setattr(service_mod, "judge_rehearsal", judge)
    return calls


def test_record_analyse_report_and_qa(owner_client, fake_ai):
    c = owner_client
    deck = c.post("/api/decks", json={"title": "Seed", "slides": [
        {"title": "Problem", "time_budget_sec": 5}, {"title": "Solution", "time_budget_sec": 5}, {"title": "Ask", "time_budget_sec": 5},
    ]}).json()
    reh = c.post("/api/rehearsals", json={"title": "Take 1", "deck_id": deck["id"], "target_seconds": 15}).json()
    assert reh["status"] == "ready"

    pcm = voiced(3) + silence(0.8) + voiced(3) + silence(0.8) + voiced(3) + silence(0.5)
    with c.websocket_connect("/ws/rehearsal") as ws:
        ws.send_json({"type": "start", "sample_rate": SR, "rehearsal_id": reh["id"]})
        assert ws.receive_json() == {"type": "started", "rehearsal_id": reh["id"]}
        for i in range(0, len(pcm), 3200):
            ws.send_bytes(pcm[i:i + 3200])
            if i == 3200 * 19:
                ws.send_json({"type": "slide", "index": 1, "t": 3.8})
            if i == 3200 * 38:
                ws.send_json({"type": "slide", "index": 2, "t": 7.6})
        ws.send_json({"type": "stop", "sensors": {"camera": False, "timeline": []}})
        events = []
        while True:
            m = ws.receive_json()
            events.append(m)
            if m["type"] in ("analysis_done", "error"):
                break

    kinds = [e["type"] for e in events]
    assert "words" in kinds and "transcript_final" in kinds and kinds[-1] == "analysis_done"
    assert "measuring delivery" in [e.get("step") for e in events]

    report = c.get(f"/api/rehearsals/{reh['id']}").json()
    assert report["status"] == "done", report.get("error")
    assert report["transcript"].startswith("Retailers overpay")
    assert [s["slide_idx"] for s in report["segments"]] == [0, 1, 2]
    assert report["judge"]["summary"] == "Solid."
    assert report["scores"]["content"] == 72 and report["scores"]["delivery"] is not None
    assert report["overall_score"] == pytest.approx((72 + report["scores"]["delivery"]) / 2, abs=0.1)
    assert report["analysis"]["pace"]["wpm"] > 0
    assert report["has_audio"]
    audio = c.get(f"/api/rehearsals/{reh['id']}/audio")
    assert audio.status_code == 200 and len(audio.content) > 1000

    listed = c.get(f"/api/rehearsals?deck_id={deck['id']}").json()["rehearsals"]
    assert [r["id"] for r in listed] == [reh["id"]]

    qa = c.post(f"/api/rehearsals/{reh['id']}/qa").json()
    session = c.get(f"/api/sessions/{qa['session_id']}").json()["session"]
    assert session["current_mode"] == "investor_qa"
    assert "Retailers overpay" in session["context_md"] and "Slide 1: Problem" in session["context_md"]

    assert c.delete(f"/api/rehearsals/{reh['id']}").status_code == 200
    assert c.get(f"/api/rehearsals/{reh['id']}").status_code == 404


def test_reanalyse_uses_stored_audio(owner_client, fake_ai, monkeypatch):
    c = owner_client
    reh = c.post("/api/rehearsals", json={"title": "Take"}).json()
    with c.websocket_connect("/ws/rehearsal") as ws:
        ws.send_json({"type": "start", "sample_rate": SR, "rehearsal_id": reh["id"]})
        ws.receive_json()
        ws.send_bytes(voiced(3) + silence(0.6))
        ws.send_json({"type": "stop", "sensors": {}})
        while ws.receive_json()["type"] != "analysis_done":
            pass

    async def better_judge(analysis, slides, sensors):
        return {"scorecard": {"overall": 90, **{k: {"score": 9, "feedback": ""} for k in
                ("clarity", "structure", "persuasion", "delivery", "confidence", "relevance")}},
                "summary": "Better model.", "strengths": [], "priorities": [], "slide_feedback": [], "rewrite": None}
    monkeypatch.setattr(service_mod, "judge_rehearsal", better_judge)
    again = c.post(f"/api/rehearsals/{reh['id']}/reanalyse").json()
    assert again["judge"]["summary"] == "Better model." and again["scores"]["content"] == 90
    assert again["analysis"]["prosody"]["range_st"] > 0  # audio was decoded back from Opus


def test_report_survives_missing_ai(owner_client, fake_ai, monkeypatch):
    async def no_ai(*a, **k):
        raise RuntimeError("no model configured")
    monkeypatch.setattr(service_mod, "judge_rehearsal", no_ai)
    c = owner_client
    reh = c.post("/api/rehearsals", json={"title": "Offline"}).json()
    with c.websocket_connect("/ws/rehearsal") as ws:
        ws.send_json({"type": "start", "sample_rate": SR, "rehearsal_id": reh["id"]})
        ws.receive_json()
        ws.send_bytes(voiced(3) + silence(0.6))
        ws.send_json({"type": "stop", "sensors": {}})
        while ws.receive_json()["type"] != "analysis_done":
            pass
    report = c.get(f"/api/rehearsals/{reh['id']}").json()
    assert report["status"] == "done" and report["judge"] is None
    assert "AI feedback unavailable" in report["error"]
    assert report["overall_score"] == report["scores"]["delivery"]


def test_rehearsals_are_profile_scoped(owner_client):
    c = owner_client
    reh = c.post("/api/rehearsals", json={"title": "Mine"}).json()
    other = c.post("/api/profiles", json={"name": "Other"}).json()
    c.post(f"/api/profiles/{other['id']}/select", json={})
    assert c.get(f"/api/rehearsals/{reh['id']}").status_code == 404
    with c.websocket_connect("/ws/rehearsal") as ws:
        ws.send_json({"type": "start", "sample_rate": SR, "rehearsal_id": reh["id"]})
        assert ws.receive_json() == {"type": "error", "message": "Rehearsal not found."}
    assert c.post("/api/rehearsals", json={"title": "x", "deck_id": "nope"}).status_code == 404

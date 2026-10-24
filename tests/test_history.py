from datetime import timedelta

from backend.db import database
from backend.db.models import Rehearsal
from backend.db.types import utcnow
from backend.rehearsals import privacy
from backend.rehearsals.compare import compare_cues, compare_scores
from backend.rehearsals.service import recordings_dir
from tests.conftest import run


def test_compare_scores_and_cues():
    pairs = {p["metric"]: p for p in compare_scores({"overall": 60, "pace": 50, "eye_contact": None},
                                                    {"overall": 72, "pace": 40, "eye_contact": 80})}
    assert pairs["overall"]["delta"] == 12 and pairs["pace"]["delta"] == -10
    assert pairs["eye_contact"]["delta"] is None and pairs["eye_contact"]["b"] == 80
    changes = compare_cues(
        [{"id": "pace_fast@1", "message": "fast"}, {"id": "monotone", "message": "flat"}],
        [{"id": "monotone", "message": "flat"}, {"id": "hedging", "message": "hedges"}],
    )
    assert [(c["id"], c["status"]) for c in changes] == [("pace_fast@1", "resolved"), ("monotone", "persisting"), ("hedging", "new")]


async def _make_done(profile_id, overall, scores, cues, wpm, eye, created=None, audio=None):
    async with database.session() as s:
        r = Rehearsal(profile_id=profile_id, title=f"take {overall}", status="done", overall_score=overall,
                      scores_json=scores, cues_json=cues, duration_sec=60,
                      metrics_json={"pace": {"wpm": wpm}, "fillers": {"per_minute": 2}, "prosody": {"range_st": 5},
                                    "language": {"hedges_per_minute": 1}},
                      sensors_json={"eye_contact_pct": eye, "posture_avg": 80}, audio_path=audio)
        if created:
            r.created_at = created
        s.add(r)
        await s.flush()
        return r.id


def test_compare_endpoint(owner_client, default_profile_id):
    a = run(_make_done(default_profile_id, 60, {"pace": 50, "delivery": 60}, [{"id": "monotone", "message": "flat"}], 190, 40))
    b = run(_make_done(default_profile_id, 75, {"pace": 90, "delivery": 80}, [], 150, 70))
    res = owner_client.get(f"/api/rehearsals/compare?a={a}&b={b}")
    assert res.status_code == 200
    body = res.json()
    scores = {s["metric"]: s for s in body["scores"]}
    assert scores["overall"]["delta"] == 15 and scores["pace"]["delta"] == 40
    facts = {f["metric"]: f for f in body["facts"]}
    assert facts["wpm"]["delta"] == -40 and facts["eye_contact_pct"]["delta"] == 30
    assert body["cues"] == [{"id": "monotone", "status": "resolved", "message": "flat"}]
    assert owner_client.get(f"/api/rehearsals/compare?a={a}&b=nope").status_code == 404


def test_goals_are_checked_against_the_latest_take(owner_client, default_profile_id):
    c = owner_client
    run(_make_done(default_profile_id, 60, {"eye_contact": 40}, [], 190, 40, created=utcnow() - timedelta(days=1)))
    run(_make_done(default_profile_id, 75, {"eye_contact": 70}, [], 150, 70))
    assert c.post("/api/rehearsals/goals", json={"metric": "eye_contact_pct", "op": "gte", "target": 65}).status_code == 200
    assert c.post("/api/rehearsals/goals", json={"metric": "wpm", "op": "between", "target": 130, "target_max": 145}).status_code == 200
    assert c.post("/api/rehearsals/goals", json={"metric": "bogus", "op": "gte", "target": 1}).status_code == 422
    assert c.post("/api/rehearsals/goals", json={"metric": "wpm", "op": "between", "target": 160, "target_max": 130}).status_code == 422
    goals = {g["metric"]: g for g in c.get("/api/rehearsals/goals").json()["goals"]}
    assert goals["eye_contact_pct"]["status"] == "met" and goals["eye_contact_pct"]["latest"] == 70
    assert goals["eye_contact_pct"]["recent_avg"] == 55
    assert goals["wpm"]["status"] == "not_met"
    assert c.delete(f"/api/rehearsals/goals/{goals['wpm']['id']}").status_code == 200


def test_privacy_retention_and_manual_delete(owner_client, default_profile_id):
    c = owner_client
    folder = recordings_dir()
    (folder / "old.ogg").write_bytes(b"x")
    (folder / "new.ogg").write_bytes(b"y")
    old = run(_make_done(default_profile_id, 60, {}, [], 150, 50, created=utcnow() - timedelta(days=40), audio="old.ogg"))
    new = run(_make_done(default_profile_id, 70, {}, [], 150, 50, audio="new.ogg"))

    assert c.get("/api/rehearsals/privacy").json() == {"keep_recordings": True, "recording_retention_days": None}
    assert c.put("/api/rehearsals/privacy", json={"keep_recordings": True, "recording_retention_days": 30}).status_code == 200
    assert not (folder / "old.ogg").exists() and (folder / "new.ogg").exists()
    assert c.get(f"/api/rehearsals/{old}").json()["has_audio"] is False
    assert c.get(f"/api/rehearsals/{old}").json()["overall_score"] == 60  # report kept

    assert c.delete(f"/api/rehearsals/{new}/audio").status_code == 200
    assert not (folder / "new.ogg").exists()
    assert c.delete(f"/api/rehearsals/{new}/audio").status_code == 404

    c.put("/api/rehearsals/privacy", json={"keep_recordings": False})
    assert run(privacy.get_settings())["keep_recordings"] is False


def test_takes_are_not_kept_when_recordings_are_off(owner_client, monkeypatch):
    import numpy as np

    from backend.rehearsals import service as service_mod
    from backend.voice import streaming
    from backend.voice.stt import TranscriptionResult
    from tests.test_rehearsals import silence, voiced  # noqa: F401  (reuse the signal helpers)

    async def transcribe(wav, context):
        return TranscriptionResult(text="We ship.", confidence=1, language="en", segments=[],
                                   words=[{"w": "We", "start": 0.1, "end": 0.3, "prob": 1}, {"w": "ship.", "start": 0.4, "end": 0.8, "prob": 1}])

    async def judge(*a, **k):
        return None
    monkeypatch.setattr(streaming, "default_transcriber", transcribe)
    monkeypatch.setattr(service_mod, "judge_rehearsal", judge)
    c = owner_client
    c.put("/api/rehearsals/privacy", json={"keep_recordings": False})
    reh = c.post("/api/rehearsals", json={"title": "private"}).json()
    with c.websocket_connect("/ws/rehearsal") as ws:
        ws.send_json({"type": "start", "sample_rate": 16000, "rehearsal_id": reh["id"]})
        ws.receive_json()
        t = np.arange(16000 * 3) / 16000
        ws.send_bytes((0.3 * np.sin(2 * np.pi * 160 * t) * 32767).astype("<i2").tobytes() + bytes(16000))
        ws.send_json({"type": "stop", "sensors": {}})
        while ws.receive_json()["type"] != "analysis_done":
            pass
    report = c.get(f"/api/rehearsals/{reh['id']}").json()
    assert report["status"] == "done" and report["has_audio"] is False
    assert report["analysis"]["prosody"]["range_st"] >= 0  # still analysed from the live audio
    assert not list(recordings_dir().glob(f"{reh['id']}.*"))

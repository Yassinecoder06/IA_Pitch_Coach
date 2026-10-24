from datetime import timedelta

from backend.arcade.service import ArcadeService
from backend.db import database
from backend.db.models import GameRun, Rehearsal
from backend.db.types import utcnow
from tests.conftest import run


def test_runs_personal_bests_and_validation(owner_client):
    c = owner_client
    first = c.post("/api/arcade/runs", json={"game": "eye_lock", "difficulty": "easy", "score": 60}).json()
    assert first["personal_best"] is True
    assert c.post("/api/arcade/runs", json={"game": "eye_lock", "difficulty": "easy", "score": 40}).json()["personal_best"] is False
    better = c.post("/api/arcade/runs", json={"game": "eye_lock", "difficulty": "easy", "score": 75}).json()
    assert better["personal_best"] is True and better["previous_best"] == 60
    assert c.post("/api/arcade/runs", json={"game": "tetris", "score": 1}).status_code == 422
    assert c.post("/api/arcade/runs", json={"game": "eye_lock", "difficulty": "insane", "score": 1}).status_code == 422

    summary = c.get("/api/arcade").json()
    eye = next(g for g in summary["games"] if g["id"] == "eye_lock")
    assert eye["bests"] == {"easy": 75}
    assert summary["streak_days"] == 1 and summary["played_today"] is True
    assert len(summary["recent"]) == 3


def test_streak_counts_consecutive_days(owner_client, default_profile_id):
    async def seed():
        async with database.session() as s:
            for d in (0, 1, 2, 4):
                s.add(GameRun(profile_id=default_profile_id, game="statue", difficulty="medium", score=10,
                              created_at=utcnow() - timedelta(days=d)))
    run(seed())
    assert owner_client.get("/api/arcade").json()["streak_days"] == 3


def test_queue_targets_weakest_metrics(owner_client, default_profile_id):
    async def seed():
        async with database.session() as s:
            s.add(Rehearsal(profile_id=default_profile_id, title="t", status="done",
                            scores_json={"fillers": 20, "eye_contact": 90, "vocal_variety": 45, "posture": 30, "pace": 95}))
    run(seed())
    queue = owner_client.get("/api/arcade").json()["queue"]
    assert [q["game"] for q in queue] == ["filler_ninja", "statue", "pitch_surfer"]
    assert "fillers averaged 20/100" in queue[0]["reason"]


def test_default_queue_without_rehearsals():
    assert [q["game"] for q in ArcadeService.queue([])] == ["filler_ninja", "eye_lock", "pitch_surfer"]

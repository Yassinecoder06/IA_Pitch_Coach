"""Game runs, personal bests, practice streak and the daily drill queue."""

from __future__ import annotations

from datetime import date, timedelta
from statistics import mean
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from backend.db import database
from backend.db.models import GameRun, Rehearsal

GAMES = {
    "filler_ninja": {"title": "Filler Ninja", "metric": "fillers", "needs": ["mic"]},
    "pitch_surfer": {"title": "Pitch Surfer", "metric": "vocal_variety", "needs": ["mic"]},
    "eye_lock": {"title": "Eye Lock", "metric": "eye_contact", "needs": ["camera"]},
    "statue": {"title": "Stage Presence", "metric": "composure", "needs": ["camera"]},
    "pace_racer": {"title": "Pace Racer", "metric": "pace", "needs": ["mic"]},
    "pause_power": {"title": "Pause Power", "metric": "pauses", "needs": ["mic"]},
}
DIFFICULTIES = ("easy", "medium", "hard")
# Metrics that point at the same skill as a game.
RELATED = {"energy": "pitch_surfer", "posture": "statue", "timing": "pace_racer"}
DEFAULT_QUEUE = ["filler_ninja", "eye_lock", "pitch_surfer"]


class ArcadeService:
    async def record(self, profile_id: str, game: str, difficulty: str, score: float, stats: Optional[dict]) -> Dict[str, Any]:
        if game not in GAMES:
            raise ValueError(f"Unknown game: {game}")
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"difficulty must be one of {', '.join(DIFFICULTIES)}")
        async with database.session() as s:
            best_before = (await s.execute(
                select(GameRun.score).where(GameRun.profile_id == profile_id, GameRun.game == game,
                                            GameRun.difficulty == difficulty).order_by(GameRun.score.desc()).limit(1)
            )).scalar_one_or_none()
            run = GameRun(profile_id=profile_id, game=game, difficulty=difficulty, score=round(score, 1), stats_json=stats)
            s.add(run)
            await s.flush()
            return {**run.to_dict(), "personal_best": best_before is None or score > best_before, "previous_best": best_before}

    async def summary(self, profile_id: str) -> Dict[str, Any]:
        async with database.session() as s:
            runs = (await s.execute(
                select(GameRun).where(GameRun.profile_id == profile_id).order_by(GameRun.created_at.desc()).limit(500)
            )).scalars().all()
            rehearsals = (await s.execute(
                select(Rehearsal.scores_json).where(Rehearsal.profile_id == profile_id, Rehearsal.status == "done")
                .order_by(Rehearsal.created_at.desc()).limit(3)
            )).scalars().all()

        bests: Dict[str, Dict[str, float]] = {}
        for r in runs:
            per = bests.setdefault(r.game, {})
            per[r.difficulty] = max(per.get(r.difficulty, 0.0), r.score)

        # Streak: consecutive days (ending today or yesterday) with at least one game or rehearsal.
        # Days are local calendar days: created_at is UTC, date.today() is local.
        days = sorted({r.created_at.astimezone().date() for r in runs}, reverse=True)
        streak, cursor = 0, date.today()
        if days and days[0] < cursor:
            cursor -= timedelta(days=1)
        for d in days:
            if d == cursor:
                streak += 1
                cursor -= timedelta(days=1)
            elif d < cursor:
                break

        return {
            "games": [{"id": g, **info, "bests": bests.get(g, {})} for g, info in GAMES.items()],
            "recent": [r.to_dict() for r in runs[:10]],
            "streak_days": streak,
            "played_today": bool(days) and days[0] == date.today(),
            "queue": self.queue([r for r in rehearsals if r]),
        }

    @staticmethod
    def queue(recent_scores: List[Dict[str, Any]], size: int = 3) -> List[Dict[str, Any]]:
        """Games for the weakest metrics across the last rehearsals (lowest average first)."""
        averages: Dict[str, float] = {}
        for metric in {*(g["metric"] for g in GAMES.values()), *RELATED}:
            values = [s[metric] for s in recent_scores if s.get(metric) is not None]
            if values:
                averages[metric] = mean(values)
        picked: List[Dict[str, Any]] = []
        for metric, avg in sorted(averages.items(), key=lambda kv: kv[1]):
            game = RELATED.get(metric) or next(g for g, i in GAMES.items() if i["metric"] == metric)
            if game not in (p["game"] for p in picked):
                picked.append({"game": game, "reason": f"{metric.replace('_', ' ')} averaged {avg:.0f}/100 in your last rehearsals"})
            if len(picked) == size:
                return picked
        for game in DEFAULT_QUEUE:
            if len(picked) == size:
                break
            if game not in (p["game"] for p in picked):
                picked.append({"game": game, "reason": "a core skill for every pitch"})
        return picked


arcade_service = ArcadeService()

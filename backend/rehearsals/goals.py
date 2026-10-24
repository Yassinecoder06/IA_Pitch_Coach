"""Practice goals per profile, checked against recent rehearsals."""

from __future__ import annotations

from statistics import mean
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from backend.analysis.scores import METRICS
from backend.db import database
from backend.db.models import Goal, Rehearsal
from backend.rehearsals.compare import flat_scores, raw_facts

RAW = ("wpm", "fillers_per_min", "pitch_range_st", "eye_contact_pct", "posture_avg", "hedges_per_min", "duration_s")
GOAL_METRICS = ("overall", "delivery", "content", *METRICS, *RAW)
OPS = ("gte", "lte", "between")
RECENT = 3


def _value(r: Dict[str, Any], metric: str) -> Optional[float]:
    return raw_facts(r).get(metric) if metric in RAW else flat_scores(r).get(metric)


def _met(op: str, target: float, target_max: Optional[float], value: float) -> bool:
    if op == "gte":
        return value >= target
    if op == "lte":
        return value <= target
    return target <= value <= (target_max if target_max is not None else target)


class GoalService:
    async def list(self, profile_id: str) -> List[Dict[str, Any]]:
        async with database.session() as s:
            goals = (await s.execute(select(Goal).where(Goal.profile_id == profile_id).order_by(Goal.created_at))).scalars().all()
            rows = (await s.execute(
                select(Rehearsal).where(Rehearsal.profile_id == profile_id, Rehearsal.status == "done")
                .order_by(Rehearsal.created_at.desc()).limit(RECENT)
            )).scalars().all()
            recent = [
                {"scores": r.scores_json, "overall_score": r.overall_score, "analysis": r.metrics_json,
                 "sensors": r.sensors_json, "duration_sec": r.duration_sec}
                for r in rows
            ]
        out = []
        for g in goals:
            values = [v for v in (_value(r, g.metric) for r in recent) if v is not None]
            latest = _value(recent[0], g.metric) if recent else None
            avg = round(mean(values), 1) if values else None
            status = "no_data" if latest is None else ("met" if _met(g.op, g.target, g.target_max, latest) else "not_met")
            out.append({**g.to_dict(), "latest": latest, "recent_avg": avg, "status": status})
        return out

    async def create(self, profile_id: str, metric: str, op: str, target: float, target_max: Optional[float]) -> Dict[str, Any]:
        if metric not in GOAL_METRICS:
            raise ValueError(f"Unknown metric: {metric}")
        if op not in OPS:
            raise ValueError(f"op must be one of {', '.join(OPS)}")
        if op == "between" and (target_max is None or target_max < target):
            raise ValueError("A 'between' goal needs target_max >= target")
        async with database.session() as s:
            g = Goal(profile_id=profile_id, metric=metric, op=op, target=target, target_max=target_max)
            s.add(g)
            await s.flush()
            return g.to_dict()

    async def delete(self, profile_id: str, goal_id: str) -> bool:
        async with database.session() as s:
            g = await s.get(Goal, goal_id)
            if not g or g.profile_id != profile_id:
                return False
            await s.delete(g)
            return True


goal_service = GoalService()

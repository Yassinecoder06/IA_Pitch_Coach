"""
Rehearsal lifecycle: create, record (via /ws/rehearsal), analyse, report, follow-up Q&A.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
from pathlib import Path
from statistics import mean
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from sqlalchemy import delete, select
from sqlalchemy.orm import selectinload

from backend.analysis.judge import judge_rehearsal
from backend.analysis.report import analyse_take
from backend.config.settings import settings
from backend.db import database
from backend.db.models import Rehearsal, RehearsalSlideSegment, iso
from backend.db.types import utcnow
from backend.decks.repository import deck_repository
from backend.voice.streaming import ChunkResult

logger = logging.getLogger(__name__)

Progress = Callable[[str], Awaitable[None]]
_running: Dict[str, asyncio.Task] = {}


def recordings_dir() -> Path:
    path = Path(settings.data_dir) / "recordings"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _encode_audio(rehearsal_id: str, wav: bytes) -> str:
    """Opus in Ogg when ffmpeg is available (about 1/20 of the WAV size), else the WAV itself."""
    folder = recordings_dir()
    if shutil.which("ffmpeg"):
        out = folder / f"{rehearsal_id}.ogg"
        try:
            subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-y", "-f", "wav", "-i", "pipe:0", "-c:a", "libopus", "-b:a", "24k", str(out)],
                input=wav, check=True, timeout=120,
            )
            return out.name
        except Exception as e:
            logger.warning(f"Opus encoding failed, keeping WAV: {e}")
    out = folder / f"{rehearsal_id}.wav"
    out.write_bytes(wav)
    return out.name


def summary_dict(r: Rehearsal) -> Dict[str, Any]:
    return {
        "id": r.id,
        "deck_id": r.deck_id,
        "title": r.title,
        "status": r.status,
        "progress": r.progress,
        "target_seconds": r.target_seconds,
        "duration_sec": r.duration_sec,
        "overall_score": r.overall_score,
        "scores": r.scores_json,
        "cue_ids": [c["id"] for c in (r.cues_json or [])],
        "error": r.error,
        "created_at": iso(r.created_at),
        "completed_at": iso(r.completed_at),
    }


def full_dict(r: Rehearsal) -> Dict[str, Any]:
    sensors = r.sensors_json or {}
    return {
        **summary_dict(r),
        "transcript": r.transcript,
        "words": r.words_json or [],
        "analysis": r.metrics_json,
        "sensors": sensors,
        "cues": r.cues_json or [],
        "judge": r.judge_json,
        "segments": [
            {"slide_idx": s.slide_idx, "start": s.start_sec, "end": s.end_sec, "stats": s.metrics_json}
            for s in r.segments
        ],
        "has_audio": bool(r.audio_path),
    }


class RehearsalService:
    async def _get(self, s, rehearsal_id: str, profile_id: str) -> Optional[Rehearsal]:
        return (await s.execute(
            select(Rehearsal).options(selectinload(Rehearsal.segments))
            .where(Rehearsal.id == rehearsal_id, Rehearsal.profile_id == profile_id)
        )).scalar_one_or_none()

    async def create(self, profile_id: str, title: str, deck_id: Optional[str], target_seconds: Optional[int]) -> Dict[str, Any]:
        if deck_id and not await deck_repository.get(deck_id, profile_id):
            raise ValueError("Deck not found")
        async with database.session() as s:
            r = Rehearsal(profile_id=profile_id, title=title, deck_id=deck_id, target_seconds=target_seconds, status="ready")
            s.add(r)
            await s.flush()
            return summary_dict(r)

    async def begin(self, rehearsal_id: str, profile_id: str) -> bool:
        async with database.session() as s:
            r = await self._get(s, rehearsal_id, profile_id)
            if not r or r.status not in ("ready", "recording", "failed"):
                return False
            r.status, r.error = "recording", None
            return True

    async def list(self, profile_id: str, deck_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        async with database.session() as s:
            q = select(Rehearsal).where(Rehearsal.profile_id == profile_id)
            if deck_id:
                q = q.where(Rehearsal.deck_id == deck_id)
            rows = (await s.execute(q.order_by(Rehearsal.created_at.desc()).limit(limit))).scalars()
            return [summary_dict(r) for r in rows]

    async def get(self, rehearsal_id: str, profile_id: str) -> Optional[Dict[str, Any]]:
        async with database.session() as s:
            r = await self._get(s, rehearsal_id, profile_id)
            return full_dict(r) if r else None

    async def audio_path(self, rehearsal_id: str, profile_id: str) -> Optional[Path]:
        async with database.session() as s:
            r = await self._get(s, rehearsal_id, profile_id)
            return recordings_dir() / r.audio_path if r and r.audio_path else None

    async def delete(self, rehearsal_id: str, profile_id: str) -> bool:
        async with database.session() as s:
            r = await self._get(s, rehearsal_id, profile_id)
            if not r:
                return False
            if r.audio_path:
                (recordings_dir() / r.audio_path).unlink(missing_ok=True)
            await s.delete(r)
            return True

    async def _set(self, rehearsal_id: str, **fields: Any) -> None:
        async with database.session() as s:
            r = await s.get(Rehearsal, rehearsal_id)
            if r:
                for k, v in fields.items():
                    setattr(r, k, v)

    async def complete_take(
        self,
        rehearsal_id: str,
        profile_id: str,
        wav: bytes,
        transcript: ChunkResult,
        duration: float,
        slide_events: Sequence[dict],
        sensors: Dict[str, Any],
        on_progress: Optional[Progress] = None,
    ) -> None:
        """Stores the take and runs the analysis. Keeps running if the browser disconnects."""
        audio_name = await asyncio.to_thread(_encode_audio, rehearsal_id, wav)
        async with database.session() as s:
            r = await self._get(s, rehearsal_id, profile_id)
            if not r:
                raise ValueError("Rehearsal not found")
            r.status, r.progress = "processing", "queued"
            r.duration_sec = round(duration, 1)
            r.transcript = transcript.text
            r.words_json = transcript.words
            r.sensors_json = {**sensors, "slide_events": list(slide_events)}
            r.audio_path = audio_name

        task = asyncio.create_task(self._analyse(rehearsal_id, profile_id, wav, on_progress))
        _running[rehearsal_id] = task
        try:
            await asyncio.shield(task)
        finally:
            if task.done():
                _running.pop(rehearsal_id, None)

    async def _analyse(self, rehearsal_id: str, profile_id: str, wav: Optional[bytes], on_progress: Optional[Progress]) -> None:
        async def progress(step: str) -> None:
            await self._set(rehearsal_id, progress=step)
            if on_progress:
                try:
                    await on_progress(step)
                except Exception:
                    pass  # the browser may have gone away; the analysis continues

        try:
            async with database.session() as s:
                r = await self._get(s, rehearsal_id, profile_id)
                words, sensors, deck_id, target = r.words_json or [], r.sensors_json or {}, r.deck_id, r.target_seconds
                duration = r.duration_sec or 0.0
                if wav is None and r.audio_path:
                    wav = await asyncio.to_thread(_decode_audio, recordings_dir() / r.audio_path)
            deck = await deck_repository.get(deck_id, profile_id) if deck_id else None
            slides = deck["slides"] if deck else []

            await progress("measuring delivery")
            analysis = await asyncio.to_thread(
                analyse_take, words, wav, duration, sensors.get("slide_events") or [], slides, sensors, target
            )

            await progress("coaching feedback")
            judge = None
            judge_error = None
            try:
                judge = await judge_rehearsal(analysis, slides, sensors)
            except Exception as e:  # no AI available: the measured report still stands
                judge_error = f"AI feedback unavailable: {e}"
                logger.warning(judge_error)

            delivery = analysis["delivery_score"]
            content = judge["scorecard"]["overall"] if judge else None
            overall = round(mean([v for v in (delivery, content) if v is not None]), 1) if (delivery or content) else None

            async with database.session() as s:
                r = await self._get(s, rehearsal_id, profile_id)
                await s.execute(delete(RehearsalSlideSegment).where(RehearsalSlideSegment.rehearsal_id == rehearsal_id))
                for seg in analysis["segments"]:
                    s.add(RehearsalSlideSegment(rehearsal_id=rehearsal_id, slide_idx=seg["slide_idx"],
                                                start_sec=seg["start"], end_sec=seg["end"], metrics_json=seg["stats"]))
                r.metrics_json = {k: v for k, v in analysis.items() if k not in ("cues", "segments")}
                r.cues_json = analysis["cues"]
                r.scores_json = {**analysis["scores"], "delivery": delivery, "content": content}
                r.judge_json = judge
                r.overall_score = overall
                r.status, r.progress, r.error = "done", None, judge_error
                r.completed_at = utcnow()
            from backend.rehearsals.privacy import drop_recording, get_settings
            if not (await get_settings())["keep_recordings"]:
                await drop_recording(rehearsal_id)  # analysed already; the audio itself is not kept
            await progress("done")
        except Exception as e:
            logger.exception(f"Rehearsal analysis failed: {e}")
            await self._set(rehearsal_id, status="failed", progress=None, error=str(e))
            raise

    async def reanalyse(self, rehearsal_id: str, profile_id: str) -> Optional[Dict[str, Any]]:
        """Re-runs the analysis and AI feedback on the stored recording (e.g. after adding a better model)."""
        async with database.session() as s:
            r = await self._get(s, rehearsal_id, profile_id)
            if not r or r.status == "processing":
                return None
            r.status = "processing"
        await self._analyse(rehearsal_id, profile_id, None, None)
        return await self.get(rehearsal_id, profile_id)

    async def qa_context(self, rehearsal_id: str, profile_id: str) -> Optional[Dict[str, Any]]:
        """Markdown context for an investor Q&A session aimed at the weakest parts of this rehearsal."""
        r = await self.get(rehearsal_id, profile_id)
        if not r or r["status"] != "done":
            return None
        deck = await deck_repository.get(r["deck_id"], profile_id) if r["deck_id"] else None
        lines = [f"# Rehearsal: {r['title']}", ""]
        if deck:
            lines.append("## Deck outline")
            for sl in deck["slides"]:
                bullets = "; ".join(sl["bullets"]) if sl["bullets"] else (sl["text"] or "")[:200]
                lines.append(f"- Slide {sl['idx'] + 1}: {sl['title']} — {bullets}")
            lines.append("")
        lines += ["## What the founder said", (r["transcript"] or "")[:6000], ""]
        if r["judge"]:
            lines.append("## Weakest points found in review")
            for p in r["judge"].get("priorities", []):
                lines.append(f"- {p['title']}: {p['detail']}")
        if r["cues"]:
            lines.append("## Measured issues")
            lines += [f"- {c['message']}" for c in r["cues"][:8]]
        lines += ["", "Ask hard investor questions that target these weak points, one at a time."]
        return {"title": f"Q&A: {r['title']}", "context_md": "\n".join(lines)}


def _decode_audio(path: Path) -> bytes:
    if path.suffix == ".wav":
        return path.read_bytes()
    out = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1"],
        check=True, capture_output=True, timeout=120,
    )
    return out.stdout


rehearsal_service = RehearsalService()

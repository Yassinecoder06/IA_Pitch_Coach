"""Profile-scoped storage for decks and slides. Slide images live on disk, not in the database."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from backend.config.settings import settings
from backend.db import database
from backend.db.models import Deck, Slide, iso
from backend.db.types import utcnow
from backend.decks.schemas import SlideIn

IMAGE_TYPES = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}
MAGIC = {"jpg": (b"\xff\xd8\xff",), "png": (b"\x89PNG\r\n\x1a\n",), "webp": (b"RIFF",)}
MAX_IMAGE_BYTES = 8 * 1024 * 1024


def deck_dir(deck_id: str) -> Path:
    return Path(settings.data_dir) / "decks" / deck_id


def slide_dict(s: Slide) -> Dict[str, Any]:
    return {
        "idx": s.idx,
        "kind": s.kind,
        "title": s.title,
        "bullets": s.bullets_json or [],
        "notes": s.notes,
        "text": s.text,
        "has_image": bool(s.image_path),
        "time_budget_sec": s.time_budget_sec,
    }


def deck_dict(d: Deck, with_slides: bool = True) -> Dict[str, Any]:
    out = {
        "id": d.id,
        "title": d.title,
        "source": d.source,
        "brief": d.brief_json,
        "critique": d.critique_json,
        "created_at": iso(d.created_at),
        "updated_at": iso(d.updated_at),
    }
    if with_slides:
        out["slides"] = [slide_dict(s) for s in d.slides]
        out["slide_count"] = len(d.slides)
    return out


class DeckRepository:
    async def _owned(self, s, deck_id: str, profile_id: str) -> Optional[Deck]:
        d = (await s.execute(
            select(Deck).options(selectinload(Deck.slides)).where(Deck.id == deck_id, Deck.profile_id == profile_id)
        )).scalar_one_or_none()
        return d

    async def list(self, profile_id: str) -> List[Dict[str, Any]]:
        async with database.session() as s:
            counts = (
                select(Slide.deck_id, func.count().label("n")).group_by(Slide.deck_id).subquery()
            )
            rows = (await s.execute(
                select(Deck, counts.c.n)
                .outerjoin(counts, counts.c.deck_id == Deck.id)
                .where(Deck.profile_id == profile_id)
                .order_by(Deck.updated_at.desc())
            )).all()
            return [{**deck_dict(d, with_slides=False), "slide_count": n or 0} for d, n in rows]

    async def get(self, deck_id: str, profile_id: str) -> Optional[Dict[str, Any]]:
        async with database.session() as s:
            d = await self._owned(s, deck_id, profile_id)
            return deck_dict(d) if d else None

    async def create(
        self, profile_id: str, title: str, source: str, slides: Sequence[SlideIn], brief: Optional[dict] = None
    ) -> Dict[str, Any]:
        async with database.session() as s:
            d = Deck(profile_id=profile_id, title=title, source=source, brief_json=brief)
            s.add(d)
            await s.flush()
            for i, sl in enumerate(slides):
                s.add(Slide(deck_id=d.id, idx=i, kind=sl.kind, title=sl.title, bullets_json=list(sl.bullets),
                            notes=sl.notes, text=sl.text, time_budget_sec=sl.time_budget_sec))
            await s.flush()
            deck_id = d.id
        return await self.get(deck_id, profile_id)  # type: ignore[return-value]

    async def update(self, deck_id: str, profile_id: str, title: Optional[str] = None) -> Optional[Dict[str, Any]]:
        async with database.session() as s:
            d = await self._owned(s, deck_id, profile_id)
            if not d:
                return None
            if title is not None:
                d.title = title
            d.updated_at = utcnow()
        return await self.get(deck_id, profile_id)

    async def replace_slides(self, deck_id: str, profile_id: str, slides: Sequence[SlideIn]) -> Optional[Dict[str, Any]]:
        """Replaces slide text; images stay attached to their index."""
        async with database.session() as s:
            d = await self._owned(s, deck_id, profile_id)
            if not d:
                return None
            images = {sl.idx: sl.image_path for sl in d.slides}
            for old in list(d.slides):
                await s.delete(old)
            await s.flush()
            for i, sl in enumerate(slides):
                s.add(Slide(deck_id=d.id, idx=i, kind=sl.kind, title=sl.title, bullets_json=list(sl.bullets),
                            notes=sl.notes, text=sl.text, time_budget_sec=sl.time_budget_sec, image_path=images.get(i)))
            d.updated_at = utcnow()
            d.critique_json = None  # stale once the slides change
        return await self.get(deck_id, profile_id)

    async def delete(self, deck_id: str, profile_id: str) -> bool:
        async with database.session() as s:
            d = await self._owned(s, deck_id, profile_id)
            if not d:
                return False
            await s.delete(d)
        shutil.rmtree(deck_dir(deck_id), ignore_errors=True)
        return True

    async def set_image(self, deck_id: str, profile_id: str, idx: int, data: bytes, content_type: str) -> bool:
        ext = IMAGE_TYPES.get(content_type)
        if not ext:
            raise ValueError("Slide images must be JPEG, PNG or WebP.")
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError("Slide image is too large (8 MB max).")
        if not data.startswith(MAGIC[ext]):
            raise ValueError("File content does not match its image type.")
        async with database.session() as s:
            d = await self._owned(s, deck_id, profile_id)
            slide = next((sl for sl in d.slides if sl.idx == idx), None) if d else None
            if not slide:
                return False
            folder = deck_dir(deck_id)
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{idx}.{ext}"  # server-chosen name: no user input in the path
            for old in folder.glob(f"{idx}.*"):
                old.unlink()
            path.write_bytes(data)
            slide.image_path = path.name
        return True

    async def image_path(self, deck_id: str, profile_id: str, idx: int) -> Optional[Path]:
        async with database.session() as s:
            d = await self._owned(s, deck_id, profile_id)
            slide = next((sl for sl in d.slides if sl.idx == idx), None) if d else None
            if not slide or not slide.image_path:
                return None
            return deck_dir(deck_id) / slide.image_path

    async def set_critique(self, deck_id: str, profile_id: str, critique: dict) -> None:
        async with database.session() as s:
            d = await self._owned(s, deck_id, profile_id)
            if d:
                d.critique_json = critique


deck_repository = DeckRepository()

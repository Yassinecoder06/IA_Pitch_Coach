"""
Decks: list, create (outline / PDF pages / PPTX import / AI generation), edit, critique, export.

PDF and PPTX are rendered to images in the browser with pdf.js; the server only converts
PPTX to PDF (LibreOffice) and stores the page images and text it receives.
"""

from __future__ import annotations

import secrets
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from backend.api.deps import get_profile_id
from backend.config.settings import settings
from backend.decks import generator, pptx_tools
from backend.decks.repository import MAX_IMAGE_BYTES, deck_repository
from backend.decks.schemas import DeckBrief, SlideIn

router = APIRouter(prefix="/api/decks", tags=["decks"])

SOURCES = ("outline", "pdf", "pptx", "ai")
CONVERTED_TTL_S = 15 * 60
_converted: Dict[str, Tuple[str, Path, float]] = {}  # token -> (profile_id, pdf path, expiry)


class DeckCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    source: str = "outline"
    slides: List[SlideIn] = Field(default_factory=list, max_length=200)


class DeckUpdate(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=200)


class SlidesReplace(BaseModel):
    slides: List[SlideIn] = Field(max_length=200)


def _not_found():
    return HTTPException(status_code=404, detail="Deck not found")


@router.get("")
async def list_decks(profile_id: str = Depends(get_profile_id)):
    return {"decks": await deck_repository.list(profile_id)}


@router.post("")
async def create_deck(payload: DeckCreate, profile_id: str = Depends(get_profile_id)):
    if payload.source not in SOURCES:
        raise HTTPException(status_code=422, detail=f"source must be one of {', '.join(SOURCES)}")
    return await deck_repository.create(profile_id, payload.title.strip(), payload.source, payload.slides)


@router.post("/generate")
async def generate_deck(brief: DeckBrief, profile_id: str = Depends(get_profile_id)):
    try:
        result = await generator.generate_deck(brief)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Deck generation failed: {e}") from e
    deck = await deck_repository.create(
        profile_id, result["title"], "ai", result["slides"], brief={**brief.model_dump(), "model": result["model"]}
    )
    return deck


@router.post("/convert-pptx")
async def convert_pptx(file: UploadFile = File(...), profile_id: str = Depends(get_profile_id)):
    """Converts a PPTX to PDF for in-browser rendering; returns its slide titles and speaker notes."""
    data = await file.read(pptx_tools.MAX_PPTX_BYTES + 1)
    if len(data) > pptx_tools.MAX_PPTX_BYTES:
        raise HTTPException(status_code=413, detail="PPTX is too large (60 MB max).")
    if not data.startswith(pptx_tools.PPTX_MAGIC):
        raise HTTPException(status_code=415, detail="That file is not a PPTX.")
    try:
        slides = pptx_tools.extract_pptx_text(data)
    except Exception as e:
        raise HTTPException(status_code=415, detail=f"Could not read the PPTX: {e}") from e
    try:
        pdf = await pptx_tools.pptx_to_pdf(data)
    except RuntimeError as e:
        raise HTTPException(status_code=501, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PPTX conversion failed: {e}") from e

    now = time.time()
    for token, (_, path, exp) in list(_converted.items()):
        if exp < now:
            path.unlink(missing_ok=True)
            _converted.pop(token, None)
    folder = Path(settings.data_dir) / "tmp"
    folder.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(16)
    path = folder / f"{token}.pdf"
    path.write_bytes(pdf)
    _converted[token] = (profile_id, path, now + CONVERTED_TTL_S)
    return {"pdf_url": f"/api/decks/converted/{token}", "slides": slides}


@router.get("/converted/{token}")
async def converted_pdf(token: str, profile_id: str = Depends(get_profile_id)):
    entry = _converted.get(token)
    if not entry or entry[0] != profile_id or entry[2] < time.time() or not entry[1].exists():
        raise HTTPException(status_code=404, detail="Converted file expired; upload the PPTX again.")
    return FileResponse(entry[1], media_type="application/pdf")


@router.get("/{deck_id}")
async def get_deck(deck_id: str, profile_id: str = Depends(get_profile_id)):
    deck = await deck_repository.get(deck_id, profile_id)
    if not deck:
        raise _not_found()
    return deck


@router.patch("/{deck_id}")
async def update_deck(deck_id: str, payload: DeckUpdate, profile_id: str = Depends(get_profile_id)):
    deck = await deck_repository.update(deck_id, profile_id, title=payload.title.strip() if payload.title else None)
    if not deck:
        raise _not_found()
    return deck


@router.put("/{deck_id}/slides")
async def replace_slides(deck_id: str, payload: SlidesReplace, profile_id: str = Depends(get_profile_id)):
    deck = await deck_repository.replace_slides(deck_id, profile_id, payload.slides)
    if not deck:
        raise _not_found()
    return deck


@router.delete("/{deck_id}")
async def delete_deck(deck_id: str, profile_id: str = Depends(get_profile_id)):
    if not await deck_repository.delete(deck_id, profile_id):
        raise _not_found()
    return {"ok": True}


@router.put("/{deck_id}/slides/{idx}/image")
async def upload_slide_image(deck_id: str, idx: int, request: Request, profile_id: str = Depends(get_profile_id)):
    """Raw image body (image/jpeg, image/png or image/webp) for one slide."""
    content_type = (request.headers.get("content-type") or "").split(";")[0].strip()
    if int(request.headers.get("content-length") or 0) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Slide image is too large (8 MB max).")
    data = await request.body()
    try:
        ok = await deck_repository.set_image(deck_id, profile_id, idx, data, content_type)
    except ValueError as e:
        raise HTTPException(status_code=415, detail=str(e)) from e
    if not ok:
        raise HTTPException(status_code=404, detail="Slide not found")
    return {"ok": True}


@router.get("/{deck_id}/slides/{idx}/image")
async def slide_image(deck_id: str, idx: int, profile_id: str = Depends(get_profile_id)):
    path = await deck_repository.image_path(deck_id, profile_id, idx)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="No image for this slide")
    return FileResponse(path, headers={"Cache-Control": "private, max-age=3600"})


@router.post("/{deck_id}/critique")
async def critique(deck_id: str, profile_id: str = Depends(get_profile_id)):
    deck = await deck_repository.get(deck_id, profile_id)
    if not deck:
        raise _not_found()
    if not deck["slides"]:
        raise HTTPException(status_code=400, detail="Add slides before asking for a critique.")
    try:
        result = await generator.critique_deck(deck)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Critique failed: {e}") from e
    await deck_repository.set_critique(deck_id, profile_id, result)
    return result


@router.get("/{deck_id}/export.pptx")
async def export(deck_id: str, profile_id: str = Depends(get_profile_id)):
    deck = await deck_repository.get(deck_id, profile_id)
    if not deck:
        raise _not_found()
    images = {}
    for s in deck["slides"]:
        if s["has_image"]:
            path = await deck_repository.image_path(deck_id, profile_id, s["idx"])
            if path:
                images[s["idx"]] = path
    data = pptx_tools.export_pptx(deck, images)
    safe = "".join(c for c in deck["title"] if c.isalnum() or c in " -_").strip() or "deck"
    return Response(
        data,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        headers={"Content-Disposition": f'attachment; filename="{safe}.pptx"'},
    )

"""
Text-to-Speech synthesis router.
"""

from __future__ import annotations

import base64
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.api.deps import get_current_user_id
from backend.voice.tts import check_piper_available, async_synthesize_speech

router = APIRouter(prefix="/api/tts", tags=["tts"])


class ReadAloudPayload(BaseModel):
    text: str
    voice: str = "en_US-lessac-medium"


@router.post("/read-aloud")
async def read_aloud(payload: ReadAloudPayload, user_id: str = Depends(get_current_user_id)):
    available, msg = check_piper_available()
    if not available:
        raise HTTPException(status_code=503, detail=msg)

    clean_text = payload.text.replace("*", "").strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Text is empty.")

    audio_bytes = await async_synthesize_speech(clean_text, payload.voice)
    if not audio_bytes:
        raise HTTPException(status_code=500, detail="Failed to synthesize speech.")

    return {
        "format": "wav",
        "data": base64.b64encode(audio_bytes).decode("utf-8")
    }

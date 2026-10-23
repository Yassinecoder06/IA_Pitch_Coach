"""
System health and modes discovery router.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from backend.coaching.modes import get_mode_catalog
from backend.config.settings import settings
from backend.voice.stt import get_engine as get_stt_engine
from backend.voice.tts import check_piper_available

router = APIRouter(tags=["system"])


@router.get("/health")
async def health_check():
    tts_available, tts_msg = check_piper_available()
    stt_loaded = get_stt_engine().is_loaded()
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": {
            "stt": "ready" if stt_loaded else "standby",
            "tts": "ready" if tts_available else "unavailable",
            "database": "ready"
        }
    }


@router.get("/api/status")
async def get_system_status():
    tts_available, tts_msg = check_piper_available()
    return {
        "stt": {
            "status": "ready",
            "model": settings.whisper_model_size,
            "device": settings.whisper_device
        },
        "tts": {
            "status": "ready" if tts_available else "unavailable",
            "message": tts_msg
        },
        "default_llm": settings.default_llm
    }


@router.get("/api/modes")
async def list_modes():
    """Returns the catalog of 8 SOTA coaching modes."""
    return {"modes": get_mode_catalog()}

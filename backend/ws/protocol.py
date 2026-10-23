"""
Typed WebSocket protocol definitions for real-time coaching.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel


class WSClientMessage(BaseModel):
    type: Literal[
        "auth",
        "config",
        "start",
        "audio",
        "stop",
        "text",
        "edit_last_user",
        "read_aloud",
        "ping"
    ]
    data: Optional[str] = None
    token: Optional[str] = None
    mode: Optional[str] = None
    model: Optional[str] = None
    session_id: Optional[str] = None
    text: Optional[str] = None


class WSTranscriptEvent(BaseModel):
    type: Literal["transcript"] = "transcript"
    text: str
    confidence: float = 1.0
    final: bool = True


class WSScorecardEvent(BaseModel):
    type: Literal["scorecard"] = "scorecard"
    data: Dict[str, Any]


class WSSpeechMetricsEvent(BaseModel):
    type: Literal["speech_metrics"] = "speech_metrics"
    data: Dict[str, Any]


class WSCoachingStreamEvent(BaseModel):
    type: Literal["coaching"] = "coaching"
    text: str
    streaming: bool = True
    complete: bool = False

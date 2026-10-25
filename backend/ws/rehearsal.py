"""
Live rehearsal stream.

Client → server
    {"type": "start", "sample_rate": 16000, "rehearsal_id": "<id or null>"}
    <binary PCM16 frames>
    {"type": "slide", "index": 2, "t": 41.3}          slide changes (seconds since start)
    {"type": "stop", "sensors": {...}}                 camera/voice summary from the browser
    {"type": "cancel"}

Server → client
    {"type": "words", "start", "end", "text", "words": [...], "fillers": [...]}   live captions
    {"type": "transcript_final", "text", "words", "duration"}
    {"type": "analysis_progress", "step"} / {"type": "analysis_done", "rehearsal_id"}
    {"type": "error", "message"}

Without a rehearsal_id the take is a free practice run: live captions only, nothing stored.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import WebSocket

from backend.analysis.fillers import find_fillers_in_words
from backend.voice.streaming import ChunkResult, StreamingTranscriber
from backend.ws.coach import MAX_PCM_BYTES, PCM_SAMPLE_RATE, authenticate_websocket

logger = logging.getLogger(__name__)


async def handle_rehearsal_connection(websocket: WebSocket) -> None:
    profile_id = await authenticate_websocket(websocket)
    if not profile_id:
        return

    stream: Optional[StreamingTranscriber] = None
    rehearsal_id: Optional[str] = None
    slide_events: List[Dict[str, Any]] = []
    received = 0

    async def push_chunk(chunk: ChunkResult) -> None:
        await websocket.send_json({
            "type": "words",
            "start": chunk.start,
            "end": chunk.end,
            "text": chunk.text,
            "words": chunk.words,
            "fillers": find_fillers_in_words(chunk.words),
        })

    try:
        while True:
            frame = await websocket.receive()
            if frame["type"] == "websocket.disconnect":
                break

            if frame.get("bytes") is not None:
                if stream is not None:
                    received += len(frame["bytes"])
                    if received > MAX_PCM_BYTES:
                        await websocket.send_json({"type": "error", "message": "Recording is too long."})
                        stream = None
                    else:
                        stream.feed(frame["bytes"])
                continue

            msg = json.loads(frame.get("text") or "{}")
            kind = msg.get("type")

            if kind == "start":
                sample_rate = int(msg.get("sample_rate") or PCM_SAMPLE_RATE)
                stream = StreamingTranscriber(on_chunk=push_chunk, sample_rate=sample_rate)
                rehearsal_id = msg.get("rehearsal_id")
                slide_events = [{"index": 0, "t": 0.0}]
                received = 0
                if rehearsal_id:
                    from backend.rehearsals.service import rehearsal_service
                    if not await rehearsal_service.begin(rehearsal_id, profile_id):
                        await websocket.send_json({"type": "error", "message": "Rehearsal not found."})
                        stream, rehearsal_id = None, None
                        continue
                await websocket.send_json({"type": "started", "rehearsal_id": rehearsal_id})

            elif kind == "slide" and stream is not None:
                slide_events.append({"index": int(msg.get("index", 0)), "t": float(msg.get("t", stream.duration))})

            elif kind == "cancel":
                stream, rehearsal_id = None, None

            elif kind == "stop" and stream is not None:
                take, stream = stream, None
                final = await take.finish()
                await websocket.send_json({
                    "type": "transcript_final",
                    "text": final.text,
                    "words": final.words,
                    "duration": take.duration,
                })
                if rehearsal_id:
                    from backend.rehearsals.service import rehearsal_service

                    async def progress(step: str) -> None:
                        await websocket.send_json({"type": "analysis_progress", "step": step})

                    await rehearsal_service.complete_take(
                        rehearsal_id=rehearsal_id,
                        profile_id=profile_id,
                        wav=take.wav(),
                        transcript=final,
                        duration=take.duration,
                        slide_events=slide_events,
                        sensors=msg.get("sensors") or {},
                        on_progress=progress,
                    )
                    await websocket.send_json({"type": "analysis_done", "rehearsal_id": rehearsal_id})
                rehearsal_id = None

            elif kind == "ping":
                await websocket.send_json({"type": "pong"})

    except Exception as e:
        logger.exception(f"[Rehearsal WS] Error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass

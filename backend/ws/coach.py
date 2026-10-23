"""
Real-time WebSocket connection handler for audio and text coaching turns.

Protocol: JSON text frames for control messages, binary frames for raw
16 kHz mono PCM16 audio between {"type": "start"} and {"type": "stop"}.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import wave
from typing import Dict, List, Optional

from fastapi import WebSocket

from backend.analysis.filler_detection import count_filler_words
from backend.analysis.speech_metrics import analyze_speech_metrics
from backend.auth.owner import SESSION_COOKIE, owner_service
from backend.coaching.engine import coaching_engine
from backend.coaching.modes import COACHING_MODES, normalize_mode
from backend.config.settings import settings
from backend.storage.profile_repository import profile_repository
from backend.storage.session_repository import session_repository
from backend.voice.stt import async_transcribe_detailed
from backend.voice.tts import async_synthesize_speech, split_into_sentences

logger = logging.getLogger(__name__)

PCM_SAMPLE_RATE = 16000
MAX_RECORDING_SECONDS = 15 * 60
MAX_PCM_BYTES = PCM_SAMPLE_RATE * 2 * MAX_RECORDING_SECONDS  # 16-bit mono

WS_UNAUTHORIZED = 4401
WS_FORBIDDEN_ORIGIN = 4403
WS_PROFILE_REQUIRED = 4409


def combine_audio_chunks(chunks: List[bytes], sample_rate: int = PCM_SAMPLE_RATE, channels: int = 1) -> Optional[bytes]:
    """Wraps raw 16-bit little-endian PCM frames (from the client AudioWorklet) in a WAV container."""
    if not chunks:
        return None

    raw_data = b"".join(chunks)
    if len(raw_data) < sample_rate // 10:  # under ~50 ms of audio
        return None

    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(raw_data)

    wav_io.seek(0)
    return wav_io.read()


def origin_allowed(websocket: WebSocket) -> bool:
    """Reject cross-site WebSocket hijacking: the Origin must be this host or a configured dev origin."""
    origin = websocket.headers.get("origin")
    if not origin:
        return True  # non-browser clients send no Origin
    host = websocket.headers.get("host", "")
    return origin in (f"http://{host}", f"https://{host}") or origin in settings.cors_origins


async def authenticate_websocket(websocket: WebSocket) -> Optional[str]:
    """Accepts the socket and returns the active profile id, or closes it with an error code."""
    await websocket.accept()
    if not origin_allowed(websocket):
        await websocket.close(code=WS_FORBIDDEN_ORIGIN)
        return None
    ctx = await owner_service.verify_session_token(websocket.cookies.get(SESSION_COOKIE))
    if ctx is None:
        await websocket.close(code=WS_UNAUTHORIZED)
        return None
    if not ctx.profile_id or not await profile_repository.get(ctx.profile_id):
        await websocket.close(code=WS_PROFILE_REQUIRED)
        return None
    return ctx.profile_id


async def handle_websocket_connection(websocket: WebSocket) -> None:
    profile_id = await authenticate_websocket(websocket)
    if not profile_id:
        return

    # Connection state
    audio_chunks: List[bytes] = []
    audio_bytes_received = 0
    sample_rate = PCM_SAMPLE_RATE
    is_recording = False
    mode = "pitch_analysis"
    model_ref: Optional[str] = None
    session_id: Optional[str] = None
    conversation_history: List[Dict[str, str]] = []
    latest_assistant_response: str = ""

    try:
        while True:
            frame = await websocket.receive()
            if frame["type"] == "websocket.disconnect":
                break

            # Binary frames carry raw PCM16 audio for the current recording
            if frame.get("bytes") is not None:
                if is_recording:
                    chunk = frame["bytes"]
                    audio_bytes_received += len(chunk)
                    if audio_bytes_received > MAX_PCM_BYTES:
                        is_recording = False
                        audio_chunks = []
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Recording exceeds {MAX_RECORDING_SECONDS // 60} minutes."
                        })
                    else:
                        audio_chunks.append(chunk)
                continue

            msg = json.loads(frame.get("text") or "{}")
            msg_type = msg.get("type")

            if msg_type == "config":
                if "mode" in msg:
                    mode = normalize_mode(msg["mode"])
                if "model" in msg:
                    model_ref = msg["model"]
                if "session_id" in msg:
                    requested = msg["session_id"]
                    # Only bind to sessions owned by the active profile
                    if requested and await session_repository.get_session(requested, profile_id):
                        session_id = requested
                        recent = await session_repository.get_recent_messages(session_id, profile_id, limit=20)
                        conversation_history = [{"role": m["role"], "content": m["content"]} for m in recent]
                    else:
                        session_id = None
                        conversation_history = []

                await websocket.send_json({
                    "type": "config_ack",
                    "config": {"mode": mode, "model": model_ref, "session_id": session_id}
                })

            elif msg_type == "start":
                audio_chunks = []
                audio_bytes_received = 0
                sample_rate = int(msg.get("sample_rate") or PCM_SAMPLE_RATE)
                is_recording = True
                await websocket.send_json({"type": "status", "message": "Listening..."})

            elif msg_type == "stop":
                if not is_recording:
                    continue
                is_recording = False
                await websocket.send_json({"type": "status", "message": "Transcribing speech..."})

                combined = combine_audio_chunks(audio_chunks, sample_rate=sample_rate)
                audio_chunks = []
                if not combined:
                    await websocket.send_json({"type": "error", "message": "No audio received."})
                    continue

                result = await async_transcribe_detailed(combined)
                transcript = result.text
                if not transcript.strip():
                    await websocket.send_json({"type": "error", "message": "No speech detected in audio."})
                    continue

                await websocket.send_json({
                    "type": "transcript",
                    "text": transcript,
                    "confidence": result.confidence,
                    "words": result.words,
                    "final": True
                })

                filler_details = count_filler_words(transcript)
                total_fillers = sum(filler_details.values())
                word_count = len(transcript.split())

                try:
                    metrics = analyze_speech_metrics(combined, transcript, filler_count=total_fillers)
                except Exception as e:
                    logger.warning(f"[WebSocket] Speech metrics failed: {e}")
                    duration_s = (len(combined) - 44) / (2 * sample_rate)
                    metrics = {
                        "words_per_minute": round(word_count / (duration_s / 60), 1) if duration_s > 0 else 0.0
                    }
                metrics["filler_count"] = total_fillers

                await websocket.send_json({
                    "type": "filler_words",
                    "count": total_fillers,
                    "details": filler_details,
                    "word_count": word_count
                })
                await websocket.send_json({"type": "speech_metrics", "data": metrics})

                latest_assistant_response = await _run_coaching_turn(
                    websocket=websocket,
                    transcript=transcript,
                    mode=mode,
                    conversation_history=conversation_history,
                    speech_metrics=metrics,
                    model_ref=model_ref,
                    profile_id=profile_id,
                    session_id=session_id
                )

            elif msg_type == "text":
                text_content = (msg.get("text") or "").strip()
                if not text_content:
                    await websocket.send_json({"type": "error", "message": "Message is empty."})
                    continue

                await websocket.send_json({
                    "type": "transcript",
                    "text": text_content,
                    "confidence": 1.0,
                    "final": True
                })

                filler_details = count_filler_words(text_content)
                total_fillers = sum(filler_details.values())
                # Typed text has no timing or audio: only report what can actually be measured.
                metrics = {"filler_count": total_fillers, "source": "text"}

                await websocket.send_json({
                    "type": "filler_words",
                    "count": total_fillers,
                    "details": filler_details,
                    "word_count": len(text_content.split())
                })

                latest_assistant_response = await _run_coaching_turn(
                    websocket=websocket,
                    transcript=text_content,
                    mode=mode,
                    conversation_history=conversation_history,
                    speech_metrics=metrics,
                    model_ref=model_ref,
                    profile_id=profile_id,
                    session_id=session_id
                )

            elif msg_type == "read_aloud":
                tts_text = (msg.get("text") or latest_assistant_response or "").strip()
                if tts_text:
                    await _stream_tts(websocket, tts_text)
                    await websocket.send_json({"type": "read_aloud_complete"})

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

    except Exception as e:
        logger.exception(f"[WebSocket] Error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


async def _run_coaching_turn(
    websocket: WebSocket,
    transcript: str,
    mode: str,
    conversation_history: List[Dict[str, str]],
    speech_metrics: Dict,
    model_ref: Optional[str],
    profile_id: str,
    session_id: Optional[str]
) -> str:
    await websocket.send_json({"type": "status", "message": "AI Coach is analyzing..."})

    full_response = ""
    try:
        async for chunk in coaching_engine.stream_coaching(
            transcript=transcript,
            mode=mode,
            conversation_history=conversation_history,
            speech_metrics=speech_metrics,
            model_ref=model_ref,
        ):
            full_response += chunk
            await websocket.send_json({"type": "coaching", "text": chunk, "streaming": True})
    except Exception as e:
        logger.warning(f"[WebSocket] Coaching stream failed: {e}")
        await websocket.send_json({"type": "error", "message": f"AI coach unavailable: {e}"})
        await websocket.send_json({"type": "complete"})
        return ""

    await websocket.send_json({"type": "coaching", "text": "", "streaming": False, "complete": True})

    conversation_history.append({"role": "user", "content": transcript})
    conversation_history.append({"role": "assistant", "content": full_response})

    scorecard = None
    mode_info = COACHING_MODES.get(mode)
    if mode_info and mode_info.produces_scorecard:
        await websocket.send_json({"type": "status", "message": "Scoring your pitch..."})
        scorecard = await coaching_engine.score_turn(transcript, full_response, speech_metrics, model_ref)
        if scorecard:
            await websocket.send_json({"type": "scorecard", "data": scorecard})

    if session_id:
        await session_repository.append_message(
            session_id=session_id, profile_id=profile_id, role="user", content=transcript, transcript=transcript
        )
        await session_repository.append_message(
            session_id=session_id,
            profile_id=profile_id,
            role="assistant",
            content=full_response,
            scores=scorecard,
            metrics=speech_metrics,
        )

    if mode_info and mode_info.default_tts:
        await _stream_tts(websocket, full_response)

    await websocket.send_json({"type": "complete"})
    return full_response


async def _stream_tts(websocket: WebSocket, text: str) -> None:
    """Synthesizes text in sentences and streams WAV audio chunks over WebSocket."""
    for sentence in split_into_sentences(text)[:3]:  # first sentences keep the cadence conversational
        clean = sentence.replace("*", "").strip()
        if not clean:
            continue
        audio_bytes = await async_synthesize_speech(clean)
        if audio_bytes:
            await websocket.send_json({"type": "audio", "data": base64.b64encode(audio_bytes).decode("utf-8")})

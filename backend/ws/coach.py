"""
Real-time WebSocket connection handler for audio and text coaching turns.
"""

from __future__ import annotations

import io
import json
import base64
import wave
import logging
from typing import Dict, List, Optional
from fastapi import WebSocket, WebSocketDisconnect

from backend.config.settings import settings
from backend.auth.owner import owner_service, SESSION_COOKIE
from backend.storage.session_repository import session_repository
from backend.voice.stt import async_transcribe_audio
from backend.voice.tts import split_into_sentences, async_synthesize_speech
from backend.analysis.filler_detection import count_filler_words
from backend.analysis.speech_metrics import analyze_speech_metrics
from backend.coaching.engine import coaching_engine
from backend.coaching.modes import COACHING_MODES, normalize_mode
from backend.coaching.rubric import parse_scorecard_from_text

logger = logging.getLogger(__name__)


PCM_SAMPLE_RATE = 16000
MAX_RECORDING_SECONDS = 15 * 60
MAX_PCM_BYTES = PCM_SAMPLE_RATE * 2 * MAX_RECORDING_SECONDS  # 16-bit mono


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


WS_UNAUTHORIZED = 4401
WS_FORBIDDEN_ORIGIN = 4403


def _origin_allowed(websocket: WebSocket) -> bool:
    """Reject cross-site WebSocket hijacking: the Origin must be this host or a configured dev origin."""
    origin = websocket.headers.get("origin")
    if not origin:
        return True  # non-browser clients send no Origin
    host = websocket.headers.get("host", "")
    return origin in (f"http://{host}", f"https://{host}") or origin in settings.cors_origins


async def handle_websocket_connection(websocket: WebSocket) -> None:
    await websocket.accept()

    if not _origin_allowed(websocket):
        await websocket.close(code=WS_FORBIDDEN_ORIGIN)
        return

    current_user_id = owner_service.verify_session_token(websocket.cookies.get(SESSION_COOKIE))
    if not current_user_id:
        await websocket.close(code=WS_UNAUTHORIZED)
        return

    # Session state
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
                    # Only bind to sessions owned by the authenticated user
                    if requested and session_repository.get_session(requested, current_user_id):
                        session_id = requested
                        recent = session_repository.get_recent_messages(session_id, current_user_id, limit=20)
                        conversation_history = [
                            {"role": m["role"], "content": m["content"]}
                            for m in recent
                        ]
                    else:
                        session_id = None
                        conversation_history = []

                await websocket.send_json({
                    "type": "config_ack",
                    "config": {
                        "mode": mode,
                        "model": model_ref,
                        "session_id": session_id
                    }
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

                # Run transcription
                transcript, confidence = await async_transcribe_audio(combined)
                if not transcript.strip():
                    await websocket.send_json({"type": "error", "message": "No speech detected in audio."})
                    continue

                # Send transcript to client
                await websocket.send_json({
                    "type": "transcript",
                    "text": transcript,
                    "confidence": confidence,
                    "final": True
                })

                # Compute speech metrics & filler words
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

                # Run LLM Coaching Stream
                latest_assistant_response = await _run_coaching_turn(
                    websocket=websocket,
                    transcript=transcript,
                    mode=mode,
                    conversation_history=conversation_history,
                    speech_metrics=metrics,
                    model_ref=model_ref,
                    user_id=current_user_id,
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
                word_count = len(text_content.split())
                # Typed text has no timing or audio: only report what can actually be measured.
                metrics = {"filler_count": total_fillers, "source": "text"}

                await websocket.send_json({
                    "type": "filler_words",
                    "count": total_fillers,
                    "details": filler_details,
                    "word_count": word_count
                })

                latest_assistant_response = await _run_coaching_turn(
                    websocket=websocket,
                    transcript=text_content,
                    mode=mode,
                    conversation_history=conversation_history,
                    speech_metrics=metrics,
                    model_ref=model_ref,
                    user_id=current_user_id,
                    session_id=session_id
                )

            elif msg_type == "read_aloud":
                tts_text = (msg.get("text") or latest_assistant_response or "").strip()
                if tts_text:
                    await _stream_tts(websocket, tts_text)
                    await websocket.send_json({"type": "read_aloud_complete"})

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info("[WebSocket] Client disconnected")
    except Exception as e:
        logger.error(f"[WebSocket] Error: {e}")
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
    user_id: str,
    session_id: Optional[str]
) -> str:
    await websocket.send_json({"type": "status", "message": "AI Coach is analyzing..."})

    full_response = ""
    async for chunk in coaching_engine.stream_coaching(
        transcript=transcript,
        mode=mode,
        conversation_history=conversation_history,
        speech_metrics=speech_metrics,
        model_ref=model_ref,
        user_id=user_id
    ):
        full_response += chunk
        await websocket.send_json({
            "type": "coaching",
            "text": chunk,
            "streaming": True
        })

    await websocket.send_json({
        "type": "coaching",
        "text": "",
        "streaming": False,
        "complete": True
    })

    # Update turn history
    conversation_history.append({"role": "user", "content": transcript})
    conversation_history.append({"role": "assistant", "content": full_response})

    # Parse and send scorecard
    scorecard = parse_scorecard_from_text(full_response)
    if scorecard:
        await websocket.send_json({
            "type": "scorecard",
            "data": scorecard
        })

    # Persist in database if session exists
    if session_id:
        session_repository.append_message(
            session_id=session_id,
            user_id=user_id,
            role="user",
            content=transcript,
            transcript=transcript
        )
        session_repository.append_message(
            session_id=session_id,
            user_id=user_id,
            role="assistant",
            content=full_response,
            scores=scorecard,
            metrics=speech_metrics
        )
        if speech_metrics.get("source") != "text":
            session_repository.append_speech_metrics(
                session_id=session_id,
                user_id=user_id,
                metrics=speech_metrics
            )

    # If mode has default_tts enabled, synthesize response
    mode_info = COACHING_MODES.get(mode)
    if mode_info and mode_info.default_tts:
        # Speak first couple sentences for speed
        await _stream_tts(websocket, full_response)

    await websocket.send_json({"type": "complete"})
    return full_response


async def _stream_tts(websocket: WebSocket, text: str) -> None:
    """Synthesizes text in sentences and streams WAV audio chunks over WebSocket."""
    sentences = split_into_sentences(text)
    for sentence in sentences[:3]:  # Top 3 sentences for conversational cadence
        clean = sentence.replace("*", "").strip()
        if not clean:
            continue
        audio_bytes = await async_synthesize_speech(clean)
        if audio_bytes:
            b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
            await websocket.send_json({
                "type": "audio",
                "data": b64_audio
            })

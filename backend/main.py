"""
AI Pitch Coach - FastAPI Backend
================================
Real-time voice-based pitch coaching using WebSockets.
Supports multiple LLM providers and coaching modes.

Pipeline:
1. Browser records audio
2. Audio chunks sent via WebSocket
3. faster-whisper converts speech to text
4. LLM (configurable provider) analyzes pitch
5. Piper generates audio response
6. Audio sent back to browser
"""

import os
import sys
import io
import json
import wave
import asyncio
import base64
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration
from backend.config import settings

# Import voice modules
from backend.voice.stt import load_model as load_stt_model, transcribe_audio
from backend.voice.tts import (
    check_piper_available,
    preload_tts_model,
    synthesize_speech,
    split_into_sentences
)
from backend.voice.voice_loop import VoiceLoop, VoiceLoopState, VoiceLoopConfig

# Import analysis modules
from backend.analysis.filler_detection import count_filler_words, get_total_filler_count
from backend.analysis.speech_metrics import analyze_speech_metrics
from backend.analysis.pitch_analysis import (
    PitchAnalyzer,
    CoachingMode,
    PITCH_COACH_SYSTEM_PROMPT,
    parse_scores_from_response
)
from backend.storage.session_manager import SessionManager
from backend.auth.supabase_auth import (
    extract_bearer_token,
    get_user_id_from_token,
    require_user_id,
)

# Import LLM modules
from backend.llm import get_provider, list_providers, get_available_providers


# Create FastAPI app
app = FastAPI(
    title="AI Pitch Coach",
    description="Real-time voice-based pitch coaching with multiple LLM providers",
    version="2.0.0"
)

# CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
startup_complete = False
session_manager = SessionManager()
SESSION_CONTEXT_WINDOW = int(os.getenv("SESSION_CONTEXT_WINDOW", "8"))


def _sanitize_llm_output_text(text: str) -> str:
    """Sanitize model output so TTS does not read markdown symbols aloud."""
    if not text:
        return ""
    return text.replace("*", "")


def _parse_mode(mode_str: str) -> CoachingMode:
    aliases = {
        "interactive": CoachingMode.INTERACTIVE,
        "interactive-coaching": CoachingMode.INTERACTIVE_COACHING,
        "interactive_coaching": CoachingMode.INTERACTIVE_COACHING,
        "pitch_analysis": CoachingMode.PITCH_ANALYSIS,
        "pitch-analysis": CoachingMode.PITCH_ANALYSIS,
        "investor_qa": CoachingMode.INVESTOR_QA,
        "investor-qa": CoachingMode.INVESTOR_QA,
        "conversation": CoachingMode.CONVERSATION,
    }
    return aliases.get((mode_str or "").strip().lower(), CoachingMode.PITCH_ANALYSIS)


class CreateSessionRequest(BaseModel):
    title: str = "Startup Pitch Practice"
    mode: str = CoachingMode.PITCH_ANALYSIS.value


class UpdateSessionModeRequest(BaseModel):
    mode: str


class ReadAloudRequest(BaseModel):
    text: str


# ============================================================================
# Startup Event - Load Models Once
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load models on startup to avoid loading during requests."""
    global startup_complete

    print("=" * 60)
    print("AI Pitch Coach - Starting Up")
    print("=" * 60)

    # Load STT model
    print("\n[Startup] Loading Speech-to-Text model...")
    try:
        load_stt_model()
        print("[Startup] STT model loaded successfully")
    except Exception as e:
        print(f"[Startup] Warning: STT model failed to load: {e}")

    # Check LLM providers
    print("\n[Startup] Checking LLM providers...")
    providers = await get_available_providers()
    for name, status in providers.items():
        if status.get("available"):
            print(f"[Startup] {name}: Available")
        else:
            print(f"[Startup] {name}: {status.get('message', 'Not available')}")

    default_provider = settings.default_provider
    print(f"[Startup] Default provider: {default_provider}")

    # Check TTS status
    print("\n[Startup] Checking Text-to-Speech...")
    tts_available, tts_msg = check_piper_available()
    if tts_available:
        print("[Startup] Piper TTS is available")
    else:
        print(f"[Startup] Warning: {tts_msg}")

    if os.getenv("PRELOAD_TTS_MODEL", "false").lower() == "true":
        print("\n[Startup] Preloading TTS voice model...")
        if preload_tts_model():
            print("[Startup] TTS voice model loaded")
        else:
            print("[Startup] Warning: failed to preload TTS voice model")

    startup_complete = True
    print("\n" + "=" * 60)
    print(f"Startup complete! Open http://localhost:{settings.server.port} in your browser")
    print("=" * 60 + "\n")


# ============================================================================
# REST API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Serve the frontend index.html."""
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return JSONResponse({"error": "Frontend not found"}, status_code=404)


@app.get("/login")
@app.get("/login.html")
async def login_page():
    """Serve the dedicated sign-in page."""
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "login.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return JSONResponse({"error": "Login page not found"}, status_code=404)


@app.get("/signup")
@app.get("/signup.html")
async def signup_page():
    """Serve the dedicated sign-up page."""
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "signup.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return JSONResponse({"error": "Signup page not found"}, status_code=404)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    providers = await get_available_providers()
    tts_available, tts_msg = check_piper_available()

    return {
        "status": "ok" if startup_complete else "starting",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "stt": "ready",
            "llm": {
                "default": settings.default_provider,
                "providers": {
                    name: status.get("available", False)
                    for name, status in providers.items()
                }
            },
            "tts": {"available": tts_available, "message": tts_msg}
        }
    }


@app.get("/api/status")
async def get_status():
    """Get detailed system status."""
    providers = await get_available_providers()
    tts_available, tts_msg = check_piper_available()

    return {
        "stt": {
            "status": "ready",
            "model": settings.stt.model_size
        },
        "llm": {
            "status": "ready" if any(p.get("available") for p in providers.values()) else "unavailable",
            "default_provider": settings.default_provider,
            "available": True
        },
        "tts": {
            "status": "ready" if tts_available else "unavailable",
            "message": tts_msg
        }
    }


@app.get("/api/providers")
async def get_providers():
    """Get available LLM providers and their models."""
    providers = await get_available_providers()

    result = {}
    for name, status in providers.items():
        if status.get("available"):
            result[name] = {
                "name": name,
                "available": True,
                "models": status.get("models", [])
            }

    return {
        "providers": result,
        "default": settings.default_provider
    }


@app.get("/api/models/{provider}")
async def get_provider_models(provider: str):
    """Get available models for a specific provider."""
    llm_provider = get_provider(provider)
    if llm_provider is None:
        raise HTTPException(status_code=404, detail=f"Provider '{provider}' not found")

    try:
        models = await llm_provider.list_models()
        return {"provider": provider, "models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/settings")
async def get_settings():
    """Get current settings (safe subset)."""
    return settings.to_dict()


@app.get("/api/sessions")
async def list_sessions(limit: int = 50, user_id: str = Depends(require_user_id)):
    if not session_manager.enabled:
        return {"enabled": False, "sessions": []}
    return {"enabled": True, "sessions": session_manager.list_sessions(user_id, limit=limit)}


@app.post("/api/sessions")
async def create_session(payload: CreateSessionRequest, user_id: str = Depends(require_user_id)):
    if not session_manager.enabled:
        raise HTTPException(status_code=503, detail="Supabase is not configured")

    created = session_manager.create_session(payload.title, _parse_mode(payload.mode).value, user_id)
    if not created:
        raise HTTPException(status_code=500, detail="Failed to create session")
    return created


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str, message_limit: int = 100, user_id: str = Depends(require_user_id)):
    if not session_manager.enabled:
        raise HTTPException(status_code=503, detail="Supabase is not configured")

    session = session_manager.get_session(session_id, user_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session": session,
        "messages": session_manager.get_recent_messages(session_id, user_id, limit=message_limit),
    }


@app.patch("/api/sessions/{session_id}/mode")
async def update_session_mode(session_id: str, payload: UpdateSessionModeRequest, user_id: str = Depends(require_user_id)):
    if not session_manager.enabled:
        raise HTTPException(status_code=503, detail="Supabase is not configured")

    session_manager.update_mode(session_id, user_id, _parse_mode(payload.mode).value)
    return {"ok": True}


@app.post("/api/sessions/{session_id}/summary")
async def summarize_session(session_id: str, user_id: str = Depends(require_user_id)):
    if not session_manager.enabled:
        raise HTTPException(status_code=503, detail="Supabase is not configured")

    summary_md = session_manager.generate_session_summary_markdown(session_id, user_id)
    if not summary_md:
        raise HTTPException(status_code=404, detail="Session not found")

    session_manager.update_context_markdown(session_id, user_id, summary_md)
    return {"session_id": session_id, "context_md": summary_md}


@app.post("/api/read-aloud")
async def read_aloud(payload: ReadAloudRequest):
    tts_available, tts_msg = check_piper_available()
    if not tts_available:
        raise HTTPException(status_code=503, detail=tts_msg)

    audio = synthesize_speech(payload.text)
    if not audio:
        raise HTTPException(status_code=500, detail="Failed to synthesize speech")

    return {
        "format": "wav",
        "data": base64.b64encode(audio).decode("utf-8")
    }


# ============================================================================
# WebSocket Handler - Main Communication Channel
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio processing.

    Protocol:
    - Client sends: {"type": "audio", "data": base64_audio_chunk}
    - Client sends: {"type": "stop"} to end recording
    - Client sends: {"type": "config", "provider": "...", "model": "...", "mode": "..."}
    - Server sends: {"type": "transcript", "text": "...", "final": bool}
    - Server sends: {"type": "analysis", "text": "...", "streaming": bool}
    - Server sends: {"type": "scores", "data": {...}}
    - Server sends: {"type": "filler_words", "count": N, "details": {...}}
    - Server sends: {"type": "audio", "data": base64_wav}
    - Server sends: {"type": "error", "message": "..."}
    """
    await websocket.accept()
    print("[WebSocket] Client connected")

    auth_header = websocket.headers.get("authorization")
    auth_token = websocket.query_params.get("token") or extract_bearer_token(auth_header)
    current_user_id: Optional[str] = get_user_id_from_token(auth_token)

    if auth_token and not current_user_id:
        await websocket.send_json({"type": "error", "message": "Authentication failed. Sign in again."})

    # Session state
    audio_chunks: List[bytes] = []
    is_recording = False
    session_config = {
        "provider": settings.default_provider,
        "model": None,
        "mode": CoachingMode.PITCH_ANALYSIS
    }
    conversation_history: List[Dict] = []
    current_session_id: Optional[str] = None
    session_context_md: str = ""
    latest_assistant_response: str = ""

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type")

            if msg_type == "config":
                # Update session configuration
                if "provider" in message:
                    session_config["provider"] = message["provider"]
                if "model" in message:
                    session_config["model"] = message["model"]
                if "mode" in message:
                    session_config["mode"] = _parse_mode(message["mode"])
                if "session_id" in message:
                    current_session_id = message.get("session_id")
                    if current_session_id and session_manager.enabled and current_user_id:
                        session_manager.update_mode(current_session_id, current_user_id, session_config["mode"].value)
                print(f"[WebSocket] Config updated: {session_config}")
                await websocket.send_json({"type": "config_ack", "config": {
                    "provider": session_config["provider"],
                    "model": session_config["model"],
                    "mode": session_config["mode"].value,
                    "session_id": current_session_id,
                }})

            elif msg_type == "auth":
                auth_token = message.get("token")
                current_user_id = get_user_id_from_token(auth_token)
                await websocket.send_json({
                    "type": "auth_ack",
                    "authenticated": bool(current_user_id),
                    "user_id": current_user_id,
                })

            elif msg_type == "create_session":
                if not session_manager.enabled:
                    await websocket.send_json({"type": "error", "message": "Supabase is not configured"})
                    continue
                if not current_user_id:
                    await websocket.send_json({"type": "error", "message": "Sign in to create sessions"})
                    continue
                title = (message.get("title") or "Startup Pitch Practice").strip()
                created = session_manager.create_session(title=title, mode=session_config["mode"].value, user_id=current_user_id)
                if not created:
                    await websocket.send_json({"type": "error", "message": "Failed to create session"})
                    continue
                current_session_id = created.get("id")
                conversation_history = []
                session_context_md = ""
                await websocket.send_json({"type": "session_created", "session": created})

            elif msg_type == "resume_session":
                requested_session_id = message.get("session_id")
                if not requested_session_id:
                    await websocket.send_json({"type": "error", "message": "session_id is required"})
                    continue
                if not session_manager.enabled:
                    await websocket.send_json({"type": "error", "message": "Supabase is not configured"})
                    continue
                if not current_user_id:
                    await websocket.send_json({"type": "error", "message": "Sign in to resume sessions"})
                    continue

                context = session_manager.build_context_window(requested_session_id, current_user_id, last_n=SESSION_CONTEXT_WINDOW)
                if not context:
                    await websocket.send_json({"type": "error", "message": "Session not found"})
                    continue

                current_session_id = requested_session_id
                session_config["mode"] = _parse_mode(context.mode)
                session_context_md = context.context_md
                conversation_history = context.recent_messages

                await websocket.send_json({
                    "type": "session_resumed",
                    "session_id": current_session_id,
                    "mode": session_config["mode"].value,
                    "context_md": session_context_md,
                    "history": conversation_history,
                })

            elif msg_type == "start":
                # Start new recording session
                audio_chunks = []
                is_recording = True
                print("[WebSocket] Recording started")
                await websocket.send_json({"type": "status", "message": "Recording started"})

            elif msg_type == "audio":
                # Receive audio chunk
                if is_recording:
                    audio_base64 = message.get("data")
                    if audio_base64:
                        chunk = base64.b64decode(audio_base64)
                        audio_chunks.append(chunk)

            elif msg_type == "stop":
                # Stop recording and process
                is_recording = False
                print(f"[WebSocket] Recording stopped, processing {len(audio_chunks)} chunks")

                if audio_chunks:
                    latest_assistant_response = await process_audio_pipeline(
                        websocket,
                        audio_chunks,
                        session_config,
                        conversation_history,
                        current_session_id,
                        session_context_md,
                        current_user_id,
                    )
                    if current_session_id and session_manager.enabled and current_user_id:
                        refreshed = session_manager.build_context_window(current_session_id, current_user_id, last_n=SESSION_CONTEXT_WINDOW)
                        if refreshed:
                            session_context_md = refreshed.context_md
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": "No audio received"
                    })

            elif msg_type == "reset":
                # Reset conversation history
                conversation_history = []
                session_context_md = ""
                await websocket.send_json({"type": "reset_ack"})

            elif msg_type == "text":
                text_content = (message.get("text") or "").strip()
                if not text_content:
                    await websocket.send_json({"type": "error", "message": "Text message is empty"})
                    continue

                latest_assistant_response = await process_text_pipeline(
                    websocket,
                    text_content,
                    session_config,
                    conversation_history,
                    current_session_id,
                    session_context_md,
                    current_user_id,
                )
                if current_session_id and session_manager.enabled and current_user_id:
                    refreshed = session_manager.build_context_window(current_session_id, current_user_id, last_n=SESSION_CONTEXT_WINDOW)
                    if refreshed:
                        session_context_md = refreshed.context_md

            elif msg_type == "read_aloud":
                tts_text = (message.get("text") or latest_assistant_response or "").strip()
                if not tts_text:
                    await websocket.send_json({"type": "error", "message": "No text available for read aloud"})
                    continue

                await stream_tts_for_text(websocket, tts_text)
                await websocket.send_json({"type": "read_aloud_complete"})

            elif msg_type == "ping":
                # Keep-alive ping
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected")
    except Exception as e:
        print(f"[WebSocket] Error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


async def process_audio_pipeline(
    websocket: WebSocket,
    audio_chunks: List[bytes],
    config: Dict,
    conversation_history: List[Dict],
    session_id: Optional[str],
    session_context_md: str,
    user_id: Optional[str],
) -> str:
    """
    Process audio through the full pipeline:
    1. Combine audio chunks
    2. Transcribe with faster-whisper
    3. Count filler words
    4. Analyze with LLM
    5. Generate TTS response
    """

    # Step 1: Combine audio chunks into WAV format
    print("[Pipeline] Combining audio chunks...")
    combined_audio = combine_audio_chunks(audio_chunks)

    if not combined_audio:
        await websocket.send_json({
            "type": "error",
            "message": "Failed to process audio"
        })
        return ""

    # Step 2: Transcribe audio
    print("[Pipeline] Transcribing audio...")
    await websocket.send_json({"type": "status", "message": "Transcribing..."})

    try:
        transcript, confidence = transcribe_audio(combined_audio)
        print(f"[Pipeline] Transcript: {transcript[:100]}...")
    except Exception as e:
        print(f"[Pipeline] Transcription error: {e}")
        await websocket.send_json({
            "type": "error",
            "message": f"Transcription failed: {e}"
        })
        return ""

    if not transcript.strip():
        await websocket.send_json({
            "type": "error",
            "message": "No speech detected in audio"
        })
        return ""

    # Send transcript to client
    await websocket.send_json({
        "type": "transcript",
        "text": transcript,
        "confidence": confidence,
        "final": True
    })

    # Step 3: Count filler words
    filler_details = count_filler_words(transcript)
    total_fillers = sum(filler_details.values())
    word_count = len(transcript.split())
    speech_metrics = analyze_speech_metrics(combined_audio, transcript, filler_count=total_fillers)

    await websocket.send_json({
        "type": "filler_words",
        "count": total_fillers,
        "details": filler_details,
        "word_count": word_count
    })
    await websocket.send_json({"type": "speech_metrics", "data": speech_metrics})

    # Step 4: Analyze with LLM (streaming)
    print(f"[Pipeline] Analyzing with {config['provider']}...")
    await websocket.send_json({"type": "status", "message": "Analyzing pitch..."})

    # Create analyzer with configured provider
    analyzer = PitchAnalyzer(
        provider_name=config["provider"],
        model=config["model"]
    )

    full_response = ""

    async for chunk in analyzer.analyze(
        transcript,
        filler_count=total_fillers,
        word_count=word_count,
        mode=config["mode"],
        conversation_history=conversation_history,
        context_md=session_context_md,
        recent_messages=conversation_history[-SESSION_CONTEXT_WINDOW:],
        speech_metrics=speech_metrics,
    ):
        clean_chunk = _sanitize_llm_output_text(chunk)
        full_response += clean_chunk

        # Send streaming chunk
        await websocket.send_json({
            "type": "analysis",
            "text": clean_chunk,
            "streaming": True
        })

    # Send final analysis marker
    await websocket.send_json({
        "type": "analysis",
        "text": "",
        "streaming": False,
        "complete": True
    })

    # Update conversation history
    conversation_history.append({"role": "user", "content": transcript})
    conversation_history.append({"role": "assistant", "content": full_response})

    if session_id and session_manager.enabled and user_id:
        session_manager.append_message(
            session_id,
            user_id,
            role="user",
            content=transcript,
            transcript=transcript,
        )
        session_manager.append_message(
            session_id,
            user_id,
            role="assistant",
            content=full_response,
        )
        session_manager.append_speech_metrics(session_id, user_id, speech_metrics)
        context_md = session_manager.generate_session_summary_markdown(session_id, user_id)
        if context_md:
            session_manager.update_context_markdown(session_id, user_id, context_md)

    # Parse and send scores (for pitch analysis mode)
    if config["mode"] == CoachingMode.PITCH_ANALYSIS:
        scores = parse_scores_from_response(full_response)
        await websocket.send_json({
            "type": "scores",
            "data": scores
        })

    # Step 5: Generate TTS only for conversation mode by default.
    if config["mode"] == CoachingMode.CONVERSATION:
        await stream_tts_for_text(websocket, full_response)

    # Signal completion
    await websocket.send_json({"type": "complete"})
    print("[Pipeline] Processing complete")
    return full_response


async def process_text_pipeline(
    websocket: WebSocket,
    text: str,
    config: Dict,
    conversation_history: List[Dict],
    session_id: Optional[str],
    session_context_md: str,
    user_id: Optional[str],
) -> str:
    """Process direct text input with the same LLM/session flow as voice."""
    await websocket.send_json({
        "type": "transcript",
        "text": text,
        "confidence": 1.0,
        "final": True,
    })

    filler_details = count_filler_words(text)
    total_fillers = sum(filler_details.values())
    word_count = len(text.split())

    await websocket.send_json({
        "type": "filler_words",
        "count": total_fillers,
        "details": filler_details,
        "word_count": word_count
    })

    analyzer = PitchAnalyzer(provider_name=config["provider"], model=config["model"])
    full_response = ""

    async for chunk in analyzer.analyze(
        text,
        filler_count=total_fillers,
        word_count=word_count,
        mode=config["mode"],
        conversation_history=conversation_history,
        context_md=session_context_md,
        recent_messages=conversation_history[-SESSION_CONTEXT_WINDOW:],
        speech_metrics=None,
    ):
        clean_chunk = _sanitize_llm_output_text(chunk)
        full_response += clean_chunk
        await websocket.send_json({"type": "analysis", "text": clean_chunk, "streaming": True})

    await websocket.send_json({"type": "analysis", "text": "", "streaming": False, "complete": True})

    conversation_history.append({"role": "user", "content": text})
    conversation_history.append({"role": "assistant", "content": full_response})

    if session_id and session_manager.enabled and user_id:
        session_manager.append_message(session_id, user_id, role="user", content=text, transcript=None)
        session_manager.append_message(session_id, user_id, role="assistant", content=full_response)
        context_md = session_manager.generate_session_summary_markdown(session_id, user_id)
        if context_md:
            session_manager.update_context_markdown(session_id, user_id, context_md)

    if config["mode"] == CoachingMode.PITCH_ANALYSIS:
        scores = parse_scores_from_response(full_response)
        await websocket.send_json({"type": "scores", "data": scores})

    if config["mode"] == CoachingMode.CONVERSATION:
        await stream_tts_for_text(websocket, full_response)

    await websocket.send_json({"type": "complete"})
    return full_response


async def stream_tts_for_text(websocket: WebSocket, text: str):
    tts_available, _ = check_piper_available()
    if not tts_available:
        return

    tts_text = (text or "").strip()
    if not tts_text:
        return

    await websocket.send_json({"type": "status", "message": "Generating voice response..."})

    for sentence in split_into_sentences(tts_text):
        audio = synthesize_speech(sentence)
        if audio:
            audio_base64 = base64.b64encode(audio).decode("utf-8")
            await websocket.send_json({
                "type": "audio",
                "data": audio_base64,
                "format": "wav"
            })


def combine_audio_chunks(chunks: List[bytes]) -> Optional[bytes]:
    """
    Combine received audio chunks into a valid WAV file.
    """
    if not chunks:
        return None

    # Combine all chunks
    combined = b"".join(chunks)

    # If it's already a valid WAV, return as-is
    if combined[:4] == b"RIFF" and combined[8:12] == b"WAVE":
        return combined

    # If it's raw PCM data, wrap in WAV header
    try:
        output = io.BytesIO()

        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)  # Mono
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(16000)  # 16kHz
            wav.writeframes(combined)

        return output.getvalue()
    except Exception as e:
        print(f"[Audio] Error combining chunks: {e}")
        return combined


# ============================================================================
# Static Files (Frontend)
# ============================================================================

# Mount frontend static files
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/style.css")
async def get_css():
    css_path = os.path.join(frontend_dir, "style.css")
    if os.path.exists(css_path):
        return FileResponse(css_path, media_type="text/css")
    return JSONResponse({"error": "CSS not found"}, status_code=404)


@app.get("/script.js")
async def get_js():
    js_path = os.path.join(frontend_dir, "script.js")
    if os.path.exists(js_path):
        return FileResponse(js_path, media_type="application/javascript")
    return JSONResponse({"error": "JS not found"}, status_code=404)


@app.get("/auth.js")
async def get_auth_js():
    js_path = os.path.join(frontend_dir, "auth.js")
    if os.path.exists(js_path):
        return FileResponse(js_path, media_type="application/javascript")
    return JSONResponse({"error": "Auth JS not found"}, status_code=404)


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = settings.server.port
    host = settings.server.host

    print(f"Starting server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)

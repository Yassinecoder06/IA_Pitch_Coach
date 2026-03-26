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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration
from backend.config import settings

# Import voice modules
from backend.voice.stt import load_model as load_stt_model, transcribe_audio
from backend.voice.tts import (
    check_piper_available,
    synthesize_speech,
    split_into_sentences
)
from backend.voice.voice_loop import VoiceLoop, VoiceLoopState, VoiceLoopConfig

# Import analysis modules
from backend.analysis.filler_detection import count_filler_words, get_total_filler_count
from backend.analysis.pitch_analysis import (
    PitchAnalyzer,
    CoachingMode,
    PITCH_COACH_SYSTEM_PROMPT,
    parse_scores_from_response
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

    # Session state
    audio_chunks: List[bytes] = []
    is_recording = False
    session_config = {
        "provider": settings.default_provider,
        "model": None,
        "mode": CoachingMode.PITCH_ANALYSIS
    }
    conversation_history: List[Dict] = []

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
                    mode_str = message["mode"]
                    try:
                        session_config["mode"] = CoachingMode(mode_str)
                    except ValueError:
                        session_config["mode"] = CoachingMode.PITCH_ANALYSIS
                print(f"[WebSocket] Config updated: {session_config}")
                await websocket.send_json({"type": "config_ack", "config": {
                    "provider": session_config["provider"],
                    "model": session_config["model"],
                    "mode": session_config["mode"].value
                }})

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
                    await process_audio_pipeline(
                        websocket,
                        audio_chunks,
                        session_config,
                        conversation_history
                    )
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": "No audio received"
                    })

            elif msg_type == "reset":
                # Reset conversation history
                conversation_history = []
                await websocket.send_json({"type": "reset_ack"})

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
    conversation_history: List[Dict]
):
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
        return

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
        return

    if not transcript.strip():
        await websocket.send_json({
            "type": "error",
            "message": "No speech detected in audio"
        })
        return

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

    await websocket.send_json({
        "type": "filler_words",
        "count": total_fillers,
        "details": filler_details,
        "word_count": word_count
    })

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
        conversation_history=conversation_history
    ):
        full_response += chunk

        # Send streaming chunk
        await websocket.send_json({
            "type": "analysis",
            "text": chunk,
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

    # Parse and send scores (for pitch analysis mode)
    if config["mode"] == CoachingMode.PITCH_ANALYSIS:
        scores = parse_scores_from_response(full_response)
        await websocket.send_json({
            "type": "scores",
            "data": scores
        })

    # Step 5: Generate TTS for response
    print("[Pipeline] Generating speech response...")
    tts_available, _ = check_piper_available()

    if tts_available:
        tts_text = PitchAnalyzer.extract_tts_summary(full_response)
        print(f"[Pipeline] TTS text: {tts_text[:100] if tts_text else 'None'}...")

        if tts_text:
            await websocket.send_json({"type": "status", "message": "Generating voice response..."})

            sentences = split_into_sentences(tts_text)
            for i, sentence in enumerate(sentences[:3]):
                audio = synthesize_speech(sentence)
                if audio:
                    audio_base64 = base64.b64encode(audio).decode("utf-8")
                    await websocket.send_json({
                        "type": "audio",
                        "data": audio_base64,
                        "format": "wav"
                    })

    # Signal completion
    await websocket.send_json({"type": "complete"})
    print("[Pipeline] Processing complete")


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


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = settings.server.port
    host = settings.server.host

    print(f"Starting server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)

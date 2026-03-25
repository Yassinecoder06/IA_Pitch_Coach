"""
AI Pitch Coach - FastAPI Backend
================================
Real-time voice-based pitch coaching using WebSockets.

Pipeline:
1. Browser records audio
2. Audio chunks sent via WebSocket
3. faster-whisper converts speech to text
4. LLM (qwen3.5:2b via Ollama) analyzes pitch
5. Piper generates audio response
6. Audio sent back to browser
"""

import os
import io
import json
import wave
import asyncio
import base64
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Import our modules
from stt import load_model as load_stt_model, transcribe_audio
from llm import (
    check_ollama_status,
    analyze_pitch,
    stream_llm_response,
    count_filler_words,
    get_total_filler_count,
    parse_scores_from_response,
    PITCH_COACH_SYSTEM_PROMPT
)
from tts import (
    check_piper_available,
    synthesize_speech,
    split_into_sentences,
    get_voice_model_path
)

# Create FastAPI app
app = FastAPI(
    title="AI Pitch Coach",
    description="Real-time voice-based pitch coaching",
    version="1.0.0"
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

    # Check Ollama status
    print("\n[Startup] Checking Ollama LLM...")
    ollama_status = await check_ollama_status()
    if ollama_status["status"] == "ok":
        print(f"[Startup] Ollama is running, model: {ollama_status['model']}")
        if not ollama_status["model_available"]:
            print(f"[Startup] Warning: Model not found. Run: ollama pull {ollama_status['model']}")
    else:
        print(f"[Startup] Warning: {ollama_status['message']}")

    # Check TTS status
    print("\n[Startup] Checking Text-to-Speech...")
    tts_available, tts_msg = check_piper_available()
    if tts_available:
        print(f"[Startup] Piper TTS is available")
    else:
        print(f"[Startup] Warning: {tts_msg}")

    startup_complete = True
    print("\n" + "=" * 60)
    print("Startup complete! Open http://localhost:8000 in your browser")
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
    ollama_status = await check_ollama_status()
    tts_available, tts_msg = check_piper_available()

    return {
        "status": "ok" if startup_complete else "starting",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "stt": "ready",
            "llm": ollama_status,
            "tts": {"available": tts_available, "message": tts_msg}
        }
    }


@app.get("/api/status")
async def get_status():
    """Get detailed system status."""
    ollama_status = await check_ollama_status()
    tts_available, tts_msg = check_piper_available()

    return {
        "stt": {
            "status": "ready",
            "model": os.getenv("WHISPER_MODEL_SIZE", "tiny")
        },
        "llm": {
            "status": ollama_status["status"],
            "model": ollama_status.get("model"),
            "available": ollama_status.get("model_available", False)
        },
        "tts": {
            "status": "ready" if tts_available else "unavailable",
            "message": tts_msg
        }
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
    - Server sends: {"type": "transcript", "text": "...", "final": bool}
    - Server sends: {"type": "analysis", "text": "...", "streaming": bool}
    - Server sends: {"type": "scores", "data": {...}}
    - Server sends: {"type": "filler_words", "count": N, "details": {...}}
    - Server sends: {"type": "audio", "data": base64_wav}
    - Server sends: {"type": "error", "message": "..."}
    """
    await websocket.accept()
    print("[WebSocket] Client connected")

    # Buffer for accumulating audio chunks
    audio_chunks: List[bytes] = []
    is_recording = False

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type")

            if msg_type == "start":
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
                    await process_audio_pipeline(websocket, audio_chunks)
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": "No audio received"
                    })

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


async def process_audio_pipeline(websocket: WebSocket, audio_chunks: List[bytes]):
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
    print("[Pipeline] Analyzing pitch with LLM...")
    await websocket.send_json({"type": "status", "message": "Analyzing pitch..."})

    full_response = ""

    async for chunk in analyze_pitch(transcript, total_fillers, word_count):
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

    # Parse and send scores
    scores = parse_scores_from_response(full_response)
    await websocket.send_json({
        "type": "scores",
        "data": scores
    })

    # Step 5: Generate TTS for response (sentence by sentence)
    print("[Pipeline] Generating speech response...")
    tts_available, tts_msg = check_piper_available()
    print(f"[Pipeline] TTS available: {tts_available}, {tts_msg}")

    if tts_available:
        # Extract a summary sentence for TTS (keep it short)
        tts_text = extract_tts_summary(full_response)
        print(f"[Pipeline] TTS text extracted: {tts_text[:100] if tts_text else 'None'}...")

        if tts_text:
            await websocket.send_json({"type": "status", "message": "Generating voice response..."})

            sentences = split_into_sentences(tts_text)
            print(f"[Pipeline] TTS sentences: {len(sentences)}")

            for i, sentence in enumerate(sentences[:3]):  # Limit to 3 sentences for speed
                print(f"[Pipeline] Synthesizing sentence {i+1}: {sentence[:50]}...")
                audio = synthesize_speech(sentence)
                if audio:
                    print(f"[Pipeline] Audio generated: {len(audio)} bytes")
                    audio_base64 = base64.b64encode(audio).decode("utf-8")
                    await websocket.send_json({
                        "type": "audio",
                        "data": audio_base64,
                        "format": "wav"
                    })
                else:
                    print(f"[Pipeline] Failed to generate audio for sentence {i+1}")
        else:
            print("[Pipeline] No TTS text extracted from response")
    else:
        print(f"[Pipeline] TTS not available: {tts_msg}")

    # Signal completion
    await websocket.send_json({"type": "complete"})
    print("[Pipeline] Processing complete")


def combine_audio_chunks(chunks: List[bytes]) -> Optional[bytes]:
    """
    Combine received audio chunks into a valid WAV file.

    The browser typically sends webm/opus or wav chunks.
    We need to convert to a format faster-whisper can process.
    """
    if not chunks:
        return None

    # Combine all chunks
    combined = b"".join(chunks)

    # If it's already a valid WAV, return as-is
    if combined[:4] == b"RIFF" and combined[8:12] == b"WAVE":
        return combined

    # If it's raw PCM data, wrap in WAV header
    # Assume 16-bit mono audio at 16kHz (common for speech)
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
        return combined  # Return raw data as fallback


def extract_tts_summary(response: str) -> str:
    """
    Extract a short summary from LLM response for TTS.
    Keeps the spoken response concise.
    """
    import re

    if not response or not response.strip():
        return ""

    text = response.strip()

    # Remove common reasoning tags that some models include.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<\/?analysis>", "", text, flags=re.IGNORECASE)

    # Prefer ANALYSIS section when present.
    analysis_match = re.search(
        r"ANALYSIS:\s*\n?(.+?)(?=\n\s*ADVICE:|\n\s*SCORES:|$)",
        text,
        re.IGNORECASE | re.DOTALL
    )
    if analysis_match:
        analysis_text = " ".join(analysis_match.group(1).split())
        if analysis_text:
            return analysis_text

    # If no analysis block, collect meaningful lines (including bullet advice text).
    lines = text.split("\n")
    candidate_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Strip markdown bullet/numbering while keeping content.
        line = re.sub(r"^[-*•]\s+", "", line)
        line = re.sub(r"^\d+[.)]\s+", "", line)

        # Skip pure headers or score lines.
        if re.match(r"^(SCORES|ANALYSIS|ADVICE):?$", line, re.IGNORECASE):
            continue
        if re.match(r"^(Clarity|Language|Confidence|Topic Relevance|Filler\s*Words?)\s*:\s*\d+\s*/\s*10\s*$", line, re.IGNORECASE):
            continue

        # Skip markdown formatting lines.
        if re.match(r"^[#`*_\-]+$", line):
            continue

        if len(line) >= 8:
            candidate_lines.append(line)
        if len(candidate_lines) >= 2:
            break

    if candidate_lines:
        return " ".join(candidate_lines)

    # Last fallback: speak first one or two non-empty sentences from cleaned text.
    cleaned = " ".join(text.split())
    sentences = split_into_sentences(cleaned)
    if sentences:
        return " ".join(sentences[:2])

    return cleaned[:220].strip()


# ============================================================================
# Static Files (Frontend)
# ============================================================================

# Mount frontend static files
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


# Serve frontend files directly
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

    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    print(f"Starting server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)

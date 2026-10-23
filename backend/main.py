"""
AI Pitch Coach - Main Application Entrypoint
============================================
FastAPI backend with local-first Owner authentication, multi-AI provider registry,
streaming speech-to-text, real-time coaching evaluation, and text-to-speech.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Ensure backend package is in python path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.config.settings import settings
from backend.storage.database import db
from backend.auth.owner import owner_service
from backend.storage.provider_repository import migrate_legacy_encryption
from backend.voice.stt import load_model as load_stt_model
from backend.voice.tts import check_piper_available, preload_tts_model

# Import API routers
from backend.api.routers.auth import router as auth_router
from backend.api.routers.ai import router as ai_router
from backend.api.routers.sessions import router as sessions_router
from backend.api.routers.tts import router as tts_router
from backend.api.routers.system import router as system_router
from backend.ws.coach import handle_websocket_connection


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle management."""
    print("\n" + "=" * 65)
    print("AI Pitch Coach (SOTA Edition) - Starting Up")
    print("=" * 65)

    # 1. Initialize Database Schema
    try:
        db.init_schema()
        print("[Database] Schema initialized successfully.")
        migrated = migrate_legacy_encryption()
        if migrated:
            print(f"[Security] Re-encrypted {migrated} provider key(s) with the new secret key.")
    except Exception as e:
        print(f"[Database] Error initializing schema: {e}")

    # 2. Check Owner Configuration (from Muse)
    if not owner_service.is_configured():
        code = owner_service.get_or_create_setup_code()
        print("\n" + "#" * 65)
        print("  FIRST-RUN SETUP CODE (CLAIM YOUR LOCAL PITCH COACH):")
        print(f"  >>>  {code}  <<<")
        print("  Enter this code in your browser to set your owner passphrase.")
        print("#" * 65 + "\n")
    else:
        print("[Auth] Owner passphrase is configured. Ready to unlock.")

    # 3. Preload Models if enabled
    if settings.preload_stt_model:
        print("[STT] Preloading speech-to-text model...")
        try:
            load_stt_model()
            print("[STT] Speech-to-text model preloaded.")
        except Exception as e:
            print(f"[STT] Warning: STT preload failed: {e}")

    if settings.preload_tts_model:
        print("[TTS] Preloading Piper TTS voice model...")
        try:
            preload_tts_model()
            print("[TTS] Voice model preloaded.")
        except Exception as e:
            print(f"[TTS] Warning: TTS preload failed: {e}")

    print(f"\n[Ready] Application running on http://{settings.host}:{settings.port}")
    print("=" * 65 + "\n")

    yield

    print("[Shutdown] AI Pitch Coach shutting down.")


# Create FastAPI application
app = FastAPI(
    title="AI Pitch Coach SOTA",
    description="Real-time voice-first startup pitch coaching with multi-AI orchestration and local owner security.",
    version="3.0.0",
    lifespan=lifespan
)

# CORS only for the local Vite dev server; the production client is served same-origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register REST Routers
app.include_router(auth_router)
app.include_router(ai_router)
app.include_router(sessions_router)
app.include_router(tts_router)
app.include_router(system_router)


# Register WebSocket endpoint
@app.websocket("/ws")
async def websocket_route(websocket: WebSocket):
    await handle_websocket_connection(websocket)


# Mount Web Client (React Vite build)
web_dist = ROOT_DIR / "web" / "dist"
old_frontend = ROOT_DIR / "frontend"

if web_dist.exists():
    app.mount("/assets", StaticFiles(directory=str(web_dist / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = web_dist / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(web_dist / "index.html")
elif old_frontend.exists():
    app.mount("/static", StaticFiles(directory=str(old_frontend)), name="static")

    @app.get("/")
    async def serve_old_index():
        return FileResponse(old_frontend / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.host, port=settings.port, reload=settings.debug)

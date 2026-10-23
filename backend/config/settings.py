"""
Application configuration for AI Pitch Coach.
Supports .env and .env.local with sensible local-first defaults.
"""

from __future__ import annotations

import os
import secrets
from functools import cached_property
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

ROOT_DIR = Path(__file__).resolve().parents[2]

# Value shipped in older .env.example files. Treated as "unset" so it is never used to sign or encrypt.
LEGACY_DEFAULT_SECRET = "pitch-coach-local-session-secret-key-32chars"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(ROOT_DIR / ".env.local"), str(ROOT_DIR / ".env")),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server configuration
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    debug: bool = Field(default=False, alias="DEBUG")
    secret_key_env: Optional[str] = Field(default=None, alias="SECRET_KEY")
    data_dir: str = Field(default=str(ROOT_DIR / "data"), alias="DATA_DIR")
    cors_origins: List[str] = Field(
        default=["http://localhost:5173", "http://127.0.0.1:5173"],
        alias="CORS_ORIGINS"
    )
    secure_cookies: bool = Field(default=False, alias="SECURE_COOKIES")

    # Database configuration (PostgreSQL in Docker, SQLite locally)
    database_url: str = Field(
        default="sqlite:///" + str(ROOT_DIR / "data" / "pitch_coach.db"),
        alias="DATABASE_URL"
    )

    # Default LLM Provider & Model
    default_llm: str = Field(default="ollama", alias="DEFAULT_LLM")
    default_model: Optional[str] = Field(default=None, alias="DEFAULT_MODEL")
    ollama_endpoint: str = Field(default="http://localhost:11434", alias="OLLAMA_ENDPOINT")

    # Speech-to-Text (faster-whisper)
    whisper_model_size: str = Field(default="tiny.en", alias="WHISPER_MODEL_SIZE")
    whisper_device: str = Field(default="cpu", alias="WHISPER_DEVICE")
    whisper_compute_type: str = Field(default="int8", alias="WHISPER_COMPUTE_TYPE")
    whisper_language: str = Field(default="en", alias="WHISPER_LANGUAGE")
    whisper_models_dir: str = Field(default=str(ROOT_DIR / "models" / "whisper"), alias="WHISPER_MODELS_DIR")
    preload_stt_model: bool = Field(default=False, alias="PRELOAD_STT_MODEL")

    # Text-to-Speech (Piper)
    piper_models_dir: str = Field(default=str(ROOT_DIR / "models" / "piper"), alias="PIPER_MODELS_DIR")
    piper_voice: str = Field(default="en_US-lessac-medium", alias="PIPER_VOICE")
    preload_tts_model: bool = Field(default=False, alias="PRELOAD_TTS_MODEL")

    # MCP Web Search Tool
    mcp_enabled: bool = Field(default=True, alias="MCP_ENABLED")
    mcp_auto_search: bool = Field(default=True, alias="MCP_AUTO_SEARCH")
    mcp_max_results: int = Field(default=5, alias="MCP_MAX_RESULTS")
    mcp_server_cmd: str = Field(default="python", alias="MCP_SERVER_CMD")
    mcp_server_args: str = Field(default="services/mcp_server/server.py", alias="MCP_SERVER_ARGS")

    # Session window
    session_context_window: int = Field(default=8, alias="SESSION_CONTEXT_WINDOW")

    @cached_property
    def secret_key(self) -> str:
        """SECRET_KEY from the environment, or a random key persisted to <data_dir>/secret.key."""
        if self.secret_key_env and self.secret_key_env != LEGACY_DEFAULT_SECRET:
            return self.secret_key_env

        key_file = Path(self.data_dir) / "secret.key"
        if key_file.exists():
            return key_file.read_text(encoding="utf-8").strip()

        key_file.parent.mkdir(parents=True, exist_ok=True)
        key = secrets.token_urlsafe(48)
        fd = os.open(str(key_file), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(key)
        return key

    @property
    def stt(self) -> SimpleNamespace:
        return SimpleNamespace(
            model_size=self.whisper_model_size,
            device=self.whisper_device,
            compute_type=self.whisper_compute_type,
            language=self.whisper_language,
            models_dir=Path(self.whisper_models_dir),
        )

    @property
    def tts(self) -> SimpleNamespace:
        return SimpleNamespace(
            voice=self.piper_voice,
            models_dir=Path(self.piper_models_dir),
        )


settings = Settings()

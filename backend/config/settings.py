"""
Settings Module - Environment-Based Configuration
=================================================
Loads all model providers from environment variables.
Auto-detects available providers based on configured API keys.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    # Ensure local .env changes take precedence over stale shell vars.
    load_dotenv(env_path, override=True)


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""
    name: str
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    api_version: Optional[str] = None
    project_id: Optional[str] = None
    enabled: bool = False
    models: List[str] = field(default_factory=list)


@dataclass
class STTConfig:
    """Speech-to-Text configuration."""
    model_size: str = "tiny"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "en"
    models_dir: Optional[Path] = None


@dataclass
class TTSConfig:
    """Text-to-Speech configuration."""
    voice: str = "en_US-lessac-medium"
    models_dir: Optional[Path] = None


@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False


class Settings:
    """
    Central settings manager for AI Pitch Coach.
    Automatically loads and validates configuration from environment.
    """

    def __init__(self):
        self._providers: Dict[str, ProviderConfig] = {}
        self._load_providers()
        self._load_stt_config()
        self._load_tts_config()
        self._load_server_config()

    def _load_providers(self):
        """Load all LLM provider configurations from environment."""

        # OpenAI
        self._providers["openai"] = ProviderConfig(
            name="OpenAI",
            endpoint=os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1"),
            api_key=os.getenv("OPENAI_API_KEY"),
            enabled=bool(os.getenv("OPENAI_API_KEY")),
            models=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
        )

        # Anthropic
        self._providers["anthropic"] = ProviderConfig(
            name="Anthropic",
            endpoint=os.getenv("ANTHROPIC_ENDPOINT", "https://api.anthropic.com"),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            enabled=bool(os.getenv("ANTHROPIC_API_KEY")),
            models=["claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"]
        )

        # Google (Gemini)
        self._providers["google"] = ProviderConfig(
            name="Google",
            api_key=os.getenv("GOOGLE_API_KEY"),
            enabled=bool(os.getenv("GOOGLE_API_KEY")),
            models=["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
        )

        # Azure OpenAI
        self._providers["azure"] = ProviderConfig(
            name="Azure OpenAI",
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
            enabled=bool(os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT")),
            models=["gpt-4o", "gpt-4-turbo", "gpt-35-turbo"]
        )

        # DeepSeek
        self._providers["deepseek"] = ProviderConfig(
            name="DeepSeek",
            endpoint=os.getenv("DEEPSEEK_ENDPOINT", "https://api.deepseek.com"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            enabled=bool(os.getenv("DEEPSEEK_API_KEY")),
            models=["deepseek-chat", "deepseek-coder"]
        )

        # Mistral
        self._providers["mistral"] = ProviderConfig(
            name="Mistral",
            endpoint=os.getenv("MISTRAL_ENDPOINT", "https://api.mistral.ai/v1"),
            api_key=os.getenv("MISTRAL_API_KEY"),
            enabled=bool(os.getenv("MISTRAL_API_KEY")),
            models=["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest"]
        )

        # Ollama (local)
        ollama_endpoint = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
        self._providers["ollama"] = ProviderConfig(
            name="Ollama",
            endpoint=ollama_endpoint,
            enabled=True,  # Always enabled, availability checked at runtime
            models=[]  # Populated dynamically
        )

        # Alibaba (DashScope)
        self._providers["alibaba"] = ProviderConfig(
            name="Alibaba",
            endpoint=os.getenv("ALIBABA_ENDPOINT", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            api_key=os.getenv("ALIBABA_API_KEY"),
            enabled=bool(os.getenv("ALIBABA_API_KEY")),
            models=["qwen-turbo", "qwen-plus", "qwen-max"]
        )

        # ModelScope
        self._providers["modelscope"] = ProviderConfig(
            name="ModelScope",
            endpoint=os.getenv("MODELSCOPE_ENDPOINT", "https://api-inference.modelscope.cn/v1"),
            api_key=os.getenv("MODELSCOPE_API_KEY"),
            enabled=bool(os.getenv("MODELSCOPE_API_KEY")),
            models=["qwen2.5-coder-32b-instruct"]
        )

        # Moonshot
        self._providers["moonshot"] = ProviderConfig(
            name="Moonshot",
            endpoint=os.getenv("MOONSHOT_ENDPOINT", "https://api.moonshot.cn/v1"),
            api_key=os.getenv("MOONSHOT_API_KEY"),
            enabled=bool(os.getenv("MOONSHOT_API_KEY")),
            models=["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"]
        )

        # Unbound
        self._providers["unbound"] = ProviderConfig(
            name="Unbound",
            endpoint=os.getenv("UNBOUND_ENDPOINT", "https://api.getunbound.ai"),
            api_key=os.getenv("UNBOUND_API_KEY"),
            enabled=bool(os.getenv("UNBOUND_API_KEY")),
            models=[]
        )

        # SiliconFlow
        self._providers["siliconflow"] = ProviderConfig(
            name="SiliconFlow",
            endpoint=os.getenv("SiliconFLOW_ENDPOINT", "https://api.siliconflow.cn/v1/"),
            api_key=os.getenv("SiliconFLOW_API_KEY"),
            enabled=bool(os.getenv("SiliconFLOW_API_KEY")),
            models=["deepseek-ai/DeepSeek-V3", "Qwen/Qwen2.5-72B-Instruct"]
        )

        # IBM
        self._providers["ibm"] = ProviderConfig(
            name="IBM",
            endpoint=os.getenv("IBM_ENDPOINT", "https://us-south.ml.cloud.ibm.com"),
            api_key=os.getenv("IBM_API_KEY"),
            project_id=os.getenv("IBM_PROJECT_ID"),
            enabled=bool(os.getenv("IBM_API_KEY") and os.getenv("IBM_PROJECT_ID")),
            models=["ibm/granite-13b-chat-v2", "meta-llama/llama-3-70b-instruct"]
        )

        # Grok (xAI)
        self._providers["grok"] = ProviderConfig(
            name="Grok",
            endpoint=os.getenv("GROK_ENDPOINT", "https://api.x.ai/v1"),
            api_key=os.getenv("GROK_API_KEY"),
            enabled=bool(os.getenv("GROK_API_KEY")),
            models=["grok-2", "grok-2-mini"]
        )

    def _load_stt_config(self):
        """Load STT configuration."""
        base_dir = Path(__file__).parent.parent.parent
        self.stt = STTConfig(
            model_size=os.getenv("WHISPER_MODEL_SIZE", "tiny"),
            device=os.getenv("WHISPER_DEVICE", "cpu"),
            compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
            language=os.getenv("WHISPER_LANGUAGE", "en"),
            models_dir=Path(os.getenv("WHISPER_MODELS_DIR", base_dir / "models" / "whisper"))
        )

    def _load_tts_config(self):
        """Load TTS configuration."""
        base_dir = Path(__file__).parent.parent.parent
        self.tts = TTSConfig(
            voice=os.getenv("PIPER_VOICE", "en_US-lessac-medium"),
            models_dir=Path(os.getenv("PIPER_MODELS_DIR", base_dir / "models" / "piper"))
        )

    def _load_server_config(self):
        """Load server configuration."""
        self.server = ServerConfig(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            debug=os.getenv("DEBUG", "false").lower() == "true"
        )

    @property
    def default_provider(self) -> str:
        """Get the default LLM provider."""
        default = os.getenv("DEFAULT_LLM", "ollama").lower()
        if default in self._providers and self._providers[default].enabled:
            return default
        # Fallback to first enabled provider
        for name, config in self._providers.items():
            if config.enabled:
                return name
        return "ollama"

    @property
    def default_model(self) -> str:
        """Get the default model for the default provider."""
        return os.getenv("DEFAULT_MODEL", "")

    def get_provider(self, name: str) -> Optional[ProviderConfig]:
        """Get configuration for a specific provider."""
        return self._providers.get(name.lower())

    def get_enabled_providers(self) -> Dict[str, ProviderConfig]:
        """Get all enabled providers."""
        return {
            name: config
            for name, config in self._providers.items()
            if config.enabled
        }

    def get_all_providers(self) -> Dict[str, ProviderConfig]:
        """Get all provider configurations."""
        return self._providers.copy()

    def list_available_models(self) -> Dict[str, List[str]]:
        """List all available models grouped by provider."""
        result = {}
        for name, config in self._providers.items():
            if config.enabled and config.models:
                result[name] = config.models
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Export settings as dictionary."""
        return {
            "providers": {
                name: {
                    "name": config.name,
                    "enabled": config.enabled,
                    "models": config.models
                }
                for name, config in self._providers.items()
            },
            "default_provider": self.default_provider,
            "default_model": self.default_model,
            "stt": {
                "model_size": self.stt.model_size,
                "device": self.stt.device,
                "compute_type": self.stt.compute_type
            },
            "tts": {
                "voice": self.tts.voice
            },
            "server": {
                "host": self.server.host,
                "port": self.server.port
            }
        }


# Global settings instance
settings = Settings()

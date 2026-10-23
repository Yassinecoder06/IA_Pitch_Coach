"""
Type definitions for LLM providers and model discovery.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

ProviderKind = Literal[
    "openai",
    "anthropic",
    "gemini",
    "ollama_local",
    "ollama_cloud",
    "openai_compatible",
    "azure"
]

PROVIDER_KINDS = [
    "openai",
    "anthropic",
    "gemini",
    "ollama_local",
    "ollama_cloud",
    "openai_compatible",
    "azure"
]

KIND_LABELS: Dict[str, str] = {
    "openai": "OpenAI",
    "anthropic": "Anthropic Claude",
    "gemini": "Google Gemini",
    "ollama_local": "Local Ollama",
    "ollama_cloud": "Ollama Cloud",
    "openai_compatible": "OpenAI-Compatible (Groq, DeepSeek, Mistral, OpenRouter)",
    "azure": "Azure OpenAI"
}

NEEDS_KEY: Dict[str, bool] = {
    "openai": True,
    "anthropic": True,
    "gemini": True,
    "ollama_local": False,
    "ollama_cloud": True,
    "openai_compatible": False,
    "azure": True
}


class ModelInfo(BaseModel):
    id: str
    label: str
    capabilities: List[str] = ["chat"]
    description: Optional[str] = None


class LLMMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


def make_model_ref(provider_id: str, model_id: str) -> str:
    return f"{provider_id}:{model_id}"


def parse_model_ref(ref: str) -> tuple[str, str]:
    if ":" in ref:
        parts = ref.split(":", 1)
        return parts[0], parts[1]
    return "default", ref

"""
Model discovery service for AI providers.
Queries provider APIs to fetch available models.
"""

from __future__ import annotations

import httpx
from typing import Any, Dict, List, Optional
from backend.llm.types import ModelInfo, ProviderKind

TIMEOUT = 10.0


async def discover_models(
    kind: ProviderKind,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None
) -> List[ModelInfo]:
    """Discover available models for a given provider."""
    models: List[ModelInfo] = []

    if kind == "ollama_local":
        url = (base_url or "http://localhost:11434").rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                res = await client.get(f"{url}/api/tags")
                if res.status_code == 200:
                    data = res.json()
                    for item in data.get("models", []):
                        name = item.get("name")
                        if name:
                            models.append(ModelInfo(id=name, label=name, capabilities=["chat"]))
        except Exception as e:
            # Fallback local models if Ollama offline
            models = [
                ModelInfo(id="qwen3:0.6b", label="Qwen 3 0.6B (Fast Local)", capabilities=["chat"]),
                ModelInfo(id="llama3.2:3b", label="Llama 3.2 3B", capabilities=["chat"]),
            ]
        return models

    if kind == "gemini":
        # Known standard Gemini models
        return [
            ModelInfo(id="gemini-2.5-flash", label="Gemini 2.5 Flash (Fast & Accurate)", capabilities=["chat"]),
            ModelInfo(id="gemini-2.5-pro", label="Gemini 2.5 Pro (Deep Reasoning)", capabilities=["chat"]),
            ModelInfo(id="gemini-1.5-flash", label="Gemini 1.5 Flash", capabilities=["chat"]),
            ModelInfo(id="gemini-1.5-pro", label="Gemini 1.5 Pro", capabilities=["chat"]),
        ]

    if kind == "anthropic":
        return [
            ModelInfo(id="claude-3-5-sonnet-latest", label="Claude 3.5 Sonnet (SOTA Coach)", capabilities=["chat"]),
            ModelInfo(id="claude-3-7-sonnet-latest", label="Claude 3.7 Sonnet (Hybrid Reasoning)", capabilities=["chat"]),
            ModelInfo(id="claude-3-5-haiku-latest", label="Claude 3.5 Haiku (Ultra Fast)", capabilities=["chat"]),
        ]

    if kind in ("openai", "openai_compatible", "ollama_cloud"):
        url = (base_url or "https://api.openai.com/v1").rstrip("/")
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                res = await client.get(f"{url}/models", headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    raw_models = data.get("data", [])
                    for item in raw_models:
                        model_id = item.get("id")
                        if model_id and not any(ext in model_id for ext in ["dall-e", "whisper", "tts", "embedding", "moderation"]):
                            models.append(ModelInfo(id=model_id, label=model_id, capabilities=["chat"]))
        except Exception:
            pass

        if not models:
            if kind == "openai":
                models = [
                    ModelInfo(id="gpt-4o", label="GPT-4o (Omni SOTA)", capabilities=["chat"]),
                    ModelInfo(id="gpt-4o-mini", label="GPT-4o Mini (Fast & Smart)", capabilities=["chat"]),
                    ModelInfo(id="o1-mini", label="o1-mini (Reasoning)", capabilities=["chat"]),
                ]
            else:
                models = [
                    ModelInfo(id="llama-3.3-70b-versatile", label="Llama 3.3 70B (Groq)", capabilities=["chat"]),
                    ModelInfo(id="deepseek-chat", label="DeepSeek-V3", capabilities=["chat"]),
                    ModelInfo(id="mistral-large-latest", label="Mistral Large", capabilities=["chat"]),
                ]
        return models

    if kind == "azure":
        return [
            ModelInfo(id="gpt-4o", label="Azure GPT-4o", capabilities=["chat"]),
            ModelInfo(id="gpt-4o-mini", label="Azure GPT-4o Mini", capabilities=["chat"]),
        ]

    return models

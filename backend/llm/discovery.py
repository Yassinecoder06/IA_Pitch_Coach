"""
Model discovery service for AI providers.
Queries provider APIs to fetch available models.
"""

from __future__ import annotations

import logging
import httpx
from typing import Any, Dict, List, Optional
from backend.config.settings import settings
from backend.llm.types import ModelInfo, ProviderKind

logger = logging.getLogger(__name__)
TIMEOUT = 10.0


async def discover_models(
    kind: ProviderKind,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None
) -> List[ModelInfo]:
    """Discover available models for a given provider."""
    models: List[ModelInfo] = []

    if kind == "ollama_local":
        # Host bridge detection: Try provided base_url, settings.ollama_endpoint, or common host bridges
        candidates = []
        if base_url:
            candidates.append(base_url.rstrip("/"))
        else:
            if settings.ollama_endpoint:
                candidates.append(settings.ollama_endpoint.rstrip("/"))
            candidates.append("http://localhost:11434")
            candidates.append("http://host.docker.internal:11434")

        # Deduplicate while preserving order
        endpoints: List[str] = []
        for ep in candidates:
            if ep not in endpoints:
                endpoints.append(ep)

        for ep in endpoints:
            try:
                async with httpx.AsyncClient(timeout=4.0) as client:
                    res = await client.get(f"{ep}/api/tags")
                    if res.status_code == 200:
                        data = res.json()
                        for item in data.get("models", []):
                            name = item.get("name") or item.get("model")
                            if name:
                                caps = ["embedding"] if "embed" in name.lower() else ["chat"]
                                models.append(ModelInfo(id=name, label=name, capabilities=caps))
                        if models:
                            logger.info(f"Ollama bridge detected {len(models)} local models at {ep}")
                            return models
            except Exception as e:
                logger.debug(f"Ollama local probe failed at {ep}: {e}")

        # If Ollama service is not installed or offline on the host machine,
        # return empty list as requested by user.
        return []

    if kind == "ollama_cloud":
        url = (base_url or "https://ollama.com").rstrip("/")
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                res = await client.get(f"{url}/api/tags", headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    for item in data.get("models", []):
                        name = item.get("name") or item.get("model")
                        if name:
                            caps = ["embedding"] if "embed" in name.lower() else ["chat"]
                            models.append(ModelInfo(id=name, label=name, capabilities=caps))
                    if models:
                        return models
        except Exception as e:
            logger.warning(f"Ollama Cloud discovery error at {url}: {e}")

        # Standard curated Ollama Cloud models if tag endpoint returned empty but key is set
        if api_key:
            return [
                ModelInfo(id="llama3.3:70b", label="Llama 3.3 70B (Ollama Cloud)", capabilities=["chat"]),
                ModelInfo(id="qwen2.5:72b", label="Qwen 2.5 72B (Ollama Cloud)", capabilities=["chat"]),
                ModelInfo(id="deepseek-r1:70b", label="DeepSeek-R1 70B (Ollama Cloud)", capabilities=["chat"]),
                ModelInfo(id="mistral-large:latest", label="Mistral Large (Ollama Cloud)", capabilities=["chat"]),
            ]
        return []

    if kind == "gemini":
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

    if kind in ("openai", "openai_compatible"):
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

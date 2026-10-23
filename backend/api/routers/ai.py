"""
AI Providers and model preferences management router adapted from Muse.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.api.deps import get_current_user_id
from backend.storage.provider_repository import provider_repository
from backend.llm.types import (
    PROVIDER_KINDS,
    KIND_LABELS,
    NEEDS_KEY,
    ProviderKind,
    make_model_ref
)
from backend.llm.discovery import discover_models
from backend.llm.router import llm_router

router = APIRouter(prefix="/api/ai", tags=["ai"])


class CreateProviderRequest(BaseModel):
    kind: ProviderKind
    name: Optional[str] = None
    baseUrl: Optional[str] = None
    apiKey: Optional[str] = None


class UpdateProviderRequest(BaseModel):
    name: Optional[str] = None
    baseUrl: Optional[str] = None
    apiKey: Optional[str] = None
    enabled: Optional[bool] = None


class TestModelRequest(BaseModel):
    ref: str
    capability: str = "chat"


class PreferencesRequest(BaseModel):
    chat_models: List[str] = Field(default_factory=list)
    fallback_models: List[str] = Field(default_factory=list)
    tts_ref: Optional[str] = "builtin:piper"
    tts_voice: Optional[str] = "en_US-lessac-medium"
    stt_ref: Optional[str] = "builtin:whisper"


BUILTIN_OPTIONS = [
    {
        "ref": "builtin:default",
        "providerId": "builtin",
        "providerName": "Host Ollama (Bridge)",
        "kind": "ollama_local",
        "model": "default",
        "label": "Host Ollama (Auto-Bridge)",
        "capabilities": ["chat"]
    },
    {
        "ref": "builtin:whisper",
        "providerId": "builtin",
        "providerName": "Fast Whisper",
        "kind": "builtin",
        "model": "whisper",
        "label": "Faster-Whisper (Local CPU/GPU)",
        "capabilities": ["stt"]
    },
    {
        "ref": "builtin:piper",
        "providerId": "builtin",
        "providerName": "Piper Neural Voice",
        "kind": "builtin",
        "model": "piper",
        "label": "Piper TTS (Local)",
        "capabilities": ["tts"]
    }
]


def build_catalog(providers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    catalog = []
    for p in providers:
        if not p.get("enabled", True):
            continue
        p_id = p["id"]
        p_name = p["name"]
        p_kind = p["kind"]
        for m in p.get("models", []):
            m_id = m.get("id")
            m_label = m.get("label", m_id)
            catalog.append({
                "ref": make_model_ref(p_id, m_id),
                "providerId": p_id,
                "providerName": p_name,
                "kind": p_kind,
                "model": m_id,
                "label": m_label,
                "capabilities": m.get("capabilities", ["chat"])
            })

    catalog.extend(BUILTIN_OPTIONS)
    return catalog


@router.get("/providers")
async def get_providers_state(user_id: str = Depends(get_current_user_id)):
    """Returns the full AI providers state: configured providers, model catalog, and preferences."""
    providers = provider_repository.list_providers(user_id)
    preferences = provider_repository.get_preferences(user_id)

    # If no providers configured yet, check if local Ollama host bridge is active
    if not providers:
        local_models = await discover_models("ollama_local")
        if local_models:
            p = provider_repository.create_provider(
                kind="ollama_local",
                name="Local Ollama (Host)",
                user_id=user_id
            )
            provider_repository.save_models(p["id"], [m.model_dump() for m in local_models], None, user_id)
            providers = provider_repository.list_providers(user_id)
            if not preferences.get("chat_models"):
                first_ref = make_model_ref(p["id"], local_models[0].id)
                preferences = provider_repository.save_preferences(
                    chat_models=[first_ref],
                    fallback_models=[],
                    user_id=user_id
                )

    catalog = build_catalog(providers)

    kinds_info = [
        {
            "kind": k,
            "label": KIND_LABELS.get(k, k),
            "needsKey": NEEDS_KEY.get(k, False),
            "needsBaseUrl": k.endswith("_compatible")
        }
        for k in PROVIDER_KINDS
    ]

    return {
        "providers": providers,
        "preferences": preferences,
        "catalog": catalog,
        "kinds": kinds_info
    }


@router.post("/providers")
async def add_provider(payload: CreateProviderRequest, user_id: str = Depends(get_current_user_id)):
    """Add a new AI provider, securely storing the API key."""
    name = payload.name.strip() if payload.name else KIND_LABELS.get(payload.kind, payload.kind)
    provider = provider_repository.create_provider(
        kind=payload.kind,
        name=name,
        base_url=payload.baseUrl.strip() if payload.baseUrl else None,
        api_key=payload.apiKey.strip() if payload.apiKey else None,
        user_id=user_id
    )

    # Immediately attempt model discovery
    try:
        discovered = await discover_models(
            kind=payload.kind,
            base_url=provider.get("base_url"),
            api_key=payload.apiKey.strip() if payload.apiKey else None
        )
        provider_repository.save_models(provider["id"], [m.model_dump() for m in discovered], None, user_id)
    except Exception as e:
        provider_repository.save_models(provider["id"], [], str(e), user_id)

    return await get_providers_state(user_id)


@router.patch("/providers/{provider_id}")
async def update_provider(
    provider_id: str,
    payload: UpdateProviderRequest,
    user_id: str = Depends(get_current_user_id)
):
    """Update provider settings or replace API key."""
    updated = provider_repository.update_provider(
        provider_id=provider_id,
        name=payload.name,
        base_url=payload.baseUrl,
        api_key=payload.apiKey,
        enabled=payload.enabled,
        user_id=user_id
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Provider not found")
    return await get_providers_state(user_id)


@router.delete("/providers/{provider_id}")
async def delete_provider(provider_id: str, user_id: str = Depends(get_current_user_id)):
    """Delete a configured provider."""
    provider_repository.delete_provider(provider_id, user_id)
    return await get_providers_state(user_id)


@router.post("/providers/{provider_id}/fetch-models")
async def fetch_models(provider_id: str, user_id: str = Depends(get_current_user_id)):
    """Query the provider to refresh available models."""
    provider = provider_repository.get_provider(provider_id, user_id)
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")

    try:
        discovered = await discover_models(
            kind=provider["kind"],
            base_url=provider.get("base_url"),
            api_key=provider.get("api_key")
        )
        provider_repository.save_models(provider_id, [m.model_dump() for m in discovered], None, user_id)
    except Exception as e:
        provider_repository.save_models(provider_id, [], str(e), user_id)
        raise HTTPException(status_code=500, detail=f"Failed to fetch models: {e}")

    return await get_providers_state(user_id)


@router.post("/test")
async def test_model(payload: TestModelRequest, user_id: str = Depends(get_current_user_id)):
    """Test model connectivity and measure response time."""
    ok, text, elapsed_ms = await llm_router.test_model(payload.ref, user_id)
    return {
        "ok": ok,
        "detail": text if ok else None,
        "error": None if ok else text,
        "ms": round(elapsed_ms, 1)
    }


@router.get("/preferences")
async def get_preferences(user_id: str = Depends(get_current_user_id)):
    return provider_repository.get_preferences(user_id)


@router.put("/preferences")
async def save_preferences(payload: PreferencesRequest, user_id: str = Depends(get_current_user_id)):
    saved = provider_repository.save_preferences(
        chat_models=payload.chat_models,
        fallback_models=payload.fallback_models,
        tts_ref=payload.tts_ref,
        tts_voice=payload.tts_voice,
        stt_ref=payload.stt_ref,
        user_id=user_id
    )
    return saved

"""
AI Providers and model preferences management router adapted from Muse.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.api.deps import get_auth
from backend.auth.owner import AuthContext
from backend.llm.discovery import discover_models
from backend.llm.router import llm_router
from backend.llm.types import KIND_LABELS, NEEDS_KEY, PROVIDER_KINDS, ProviderKind, make_model_ref
from backend.storage.provider_repository import AI_TASKS, provider_repository

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
    task_models: Optional[Dict[str, str]] = None
    tts_voice: Optional[str] = None


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
async def get_providers_state(ctx: AuthContext = Depends(get_auth)):
    """Returns the full AI providers state: configured providers, model catalog, and preferences."""
    return await _providers_state()


async def _providers_state() -> Dict[str, Any]:
    providers = await provider_repository.list_providers()
    preferences = await provider_repository.get_preferences()

    # If no providers configured yet, check if local Ollama host bridge is active
    if not providers:
        local_models = await discover_models("ollama_local")
        if local_models:
            p = await provider_repository.create_provider(kind="ollama_local", name="Local Ollama (Host)")
            await provider_repository.save_models(p["id"], [m.model_dump() for m in local_models], None)
            providers = await provider_repository.list_providers()
            if not preferences.get("chat_models"):
                first_ref = make_model_ref(p["id"], local_models[0].id)
                preferences = await provider_repository.save_preferences(chat_models=[first_ref], fallback_models=[])

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
        "kinds": kinds_info,
        "tasks": list(AI_TASKS),
    }


@router.post("/providers")
async def add_provider(payload: CreateProviderRequest, ctx: AuthContext = Depends(get_auth)):
    """Add a new AI provider, securely storing the API key."""
    name = payload.name.strip() if payload.name else KIND_LABELS.get(payload.kind, payload.kind)
    provider = await provider_repository.create_provider(
        kind=payload.kind,
        name=name,
        base_url=payload.baseUrl.strip() if payload.baseUrl else None,
        api_key=payload.apiKey.strip() if payload.apiKey else None,
    )

    # Immediately attempt model discovery
    try:
        discovered = await discover_models(
            kind=payload.kind,
            base_url=provider.get("base_url"),
            api_key=payload.apiKey.strip() if payload.apiKey else None
        )
        await provider_repository.save_models(provider["id"], [m.model_dump() for m in discovered], None)
    except Exception as e:
        await provider_repository.save_models(provider["id"], [], str(e))

    return await _providers_state()


@router.patch("/providers/{provider_id}")
async def update_provider(
    provider_id: str,
    payload: UpdateProviderRequest,
    ctx: AuthContext = Depends(get_auth)
):
    """Update provider settings or replace API key."""
    updated = await provider_repository.update_provider(
        provider_id=provider_id,
        name=payload.name,
        base_url=payload.baseUrl,
        api_key=payload.apiKey,
        enabled=payload.enabled,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Provider not found")
    return await _providers_state()


@router.delete("/providers/{provider_id}")
async def delete_provider(provider_id: str, ctx: AuthContext = Depends(get_auth)):
    """Delete a configured provider."""
    await provider_repository.delete_provider(provider_id)
    return await _providers_state()


@router.post("/providers/{provider_id}/fetch-models")
async def fetch_models(provider_id: str, ctx: AuthContext = Depends(get_auth)):
    """Query the provider to refresh available models."""
    provider = await provider_repository.get_provider(provider_id)
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")

    try:
        discovered = await discover_models(
            kind=provider["kind"],
            base_url=provider.get("base_url"),
            api_key=provider.get("api_key")
        )
        await provider_repository.save_models(provider_id, [m.model_dump() for m in discovered], None)
    except Exception as e:
        await provider_repository.save_models(provider_id, [], str(e))
        raise HTTPException(status_code=502, detail=f"Failed to fetch models: {e}") from e

    return await _providers_state()


@router.post("/test")
async def test_model(payload: TestModelRequest, ctx: AuthContext = Depends(get_auth)):
    """Test model connectivity and measure response time."""
    ok, text, elapsed_ms = await llm_router.test_model(payload.ref)
    return {
        "ok": ok,
        "detail": text if ok else None,
        "error": None if ok else text,
        "ms": round(elapsed_ms, 1)
    }


@router.get("/preferences")
async def get_preferences(ctx: AuthContext = Depends(get_auth)):
    return await provider_repository.get_preferences()


@router.put("/preferences")
async def save_preferences(payload: PreferencesRequest, ctx: AuthContext = Depends(get_auth)):
    unknown = set(payload.task_models or {}) - set(AI_TASKS)
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown task(s): {', '.join(sorted(unknown))}")
    return await provider_repository.save_preferences(
        chat_models=payload.chat_models,
        fallback_models=payload.fallback_models,
        task_models=payload.task_models,
        tts_voice=payload.tts_voice,
    )

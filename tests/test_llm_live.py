"""Live provider checks. Run with GEMINI_API_KEY and/or OLLAMA_API_KEY set; skipped otherwise."""

import os

import pytest

from backend.coaching.rubric import Scorecard
from backend.llm.discovery import discover_models
from backend.llm.router import create_provider_instance, llm_router
from backend.llm.schema import inline_schema
from backend.llm.types import LLMMessage
from backend.storage.provider_repository import provider_repository
from tests.conftest import run

GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
OLLAMA_KEY = os.environ.get("OLLAMA_API_KEY")
OLLAMA_MODEL = os.environ.get("OLLAMA_CLOUD_MODEL", "beam")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

PITCH = (
    "Um, so, we built FreightPilot. Small retailers overpay for shipping by about 30 percent "
    "because they can't negotiate rates. Our app pools their volume and books carriers automatically. "
    "We have 40 paying stores and are raising 1.5 million dollars to expand to 500."
)

live = pytest.mark.live


def _judge(kind, key, model):
    provider = create_provider_instance(kind, api_key=key)
    raw = run(provider.generate_json(
        [LLMMessage(role="user", content=f"Score this pitch:\n{PITCH}")],
        model, inline_schema(Scorecard), "Scorecard",
        system_prompt="You are a strict startup pitch judge.",
    ))
    return Scorecard.model_validate_json(raw)


def _stream(kind, key, model):
    provider = create_provider_instance(kind, api_key=key)

    async def go():
        return "".join([c async for c in provider.stream_chat(
            [LLMMessage(role="user", content="Reply with exactly: ready")], model, max_tokens=400)])
    return run(go())


@live
@pytest.mark.skipif(not GEMINI_KEY, reason="GEMINI_API_KEY not set")
def test_gemini_live():
    assert "ready" in _stream("gemini", GEMINI_KEY, GEMINI_MODEL).lower()
    card = _judge("gemini", GEMINI_KEY, GEMINI_MODEL)
    assert 0 <= card.overall <= 100 and card.checklist.ask
    models = run(discover_models("gemini", api_key=GEMINI_KEY))
    assert any(m.id == GEMINI_MODEL for m in models)
    assert not any("tts" in m.id or "image" in m.id for m in models)


@live
@pytest.mark.skipif(not OLLAMA_KEY, reason="OLLAMA_API_KEY not set")
def test_ollama_cloud_live():
    assert "ready" in _stream("ollama_cloud", OLLAMA_KEY, OLLAMA_MODEL).lower()
    card = _judge("ollama_cloud", OLLAMA_KEY, OLLAMA_MODEL)
    assert 0 <= card.overall <= 100 and card.checklist.ask
    assert any(m.id == OLLAMA_MODEL for m in run(discover_models("ollama_cloud", api_key=OLLAMA_KEY)))


@live
@pytest.mark.skipif(not (GEMINI_KEY and OLLAMA_KEY), reason="needs both keys")
def test_router_falls_back_from_broken_gemini_to_ollama_live():
    bad = run(provider_repository.create_provider("gemini", "Gemini (bad key)", api_key="invalid-key"))
    good = run(provider_repository.create_provider("ollama_cloud", "Ollama Cloud", api_key=OLLAMA_KEY))
    run(provider_repository.save_preferences(
        chat_models=[f"{bad['id']}:{GEMINI_MODEL}"], fallback_models=[f"{good['id']}:{OLLAMA_MODEL}"]))
    card, used = run(llm_router.generate_structured(
        [LLMMessage(role="user", content=f"Score this pitch:\n{PITCH}")], Scorecard))
    assert used.startswith(good["id"]) and card.overall > 0

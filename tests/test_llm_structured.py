"""Structured output: each provider sends its native schema mechanism; the router validates and repairs."""

import json
from typing import List, Optional

import httpx
import pytest
from pydantic import BaseModel, Field

from backend.coaching.rubric import Scorecard
from backend.llm import router as router_mod
from backend.llm.providers import anthropic, gemini, ollama, openai_compatible
from backend.llm.schema import extract_json_text, inline_schema
from backend.llm.types import LLMMessage
from backend.storage.provider_repository import provider_repository
from tests.conftest import run


class Item(BaseModel):
    name: str
    score: float = Field(ge=0, le=10)


class Result(BaseModel):
    items: List[Item]
    note: Optional[str] = None


GOOD = {"items": [{"name": "hook", "score": 8}], "note": None}


def mock_http(monkeypatch, module, handler):
    """Route every httpx.AsyncClient created in `module` through `handler`; returns captured requests."""
    seen = []
    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        def wrapped(request: httpx.Request):
            seen.append(request)
            return handler(request)
        kwargs["transport"] = httpx.MockTransport(wrapped)
        return real(*args, **kwargs)

    monkeypatch.setattr(module.httpx, "AsyncClient", factory)
    return seen


def test_inline_schema_resolves_refs_and_nullables():
    schema = inline_schema(Result)
    assert "$defs" not in json.dumps(schema) and "$ref" not in json.dumps(schema)
    item = schema["properties"]["items"]["items"]
    assert item["properties"]["score"]["type"] == "number"
    assert schema["properties"]["note"] == {"type": "string", "nullable": True}
    assert "title" not in schema


def test_inline_schema_keeps_property_named_title():
    card = inline_schema(Scorecard)
    assert "clarity" in card["properties"]

    class HasTitle(BaseModel):
        title: str
    assert "title" in inline_schema(HasTitle)["properties"]


def test_extract_json_text():
    assert json.loads(extract_json_text('Sure!\n```json\n{"a": 1}\n```')) == {"a": 1}
    assert json.loads(extract_json_text('Here: {"a": {"b": 2}} done')) == {"a": {"b": 2}}


def test_gemini_uses_response_schema_and_header_key(monkeypatch):
    def handler(req):
        body = json.loads(req.content)
        cfg = body["generationConfig"]
        assert cfg["responseMimeType"] == "application/json"
        assert cfg["responseSchema"]["properties"]["items"]["type"] == "array"
        assert "additionalProperties" not in json.dumps(cfg["responseSchema"])
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [
            {"text": "thinking...", "thought": True}, {"text": json.dumps(GOOD)}]}}]})

    seen = mock_http(monkeypatch, gemini, handler)
    raw = run(gemini.GeminiProvider(api_key="g-key").generate_json(
        [LLMMessage(role="user", content="x")], "gemini-2.5-flash", inline_schema(Result), "Result"))
    assert json.loads(raw) == GOOD
    assert seen[0].headers["x-goog-api-key"] == "g-key"
    assert "key=" not in str(seen[0].url)


def test_ollama_uses_format_schema(monkeypatch):
    def handler(req):
        body = json.loads(req.content)
        assert body["format"]["type"] == "object" and body["stream"] is False
        return httpx.Response(200, json={"message": {"content": json.dumps(GOOD)}})

    seen = mock_http(monkeypatch, ollama, handler)
    raw = run(ollama.OllamaProvider("https://ollama.com", api_key="o-key").generate_json(
        [LLMMessage(role="user", content="x")], "gpt-oss:20b", inline_schema(Result), "Result"))
    assert json.loads(raw) == GOOD
    assert seen[0].headers["authorization"] == "Bearer o-key"


def test_ollama_retries_without_think_flag(monkeypatch):
    def handler(req):
        if "think" in json.loads(req.content):
            return httpx.Response(400, text='{"error":"think value \\"false\\" is not supported for this model"}')
        return httpx.Response(200, json={"message": {"content": json.dumps(GOOD)}})

    seen = mock_http(monkeypatch, ollama, handler)
    raw = run(ollama.OllamaProvider().generate_json(
        [LLMMessage(role="user", content="x")], "m", inline_schema(Result), "Result"))
    assert json.loads(raw) == GOOD and len(seen) == 2


def test_openai_json_schema_then_json_mode_fallback(monkeypatch):
    def handler(req):
        body = json.loads(req.content)
        if body["response_format"]["type"] == "json_schema":
            assert body["response_format"]["json_schema"]["name"] == "Result"
            return httpx.Response(400, json={"error": "json_schema not supported"})
        assert "JSON Schema" in body["messages"][0]["content"]
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(GOOD)}}]})

    seen = mock_http(monkeypatch, openai_compatible, handler)
    raw = run(openai_compatible.OpenAICompatibleProvider("https://api.groq.com/openai/v1", "k").generate_json(
        [LLMMessage(role="user", content="x")], "llama", inline_schema(Result), "Result"))
    assert json.loads(raw) == GOOD and len(seen) == 2


def test_anthropic_forces_tool_use(monkeypatch):
    def handler(req):
        body = json.loads(req.content)
        assert body["tool_choice"] == {"type": "tool", "name": "Result"}
        assert body["tools"][0]["input_schema"]["type"] == "object"
        return httpx.Response(200, json={"content": [{"type": "tool_use", "name": "Result", "input": GOOD}]})

    mock_http(monkeypatch, anthropic, handler)
    raw = run(anthropic.AnthropicProvider("a-key").generate_json(
        [LLMMessage(role="user", content="x")], "claude-sonnet-5-5", inline_schema(Result), "Result"))
    assert json.loads(raw) == GOOD


class FakeProvider:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def generate_json(self, messages, model, **kw):
        self.calls.append((model, [m.content for m in messages]))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _install(monkeypatch, providers):
    async def resolve(self, ref):
        return providers[ref], ref.split(":", 1)[1]
    monkeypatch.setattr(router_mod.LLMRouter, "resolve_provider_and_model", resolve)


def test_router_repairs_invalid_json(monkeypatch):
    fake = FakeProvider(['{"items": [{"name": "hook", "score": 42}]}', json.dumps(GOOD)])
    _install(monkeypatch, {"p:m": fake})
    run(provider_repository.save_preferences(chat_models=["p:m"], fallback_models=[]))
    result, used = run(router_mod.llm_router.generate_structured([LLMMessage(role="user", content="go")], Result))
    assert result.items[0].score == 8 and used == "p:m"
    assert "does not match the required schema" in fake.calls[1][1][-1]


def test_router_falls_back_and_prefers_task_model(monkeypatch):
    broken = FakeProvider([RuntimeError("503"), RuntimeError("503")])
    good = FakeProvider([json.dumps(GOOD)])
    _install(monkeypatch, {"a:judge": broken, "b:chat": good})
    run(provider_repository.save_preferences(chat_models=["b:chat"], fallback_models=[], task_models={"judge": "a:judge"}))
    _, used = run(router_mod.llm_router.generate_structured([LLMMessage(role="user", content="go")], Result, task="judge"))
    assert used == "b:chat"
    assert broken.calls[0][0] == "judge"  # the task model was tried first


def test_router_candidate_order():
    run(provider_repository.save_preferences(
        chat_models=["a:1", "b:2"], fallback_models=["b:2", "c:3"], task_models={"deck_gen": "d:4"}))
    assert run(router_mod.llm_router.candidates("deck_gen")) == ["d:4", "a:1", "b:2", "c:3"]
    assert run(router_mod.llm_router.candidates("chat", "x:9")) == ["x:9", "a:1", "b:2", "c:3"]


def test_stream_does_not_switch_models_mid_answer(monkeypatch):
    class Half:
        async def stream_chat(self, **kw):
            yield "partial "
            raise RuntimeError("connection reset")

    async def resolve(self, ref):
        return Half(), "m"
    monkeypatch.setattr(router_mod.LLMRouter, "resolve_provider_and_model", resolve)
    run(provider_repository.save_preferences(chat_models=["a:1", "b:2"], fallback_models=[]))

    async def consume():
        out = []
        async for chunk in router_mod.llm_router.stream_with_fallbacks([LLMMessage(role="user", content="hi")]):
            out.append(chunk)
        return out
    with pytest.raises(RuntimeError, match="connection reset"):
        run(consume())


def test_overloaded_model_cools_down(monkeypatch):
    from backend.llm.providers.base import ProviderHTTPError
    busy = FakeProvider([ProviderHTTPError("Gemini", 503, "high demand")])
    good = FakeProvider([json.dumps(GOOD), json.dumps(GOOD)])
    _install(monkeypatch, {"a:busy": busy, "b:ok": good})
    router_mod.llm_router._cooldown_until.clear()
    run(provider_repository.save_preferences(chat_models=["a:busy"], fallback_models=["b:ok"]))
    msgs = [LLMMessage(role="user", content="go")]
    assert run(router_mod.llm_router.generate_structured(msgs, Result))[1] == "b:ok"
    # Second call skips straight to the healthy model.
    assert run(router_mod.llm_router.candidates()) == ["b:ok", "a:busy"]
    assert run(router_mod.llm_router.generate_structured(msgs, Result))[1] == "b:ok"
    assert len(busy.calls) == 1
    router_mod.llm_router._cooldown_until.clear()

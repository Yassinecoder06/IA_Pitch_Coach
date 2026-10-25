"""
Google Gemini LLM provider adapter.
Direct streaming HTTP integration with Gemini REST API.
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx

from backend.llm.providers.base import JSON_TIMEOUT, PING_TIMEOUT, STREAM_TIMEOUT, BaseLLMProvider, ProviderHTTPError
from backend.llm.types import LLMMessage

# OpenAPI-subset keys accepted by Gemini's responseSchema.
_GEMINI_SCHEMA_KEYS = {
    "type", "format", "description", "nullable", "enum", "properties", "required",
    "items", "minItems", "maxItems", "minimum", "maximum", "anyOf", "propertyOrdering",
}


def to_gemini_schema(node: Any) -> Any:
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k not in _GEMINI_SCHEMA_KEYS:
                continue
            if k == "properties":
                out[k] = {name: to_gemini_schema(prop) for name, prop in v.items()}
            else:
                out[k] = to_gemini_schema(v)
        if out.get("type") == "object" and "properties" in out and "propertyOrdering" not in out:
            out["propertyOrdering"] = list(out["properties"].keys())
        return out
    if isinstance(node, list):
        return [to_gemini_schema(n) for n in node]
    return node


class GeminiProvider(BaseLLMProvider):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    def _headers(self) -> Dict[str, str]:
        # Header instead of ?key= so the key never lands in URLs or access logs.
        return {"x-goog-api-key": self.api_key or "", "Content-Type": "application/json"}

    @staticmethod
    def _contents(messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        return [
            {"role": "user" if m.role == "user" else "model", "parts": [{"text": m.content}]}
            for m in messages if m.role != "system"
        ]

    async def generate_json(
        self,
        messages: List[LLMMessage],
        model: str,
        schema: Dict[str, Any],
        schema_name: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4000
    ) -> str:
        if not self.api_key:
            raise ValueError("Google Gemini requires an API Key.")
        gemini_model = model if model.startswith("models/") else f"models/{model}"
        body: Dict[str, Any] = {
            "contents": self._contents(messages),
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
                "responseSchema": to_gemini_schema(schema),
            },
        }
        if system_prompt:
            body["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        async with httpx.AsyncClient(timeout=JSON_TIMEOUT) as client:
            res = await client.post(f"{self.base_url}/{gemini_model}:generateContent", headers=self._headers(), json=body)
        if res.status_code != 200:
            raise ProviderHTTPError("Gemini", res.status_code, res.text)
        candidates = res.json().get("candidates") or []
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        # Thinking models may return thought parts first; keep only the answer text.
        return "".join(p.get("text", "") for p in parts if not p.get("thought"))

    async def stream_chat(
        self,
        messages: List[LLMMessage],
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1500
    ) -> AsyncGenerator[str, None]:
        if not self.api_key:
            raise ValueError("Google Gemini requires an API Key.")

        # Format model name
        gemini_model = model if model.startswith("models/") else f"models/{model}"

        body: Dict = {
            "contents": self._contents(messages),
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }
        if system_prompt:
            body["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        url = f"{self.base_url}/{gemini_model}:streamGenerateContent?alt=sse"

        async with httpx.AsyncClient(timeout=STREAM_TIMEOUT) as client:
            async with client.stream("POST", url, json=body, headers=self._headers()) as response:
                if response.status_code != 200:
                    err_text = await response.aread()
                    raise ProviderHTTPError("Gemini", response.status_code, err_text.decode("utf-8", errors="ignore"))

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    try:
                        data = json.loads(data_str)
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            for p in parts:
                                text = p.get("text", "")
                                if text and not p.get("thought"):
                                    yield text
                    except Exception:
                        continue

    async def ping_test(self, model: str) -> Tuple[bool, str, float]:
        start = time.perf_counter()
        if not self.api_key:
            return False, "Missing API Key", 0.0

        gemini_model = model if model.startswith("models/") else f"models/{model}"
        url = f"{self.base_url}/{gemini_model}:generateContent"
        body = {
            "contents": [{"role": "user", "parts": [{"text": "Reply with 'OK' only."}]}],
            "generationConfig": {"maxOutputTokens": 256}
        }
        try:
            async with httpx.AsyncClient(timeout=PING_TIMEOUT) as client:
                res = await client.post(url, json=body, headers=self._headers())
                elapsed_ms = (time.perf_counter() - start) * 1000
                if res.status_code == 200:
                    data = res.json()
                    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                    text = parts[0].get("text", "").strip() if parts else ""
                    return True, f"Answered: '{text}'", elapsed_ms
                else:
                    return False, f"HTTP {res.status_code}: {res.text[:120]}", elapsed_ms
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return False, str(e), elapsed_ms

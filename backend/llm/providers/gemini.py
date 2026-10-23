"""
Google Gemini LLM provider adapter.
Direct streaming HTTP integration with Gemini REST API.
"""

from __future__ import annotations

import json
import time
import httpx
from typing import AsyncGenerator, Dict, List, Optional, Tuple
from backend.llm.types import LLMMessage
from backend.llm.providers.base import BaseLLMProvider


class GeminiProvider(BaseLLMProvider):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

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

        contents = []
        for m in messages:
            role = "user" if m.role == "user" else "model"
            contents.append({"role": role, "parts": [{"text": m.content}]})

        body: Dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }
        if system_prompt:
            body["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        url = f"{self.base_url}/{gemini_model}:streamGenerateContent?alt=sse&key={self.api_key}"

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, json=body) as response:
                if response.status_code != 200:
                    err_text = await response.aread()
                    raise RuntimeError(f"Gemini API error ({response.status_code}): {err_text.decode('utf-8', errors='ignore')}")

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
                                if text:
                                    yield text
                    except Exception:
                        continue

    async def ping_test(self, model: str) -> Tuple[bool, str, float]:
        start = time.perf_counter()
        if not self.api_key:
            return False, "Missing API Key", 0.0

        gemini_model = model if model.startswith("models/") else f"models/{model}"
        url = f"{self.base_url}/{gemini_model}:generateContent?key={self.api_key}"
        body = {
            "contents": [{"role": "user", "parts": [{"text": "Reply with 'OK' only."}]}],
            "generationConfig": {"maxOutputTokens": 10}
        }
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                res = await client.post(url, json=body)
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

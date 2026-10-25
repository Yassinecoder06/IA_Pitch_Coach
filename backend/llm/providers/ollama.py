"""
Ollama Local / Cloud LLM provider adapter.
Direct streaming HTTP integration with Ollama /api/chat.
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx

from backend.llm.providers.base import JSON_TIMEOUT, PING_TIMEOUT, STREAM_TIMEOUT, BaseLLMProvider, ProviderHTTPError
from backend.llm.schema import schema_instruction
from backend.llm.types import LLMMessage


class OllamaProvider(BaseLLMProvider):
    def __init__(self, base_url: str = "http://localhost:11434", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _messages(messages: List[LLMMessage], system_prompt: Optional[str]) -> List[Dict[str, str]]:
        out = [{"role": "system", "content": system_prompt}] if system_prompt else []
        return out + [{"role": m.role, "content": m.content} for m in messages]

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
        # Local Ollama enforces `format` during decoding, but Ollama Cloud ignores it,
        # so the schema is also spelled out in the system prompt.
        system = ((system_prompt or "") + "\n\n" + schema_instruction(schema)).strip()
        body = {
            "model": model,
            "messages": self._messages(messages, system),
            "stream": False,
            "format": schema,
            "think": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        async with httpx.AsyncClient(timeout=JSON_TIMEOUT) as client:
            res = await client.post(f"{self.base_url}/api/chat", json=body, headers=self._headers())
        if res.status_code == 400 and "think" in res.text:
            body.pop("think")  # models without a thinking toggle reject the field
            async with httpx.AsyncClient(timeout=JSON_TIMEOUT) as client:
                res = await client.post(f"{self.base_url}/api/chat", json=body, headers=self._headers())
        if res.status_code != 200:
            raise ProviderHTTPError("Ollama", res.status_code, res.text)
        return res.json().get("message", {}).get("content", "")

    async def stream_chat(
        self,
        messages: List[LLMMessage],
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1500
    ) -> AsyncGenerator[str, None]:
        body = {
            "model": model,
            "messages": self._messages(messages, system_prompt),
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        async with httpx.AsyncClient(timeout=STREAM_TIMEOUT) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=body,
                headers=self._headers()
            ) as response:
                if response.status_code != 200:
                    err_text = await response.aread()
                    raise ProviderHTTPError("Ollama", response.status_code, err_text.decode("utf-8", errors="ignore"))

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        msg = data.get("message", {})
                        content = msg.get("content", "")
                        if content:
                            yield content
                    except Exception:
                        continue

    async def ping_test(self, model: str) -> Tuple[bool, str, float]:
        start = time.perf_counter()
        body = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with 'OK' only."}],
            "stream": False,
            "options": {"num_predict": 256}
        }
        try:
            async with httpx.AsyncClient(timeout=PING_TIMEOUT) as client:
                res = await client.post(f"{self.base_url}/api/chat", json=body, headers=self._headers())
                elapsed_ms = (time.perf_counter() - start) * 1000
                if res.status_code == 200:
                    data = res.json()
                    content = data.get("message", {}).get("content", "").strip()
                    return True, f"Answered: '{content}'", elapsed_ms
                else:
                    return False, f"HTTP {res.status_code}: {res.text[:120]}", elapsed_ms
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return False, str(e), elapsed_ms

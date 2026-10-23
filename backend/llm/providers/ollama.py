"""
Ollama Local / Cloud LLM provider adapter.
Direct streaming HTTP integration with Ollama /api/chat.
"""

from __future__ import annotations

import json
import time
import httpx
from typing import AsyncGenerator, Dict, List, Optional, Tuple
from backend.llm.types import LLMMessage
from backend.llm.providers.base import BaseLLMProvider


class OllamaProvider(BaseLLMProvider):
    def __init__(self, base_url: str = "http://localhost:11434", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    async def stream_chat(
        self,
        messages: List[LLMMessage],
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1500
    ) -> AsyncGenerator[str, None]:
        payload_messages = []
        if system_prompt:
            payload_messages.append({"role": "system", "content": system_prompt})
        for m in messages:
            payload_messages.append({"role": m.role, "content": m.content})

        body = {
            "model": model,
            "messages": payload_messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with httpx.AsyncClient(timeout=90.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=body,
                headers=headers
            ) as response:
                if response.status_code != 200:
                    err_text = await response.aread()
                    raise RuntimeError(f"Ollama API error ({response.status_code}): {err_text.decode('utf-8', errors='ignore')}")

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
            "options": {"num_predict": 10}
        }
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                res = await client.post(f"{self.base_url}/api/chat", json=body, headers=headers)
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

"""
Anthropic Claude LLM provider adapter.
Direct streaming HTTP integration with Anthropic Messages API.
"""

from __future__ import annotations

import json
import time
import httpx
from typing import AsyncGenerator, Dict, List, Optional, Tuple
from backend.llm.types import LLMMessage
from backend.llm.providers.base import BaseLLMProvider


class AnthropicProvider(BaseLLMProvider):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1"

    def _get_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    async def stream_chat(
        self,
        messages: List[LLMMessage],
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1500
    ) -> AsyncGenerator[str, None]:
        if not self.api_key:
            raise ValueError("Anthropic requires an API Key.")

        anthropic_messages = []
        for m in messages:
            if m.role in ("user", "assistant"):
                anthropic_messages.append({"role": m.role, "content": m.content})

        body: Dict = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True
        }
        if system_prompt:
            body["system"] = system_prompt

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/messages",
                headers=self._get_headers(),
                json=body
            ) as response:
                if response.status_code != 200:
                    err_text = await response.aread()
                    raise RuntimeError(f"Anthropic API error ({response.status_code}): {err_text.decode('utf-8', errors='ignore')}")

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    try:
                        data = json.loads(data_str)
                        ev_type = data.get("type")
                        if ev_type == "content_block_delta":
                            delta = data.get("delta", {})
                            if delta.get("type") == "text_delta":
                                text = delta.get("text", "")
                                if text:
                                    yield text
                    except Exception:
                        continue

    async def ping_test(self, model: str) -> Tuple[bool, str, float]:
        start = time.perf_counter()
        if not self.api_key:
            return False, "Missing API Key", 0.0

        body = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with 'OK' only."}],
            "max_tokens": 10
        }
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                res = await client.post(
                    f"{self.base_url}/messages",
                    headers=self._get_headers(),
                    json=body
                )
                elapsed_ms = (time.perf_counter() - start) * 1000
                if res.status_code == 200:
                    data = res.json()
                    parts = data.get("content", [])
                    text = parts[0].get("text", "").strip() if parts else ""
                    return True, f"Answered: '{text}'", elapsed_ms
                else:
                    return False, f"HTTP {res.status_code}: {res.text[:120]}", elapsed_ms
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return False, str(e), elapsed_ms

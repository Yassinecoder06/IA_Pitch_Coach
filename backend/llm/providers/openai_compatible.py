"""
OpenAI-compatible LLM provider adapter.
Supports OpenAI, Groq, DeepSeek, Mistral, OpenRouter, LM Studio, etc.
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx

from backend.llm.providers.base import JSON_TIMEOUT, PING_TIMEOUT, STREAM_TIMEOUT, BaseLLMProvider, ProviderHTTPError
from backend.llm.schema import schema_instruction
from backend.llm.types import LLMMessage


class OpenAICompatibleProvider(BaseLLMProvider):
    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
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
        body: Dict[str, Any] = {
            "model": model,
            "messages": self._messages(messages, system_prompt),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": False},
            },
        }
        async with httpx.AsyncClient(timeout=JSON_TIMEOUT) as client:
            res = await client.post(f"{self.base_url}/chat/completions", headers=self._get_headers(), json=body)
            if res.status_code in (400, 422):
                # Many OpenAI-compatible servers only support JSON mode: put the schema in the prompt.
                body["response_format"] = {"type": "json_object"}
                body["messages"] = self._messages(
                    messages, ((system_prompt or "") + "\n\n" + schema_instruction(schema)).strip()
                )
                res = await client.post(f"{self.base_url}/chat/completions", headers=self._get_headers(), json=body)
        if res.status_code != 200:
            raise ProviderHTTPError("OpenAI-compatible", res.status_code, res.text)
        choices = res.json().get("choices") or [{}]
        return choices[0].get("message", {}).get("content") or ""

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
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }

        async with httpx.AsyncClient(timeout=STREAM_TIMEOUT) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._get_headers(),
                json=body
            ) as response:
                if response.status_code != 200:
                    err_text = await response.aread()
                    raise ProviderHTTPError("OpenAI-compatible", response.status_code, err_text.decode("utf-8", errors="ignore"))

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                    except Exception:
                        continue

    async def ping_test(self, model: str) -> Tuple[bool, str, float]:
        start = time.perf_counter()
        body = {
            "model": model,
            "messages": [{"role": "user", "content": "Respond with 'OK' only."}],
            "max_tokens": 10
        }
        try:
            async with httpx.AsyncClient(timeout=PING_TIMEOUT) as client:
                res = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._get_headers(),
                    json=body
                )
                elapsed_ms = (time.perf_counter() - start) * 1000
                if res.status_code == 200:
                    data = res.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                    return True, f"Answered: '{content}'", elapsed_ms
                else:
                    return False, f"HTTP {res.status_code}: {res.text[:120]}", elapsed_ms
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return False, str(e), elapsed_ms

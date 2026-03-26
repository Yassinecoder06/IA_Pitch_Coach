"""
OpenAI Provider
===============
LLM provider for OpenAI API (GPT models).
Also serves as base for OpenAI-compatible APIs.
"""

import json
from typing import AsyncGenerator, Dict, List, Optional, Any
import httpx

from .provider_interface import (
    LLMProvider,
    Message,
    ProviderResponse,
    GenerationConfig
)


class OpenAIProvider(LLMProvider):
    """
    Provider for OpenAI API.
    Supports GPT-4, GPT-3.5 and other OpenAI models.
    """

    DEFAULT_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        super().__init__(endpoint, api_key)
        self.endpoint = endpoint or "https://api.openai.com/v1"
        self._name = "OpenAI"

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    async def generate(
        self,
        messages: List[Message],
        model: str,
        config: Optional[GenerationConfig] = None
    ) -> ProviderResponse:
        """Generate a complete response."""
        cfg = self._get_config(config)
        payload = {
            "model": model,
            "messages": self._prepare_messages(messages),
            **cfg.to_dict(),
            "stream": False
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.endpoint}/chat/completions",
                headers=self._get_headers(),
                json=payload,
                timeout=120.0
            )
            response.raise_for_status()
            data = response.json()

            return ProviderResponse(
                content=data["choices"][0]["message"]["content"],
                model=data.get("model", model),
                provider=self.name,
                finish_reason=data["choices"][0].get("finish_reason"),
                usage=data.get("usage"),
                raw_response=data
            )

    async def stream(
        self,
        messages: List[Message],
        model: str,
        config: Optional[GenerationConfig] = None
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens."""
        cfg = self._get_config(config)
        payload = {
            "model": model,
            "messages": self._prepare_messages(messages),
            **cfg.to_dict(),
            "stream": True
        }

        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{self.endpoint}/chat/completions",
                    headers=self._get_headers(),
                    json=payload,
                    timeout=120.0
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                continue

        except httpx.ConnectError:
            yield f"Error: Cannot connect to OpenAI API at {self.endpoint}"
        except httpx.HTTPStatusError as e:
            error_text = ""
            try:
                error_text = (await e.response.aread()).decode("utf-8", errors="replace")
            except Exception:
                error_text = "Unable to read error response body"
            yield f"Error: {e.response.status_code} - {error_text}"
        except Exception as e:
            yield f"Error: {str(e)}"

    async def check_availability(self) -> Dict[str, Any]:
        """Check if OpenAI API is available."""
        if not self.api_key:
            return {
                "available": False,
                "message": "OpenAI API key not configured"
            }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.endpoint}/models",
                    headers=self._get_headers(),
                    timeout=10.0
                )

                if response.status_code == 200:
                    data = response.json()
                    models = [m["id"] for m in data.get("data", [])]
                    return {
                        "available": True,
                        "message": "OpenAI API is available",
                        "models": models
                    }
                else:
                    return {
                        "available": False,
                        "message": f"API returned status {response.status_code}"
                    }

        except Exception as e:
            return {
                "available": False,
                "message": str(e)
            }

    async def list_models(self) -> List[str]:
        """List available models."""
        try:
            status = await self.check_availability()
            if status.get("available"):
                return status.get("models", self.DEFAULT_MODELS)
        except Exception:
            pass
        return self.DEFAULT_MODELS

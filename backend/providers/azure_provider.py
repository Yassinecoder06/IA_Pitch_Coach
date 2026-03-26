"""
Azure OpenAI Provider
=====================
LLM provider for Azure OpenAI Service.
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


class AzureOpenAIProvider(LLMProvider):
    """
    Provider for Azure OpenAI Service.
    Supports deployed GPT models on Azure.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: str = "2025-01-01-preview"
    ):
        super().__init__(endpoint, api_key)
        self.api_version = api_version
        self._name = "Azure"

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        return {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }

    def _get_url(self, deployment: str, endpoint_type: str = "chat/completions") -> str:
        """Get the full URL for an API endpoint."""
        return f"{self.endpoint}/openai/deployments/{deployment}/{endpoint_type}?api-version={self.api_version}"

    async def generate(
        self,
        messages: List[Message],
        model: str,  # In Azure, this is the deployment name
        config: Optional[GenerationConfig] = None
    ) -> ProviderResponse:
        """Generate a complete response."""
        cfg = self._get_config(config)
        payload = {
            "messages": self._prepare_messages(messages),
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "top_p": cfg.top_p,
            "stream": False
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._get_url(model),
                headers=self._get_headers(),
                json=payload,
                timeout=120.0
            )
            response.raise_for_status()
            data = response.json()

            return ProviderResponse(
                content=data["choices"][0]["message"]["content"],
                model=model,
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
            "messages": self._prepare_messages(messages),
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "top_p": cfg.top_p,
            "stream": True
        }

        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    self._get_url(model),
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
            yield f"Error: Cannot connect to Azure OpenAI at {self.endpoint}"
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
        """Check if Azure OpenAI is available."""
        if not self.api_key or not self.endpoint:
            return {
                "available": False,
                "message": "Azure OpenAI endpoint or API key not configured"
            }

        try:
            # Azure doesn't have a standard models endpoint, so we check deployments
            url = f"{self.endpoint}/openai/deployments?api-version={self.api_version}"
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers=self._get_headers(),
                    timeout=10.0
                )

                if response.status_code == 200:
                    data = response.json()
                    deployments = [d.get("id", "") for d in data.get("data", [])]
                    return {
                        "available": True,
                        "message": "Azure OpenAI is available",
                        "models": deployments
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
        """List available deployments."""
        try:
            status = await self.check_availability()
            if status.get("available"):
                return status.get("models", [])
        except Exception:
            pass
        return []

"""
Anthropic Provider
==================
LLM provider for Anthropic API (Claude models).
"""

import json
from typing import AsyncGenerator, Dict, List, Optional, Any
import httpx

from .provider_interface import (
    LLMProvider,
    Message,
    MessageRole,
    ProviderResponse,
    GenerationConfig
)


class AnthropicProvider(LLMProvider):
    """
    Provider for Anthropic Claude API.
    Supports Claude 3.5, Claude 3, and newer models.
    """

    DEFAULT_MODELS = [
        "claude-sonnet-4-20250514",
        "claude-3-5-sonnet-20241022",
        "claude-3-haiku-20240307",
        "claude-3-opus-20240229"
    ]
    API_VERSION = "2023-06-01"

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        super().__init__(endpoint, api_key)
        self.endpoint = endpoint or "https://api.anthropic.com"
        self._name = "Anthropic"

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.API_VERSION,
            "Content-Type": "application/json"
        }

    def _prepare_anthropic_messages(
        self,
        messages: List[Message]
    ) -> tuple[Optional[str], List[Dict]]:
        """
        Prepare messages for Anthropic API format.
        Anthropic uses a separate system parameter.
        """
        system_prompt = None
        chat_messages = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_prompt = msg.content
            else:
                chat_messages.append({
                    "role": msg.role.value,
                    "content": msg.content
                })

        return system_prompt, chat_messages

    async def generate(
        self,
        messages: List[Message],
        model: str,
        config: Optional[GenerationConfig] = None
    ) -> ProviderResponse:
        """Generate a complete response."""
        cfg = self._get_config(config)
        system_prompt, chat_messages = self._prepare_anthropic_messages(messages)

        payload = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
        }

        if system_prompt:
            payload["system"] = system_prompt

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.endpoint}/v1/messages",
                headers=self._get_headers(),
                json=payload,
                timeout=120.0
            )
            response.raise_for_status()
            data = response.json()

            content = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    content += block.get("text", "")

            return ProviderResponse(
                content=content,
                model=data.get("model", model),
                provider=self.name,
                finish_reason=data.get("stop_reason"),
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
        system_prompt, chat_messages = self._prepare_anthropic_messages(messages)

        payload = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "stream": True
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{self.endpoint}/v1/messages",
                    headers=self._get_headers(),
                    json=payload,
                    timeout=120.0
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            try:
                                data = json.loads(data_str)
                                event_type = data.get("type")

                                if event_type == "content_block_delta":
                                    delta = data.get("delta", {})
                                    if delta.get("type") == "text_delta":
                                        text = delta.get("text", "")
                                        if text:
                                            yield text

                                elif event_type == "message_stop":
                                    break

                            except json.JSONDecodeError:
                                continue

        except httpx.ConnectError:
            yield f"Error: Cannot connect to Anthropic API"
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
        """Check if Anthropic API is available."""
        if not self.api_key:
            return {
                "available": False,
                "message": "Anthropic API key not configured"
            }

        # Anthropic doesn't have a models endpoint, so we just verify the key works
        try:
            async with httpx.AsyncClient() as client:
                # Make a minimal request to verify API key
                response = await client.post(
                    f"{self.endpoint}/v1/messages",
                    headers=self._get_headers(),
                    json={
                        "model": "claude-3-haiku-20240307",
                        "max_tokens": 1,
                        "messages": [{"role": "user", "content": "Hi"}]
                    },
                    timeout=10.0
                )

                # A valid response (even if cut off) means the key works
                if response.status_code in (200, 400):  # 400 might be rate limit
                    return {
                        "available": True,
                        "message": "Anthropic API is available",
                        "models": self.DEFAULT_MODELS
                    }
                elif response.status_code == 401:
                    return {
                        "available": False,
                        "message": "Invalid API key"
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
        return self.DEFAULT_MODELS

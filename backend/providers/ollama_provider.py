"""
Ollama Provider
===============
LLM provider for Ollama (local models).
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


class OllamaProvider(LLMProvider):
    """
    Provider for Ollama local LLM server.
    Supports any model available in Ollama.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None  # Not used for Ollama
    ):
        super().__init__(endpoint, api_key)
        self.endpoint = endpoint or "http://localhost:11434"
        self._name = "Ollama"
        self._cached_models: List[str] = []

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
            "stream": False,
            "options": {
                "temperature": cfg.temperature,
                "num_predict": cfg.max_tokens,
                "top_p": cfg.top_p,
                "num_ctx": 2048  # Context window
            }
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.endpoint}/api/chat",
                json=payload,
                timeout=120.0
            )
            response.raise_for_status()
            data = response.json()

            return ProviderResponse(
                content=data.get("message", {}).get("content", ""),
                model=data.get("model", model),
                provider=self.name,
                finish_reason="stop" if data.get("done") else None,
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0)
                },
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
            "stream": True,
            "options": {
                "temperature": cfg.temperature,
                "num_predict": cfg.max_tokens,
                "top_p": cfg.top_p,
                "num_ctx": 2048
            }
        }

        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{self.endpoint}/api/chat",
                    json=payload,
                    timeout=120.0
                ) as response:
                    if response.status_code == 404:
                        yield f"Model '{model}' not found. Run: ollama pull {model}"
                        return

                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                if "message" in data and "content" in data["message"]:
                                    content = data["message"]["content"]
                                    if content:
                                        yield content

                                if data.get("done", False):
                                    break

                            except json.JSONDecodeError:
                                continue

        except httpx.ConnectError:
            yield "Error: Cannot connect to Ollama. Ensure 'ollama serve' is running."
        except httpx.TimeoutException:
            yield "Error: Request timed out. The model may be overloaded."
        except Exception as e:
            yield f"Error: {str(e)}"

    async def check_availability(self) -> Dict[str, Any]:
        """Check if Ollama is available."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.endpoint}/api/tags",
                    timeout=5.0
                )

                if response.status_code == 200:
                    data = response.json()
                    models = [m.get("name", "") for m in data.get("models", [])]
                    self._cached_models = models

                    return {
                        "available": True,
                        "message": "Ollama is running",
                        "models": models
                    }
                else:
                    return {
                        "available": False,
                        "message": f"Unexpected response: {response.status_code}"
                    }

        except httpx.ConnectError:
            return {
                "available": False,
                "message": "Cannot connect to Ollama. Run 'ollama serve' first."
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
                return status.get("models", [])
        except Exception:
            pass
        return self._cached_models

    async def pull_model(self, model: str) -> AsyncGenerator[Dict, None]:
        """Pull a model from Ollama registry."""
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{self.endpoint}/api/pull",
                    json={"name": model},
                    timeout=None  # Pulling can take a long time
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if line:
                            try:
                                yield json.loads(line)
                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            yield {"error": str(e)}

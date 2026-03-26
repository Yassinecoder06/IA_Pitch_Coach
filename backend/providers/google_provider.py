"""
Google Provider
===============
LLM provider for Google Gemini API.
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


class GoogleProvider(LLMProvider):
    """
    Provider for Google Gemini API.
    Supports Gemini 1.5 and 2.0 models.
    """

    DEFAULT_MODELS = [
        "gemini-3.1-flash-preview",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash-8b"
    ]

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        super().__init__(endpoint, api_key)
        self.endpoint = "https://generativelanguage.googleapis.com/v1beta"
        self._name = "Google"

    def _prepare_gemini_messages(self, messages: List[Message]) -> tuple[Optional[str], List[Dict]]:
        """Convert messages to Gemini format."""
        system_instruction = None
        contents = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_instruction = msg.content
            else:
                role = "user" if msg.role == MessageRole.USER else "model"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg.content}]
                })

        return system_instruction, contents

    async def generate(
        self,
        messages: List[Message],
        model: str,
        config: Optional[GenerationConfig] = None
    ) -> ProviderResponse:
        """Generate a complete response."""
        cfg = self._get_config(config)
        system_instruction, contents = self._prepare_gemini_messages(messages)

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": cfg.temperature,
                "maxOutputTokens": cfg.max_tokens,
                "topP": cfg.top_p
            }
        }

        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        url = f"{self.endpoint}/models/{model}:generateContent?key={self.api_key}"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=payload,
                timeout=120.0
            )
            response.raise_for_status()
            data = response.json()

            content = ""
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                for part in parts:
                    content += part.get("text", "")

            return ProviderResponse(
                content=content,
                model=model,
                provider=self.name,
                finish_reason=candidates[0].get("finishReason") if candidates else None,
                usage=data.get("usageMetadata"),
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
        system_instruction, contents = self._prepare_gemini_messages(messages)

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": cfg.temperature,
                "maxOutputTokens": cfg.max_tokens,
                "topP": cfg.top_p
            }
        }

        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        url = f"{self.endpoint}/models/{model}:streamGenerateContent?key={self.api_key}&alt=sse"

        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    url,
                    json=payload,
                    timeout=120.0
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            try:
                                data = json.loads(data_str)
                                candidates = data.get("candidates", [])
                                if candidates:
                                    parts = candidates[0].get("content", {}).get("parts", [])
                                    for part in parts:
                                        text = part.get("text", "")
                                        if text:
                                            yield text
                            except json.JSONDecodeError:
                                continue

        except httpx.ConnectError:
            yield "Error: Cannot connect to Google Gemini API"
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
        """Check if Google Gemini API is available."""
        if not self.api_key:
            return {
                "available": False,
                "message": "Google API key not configured"
            }

        try:
            url = f"{self.endpoint}/models?key={self.api_key}"
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10.0)

                if response.status_code == 200:
                    data = response.json()
                    models = [
                        m.get("name", "").replace("models/", "")
                        for m in data.get("models", [])
                        if "generateContent" in m.get("supportedGenerationMethods", [])
                    ]
                    return {
                        "available": True,
                        "message": "Google Gemini API is available",
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

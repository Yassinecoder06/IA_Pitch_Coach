"""
LLM Router: Resolves models, executes streaming with fallback chains,
and runs ping tests across configured providers.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator, Dict, List, Optional, Tuple

from backend.config.settings import settings
from backend.storage.database import DEFAULT_LOCAL_USER_ID
from backend.storage.provider_repository import provider_repository
from backend.llm.types import LLMMessage, parse_model_ref
from backend.llm.providers.base import BaseLLMProvider
from backend.llm.providers.openai_compatible import OpenAICompatibleProvider
from backend.llm.providers.gemini import GeminiProvider
from backend.llm.providers.anthropic import AnthropicProvider
from backend.llm.providers.ollama import OllamaProvider

logger = logging.getLogger(__name__)


def create_provider_instance(
    kind: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None
) -> BaseLLMProvider:
    """Instantiate provider adapter based on kind and credentials."""
    if kind == "gemini":
        return GeminiProvider(api_key=api_key)
    elif kind == "anthropic":
        return AnthropicProvider(api_key=api_key)
    elif kind in ("ollama_local", "ollama_cloud"):
        return OllamaProvider(base_url=base_url or settings.ollama_endpoint)
    else:  # openai, openai_compatible, azure
        url = base_url or ("https://api.openai.com/v1" if kind == "openai" else settings.ollama_endpoint)
        return OpenAICompatibleProvider(base_url=url, api_key=api_key)


class LLMRouter:
    """Manages routing to active LLM providers with automatic fallback cascade."""

    def resolve_provider_and_model(
        self,
        ref: Optional[str],
        user_id: str = DEFAULT_LOCAL_USER_ID
    ) -> Tuple[BaseLLMProvider, str]:
        """Resolve a model ref string to an instantiated provider and model id."""
        if not ref or ref in ("builtin:default", "auto", "default"):
            # Check user preferences
            prefs = provider_repository.get_preferences(user_id)
            if prefs.get("chat_models"):
                ref = prefs["chat_models"][0]
            else:
                # Default to local Ollama or environment
                return OllamaProvider(base_url=settings.ollama_endpoint), (settings.default_model or "qwen3:0.6b")

        provider_id, model_id = parse_model_ref(ref)

        if provider_id in ("builtin", "default"):
            return OllamaProvider(base_url=settings.ollama_endpoint), model_id

        # Lookup in provider repository
        provider_data = provider_repository.get_provider(provider_id, user_id)
        if not provider_data:
            # Fallback to local Ollama
            return OllamaProvider(base_url=settings.ollama_endpoint), model_id

        instance = create_provider_instance(
            kind=provider_data["kind"],
            base_url=provider_data.get("base_url"),
            api_key=provider_data.get("api_key")
        )
        return instance, model_id

    async def stream_with_fallbacks(
        self,
        messages: List[LLMMessage],
        system_prompt: Optional[str] = None,
        model_ref: Optional[str] = None,
        user_id: str = DEFAULT_LOCAL_USER_ID,
        temperature: float = 0.7,
        max_tokens: int = 1500
    ) -> AsyncGenerator[str, None]:
        """
        Stream assistant response using the preferred model, automatically
        falling back to alternative models if the primary model errors.
        """
        prefs = provider_repository.get_preferences(user_id)
        candidates = []
        if model_ref and model_ref not in ("builtin:default", "auto"):
            candidates.append(model_ref)
        elif prefs.get("chat_models"):
            candidates.extend(prefs["chat_models"])

        # Add fallback models
        for fb in prefs.get("fallback_models", []):
            if fb not in candidates:
                candidates.append(fb)

        # Ensure at least default Ollama is in the list
        default_ref = f"builtin:{settings.default_model or 'qwen3:0.6b'}"
        if default_ref not in candidates:
            candidates.append(default_ref)

        last_error = None
        for candidate_ref in candidates:
            try:
                provider, model_id = self.resolve_provider_and_model(candidate_ref, user_id)
                stream = provider.stream_chat(
                    messages=messages,
                    model=model_id,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                chunk_received = False
                async for chunk in stream:
                    chunk_received = True
                    yield chunk

                if chunk_received:
                    return  # Successfully completed
            except Exception as e:
                logger.warning(f"Model '{candidate_ref}' failed: {e}. Trying fallback...")
                last_error = e
                continue

        if last_error:
            raise last_error

    async def test_model(
        self,
        ref: str,
        user_id: str = DEFAULT_LOCAL_USER_ID
    ) -> Tuple[bool, str, float]:
        """Run a test ping against a model ref."""
        try:
            provider, model_id = self.resolve_provider_and_model(ref, user_id)
            return await provider.ping_test(model_id)
        except Exception as e:
            return False, str(e), 0.0


llm_router = LLMRouter()

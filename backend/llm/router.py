"""
LLM Router: resolves model refs to providers, routes each task to its preferred
model, falls back through the cascade on errors, and returns schema-validated
structured output.
"""

from __future__ import annotations

import logging
import time
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel, ValidationError

from backend.config.settings import settings
from backend.llm.providers.anthropic import AnthropicProvider
from backend.llm.providers.base import BaseLLMProvider, ProviderHTTPError
from backend.llm.providers.gemini import GeminiProvider
from backend.llm.providers.ollama import OllamaProvider
from backend.llm.providers.openai_compatible import OpenAICompatibleProvider
from backend.llm.schema import extract_json_text, inline_schema
from backend.llm.types import LLMMessage, parse_model_ref
from backend.storage.provider_repository import provider_repository

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
BUILTIN_REFS = ("builtin:default", "auto", "default")

# A model that answered "overloaded" / "rate limited" is tried last for this long.
COOLDOWN_SECONDS = 60.0
COOLDOWN_STATUSES = {429, 500, 502, 503, 504}


class NoModelAvailable(RuntimeError):
    pass


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
    elif kind == "ollama_cloud":
        return OllamaProvider(base_url=base_url or "https://ollama.com", api_key=api_key)
    elif kind == "ollama_local":
        return OllamaProvider(base_url=base_url or settings.ollama_endpoint)
    else:  # openai, openai_compatible, azure
        url = base_url or ("https://api.openai.com/v1" if kind == "openai" else settings.ollama_endpoint)
        return OpenAICompatibleProvider(base_url=url, api_key=api_key)


def _default_builtin() -> Tuple[BaseLLMProvider, str]:
    return OllamaProvider(base_url=settings.ollama_endpoint), (settings.default_model or "qwen3:0.6b")


class LLMRouter:
    """Routes requests to configured providers with an automatic fallback cascade."""

    def __init__(self) -> None:
        self._cooldown_until: Dict[str, float] = {}

    def _note_failure(self, ref: str, error: Exception) -> None:
        if isinstance(error, ProviderHTTPError) and error.status in COOLDOWN_STATUSES:
            self._cooldown_until[ref] = time.monotonic() + COOLDOWN_SECONDS
        elif isinstance(error, (TimeoutError,)) or "Timeout" in type(error).__name__:
            self._cooldown_until[ref] = time.monotonic() + COOLDOWN_SECONDS

    async def resolve_provider_and_model(self, ref: Optional[str]) -> Tuple[BaseLLMProvider, str]:
        """Resolve a model ref string ("<provider_id>:<model_id>") to a provider and model id."""
        if not ref or ref in BUILTIN_REFS:
            prefs = await provider_repository.get_preferences()
            if not prefs.get("chat_models"):
                return _default_builtin()
            ref = prefs["chat_models"][0]

        provider_id, model_id = parse_model_ref(ref)
        if provider_id in ("builtin", "default"):
            return OllamaProvider(base_url=settings.ollama_endpoint), model_id

        provider_data = await provider_repository.get_provider(provider_id)
        if not provider_data:
            raise NoModelAvailable(f"Provider for '{ref}' is not configured.")
        if not provider_data.get("enabled", True):
            raise NoModelAvailable(f"Provider '{provider_data['name']}' is disabled.")

        instance = create_provider_instance(
            kind=provider_data["kind"],
            base_url=provider_data.get("base_url"),
            api_key=provider_data.get("api_key")
        )
        return instance, model_id

    async def candidates(self, task: str = "chat", model_ref: Optional[str] = None) -> List[str]:
        """Ordered model refs to try: explicit ref, task model, chat models, then fallbacks."""
        prefs = await provider_repository.get_preferences()
        ordered: List[str] = []
        if model_ref and model_ref not in BUILTIN_REFS:
            ordered.append(model_ref)
        task_ref = (prefs.get("task_models") or {}).get(task)
        if task_ref:
            ordered.append(task_ref)
        ordered.extend(prefs.get("chat_models") or [])
        ordered.extend(prefs.get("fallback_models") or [])
        seen, unique = set(), []
        for ref in ordered:
            if ref not in seen:
                seen.add(ref)
                unique.append(ref)
        if not unique:
            return [f"builtin:{settings.default_model or 'qwen3:0.6b'}"]
        now = time.monotonic()
        # Stable sort: healthy models first, cooling-down models keep their relative order at the end.
        return sorted(unique, key=lambda r: self._cooldown_until.get(r, 0.0) > now)

    async def stream_with_fallbacks(
        self,
        messages: List[LLMMessage],
        system_prompt: Optional[str] = None,
        model_ref: Optional[str] = None,
        task: str = "chat",
        temperature: float = 0.7,
        max_tokens: int = 1500
    ) -> AsyncGenerator[str, None]:
        """
        Stream assistant response using the preferred model, automatically
        falling back to alternative models if a model errors before producing output.
        """
        last_error: Optional[Exception] = None
        for candidate_ref in await self.candidates(task, model_ref):
            chunk_received = False
            try:
                provider, model_id = await self.resolve_provider_and_model(candidate_ref)
                async for chunk in provider.stream_chat(
                    messages=messages,
                    model=model_id,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                ):
                    chunk_received = True
                    yield chunk
                if chunk_received:
                    return
            except Exception as e:
                if chunk_received:
                    raise  # output already streamed; switching models mid-answer would garble it
                logger.warning(f"Model '{candidate_ref}' failed: {e}. Trying fallback...")
                self._note_failure(candidate_ref, e)
                last_error = e

        raise last_error or NoModelAvailable("No model produced a response.")

    async def generate_structured(
        self,
        messages: List[LLMMessage],
        schema: Type[T],
        system_prompt: Optional[str] = None,
        model_ref: Optional[str] = None,
        task: str = "judge",
        temperature: float = 0.2,
        max_tokens: int = 4000
    ) -> Tuple[T, str]:
        """
        Returns (validated object, model ref used). Each candidate gets one repair
        attempt when its JSON fails validation before the next candidate is tried.
        """
        json_schema = inline_schema(schema)
        schema_name = schema.__name__
        last_error: Optional[Exception] = None

        for candidate_ref in await self.candidates(task, model_ref):
            try:
                provider, model_id = await self.resolve_provider_and_model(candidate_ref)
                convo = list(messages)
                for attempt in range(2):
                    raw = await provider.generate_json(
                        messages=convo,
                        model=model_id,
                        schema=json_schema,
                        schema_name=schema_name,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    try:
                        return schema.model_validate_json(extract_json_text(raw)), candidate_ref
                    except ValidationError as ve:
                        last_error = ve
                        logger.info(f"Structured output from '{candidate_ref}' failed validation (attempt {attempt + 1}).")
                        convo = convo + [
                            LLMMessage(role="assistant", content=raw[:8000]),
                            LLMMessage(
                                role="user",
                                content=(
                                    "That JSON does not match the required schema. Errors:\n"
                                    f"{ve.errors(include_url=False)[:10]}\n"
                                    "Return the complete corrected JSON object only."
                                ),
                            ),
                        ]
            except Exception as e:
                logger.warning(f"Structured generation with '{candidate_ref}' failed: {e}")
                self._note_failure(candidate_ref, e)
                last_error = e

        raise last_error or NoModelAvailable("No model produced valid structured output.")

    async def test_model(self, ref: str) -> Tuple[bool, str, float]:
        """Run a test ping against a model ref."""
        try:
            provider, model_id = await self.resolve_provider_and_model(ref)
            return await provider.ping_test(model_id)
        except Exception as e:
            return False, str(e), 0.0


llm_router = LLMRouter()

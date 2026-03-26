"""
Provider Registry
=================
Central registry for managing LLM providers.
Auto-discovers and loads available providers based on configuration.
"""

from typing import Dict, List, Optional, Type
import asyncio

from .provider_interface import LLMProvider


class ProviderRegistry:
    """
    Registry for LLM providers.
    Manages provider instances and provides unified access.
    """

    _instance: Optional["ProviderRegistry"] = None
    _providers: Dict[str, LLMProvider] = {}
    _provider_classes: Dict[str, Type[LLMProvider]] = {}

    def __new__(cls) -> "ProviderRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register_class(cls, name: str, provider_class: Type[LLMProvider]):
        """Register a provider class for later instantiation."""
        cls._provider_classes[name.lower()] = provider_class

    @classmethod
    def register(cls, name: str, provider: LLMProvider):
        """Register an instantiated provider."""
        cls._providers[name.lower()] = provider

    @classmethod
    def get(cls, name: str) -> Optional[LLMProvider]:
        """Get a provider by name."""
        return cls._providers.get(name.lower())

    @classmethod
    def list_registered(cls) -> List[str]:
        """List all registered provider names."""
        return list(cls._providers.keys())

    @classmethod
    def get_all(cls) -> Dict[str, LLMProvider]:
        """Get all registered providers."""
        return cls._providers.copy()

    @classmethod
    async def check_all_availability(cls) -> Dict[str, Dict]:
        """Check availability of all registered providers."""
        results = {}
        for name, provider in cls._providers.items():
            try:
                results[name] = await provider.check_availability()
            except Exception as e:
                results[name] = {
                    "available": False,
                    "message": str(e)
                }
        return results

    @classmethod
    def clear(cls):
        """Clear all registered providers."""
        cls._providers.clear()


def _initialize_providers():
    """Initialize providers based on settings."""
    from backend.config import settings

    # Import provider implementations
    from .openai_provider import OpenAIProvider
    from .anthropic_provider import AnthropicProvider
    from .ollama_provider import OllamaProvider
    from .deepseek_provider import DeepSeekProvider
    from .mistral_provider import MistralProvider
    from .google_provider import GoogleProvider
    from .azure_provider import AzureOpenAIProvider
    from .grok_provider import GrokProvider

    provider_map = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "ollama": OllamaProvider,
        "deepseek": DeepSeekProvider,
        "mistral": MistralProvider,
        "google": GoogleProvider,
        "azure": AzureOpenAIProvider,
        "grok": GrokProvider,
    }

    for name, config in settings.get_enabled_providers().items():
        if name in provider_map:
            provider_class = provider_map[name]
            try:
                provider = provider_class(
                    endpoint=config.endpoint,
                    api_key=config.api_key
                )
                ProviderRegistry.register(name, provider)
            except Exception as e:
                print(f"[LLM] Failed to initialize {name} provider: {e}")


_initialized = False

def _ensure_initialized():
    """Ensure providers are initialized."""
    global _initialized
    if not _initialized:
        try:
            _initialize_providers()
            _initialized = True
        except ImportError:
            # During initial module loading, skip
            pass


def get_provider(name: Optional[str] = None) -> Optional[LLMProvider]:
    """
    Get an LLM provider by name.

    Args:
        name: Provider name (e.g., "openai", "ollama").
              If None, returns the default provider.

    Returns:
        LLMProvider instance or None if not found
    """
    _ensure_initialized()

    if name is None:
        from backend.config import settings
        name = settings.default_provider

    return ProviderRegistry.get(name)


def list_providers() -> List[str]:
    """List all available provider names."""
    _ensure_initialized()
    return ProviderRegistry.list_registered()


async def get_available_providers() -> Dict[str, Dict]:
    """Get providers with their availability status."""
    _ensure_initialized()
    return await ProviderRegistry.check_all_availability()

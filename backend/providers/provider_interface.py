"""
LLM Provider Interface
======================
Abstract base class for all LLM providers.
Implements a unified interface similar to Open WebUI architecture.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncGenerator, Dict, List, Optional, Any
from enum import Enum


class MessageRole(str, Enum):
    """Message roles for chat conversations."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """A single message in a conversation."""
    role: MessageRole
    content: str
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format."""
        result = {"role": self.role.value, "content": self.content}
        if self.name:
            result["name"] = self.name
        return result

    @classmethod
    def system(cls, content: str) -> "Message":
        """Create a system message."""
        return cls(role=MessageRole.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        """Create a user message."""
        return cls(role=MessageRole.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        """Create an assistant message."""
        return cls(role=MessageRole.ASSISTANT, content=content)


@dataclass
class ProviderResponse:
    """Response from an LLM provider."""
    content: str
    model: str
    provider: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Any] = None


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        result = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
        if self.frequency_penalty != 0:
            result["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty != 0:
            result["presence_penalty"] = self.presence_penalty
        if self.stop:
            result["stop"] = self.stop
        return result


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All providers must implement:
    - generate(): For complete responses
    - stream(): For streaming responses
    - check_availability(): To verify the provider is accessible
    - list_models(): To list available models
    """

    def __init__(self, endpoint: Optional[str] = None, api_key: Optional[str] = None):
        self.endpoint = endpoint
        self.api_key = api_key
        self._name = self.__class__.__name__.replace("Provider", "")

    @property
    def name(self) -> str:
        """Get the provider name."""
        return self._name

    @abstractmethod
    async def generate(
        self,
        messages: List[Message],
        model: str,
        config: Optional[GenerationConfig] = None
    ) -> ProviderResponse:
        """
        Generate a complete response.

        Args:
            messages: List of conversation messages
            model: Model identifier
            config: Generation configuration

        Returns:
            ProviderResponse with the complete response
        """
        pass

    @abstractmethod
    async def stream(
        self,
        messages: List[Message],
        model: str,
        config: Optional[GenerationConfig] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream response tokens.

        Args:
            messages: List of conversation messages
            model: Model identifier
            config: Generation configuration

        Yields:
            String chunks as they arrive
        """
        pass

    @abstractmethod
    async def check_availability(self) -> Dict[str, Any]:
        """
        Check if the provider is available and configured correctly.

        Returns:
            Dict with status information:
            {
                "available": bool,
                "message": str,
                "models": List[str] (if available)
            }
        """
        pass

    @abstractmethod
    async def list_models(self) -> List[str]:
        """
        List available models for this provider.

        Returns:
            List of model identifiers
        """
        pass

    def _prepare_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        """Convert Message objects to dict format for API calls."""
        return [msg.to_dict() for msg in messages]

    def _get_config(self, config: Optional[GenerationConfig]) -> GenerationConfig:
        """Get config with defaults if not provided."""
        return config or GenerationConfig()

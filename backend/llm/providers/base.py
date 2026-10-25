"""
Base LLM Provider abstract interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx

from backend.llm.types import LLMMessage

# A stalled provider should hand over to the fallback cascade quickly: the read timeout
# applies between streamed chunks, not to the whole answer.
STREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=45.0, write=30.0, pool=10.0)
JSON_TIMEOUT = httpx.Timeout(connect=10.0, read=90.0, write=30.0, pool=10.0)
PING_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


class BaseLLMProvider(ABC):
    @abstractmethod
    async def stream_chat(
        self,
        messages: List[LLMMessage],
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1500
    ) -> AsyncGenerator[str, None]:
        """Stream chunks of assistant response."""
        yield ""

    @abstractmethod
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
        """Return raw JSON text constrained to `schema` using the provider's native mechanism."""

    @abstractmethod
    async def ping_test(self, model: str) -> Tuple[bool, str, float]:
        """Test model connectivity. Returns (success, detail_or_error, elapsed_ms)."""
        return False, "Not implemented", 0.0


class ProviderHTTPError(RuntimeError):
    """A provider answered with a non-2xx status."""

    def __init__(self, provider: str, status: int, body: str):
        super().__init__(f"{provider} API error ({status}): {body[:500]}")
        self.status = status

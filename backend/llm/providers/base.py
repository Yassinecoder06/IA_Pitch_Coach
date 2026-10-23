"""
Base LLM Provider abstract interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, List, Optional, Tuple
from backend.llm.types import LLMMessage


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
    async def ping_test(self, model: str) -> Tuple[bool, str, float]:
        """Test model connectivity. Returns (success, detail_or_error, elapsed_ms)."""
        return False, "Not implemented", 0.0

"""
Grok Provider
=============
LLM provider for xAI Grok API.
Uses OpenAI-compatible API format.
"""

from typing import Optional

from .openai_provider import OpenAIProvider


class GrokProvider(OpenAIProvider):
    """
    Provider for xAI Grok API.
    Grok uses an OpenAI-compatible API.
    """

    DEFAULT_MODELS = ["grok-2", "grok-2-mini", "grok-beta"]

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        super().__init__(endpoint, api_key)
        self.endpoint = endpoint or "https://api.x.ai/v1"
        self._name = "Grok"

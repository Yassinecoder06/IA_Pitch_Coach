"""
Mistral Provider
================
LLM provider for Mistral API.
Uses OpenAI-compatible API format.
"""

from typing import Optional

from .openai_provider import OpenAIProvider


class MistralProvider(OpenAIProvider):
    """
    Provider for Mistral API.
    Mistral uses an OpenAI-compatible API.
    """

    DEFAULT_MODELS = [
        "mistral-large-latest",
        "mistral-medium-latest",
        "mistral-small-latest",
        "open-mistral-7b",
        "open-mixtral-8x7b"
    ]

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        super().__init__(endpoint, api_key)
        self.endpoint = endpoint or "https://api.mistral.ai/v1"
        self._name = "Mistral"

"""
DeepSeek Provider
=================
LLM provider for DeepSeek API.
Uses OpenAI-compatible API format.
"""

from typing import Optional

from .openai_provider import OpenAIProvider


class DeepSeekProvider(OpenAIProvider):
    """
    Provider for DeepSeek API.
    DeepSeek uses an OpenAI-compatible API.
    """

    DEFAULT_MODELS = ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"]

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        super().__init__(endpoint, api_key)
        self.endpoint = endpoint or "https://api.deepseek.com"
        self._name = "DeepSeek"

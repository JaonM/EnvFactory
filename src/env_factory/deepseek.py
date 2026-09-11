"""Backward-compatible DeepSeek client aliases.

Use :class:`env_factory.LLMClient` for new integrations.
"""

from typing import Any

from .llm import LLMClient, LLMError, LLMResponse


class DeepSeekClient(LLMClient):
    """DeepSeek-configured LLM client kept for backward compatibility."""

    def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
        super().__init__(
            api_key,
            base_url=kwargs.pop("base_url", None),
            model=kwargs.pop("model", None),
            api_key_env="DEEPSEEK_API_KEY",
            base_url_env="DEEPSEEK_BASE_URL",
            model_env="DEEPSEEK_MODEL",
            **kwargs,
        )


DeepSeekError = LLMError
DeepSeekResponse = LLMResponse

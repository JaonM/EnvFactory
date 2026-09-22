"""Small, dependency-free runtime adapter for generated sandboxes.

The adapter is intentionally separate from task-generation prompts.  A
generated sandbox may copy this module (or implement the same contract) so
User Simulator and reward evaluators share timeout, retry, JSON validation and
credential-redaction behavior.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from http.client import IncompleteRead
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class RuntimeLLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeLLMConfig:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: float = 60.0
    max_retries: int = 2
    mock: bool = False

    @classmethod
    def from_env(cls) -> "RuntimeLLMConfig":
        api_key = os.getenv("SANDBOX_LLM_API_KEY", "")
        base_url = os.getenv("SANDBOX_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        model = os.getenv("SANDBOX_LLM_MODEL", "")
        timeout = float(os.getenv("SANDBOX_LLM_TIMEOUT_SECONDS", "60"))
        retries = int(os.getenv("SANDBOX_LLM_MAX_RETRIES", "2"))
        mock = os.getenv("SANDBOX_EVALUATOR_MOCK", "").lower() in {"1", "true", "yes"}
        if not api_key and not mock:
            raise RuntimeLLMError("SANDBOX_LLM_API_KEY is not configured")
        if not model and not mock:
            raise RuntimeLLMError("SANDBOX_LLM_MODEL is not configured")
        if timeout <= 0 or retries < 0:
            raise RuntimeLLMError("invalid runtime LLM timeout or retry configuration")
        return cls(api_key, base_url, model, timeout, retries, mock)


class RuntimeLLMClient:
    def __init__(self, config: RuntimeLLMConfig | None = None, *, mock_handler: Callable[[Sequence[Mapping[str, Any]], Mapping[str, Any]], dict[str, Any]] | None = None) -> None:
        self.config = config or RuntimeLLMConfig.from_env()
        self.mock_handler = mock_handler

    def json_chat(self, messages: Sequence[Mapping[str, Any]], *, response_schema: Mapping[str, Any]) -> dict[str, Any]:
        if not messages:
            raise RuntimeLLMError("messages must not be empty")
        if self.config.mock:
            if self.mock_handler is None:
                raise RuntimeLLMError("mock mode requires an injected deterministic evaluator")
            value = self.mock_handler(messages, response_schema)
            if not isinstance(value, dict):
                raise RuntimeLLMError("mock evaluator must return an object")
            self._validate_response(value, response_schema)
            return value
        body = {
            "model": self.config.model,
            "messages": [dict(item) for item in messages],
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        for attempt in range(self.config.max_retries + 1):
            try:
                request = Request(f"{self.config.base_url}/chat/completions", data=encoded, headers=headers, method="POST")
                with urlopen(request, timeout=self.config.timeout_seconds) as response:
                    payload = json.load(response)
                content = payload["choices"][0]["message"]["content"]
                value = json.loads(content)
                if not isinstance(value, dict):
                    raise RuntimeLLMError("runtime LLM JSON response must be an object")
                self._validate_response(value, response_schema)
                return value
            except HTTPError as exc:
                if exc.code in {408, 409, 425, 429, 500, 502, 503, 504} and attempt < self.config.max_retries:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                raise RuntimeLLMError(f"runtime LLM HTTP failure: {exc.code}") from exc
            except (IncompleteRead, URLError, TimeoutError, OSError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
                if attempt >= self.config.max_retries:
                    raise RuntimeLLMError("runtime LLM request or response failed") from exc
                time.sleep(0.5 * (2 ** attempt))
        raise RuntimeLLMError("runtime LLM request failed")

    @staticmethod
    def _validate_response(value: dict[str, Any], schema: Mapping[str, Any]) -> None:
        required = schema.get("required", [])
        missing = [name for name in required if name not in value]
        if missing:
            raise RuntimeLLMError(f"runtime LLM response missing fields: {missing}")

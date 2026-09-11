"""Configurable LLM wrapper for OpenAI-compatible chat APIs."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


load_dotenv()


class LLMError(RuntimeError):
    """Raised when an LLM request fails or returns an invalid response."""


@dataclass(frozen=True)
class LLMResponse:
    """Normalized response from an OpenAI-compatible chat completion."""

    content: str
    model: str
    id: str | None = None
    finish_reason: str | None = None
    reasoning_content: str | None = None
    usage: Mapping[str, Any] | None = None


class LLMClient:
    """Call a configurable OpenAI-compatible chat completion endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        api_key_env: str = "LLM_API_KEY",
        base_url_env: str = "LLM_BASE_URL",
        model_env: str = "LLM_MODEL",
        default_params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv(api_key_env)
        if not self.api_key:
            raise ValueError(f"{api_key_env} is not configured")
        self.base_url = (
            base_url or os.getenv(base_url_env, "https://api.openai.com/v1")
        ).rstrip("/")
        self.model = model or os.getenv(model_env)
        if not self.model:
            raise ValueError(f"{model_env} is not configured")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        self.timeout = timeout
        self.default_params = dict(default_params or {})
        self.headers = dict(headers or {})

    @classmethod
    def from_env(cls, prefix: str = "LLM", **kwargs: Any) -> "LLMClient":
        """Create a client from ``<PREFIX>_API_KEY/BASE_URL/MODEL`` variables."""

        prefix = prefix.rstrip("_").upper()
        return cls(
            api_key_env=f"{prefix}_API_KEY",
            base_url_env=f"{prefix}_BASE_URL",
            model_env=f"{prefix}_MODEL",
            **kwargs,
        )

    def chat(self, messages: Sequence[Mapping[str, Any]], **parameters: Any) -> LLMResponse:
        """Generate a response and pass arbitrary provider parameters through."""

        if not messages:
            raise ValueError("messages must not be empty")
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "stream": False,
            **self.default_params,
            **parameters,
        }
        body["model"] = self.model
        body["messages"] = [dict(message) for message in messages]
        body["stream"] = False
        if isinstance(body.get("thinking"), bool) and "deepseek" in self.base_url.lower():
            body["thinking"] = {
                "type": "enabled" if body["thinking"] else "disabled"
            }
        if isinstance(body.get("response_format"), str):
            body["response_format"] = {"type": body["response_format"]}

        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                **self.headers,
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except HTTPError as exc:
            detail = self._error_detail(exc)
            raise LLMError(f"LLM returned HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise LLMError(f"Unable to reach LLM at {self.base_url}") from exc
        except json.JSONDecodeError as exc:
            raise LLMError("LLM returned invalid JSON") from exc

        try:
            choice = payload["choices"][0]
            message = choice["message"]
            return LLMResponse(
                content=message.get("content") or "",
                model=str(payload.get("model", self.model)),
                id=payload.get("id"),
                finish_reason=choice.get("finish_reason"),
                reasoning_content=message.get("reasoning_content"),
                usage=payload.get("usage"),
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("LLM response has an unexpected shape") from exc

    def complete(self, prompt: str, *, system_prompt: str | None = None, **parameters: Any) -> LLMResponse:
        """Generate a response from a single prompt."""

        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        messages: list[dict[str, str]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **parameters)

    @staticmethod
    def _error_detail(error: HTTPError) -> str:
        try:
            payload = json.load(error)
            if isinstance(payload, dict):
                error_value = payload.get("error")
                if isinstance(error_value, dict) and error_value.get("message"):
                    return str(error_value["message"])
        except (json.JSONDecodeError, OSError, TypeError):
            pass
        return str(error.reason or "request failed")

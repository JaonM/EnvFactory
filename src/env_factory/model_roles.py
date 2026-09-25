"""Resolve model-provider roles with explicit, field-wise fallbacks."""

from __future__ import annotations

import os
from typing import Mapping


DEFAULT_BASE_URL = "https://api.openai.com/v1"


def _first(environment: Mapping[str, str], *names: str, default: str = "") -> str:
    for name in names:
        value = environment.get(name, "").strip()
        if value:
            return value
    return default


def resolve_model_roles(
    environment: Mapping[str, str] | None = None,
) -> dict[str, dict[str, str]]:
    """Return generation, rollout-agent and runtime model configurations.

    Rollout and sandbox settings fall back one field at a time to generation
    settings, so overriding a model never accidentally discards the base URL
    or credential fallback for that role.
    """
    env = os.environ if environment is None else environment
    generation = {
        "api_key": _first(env, "LLM_API_KEY"),
        "base_url": _first(env, "LLM_BASE_URL", default=DEFAULT_BASE_URL).rstrip("/"),
        "model": _first(env, "LLM_MODEL"),
        "timeout_seconds": _first(env, "LLM_TIMEOUT", default="60"),
    }
    agent = {
        "api_key": _first(env, "ROLLOUT_LLM_API_KEY", "LLM_API_KEY"),
        "base_url": _first(
            env, "ROLLOUT_LLM_BASE_URL", "LLM_BASE_URL",
            default=DEFAULT_BASE_URL,
        ).rstrip("/"),
        "model": _first(env, "ROLLOUT_LLM_MODEL", "LLM_MODEL"),
        "timeout_seconds": _first(
            env, "ROLLOUT_LLM_TIMEOUT_SECONDS", "ROLLOUT_LLM_TIMEOUT",
            "LLM_TIMEOUT", default="60",
        ),
    }
    runtime = {
        "api_key": _first(env, "SANDBOX_LLM_API_KEY", "LLM_API_KEY"),
        "base_url": _first(
            env, "SANDBOX_LLM_BASE_URL", "LLM_BASE_URL",
            default=DEFAULT_BASE_URL,
        ).rstrip("/"),
        "model": _first(env, "SANDBOX_LLM_MODEL", "LLM_MODEL"),
        "timeout_seconds": _first(
            env, "SANDBOX_LLM_TIMEOUT_SECONDS", "LLM_TIMEOUT", default="60",
        ),
    }
    return {"generation": generation, "agent": agent, "runtime": runtime}

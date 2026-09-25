"""Fail-fast checks for a production material certification experiment."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable, Mapping
from urllib.parse import urlparse

from .material_attestation import (
    private_key_public_identity,
    public_key_identity,
)
from .material_consumer import BUNDLE_VERSION


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _valid_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.hostname)


def run_production_preflight(
    project: Path,
    output_parent: Path,
    *,
    signing_private_key: Path,
    trusted_public_key: Path,
    environment: Mapping[str, str] | None = None,
    runner: Runner = subprocess.run,
    which: Callable[[str], str | None] = shutil.which,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
    minimum_free_bytes: int = 10 * 1024**3,
) -> dict[str, Any]:
    """Return non-secret readiness evidence without contacting model providers."""
    env = dict(os.environ if environment is None else environment)
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, evidence: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    required_files = (
        "pyproject.toml", "uv.lock", "scripts/loop_experiment.py",
        "scripts/develop_sandbox_with_agent.sh",
    )
    missing_files = [name for name in required_files if not (project / name).is_file()]
    record("project_contract", not missing_files, {"missing": missing_files})

    executables = {name: which(name) is not None for name in ("codex", "docker", "openssl")}
    record("required_executables", all(executables.values()), executables)

    docker_ready = False
    docker_identity: dict[str, str] = {}
    if executables["docker"]:
        try:
            completed = runner(
                ["docker", "info", "--format", "{{.OSType}} {{.Architecture}}"],
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            parts = completed.stdout.strip().split()
            docker_ready = completed.returncode == 0 and len(parts) == 2
            if docker_ready:
                docker_identity = {"os": parts[0], "architecture": parts[1]}
        except (OSError, subprocess.TimeoutExpired):
            docker_ready = False
    record("docker_daemon", docker_ready, docker_identity)

    agent_model = env.get("LLM_MODEL", "").strip()
    agent_key = env.get("LLM_API_KEY", "").strip()
    runtime_model = (env.get("SANDBOX_LLM_MODEL") or agent_model).strip()
    runtime_key = (env.get("SANDBOX_LLM_API_KEY") or agent_key).strip()
    agent_url = (env.get("LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    runtime_url = (env.get("SANDBOX_LLM_BASE_URL") or agent_url).rstrip("/")
    model_configured = all((agent_model, agent_key, runtime_model, runtime_key))
    urls_valid = _valid_url(agent_url) and _valid_url(runtime_url)
    record("model_configuration", model_configured and urls_valid, {
        "agent_model_configured": bool(agent_model),
        "agent_key_configured": bool(agent_key),
        "runtime_model_configured": bool(runtime_model),
        "runtime_key_configured": bool(runtime_key),
        "agent_host": urlparse(agent_url).hostname,
        "runtime_host": urlparse(runtime_url).hostname,
        "urls_valid": urls_valid,
    })

    timeout_valid = retries_valid = False
    try:
        timeout_valid = float(
            env.get("SANDBOX_LLM_TIMEOUT_SECONDS")
            or env.get("LLM_TIMEOUT")
            or "60"
        ) > 0
    except ValueError:
        pass
    try:
        retries = int(env.get("SANDBOX_LLM_MAX_RETRIES") or "3")
        retries_valid = 0 <= retries <= 10
    except ValueError:
        pass
    record("model_runtime_limits", timeout_valid and retries_valid, {
        "timeout_valid": timeout_valid,
        "retries_valid": retries_valid,
    })

    signing_ready = False
    key_identity = None
    if executables["openssl"]:
        try:
            private_identity = private_key_public_identity(signing_private_key)
            public_identity = public_key_identity(trusted_public_key)
            signing_ready = private_identity == public_identity
            key_identity = public_identity if signing_ready else None
        except (OSError, ValueError):
            signing_ready = False
    record("bundle_signing_identity", signing_ready, {
        "key_identity_sha256": key_identity,
        "bundle_version": BUNDLE_VERSION,
    })

    free_bytes = 0
    try:
        free_bytes = int(disk_usage(output_parent).free)
    except OSError:
        pass
    record("workspace_capacity", free_bytes >= minimum_free_bytes, {
        "free_gib": round(free_bytes / 1024**3, 2),
        "minimum_gib": round(minimum_free_bytes / 1024**3, 2),
    })

    failed = [item["name"] for item in checks if not item["passed"]]
    return {
        "version": "1.0",
        "scope": "production_pre_training_material_experiment",
        "network_probe_performed": False,
        "ready": not failed,
        "failed_checks": failed,
        "checks": checks,
    }


#!/usr/bin/env python3
"""Fail fast when a task contract cannot be implemented by the shared sandbox."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

from env_factory.sandbox_runtime import (
    DeclarativeMetricEvaluator, EpisodeStore, ManifestDataStore, SandboxError,
)
from env_factory.task_quality import score_file
from env_factory.task_pipeline import PipelineGenerationError, TaskGenerationPipeline
from env_factory.task_spec import TaskSpecError, validate_task_spec


def _issue(code: str, owner: str, message: str, **evidence: Any) -> dict[str, Any]:
    return {"code": code, "owner": owner, "message": message, "evidence": evidence}


def _manifest_root(task_root: Path, manifest: dict[str, Any]) -> Path:
    """Resolve both portable and generation-workspace manifest roots.

    Generated task artifacts retain a project-relative root so the outer
    builder can locate and copy them.  Once the task directory itself is
    handed to this preflight, its schema/row files are already immediately
    below ``task_root``.  Prefer that self-contained representation, then the
    conventional task-relative and current-workspace forms.
    """
    declared = Path(str(manifest["root"]))
    if declared.is_absolute():
        return declared
    referenced = [
        item.get("schema_file") or item.get("rows_file")
        for item in manifest.get("tables", [])
        if isinstance(item, dict)
    ]
    if any(isinstance(name, str) and (task_root / name).is_file() for name in referenced):
        return task_root
    candidates = (task_root / declared, Path.cwd() / declared)
    return next((candidate for candidate in candidates if candidate.is_dir()), candidates[0])


def assess(root: Path, *, threshold: float = 8.0) -> dict[str, Any]:
    task_path = root / "task.json"
    issues: list[dict[str, Any]] = []
    try:
        task = json.loads(task_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"buildable": False, "failure_owner": "task_generation", "issues": [
            _issue("TASK_ARTIFACT_INVALID", "task_generation", str(exc))
        ]}
    quality = score_file(task_path, min_score=threshold).to_dict()
    if not quality.get("eligible") or quality.get("score", 0) < threshold:
        issues.append(_issue(
            "TASK_QUALITY_GATE", "task_generation", "task is not eligible for sandbox construction",
            score=quality.get("score"), findings=quality.get("findings", []),
        ))

    try:
        validate_task_spec(task.get("task_spec", {}))
    except (TaskSpecError, TypeError, ValueError) as exc:
        issues.append(_issue("TASK_SPEC_INVALID", "task_generation", str(exc)))

    mode = task.get("environment_plan", {}).get("mode")
    declared_modes = task.get("runtime_capabilities", {}).get("environment_modes", [])
    if isinstance(declared_modes, list) and declared_modes and mode not in declared_modes:
        issues.append(_issue(
            "ENVIRONMENT_MODE_UNDECLARED", "task_generation",
            "task environment mode is absent from its generation-time capability inventory",
            mode=mode, declared_modes=declared_modes,
        ))
    external_configured = bool(os.getenv("SANDBOX_EXTERNAL_CAPABILITY_URL", "").strip())
    fixture = os.getenv("SANDBOX_EXTERNAL_FIXTURES", "").strip()
    external_configured = external_configured or bool(fixture and Path(fixture).is_file())
    if mode == "external_capability" and not external_configured:
        issues.append(_issue(
            "EXTERNAL_CAPABILITY_UNAVAILABLE", "task_generation",
            "task requires an external capability but no provider or fixture is configured",
        ))

    manifest = task.get("artifacts", {}).get("data_manifest", {})
    if isinstance(manifest, dict) and manifest.get("root"):
        data_root = _manifest_root(root, manifest)
        if not data_root.is_dir():
            issues.append(_issue(
                "BUSINESS_DATA_MISSING", "task_generation", "declared business data root is missing",
                root=str(data_root),
            ))
        else:
            try:
                with tempfile.TemporaryDirectory(prefix="envfactory-buildability-") as directory:
                    store = EpisodeStore(Path(directory) / "episodes.sqlite3")
                    ManifestDataStore(manifest, data_root, store)
            except (SandboxError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                issues.append(_issue(
                    "BUSINESS_DATA_INVALID", "task_generation",
                    "business data violates the shared persistence contract",
                    error=str(exc), root=str(data_root),
                ))

    for index, spec in enumerate(task.get("metric_implementations", [])):
        if not isinstance(spec, dict) or not isinstance(spec.get("path"), str):
            issues.append(_issue(
                "METRIC_IMPLEMENTATION_INVALID", "task_generation",
                f"metric implementation {index} has no valid path",
            ))
            continue
        try:
            DeclarativeMetricEvaluator.path_tokens(spec["path"])
        except (SandboxError, ValueError, SyntaxError) as exc:
            issues.append(_issue(
                "METRIC_PATH_UNSUPPORTED", "platform", str(exc),
                metric_id=spec.get("metric_id"), path=spec.get("path"),
            ))

    declared_tools = {
        item.get("function", {}).get("name") for item in task.get("tools", [])
        if isinstance(item, dict) and isinstance(item.get("function"), dict)
    }
    implemented_tools = {
        item.get("tool_name") for item in task.get("tool_implementations", [])
        if isinstance(item, dict)
    }
    noise_tools = {
        item.get("name") for item in task.get("noise_tools", []) if isinstance(item, dict)
    }
    missing_handlers = sorted(declared_tools - implemented_tools - noise_tools)
    business_tools = [
        item for item in task.get("tools", [])
        if isinstance(item, dict)
        and item.get("function", {}).get("name") not in noise_tools
    ]
    try:
        TaskGenerationPipeline._validate_business_tool_semantics(
            tools=business_tools,
            implementations=[
                item for item in task.get("tool_implementations", [])
                if isinstance(item, dict)
            ],
        )
    except PipelineGenerationError as exc:
        issues.append(_issue(
            "BUSINESS_TOOL_SEMANTICS_INVALID", "task_generation", str(exc),
        ))
    report = {
        "buildable": not issues,
        "failure_owner": issues[0]["owner"] if issues else None,
        "issues": issues,
        "task_score": quality,
        "environment_mode": mode,
        "custom_business_handlers": missing_handlers,
        "external_capability_configured": external_configured,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threshold", type=float, default=8.0)
    args = parser.parse_args()
    root = args.root.resolve()
    report = assess(root, threshold=args.threshold)
    output = args.output or root / "buildability.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["buildable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

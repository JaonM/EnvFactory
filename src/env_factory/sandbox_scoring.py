"""Immutable identity and structural verification for sandbox score evidence."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


SCORE_RUBRIC = (
    ("delivery_integrity", 1.0, True),
    ("contract_and_tool_identity", 1.0, True),
    ("semantic_business_fidelity", 0.5, True),
    ("business_acceptance", 1.0, True),
    ("sandbox_pytest", 1.0, True),
    ("runtime_genericity", 0.5, True),
    ("outer_conformance", 1.0, True),
    ("mutation_resistance", 1.0, True),
    ("training_readiness", 1.0, True),
    ("declared_training_policy", 2.0, True),
)
REQUIRED_CHECKS = frozenset(name for name, _, _ in SCORE_RUBRIC)
RUBRIC_BY_NAME = {
    name: {"weight": weight, "critical": critical}
    for name, weight, critical in SCORE_RUBRIC
}


@lru_cache(maxsize=256)
def _base_evidence_digest(root_value: str, project_value: str):
    root, project = Path(root_value), Path(project_value)
    paths = [
        *sorted((project / "src/env_factory").glob("*.py")),
        *sorted((project / "scripts").glob("*.py")),
        root / "BUILD_CONTRACT.json",
        *sorted(root.glob("*.py")),
        *sorted(root.glob("*.sh")),
        *sorted((root / "tests").rglob("*.py")),
        *sorted((root / "data").rglob("*.json*")),
    ]
    digest = hashlib.sha256()
    digest.update(b"envfactory-sandbox-score-evidence-v2\0")
    for path in paths:
        if path.is_file():
            label = (
                str(path.relative_to(root))
                if path.is_relative_to(root)
                else str(path.relative_to(project))
            )
            digest.update(label.encode())
            digest.update(path.read_bytes())
    return digest


def evidence_fingerprint(
    root: Path, project: Path, *, task_path: Path | None = None
) -> str:
    """Bind scoring evidence to evaluator code and evaluated implementation."""
    digest = _base_evidence_digest(str(root.resolve()), str(project.resolve())).copy()
    task = task_path or root / "task.json"
    if task.is_file():
        digest.update(b"task.json")
        digest.update(task.read_bytes())
    return digest.hexdigest()


def valid_score_report(
    report: Any,
    *,
    root: Path,
    project: Path,
    threshold: float,
    task_path: Path | None = None,
) -> bool:
    """Recompute the deterministic score envelope without rerunning evidence."""
    if not isinstance(report, Mapping):
        return False
    checks = report.get("checks")
    if not isinstance(checks, list) or not checks:
        return False
    names: list[str] = []
    raw = 0.0
    failed_critical: list[str] = []
    for check in checks:
        if not isinstance(check, Mapping) or set(check) != {
            "name", "weight", "passed", "evidence", "critical"
        }:
            return False
        name = check.get("name")
        weight = check.get("weight")
        if (
            not isinstance(name, str) or not name
            or isinstance(weight, bool) or not isinstance(weight, (int, float))
            or weight < 0
            or not isinstance(check.get("passed"), bool)
            or not isinstance(check.get("critical"), bool)
            or not isinstance(check.get("evidence"), str)
        ):
            return False
        names.append(name)
        if check["passed"]:
            raw += float(weight)
        elif check["critical"]:
            failed_critical.append(name)
    if names != [name for name, _, _ in SCORE_RUBRIC]:
        return False
    if any(
        check["weight"] != RUBRIC_BY_NAME[check["name"]]["weight"]
        or check["critical"] is not RUBRIC_BY_NAME[check["name"]]["critical"]
        for check in checks
    ):
        return False
    score = round(raw, 2)
    eligible = not failed_critical
    return (
        report.get("mode") == "offline_executable"
        and report.get("network_used") is False
        and report.get("model_used") is False
        and report.get("live_rollout_verified") is False
        and report.get("threshold") == threshold
        and report.get("score") == score
        and report.get("eligible") is eligible
        and report.get("passed") is (eligible and score >= threshold)
        and report.get("failed_critical_gates") == failed_critical
        and report.get("evidence_fingerprint")
        == evidence_fingerprint(root, project, task_path=task_path)
        and isinstance(report.get("model"), str) and bool(report["model"])
        and isinstance(report.get("review_model"), str)
        and bool(report["review_model"])
    )

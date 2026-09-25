"""Stable identities for portable sandbox inputs and certification evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


RUNTIME_ROOT_NAMES = {
    "task.json", "BUILD_CONTRACT.json", "tools.json", "sandbox_profile.json",
    "Dockerfile", "requirements-dev.txt", "docker_image_metadata.json", "TASK_PROMPT.md",
    "IMPLEMENTATION_REPORT.md",
}
RUNTIME_ROOT_SUFFIXES = {".py", ".sh", ".toml", ".yaml", ".yml"}
RUNTIME_DIRECTORIES = ("tests", "data", ".outer_conformance")
EVIDENCE_NAMES = {
    "acceptance_result.json", "agentic_training_value.json",
    "agentic_training_value_live.json", "buildability.json",
    "data_governance.json",
    "offline_sandbox_score.json", "review_report.json", "score_summary.json",
    "training_readiness.json",
}
SKIPPED_SUFFIXES = {".pyc", ".sqlite", ".sqlite3", ".db", ".log"}


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _digest_paths(root: Path, paths: list[Path]) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(paths))
    }


def portable_artifact_digests(root: Path) -> dict[str, str]:
    """Hash immutable runtime inputs, excluding outputs produced by validation."""
    paths = [
        path for path in root.iterdir()
        if path.is_file()
        and (path.name in RUNTIME_ROOT_NAMES or path.suffix in RUNTIME_ROOT_SUFFIXES)
    ]
    for directory in RUNTIME_DIRECTORIES:
        paths.extend(
            path for path in (root / directory).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
            and path.suffix not in SKIPPED_SUFFIXES
        )
    return _digest_paths(root, paths)


def evidence_artifact_digests(root: Path) -> dict[str, str]:
    """Hash reports whose claims are consumed by production certification."""
    return _digest_paths(root, [root / name for name in EVIDENCE_NAMES if (root / name).is_file()])


def portable_artifact_digest(root: Path) -> str:
    return digest_json(portable_artifact_digests(root))

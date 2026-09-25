"""Independent checks for reproducible, portable sandbox container evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


EXACT_REQUIREMENT = re.compile(
    r"^[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?==[^\s;]+(?:\s*;.*)?$"
)
PINNED_IMAGE = re.compile(r"^[^\s]+@sha256:[0-9a-f]{64}$")
CONTENT_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_requirements(path: Path) -> bool:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    values = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    return bool(values) and all(EXACT_REQUIREMENT.fullmatch(line) for line in values)


def verify_container_provenance(root: Path) -> dict[str, Any]:
    failures = []
    metadata_path = root / "docker_image_metadata.json"
    dockerfile = root / "Dockerfile"
    requirements = root / "requirements-dev.txt"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        metadata = {}
        failures.append("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("version") != "2.0":
        failures.append("metadata_schema")
        metadata = {}
    if PINNED_IMAGE.fullmatch(str(metadata.get("base_image", ""))) is None:
        failures.append("base_image_digest")
    if CONTENT_DIGEST.fullmatch(str(metadata.get("image_id", ""))) is None:
        failures.append("image_id")
    try:
        dockerfile_text = dockerfile.read_text(encoding="utf-8")
        dockerfile_digest = file_sha256(dockerfile)
    except OSError:
        dockerfile_text = ""
        dockerfile_digest = ""
        failures.append("dockerfile")
    from_images = re.findall(r"(?im)^\s*FROM\s+([^\s]+)", dockerfile_text)
    if not from_images or any(PINNED_IMAGE.fullmatch(image) is None for image in from_images):
        failures.append("dockerfile_base_image")
    users = re.findall(r"(?im)^\s*USER\s+([^\s]+)", dockerfile_text)
    runtime_user = str(metadata.get("runtime_user", ""))
    if (
        not users
        or users[-1] in {"root", "0"}
        or runtime_user in {"", "root", "0"}
        or users[-1] != runtime_user
    ):
        failures.append("non_root_user")
    if metadata.get("dockerfile_sha256") != dockerfile_digest:
        failures.append("dockerfile_digest")
    if not exact_requirements(requirements):
        failures.append("dependency_pins")
    try:
        requirements_digest = file_sha256(requirements)
    except OSError:
        requirements_digest = ""
    if metadata.get("requirements_sha256") != requirements_digest:
        failures.append("requirements_digest")
    smoke = metadata.get("smoke_test", {})
    if not (
        isinstance(smoke, Mapping)
        and smoke.get("passed") is True
        and smoke.get("network") == "none"
        and smoke.get("read_only_root") is True
        and smoke.get("cap_drop") == "ALL"
        and smoke.get("no_new_privileges") is True
        and smoke.get("non_root_user") is True
    ):
        failures.append("container_smoke_test")
    platform = metadata.get("platform", {})
    if not (
        isinstance(platform, Mapping)
        and isinstance(platform.get("os"), str) and platform["os"]
        and isinstance(platform.get("architecture"), str) and platform["architecture"]
    ):
        failures.append("platform")
    return {
        "verified": not failures,
        "failed_gates": sorted(set(failures)),
        "base_image": metadata.get("base_image"),
        "image_id": metadata.get("image_id"),
        "platform": metadata.get("platform"),
    }

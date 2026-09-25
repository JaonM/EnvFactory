#!/usr/bin/env python3
"""Verify that certified Agentic-RL material artifacts are still immutable."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from env_factory.material_artifacts import (
    digest_json,
    evidence_artifact_digests,
    portable_artifact_digests,
)
from env_factory.execution_provenance import valid_execution_provenance


def verify_artifact_digests(root: Path, expected: Mapping[str, Any]) -> list[str]:
    failures = []
    normalized = {}
    for relative, wanted in expected.items():
        relative_path = Path(relative) if isinstance(relative, str) else None
        if (
            relative_path is None or relative_path.is_absolute()
            or ".." in relative_path.parts or not isinstance(wanted, str)
        ):
            failures.append(str(relative))
            continue
        normalized[relative] = wanted
    try:
        actual = portable_artifact_digests(root)
    except OSError as exc:
        return [f"artifact inventory unavailable: {exc}"]
    failures.extend(sorted(set(normalized) ^ set(actual)))
    failures.extend(
        relative for relative in sorted(set(normalized) & set(actual))
        if normalized[relative] != actual[relative]
    )
    return sorted(set(failures))


def verify(
    manifest: Mapping[str, Any], project: Path,
    *, fingerprint: Callable[[Path, Path], str] | None = None,
) -> dict[str, Any]:
    failures = []
    base = {key: value for key, value in manifest.items() if key != "dataset_sha256"}
    if manifest.get("dataset_sha256") != digest_json(base):
        failures.append({"gate": "dataset_digest", "message": "manifest digest changed"})
    if manifest.get("version") == "3.0" and not valid_execution_provenance(
        manifest.get("execution_provenance")
    ):
        failures.append({
            "gate": "execution_provenance",
            "message": "v3 manifest needs a valid execution environment snapshot",
        })
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        failures.append({"gate": "manifest_schema", "message": "items must be non-empty"})
        items = []
    seen = set()
    seen_roots = set()
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            failures.append({"gate": "manifest_schema", "item": index})
            continue
        task = Path(str(item.get("task_path", "")))
        root = Path(str(item.get("sandbox_root", "")))
        resolved_root = str(root.resolve())
        if resolved_root in seen_roots:
            failures.append({"gate": "duplicate_sandbox_root", "item": index})
        seen_roots.add(resolved_root)
        try:
            task_digest = hashlib.sha256(task.read_bytes()).hexdigest()
        except OSError as exc:
            failures.append({"gate": "task_artifact", "item": index, "message": str(exc)})
            continue
        if task_digest != item.get("task_sha256"):
            failures.append({"gate": "task_digest", "item": index})
        expected_fingerprint = item.get("sandbox_evidence_fingerprint")
        artifact_hashes = item.get("sandbox_artifacts_sha256")
        if manifest.get("version") in {"2.0", "3.0"} and not (
            isinstance(artifact_hashes, Mapping) and artifact_hashes
        ):
            failures.append({"gate": "manifest_schema", "item": index,
                             "message": "v2/v3 item needs sandbox_artifacts_sha256"})
            continue
        if isinstance(artifact_hashes, Mapping) and artifact_hashes:
            changed = verify_artifact_digests(root, artifact_hashes)
            if changed:
                failures.append({
                    "gate": "sandbox_digest", "item": index,
                    "message": f"changed or missing artifacts: {changed}",
                })
        else:
            # Backward compatibility for v1 manifests. V2/v3 deliberately avoid
            # recomputing a fingerprint that also depends on the current
            # evaluator source tree.
            try:
                if fingerprint is None:
                    from score_sandbox import evidence_fingerprint
                    fingerprint = evidence_fingerprint
                actual_fingerprint = fingerprint(root, project)
            except Exception as exc:  # verifier must report, not abort the batch
                failures.append({
                    "gate": "sandbox_artifact", "item": index,
                    "message": f"{type(exc).__name__}: {exc}",
                })
                continue
            if actual_fingerprint != expected_fingerprint:
                failures.append({"gate": "sandbox_digest", "item": index})
        evidence_hashes = item.get("sandbox_evidence_sha256")
        if manifest.get("version") in {"2.0", "3.0"} and not (
            isinstance(evidence_hashes, Mapping) and evidence_hashes
        ):
            failures.append({"gate": "manifest_schema", "item": index,
                             "message": "v2/v3 item needs sandbox_evidence_sha256"})
        elif isinstance(evidence_hashes, Mapping):
            actual_evidence = evidence_artifact_digests(root)
            if dict(evidence_hashes) != actual_evidence:
                failures.append({"gate": "sandbox_evidence_digest", "item": index})
        if expected_fingerprint in seen:
            failures.append({"gate": "duplicate_sandbox_identity", "item": index})
        seen.add(expected_fingerprint)
        try:
            rollout = json.loads((root / "live_rollout.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append({"gate": "rollout_artifact", "item": index, "message": str(exc)})
            continue
        if digest_json(rollout) != item.get("rollout_sha256"):
            failures.append({"gate": "rollout_digest", "item": index})
        episodes = rollout.get("episodes", []) if isinstance(rollout, Mapping) else []
        if len(episodes) != item.get("episode_count"):
            failures.append({"gate": "episode_count", "item": index})
    return {
        "verified": not failures,
        "items": len(items),
        "failed_gates": sorted({item["gate"] for item in failures}),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = verify(value, args.project.resolve())
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export a certified EnvFactory result as a portable, verified RL-material bundle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping

from env_factory.material_artifacts import digest_json
from env_factory.trajectory_schema import episode_errors


BUNDLE_MANIFEST = "bundle_manifest.json"
TRANSITIONS_FILE = "transitions.jsonl"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe material path: {value}")
    return path


def _copy_verified(source: Path, destination: Path, expected: str) -> None:
    if file_sha256(source) != expected:
        raise ValueError(f"source digest changed: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if file_sha256(destination) != expected:
        raise ValueError(f"copied digest mismatch: {destination}")


def _transition_records(
    item_id: str, item: Mapping[str, Any], rollout: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if rollout.get("schema_version") != "2.0":
        raise ValueError("rollout schema_version must be 2.0")
    if not all(
        isinstance(rollout.get(name), str) and rollout[name]
        for name in ("agent_model", "runtime_model")
    ):
        raise ValueError("rollout model provenance is incomplete")
    episodes = rollout.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("rollout episodes must be non-empty")
    records = []
    for episode_index, episode in enumerate(episodes):
        errors = episode_errors(episode)
        if errors:
            raise ValueError(
                f"rollout episode {episode_index} violates trajectory schema: {errors}"
            )
        for transition_index, transition in enumerate(episode["transitions"]):
            records.append({
                "schema_version": "2.0",
                "item_id": item_id,
                "task_sha256": item["task_sha256"],
                "category": item["category"],
                "episode_index": episode_index,
                "episode_seed": episode.get("seed"),
                "episode_success": episode.get("agent_success"),
                "episode_termination": episode.get("termination"),
                "episode_initial_reward": episode.get("initial_reward"),
                "episode_final_reward": episode.get("final_reward"),
                "agent_model": rollout.get("agent_model"),
                "runtime_model": rollout.get("runtime_model"),
                "agent_usage": episode["usage"][transition_index],
                "transition": dict(transition),
            })
    return records


def _export_bundle_uncommitted(
    certification: Mapping[str, Any], output: Path, project: Path
) -> dict[str, Any]:
    if certification.get("certified") is not True:
        raise ValueError("only a certified production report can be exported")
    if certification.get("material_verification", {}).get("verified") is not True:
        raise ValueError("source material verification is required before export")
    source_manifest = certification.get("materials_manifest")
    if not isinstance(source_manifest, Mapping) or source_manifest.get("version") != "2.0":
        raise ValueError("a v2 materials manifest is required")
    try:
        from verify_training_materials import verify
    except ModuleNotFoundError:
        spec = importlib.util.spec_from_file_location(
            "verify_training_materials", Path(__file__).with_name("verify_training_materials.py")
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load material verifier")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        verify = module.verify
    source_verification = verify(source_manifest, project)
    if source_verification.get("verified") is not True:
        raise ValueError(f"source material verification failed: {source_verification['failed_gates']}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"bundle output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    transition_path = output / TRANSITIONS_FILE
    exported_items = []
    seen_item_ids = set()
    transition_count = 0
    with transition_path.open("w", encoding="utf-8", newline="\n") as stream:
        for item in source_manifest.get("items", []):
            if not isinstance(item, Mapping):
                raise ValueError("material item is not an object")
            item_id = f"{item['task_sha256'][:16]}-{item['sandbox_evidence_fingerprint'][:16]}"
            if item_id in seen_item_ids:
                raise ValueError(f"duplicate portable item identity: {item_id}")
            seen_item_ids.add(item_id)
            source_root = Path(str(item["sandbox_root"]))
            destination_root = output / "environments" / item_id
            copied = {}
            for group in ("sandbox_artifacts_sha256", "sandbox_evidence_sha256"):
                values = item.get(group)
                if not isinstance(values, Mapping) or not values:
                    raise ValueError(f"material item lacks {group}")
                for relative, expected in values.items():
                    safe = _safe_relative(str(relative))
                    destination = destination_root / safe
                    if str(safe) in copied and copied[str(safe)] != str(expected):
                        raise ValueError(f"conflicting digest for {safe}")
                    _copy_verified(source_root / safe, destination, str(expected))
                    copied[str(safe)] = str(expected)
            rollout_source = source_root / "live_rollout.json"
            rollout = json.loads(rollout_source.read_text(encoding="utf-8"))
            if digest_json(rollout) != item.get("rollout_sha256"):
                raise ValueError(f"rollout changed for {item_id}")
            rollout_destination = destination_root / "live_rollout.json"
            rollout_destination.write_text(
                json.dumps(rollout, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8", newline="\n",
            )
            copied["live_rollout.json"] = file_sha256(rollout_destination)
            records = _transition_records(item_id, item, rollout)
            for record in records:
                stream.write(json.dumps(
                    record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ) + "\n")
            transition_count += len(records)
            exported_items.append({
                "item_id": item_id,
                "category": item["category"],
                "score": item["score"],
                "task_sha256": item["task_sha256"],
                "environment_path": f"environments/{item_id}",
                "episode_count": item["episode_count"],
                "transition_count": len(records),
                "files_sha256": copied,
            })

    files = {
        str(path.relative_to(output)): file_sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != BUNDLE_MANIFEST
    }
    manifest = {
        "version": "2.0",
        "kind": "portable_agentic_rl_training_materials",
        "source_dataset_sha256": source_manifest.get("dataset_sha256"),
        "items": exported_items,
        "item_count": len(exported_items),
        "transition_count": transition_count,
        "files_sha256": files,
    }
    manifest["bundle_sha256"] = digest_json(manifest)
    (output / BUNDLE_MANIFEST).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    return verify_bundle(output)


def verify_bundle(root: Path) -> dict[str, Any]:
    failures = []
    try:
        manifest = json.loads((root / BUNDLE_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"verified": False, "failed_gates": ["bundle_manifest"], "message": str(exc)}
    unsigned = {key: value for key, value in manifest.items() if key != "bundle_sha256"}
    if manifest.get("bundle_sha256") != digest_json(unsigned):
        failures.append("bundle_digest")
    if (
        manifest.get("version") != "2.0"
        or manifest.get("kind") != "portable_agentic_rl_training_materials"
        or not isinstance(manifest.get("source_dataset_sha256"), str)
        or len(manifest["source_dataset_sha256"]) != 64
    ):
        failures.append("bundle_schema")
    expected = manifest.get("files_sha256")
    expected_files = dict(expected) if isinstance(expected, Mapping) else {}
    actual = {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != BUNDLE_MANIFEST
    }
    if not isinstance(expected, Mapping) or expected_files != actual:
        failures.append("bundle_files")
    transition_path = root / TRANSITIONS_FILE
    records = []
    try:
        records = [json.loads(line) for line in transition_path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError):
        failures.append("transition_jsonl")
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        failures.append("bundle_items")
        items = []
    expected_records = []
    seen_item_ids = set()
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            failures.append("bundle_items")
            continue
        item_id = item.get("item_id")
        if (
            not isinstance(item_id, str)
            or re.fullmatch(r"[0-9a-f]{16}-[0-9a-f]{16}", item_id) is None
            or item_id in seen_item_ids
        ):
            failures.append("item_identity")
            continue
        seen_item_ids.add(item_id)
        try:
            environment = _safe_relative(str(item.get("environment_path", "")))
        except ValueError:
            failures.append("item_path")
            continue
        if environment != Path("environments") / item_id:
            failures.append("item_path")
        item_files = item.get("files_sha256")
        if not isinstance(item_files, Mapping) or not item_files:
            failures.append("item_files")
            continue
        for relative, digest in item_files.items():
            try:
                safe = _safe_relative(str(relative))
            except ValueError:
                failures.append("item_files")
                continue
            if expected_files.get(str(environment / safe)) != digest:
                failures.append("item_files")
        task_path = root / environment / "task.json"
        if not task_path.is_file() or file_sha256(task_path) != item.get("task_sha256"):
            failures.append("item_task")
        rollout_path = root / environment / "live_rollout.json"
        try:
            rollout = json.loads(rollout_path.read_text(encoding="utf-8"))
            projected = _transition_records(item_id, item, rollout)
        except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError):
            failures.append("transition_schema")
            continue
        if item.get("episode_count") != len(rollout.get("episodes", [])):
            failures.append("item_episode_count")
        if item.get("transition_count") != len(projected):
            failures.append("item_transition_count")
        expected_records.extend(projected)
    if records != expected_records:
        failures.append("transition_projection")
    if (
        len(records) != manifest.get("transition_count")
        or len(items) != manifest.get("item_count")
    ):
        failures.append("bundle_counts")
    if manifest.get("item_count", 0) <= 0 or manifest.get("transition_count", 0) <= 0:
        failures.append("empty_bundle")
    return {
        "verified": not failures,
        "bundle_sha256": manifest.get("bundle_sha256"),
        "source_dataset_sha256": manifest.get("source_dataset_sha256"),
        "items": manifest.get("item_count", 0),
        "transitions": len(records),
        "failed_gates": sorted(set(failures)),
    }


def export_bundle(
    certification: Mapping[str, Any], output: Path, project: Path
) -> dict[str, Any]:
    """Stage the complete bundle and publish it with one directory rename."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if any(output.iterdir()):
            raise FileExistsError(f"bundle output is not empty: {output}")
        output.rmdir()
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.", dir=output.parent
    ) as temporary:
        staged = Path(temporary) / "bundle"
        report = _export_bundle_uncommitted(certification, staged, project)
        if report.get("verified") is not True:
            raise ValueError(f"staged bundle verification failed: {report['failed_gates']}")
        staged.replace(output)
    return verify_bundle(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certification", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        report = verify_bundle(args.output.resolve())
    else:
        certification = json.loads(args.certification.read_text(encoding="utf-8"))
        report = export_bundle(certification, args.output.resolve(), args.project.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

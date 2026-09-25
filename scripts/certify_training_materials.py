#!/usr/bin/env python3
"""Certify an EnvFactory holdout as production-prepared Agentic-RL material.

This is a pre-training certification.  It verifies task/sandbox yield, runtime
integrity, reward counterfactuals and trajectory collectability; it deliberately
does not claim that an RL algorithm will improve a policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


Z_95 = 1.959963984540054
USER_OUTCOMES = {
    "goal_satisfied", "information_required", "user_correction",
    "user_rejection", "user_acceptance", "agent_off_topic",
    "agent_premature_completion", "unrecognized",
}
NEGATIVE_COUNTERFACTUALS = (
    "goal_failure", "no_tools", "noise_selection", "reordered_tools",
)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def digest_json(value: Any) -> str:
    rendered = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def wilson_lower(successes: int, total: int, *, z: float = Z_95) -> float:
    """Return the lower bound of a two-sided Wilson score interval."""
    if total <= 0:
        return 0.0
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = proportion + z * z / (2 * total)
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    )
    return max(0.0, (centre - margin) / denominator)


def _normal_text(value: Any) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"\d+(?:\.\d+)?", "#", text)
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text)


def _shingles(text: str, size: int = 3) -> set[str]:
    if len(text) <= size:
        return {text} if text else set()
    return {text[index:index + size] for index in range(len(text) - size + 1)}


def near_duplicate_rate(tasks: Iterable[Mapping[str, Any]], threshold: float = 0.9) -> float:
    """Count later tasks that are near duplicates of an earlier task."""
    vectors: list[set[str]] = []
    duplicates = 0
    for task in tasks:
        vector = _shingles(_normal_text(task.get("task")))
        if vector and any(
            len(vector & previous) / len(vector | previous) >= threshold
            for previous in vectors if previous
        ):
            duplicates += 1
        vectors.append(vector)
    return duplicates / len(vectors) if vectors else 0.0


def _artifact(result: Mapping[str, Any], name: str) -> dict[str, Any]:
    root = Path(str(result.get("output", "")))
    path = root / name
    try:
        value = load(path)
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _counterfactual_counts(reports: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    positive_total = positive_errors = negative_total = negative_errors = 0
    for report in reports:
        cases = report.get("evidence", {}).get("counterfactuals", {})
        if not isinstance(cases, Mapping):
            continue
        success = cases.get("goal_success")
        if isinstance(success, Mapping):
            positive_total += 1
            reward = success.get("reward")
            if not isinstance(reward, (int, float)) or reward < 0.6:
                positive_errors += 1
        for name, case in cases.items():
            if name == "goal_success" or not isinstance(case, Mapping):
                continue
            if not (
                name in NEGATIVE_COUNTERFACTUALS
                or name.startswith(("corrupted_arguments_", "skipped_tool_"))
            ):
                continue
            negative_total += 1
            reward = case.get("reward")
            limit = 0.0 if name == "noise_selection" else 0.2
            if isinstance(reward, (int, float)) and reward > limit:
                negative_errors += 1
    return {
        "positive_total": positive_total,
        "positive_errors": positive_errors,
        "negative_total": negative_total,
        "negative_errors": negative_errors,
    }


def complete_episode(episode: Mapping[str, Any]) -> bool:
    """Require the fields needed to reconstruct a future RL transition stream."""
    return (
        isinstance(episode.get("seed"), int)
        and isinstance(episode.get("agent_success"), bool)
        and isinstance(episode.get("termination"), str)
        and isinstance(episode.get("initial_reward"), (int, float))
        and isinstance(episode.get("final_reward"), (int, float))
        and isinstance(episode.get("trajectory"), list)
        and bool(episode["trajectory"])
        and all(
            isinstance(step, Mapping)
            and isinstance(step.get("method"), str)
            and isinstance(step.get("path"), str)
            and isinstance(step.get("status"), int)
            and "result" in step
            for step in episode["trajectory"]
        )
        and isinstance(episode.get("replay"), Mapping)
        and isinstance(episode.get("initial_state"), Mapping)
        and isinstance(episode.get("final_state"), Mapping)
        and isinstance(episode.get("usage"), list)
    )


def valid_user_turn(step: Mapping[str, Any]) -> bool:
    result = step.get("result")
    return (
        step.get("status") == 200
        and isinstance(result, Mapping)
        and isinstance(result.get("user_query"), str)
        and bool(result["user_query"].strip())
        and isinstance(result.get("should_end"), bool)
        and result.get("outcome_category") in USER_OUTCOMES
        and isinstance(result.get("reason_code"), str)
    )


def certify(history: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    holdout = history.get("holdout")
    jobs = holdout.get("jobs", []) if isinstance(holdout, Mapping) else []
    results = [job.get("result", {}) for job in jobs if isinstance(job, Mapping)]
    total = len(results)
    generated = [item for item in results if Path(str(item.get("task_path", ""))).is_file()]
    task_good = [
        item for item in results
        if item.get("task_score", {}).get("eligible") is True
        and item.get("task_score", {}).get("score", 0) >= policy["score_threshold"]
    ]
    built = [
        item for item in task_good
        if item.get("sandbox_score", {}).get("passed") is True
        and item.get("sandbox_score", {}).get("score", 0) >= policy["score_threshold"]
    ]
    qualified = [
        item for item in results
        if item.get("passed") is True and item.get("score", 0) >= policy["score_threshold"]
    ]
    rates = {
        "generation_completion": len(generated) / total if total else 0.0,
        "task_good_yield": len(task_good) / total if total else 0.0,
        "task_good_yield_ci95_lower": wilson_lower(len(task_good), total),
        "sandbox_build_yield": len(built) / len(task_good) if task_good else 0.0,
        "sandbox_build_yield_ci95_lower": wilson_lower(len(built), len(task_good)),
        "end_to_end_rate": len(qualified) / total if total else 0.0,
        "end_to_end_ci95_lower": wilson_lower(len(qualified), total),
    }

    category_counts: dict[str, dict[str, Any]] = {}
    categories = sorted({str(item.get("category", "unknown")) for item in results})
    for category in categories:
        members = [item for item in results if str(item.get("category", "unknown")) == category]
        passed = sum(item in qualified for item in members)
        category_counts[category] = {
            "requested": len(members), "qualified": passed,
            "rate": passed / len(members) if members else 0.0,
        }

    task_documents = []
    task_digests = []
    for item in generated:
        path = Path(str(item["task_path"]))
        try:
            raw = path.read_bytes()
            document = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(document, dict):
            task_documents.append(document)
            task_digests.append(hashlib.sha256(raw).hexdigest())
    exact_duplicate_rate = (
        (len(task_digests) - len(set(task_digests))) / len(task_digests)
        if task_digests else 1.0
    )

    live_reports = [
        item.get("live_rollout") for item in built
        if isinstance(item.get("live_rollout"), Mapping)
    ]
    episodes = [
        episode for report in live_reports
        for episode in report.get("episodes", []) if isinstance(episode, Mapping)
    ]
    environment_errors = sum(bool(item.get("issues")) for item in episodes)
    fallbacks = sum(
        "runtime_llm_fallback" in item.get("issues", []) for item in episodes
    )
    successes = sum(item.get("agent_success") is True for item in episodes)
    complete_episodes = sum(complete_episode(item) for item in episodes)
    user_turns = [
        step for episode in episodes for step in episode.get("trajectory", [])
        if isinstance(step, Mapping) and step.get("path") == "/v1/user_simulator"
    ]
    valid_user_turns = sum(valid_user_turn(step) for step in user_turns)
    rollout_coverage = all(
        len(item.get("live_rollout", {}).get("episodes", []))
        >= policy["min_episodes_per_qualified_sandbox"]
        for item in qualified
    ) and bool(qualified)

    readiness_reports = [_artifact(item, "training_readiness.json") for item in qualified]
    agentic_reports = [_artifact(item, "agentic_training_value.json") for item in qualified]
    runtime_integrity = bool(readiness_reports) and all(
        report.get("training_ready") is True
        and report.get("evidence", {}).get("determinism") is True
        and report.get("evidence", {}).get("observation_scan", {}).get("forbidden_paths") == []
        and all(
            report.get("evidence", {}).get("runtime_state", {}).get(key) is True
            for key in ("reset_reproducible", "episode_isolation", "replay_consistent")
        )
        for report in readiness_reports
    )
    tool_and_reward_integrity = bool(agentic_reports) and all(
        report.get("curriculum_training_ready") is True for report in agentic_reports
    )
    counterfactuals = _counterfactual_counts(agentic_reports)
    false_positive_rate = (
        counterfactuals["negative_errors"] / counterfactuals["negative_total"]
        if counterfactuals["negative_total"] else 1.0
    )
    false_negative_rate = (
        counterfactuals["positive_errors"] / counterfactuals["positive_total"]
        if counterfactuals["positive_total"] else 1.0
    )
    material_items = []
    for item in qualified:
        task_path = Path(str(item.get("task_path", "")))
        sandbox_score = item.get("sandbox_score", {})
        fingerprint = sandbox_score.get("evidence_fingerprint")
        live = item.get("live_rollout")
        if not task_path.is_file() or not isinstance(fingerprint, str) or not fingerprint:
            continue
        if not isinstance(live, Mapping):
            continue
        item_episodes = live.get("episodes", [])
        material_items.append({
            "task_path": str(task_path.resolve()),
            "task_sha256": hashlib.sha256(task_path.read_bytes()).hexdigest(),
            "sandbox_root": str(Path(str(item.get("output", ""))).resolve()),
            "sandbox_evidence_fingerprint": fingerprint,
            "category": str(item.get("category", "unknown")),
            "score": item.get("score"),
            "rollout_sha256": digest_json(live),
            "episode_count": len(item_episodes) if isinstance(item_episodes, list) else 0,
            "successful_episodes": sum(
                episode.get("agent_success") is True
                for episode in item_episodes if isinstance(episode, Mapping)
            ) if isinstance(item_episodes, list) else 0,
        })
    material_fingerprints = [
        item["sandbox_evidence_fingerprint"] for item in material_items
    ]
    material_manifest = {
        "version": "1.0",
        "kind": "agentic_rl_pretraining_materials",
        "items": material_items,
    }
    material_manifest["dataset_sha256"] = digest_json(material_manifest)

    measurements = {
        "requested_tasks": total,
        "generated_tasks": len(generated),
        "parseable_tasks": len(task_documents),
        "task_qualified": len(task_good),
        "sandbox_built": len(built),
        "training_ready": len(qualified),
        **rates,
        "by_category": category_counts,
        "exact_duplicate_rate": exact_duplicate_rate,
        "near_duplicate_rate": near_duplicate_rate(task_documents),
        "live_sandboxes": len(live_reports),
        "episodes": len(episodes),
        "successful_episodes": successes,
        "failed_episodes": len(episodes) - successes,
        "complete_episodes": complete_episodes,
        "agent_success_rate": successes / len(episodes) if episodes else 0.0,
        "environment_errors": environment_errors,
        "environment_error_rate": environment_errors / len(episodes) if episodes else 1.0,
        "llm_fallbacks": fallbacks,
        "rollout_coverage": rollout_coverage,
        "trajectory_schema_complete": bool(episodes) and complete_episodes == len(episodes),
        "user_simulator_calls": len(user_turns),
        "user_simulator_valid_calls": valid_user_turns,
        "user_simulator_protocol_rate": (
            valid_user_turns / len(user_turns) if user_turns else 0.0
        ),
        "runtime_integrity": runtime_integrity,
        "tool_and_reward_integrity": tool_and_reward_integrity,
        "reward_counterfactuals": counterfactuals,
        "reward_false_positive_rate": false_positive_rate,
        "reward_false_negative_rate": false_negative_rate,
        "fresh_holdout_verified": (
            holdout.get("summary", {}).get("fresh_tasks_verified") is True
            if isinstance(holdout, Mapping) else False
        ),
        "material_manifest_items": len(material_items),
    }

    gates = {
        "materialized_sample_size": (
            len(task_documents) >= policy["min_tasks"]
            and len(task_documents) == len(generated)
        ),
        "fresh_holdout": measurements["fresh_holdout_verified"],
        "task_good_yield": (
            rates["task_good_yield"] >= policy["min_task_yield"]
            and rates["task_good_yield_ci95_lower"] >= policy["min_task_yield_ci95_lower"]
        ),
        "sandbox_build_yield": (
            rates["sandbox_build_yield"] >= policy["min_build_yield"]
            and rates["sandbox_build_yield_ci95_lower"] >= policy["min_build_yield_ci95_lower"]
        ),
        "end_to_end_yield": (
            rates["end_to_end_rate"] >= policy["min_end_to_end_rate"]
            and rates["end_to_end_ci95_lower"] >= policy["min_end_to_end_ci95_lower"]
        ),
        "category_floor": bool(category_counts) and all(
            item["rate"] >= policy["min_category_rate"] for item in category_counts.values()
        ),
        "exact_deduplication": exact_duplicate_rate == 0,
        "near_deduplication": measurements["near_duplicate_rate"] <= policy["max_near_duplicate_rate"],
        "rollout_coverage": rollout_coverage and len(episodes) >= policy["min_total_episodes"],
        "trajectory_schema": measurements["trajectory_schema_complete"],
        "user_simulator_protocol": (
            bool(user_turns)
            and measurements["user_simulator_protocol_rate"]
            >= policy["min_user_simulator_protocol_rate"]
        ),
        "trajectory_diversity": successes > 0 and successes < len(episodes),
        "environment_integrity": (
            measurements["environment_error_rate"] <= policy["max_environment_error_rate"]
            and fallbacks == 0
        ),
        "runtime_state_integrity": runtime_integrity,
        "tool_and_reward_integrity": tool_and_reward_integrity,
        "material_identity": (
            len(material_items) == len(qualified)
            and len(material_fingerprints) == len(set(material_fingerprints))
        ),
        "reward_false_positive_rate": false_positive_rate <= policy["max_reward_false_positive_rate"],
        "reward_false_negative_rate": false_negative_rate <= policy["max_reward_false_negative_rate"],
    }
    failed = [name for name, passed in gates.items() if not passed]
    return {
        "certification": "production_prepared_for_agentic_rl",
        "scope": "pre_training_material_readiness",
        "certified": not failed,
        "does_not_certify": [
            "rl_training_convergence", "post_training_policy_improvement", "cross_model_generalization",
        ],
        "policy": dict(policy),
        "measurements": measurements,
        "gates": gates,
        "failed_gates": failed,
        "materials_manifest": material_manifest,
    }


def default_policy() -> dict[str, Any]:
    return {
        "score_threshold": 8.0,
        "min_tasks": 300,
        "min_task_yield": 0.90,
        "min_task_yield_ci95_lower": 0.85,
        "min_build_yield": 0.90,
        "min_build_yield_ci95_lower": 0.85,
        "min_end_to_end_rate": 0.85,
        "min_end_to_end_ci95_lower": 0.80,
        "min_category_rate": 0.75,
        "max_near_duplicate_rate": 0.05,
        "min_episodes_per_qualified_sandbox": 10,
        "min_total_episodes": 2500,
        "max_environment_error_rate": 0.001,
        "max_reward_false_positive_rate": 0.005,
        "max_reward_false_negative_rate": 0.02,
        "min_user_simulator_protocol_rate": 0.995,
    }


def attach_artifact_verification(
    report: dict[str, Any], verification: Mapping[str, Any]
) -> dict[str, Any]:
    """Make post-score artifact immutability part of the certification claim."""
    report["material_verification"] = dict(verification)
    report["gates"]["material_artifacts_immutable"] = verification.get("verified") is True
    report["failed_gates"] = [
        name for name, passed in report["gates"].items() if not passed
    ]
    report["certified"] = not report["failed_gates"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path, help="loop experiment history.json")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = certify(load(args.history.resolve()), default_policy())
    from verify_training_materials import verify
    report = attach_artifact_verification(
        report, verify(report["materials_manifest"], args.project.resolve())
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    output = args.output or args.history.with_name("production_readiness.json")
    output.write_text(rendered, encoding="utf-8")
    manifest_path = output.with_name("training_materials_manifest.json")
    manifest_path.write_text(
        json.dumps(report["materials_manifest"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(rendered, end="")
    return 0 if report["certified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Certify an EnvFactory holdout as production-prepared Agentic-RL material.

This is a pre-training certification.  It verifies task/sandbox yield, runtime
integrity, reward counterfactuals and trajectory collectability; it deliberately
does not claim that an RL algorithm will improve a policy.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from env_factory.material_artifacts import (
    digest_json,
    evidence_artifact_digests,
    portable_artifact_digest,
    portable_artifact_digests,
)
from env_factory.trajectory_schema import complete_episode
from env_factory.data_governance import (
    FORBIDDEN_OUTBOUND,
    OUTBOUND_SURFACES,
    SYNTHETIC_ORIGIN,
    sandbox_payloads,
    scan_payloads,
)


Z_95 = 1.959963984540054
USER_OUTCOMES = {
    "goal_satisfied", "information_required", "user_correction",
    "user_rejection", "user_acceptance", "agent_off_topic",
    "agent_premature_completion", "unrecognized",
}
NEGATIVE_COUNTERFACTUALS = (
    "goal_failure", "no_tools", "noise_selection", "reordered_tools",
)
REQUIRED_OUTBOUND_SURFACES = {
    name: set(values) for name, values in OUTBOUND_SURFACES.items()
}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_digests(root: Path) -> dict[str, str]:
    """Backward-compatible public helper used by existing report consumers."""
    return portable_artifact_digests(root)


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


def valid_data_governance(
    report: Mapping[str, Any], sandbox_root: Path, task_path: Path
) -> bool:
    """Independently recompute claims over the frozen task and fixture payloads."""
    origin = report.get("data_origin", {})
    providers = report.get("providers", {})
    surfaces = report.get("outbound_surfaces", {})
    try:
        task = load(task_path)
        declared = task.get("artifacts", {}).get("data_manifest", {}).get(
            "data_governance", {}
        )
        rescanned = scan_payloads(sandbox_payloads(sandbox_root, task))
    except (OSError, json.JSONDecodeError, TypeError, AttributeError):
        return False
    return (
        report.get("version") == "1.0"
        and report.get("eligible_for_external_model_processing") is True
        and isinstance(origin, Mapping)
        and dict(origin) == SYNTHETIC_ORIGIN
        and isinstance(declared, Mapping)
        and dict(declared) == SYNTHETIC_ORIGIN
        and report.get("credential_findings") == rescanned["credential_findings"] == []
        and report.get("pii_findings") == rescanned["pii_findings"]
        and isinstance(providers, Mapping)
        and all(
            isinstance(providers.get(name), Mapping)
            and bool(providers[name].get("host"))
            and bool(providers[name].get("model"))
            and isinstance(providers[name].get("identity_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", providers[name]["identity_sha256"])
            for name in ("agent", "user_simulator_and_reward")
        )
        and isinstance(surfaces, Mapping)
        and all(
            required <= set(surfaces.get(name, []))
            for name, required in REQUIRED_OUTBOUND_SURFACES.items()
        )
        and set(FORBIDDEN_OUTBOUND) <= set(report.get("forbidden_outbound", []))
    )


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


def valid_user_turn(step: Mapping[str, Any]) -> bool:
    result = step.get("result")
    return (
        step.get("status") == 200
        and isinstance(result, Mapping)
        and isinstance(result.get("user_query"), str)
        and bool(result["user_query"].strip())
        and isinstance(result.get("should_end"), bool)
        and result.get("match_status") in {"matched", "unmatched", "ambiguous"}
        and result.get("outcome_category") in USER_OUTCOMES
        and isinstance(result.get("reason_code"), str)
        and (
            not result.get("should_end")
            or isinstance(result.get("termination_reason"), str)
        )
    )


def category_mix(results: Iterable[Mapping[str, Any]], policy: Mapping[str, Any]) -> dict[str, Any]:
    values = list(results)
    total = len(values)
    counts = Counter(str(item.get("category", "unknown")) for item in values)
    shares = {name: count / total if total else 0.0 for name, count in counts.items()}
    minimums = policy.get("min_category_shares", {})
    passed = total > 0 and all(
        shares.get(str(name), 0.0) >= float(minimum)
        for name, minimum in minimums.items()
    )
    maximums = policy.get("max_category_shares", {})
    passed = passed and all(
        shares.get(str(name), 0.0) <= float(maximum)
        for name, maximum in maximums.items()
    )
    return {"counts": dict(counts), "shares": shares, "passed": passed}


def valid_rollout_provenance(item: Mapping[str, Any]) -> bool:
    live = item.get("live_rollout")
    task_path = Path(str(item.get("task_path", "")))
    root = Path(str(item.get("output", "")))
    if not isinstance(live, Mapping) or not task_path.is_file() or not root.is_dir():
        return False
    try:
        task_sha = hashlib.sha256(task_path.read_bytes()).hexdigest()
        artifact_sha = portable_artifact_digest(root)
    except OSError:
        return False
    return (
        live.get("schema_version") == "2.0"
        and live.get("task_sha256") == task_sha
        and live.get("sandbox_artifacts_digest") == artifact_sha
        and isinstance(live.get("agent_model"), str) and bool(live["agent_model"])
        and isinstance(live.get("runtime_model"), str) and bool(live["runtime_model"])
        and all(
            isinstance(live.get(name), str) and len(live[name]) == 64
            for name in ("agent_provider_sha256", "runtime_provider_sha256")
        )
    )


def fresh_holdout_evidence(
    holdouts: Iterable[Mapping[str, Any]], policy: Mapping[str, Any]
) -> dict[str, Any]:
    batches = list(holdouts)
    batch_ids = [item.get("holdout_batch") for item in batches]
    seen_seeds: set[int] = set()
    seen_tasks: set[str] = set()
    details = []
    all_fresh = (
        len(batches) >= policy["min_holdout_batches"]
        and all(isinstance(item, int) for item in batch_ids)
        and len(set(batch_ids)) == len(batch_ids)
    )
    for batch in batches:
        results = [
            job.get("result", {}) for job in batch.get("jobs", [])
            if isinstance(job, Mapping)
        ]
        seeds = [item.get("sample_seed") for item in results]
        digests = []
        for item in results:
            path = Path(str(item.get("task_path", "")))
            try:
                digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
            except OSError:
                continue
        batch_fresh = (
            len(results) >= policy["min_tasks"]
            and len(digests) >= policy["min_tasks"]
            and all(isinstance(seed, int) for seed in seeds)
            and len(set(seeds)) == len(seeds)
            and seen_seeds.isdisjoint(seeds)
            and len(set(digests)) == len(digests)
            and seen_tasks.isdisjoint(digests)
            and batch.get("summary", {}).get("fresh_tasks_verified") is True
        )
        details.append({
            "batch": batch.get("holdout_batch"),
            "requests": len(results),
            "materialized": len(digests),
            "fresh": batch_fresh,
        })
        all_fresh = all_fresh and batch_fresh
        seen_seeds.update(seed for seed in seeds if isinstance(seed, int))
        seen_tasks.update(digests)
    return {
        "verified": all_fresh,
        "unique_seeds": len(seen_seeds),
        "unique_tasks": len(seen_tasks),
        "batches": details,
    }


def certify(history: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    raw_holdouts = history.get("holdouts")
    if isinstance(raw_holdouts, list) and raw_holdouts:
        holdouts = [item for item in raw_holdouts if isinstance(item, Mapping)]
    else:
        holdout = history.get("holdout")
        holdouts = [holdout] if isinstance(holdout, Mapping) else []
    jobs = [job for holdout in holdouts for job in holdout.get("jobs", [])]
    results = [job.get("result", {}) for job in jobs if isinstance(job, Mapping)]
    total = len(results)
    holdout_freshness = fresh_holdout_evidence(holdouts, policy)
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
    user_outcomes = Counter(
        str(step.get("result", {}).get("outcome_category"))
        for step in user_turns if valid_user_turn(step)
    )
    interactive_outcomes = {
        "information_required", "user_correction", "user_rejection",
        "agent_off_topic", "agent_premature_completion", "unrecognized",
    }
    rollout_coverage = all(
        len(item.get("live_rollout", {}).get("episodes", []))
        >= policy["min_episodes_per_qualified_sandbox"]
        for item in qualified
    ) and bool(qualified)
    rollout_provenance = bool(qualified) and all(
        valid_rollout_provenance(item) for item in qualified
    )

    readiness_reports = [_artifact(item, "training_readiness.json") for item in qualified]
    agentic_reports = [
        _artifact(item, "agentic_training_value_live.json") for item in qualified
    ]
    governance_reports = [_artifact(item, "data_governance.json") for item in qualified]
    governed_materials = sum(
        valid_data_governance(
            report,
            Path(str(item.get("output", ""))),
            Path(str(item.get("task_path", ""))),
        )
        for item, report in zip(qualified, governance_reports)
    )
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
        report.get("curriculum_training_ready") is True
        and report.get("validation_mode") == "live_evaluator"
        for report in agentic_reports
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
    batch_measurements = []
    for batch in holdouts:
        batch_results = [
            job.get("result", {}) for job in batch.get("jobs", [])
            if isinstance(job, Mapping)
        ]
        batch_total = len(batch_results)
        batch_generated = sum(
            Path(str(item.get("task_path", ""))).is_file() for item in batch_results
        )
        batch_task_good = [
            item for item in batch_results
            if item.get("task_score", {}).get("eligible") is True
            and item.get("task_score", {}).get("score", 0) >= policy["score_threshold"]
        ]
        batch_built = [
            item for item in batch_task_good
            if item.get("sandbox_score", {}).get("passed") is True
            and item.get("sandbox_score", {}).get("score", 0) >= policy["score_threshold"]
        ]
        batch_qualified = [
            item for item in batch_results
            if item.get("passed") is True
            and item.get("score", 0) >= policy["score_threshold"]
        ]
        batch_categories = {}
        for category in sorted({str(item.get("category", "unknown")) for item in batch_results}):
            members = [
                item for item in batch_results
                if str(item.get("category", "unknown")) == category
            ]
            count = sum(item in batch_qualified for item in members)
            batch_categories[category] = count / len(members) if members else 0.0
        task_rate = len(batch_task_good) / batch_total if batch_total else 0.0
        build_rate = len(batch_built) / len(batch_task_good) if batch_task_good else 0.0
        e2e_rate = len(batch_qualified) / batch_total if batch_total else 0.0
        batch_mix = category_mix(batch_results, policy)
        batch_passed = (
            batch_generated >= policy["min_tasks"]
            and batch.get("summary", {}).get("fresh_tasks_verified") is True
            and task_rate >= policy["min_task_yield"]
            and wilson_lower(len(batch_task_good), batch_total)
                >= policy["min_task_yield_ci95_lower"]
            and build_rate >= policy["min_build_yield"]
            and wilson_lower(len(batch_built), len(batch_task_good))
                >= policy["min_build_yield_ci95_lower"]
            and e2e_rate >= policy["min_end_to_end_rate"]
            and wilson_lower(len(batch_qualified), batch_total)
                >= policy["min_end_to_end_ci95_lower"]
            and bool(batch_categories)
            and all(rate >= policy["min_category_rate"] for rate in batch_categories.values())
            and batch_mix["passed"]
        )
        batch_measurements.append({
            "batch": batch.get("holdout_batch"),
            "requested": batch_total,
            "generated": batch_generated,
            "task_good_yield": task_rate,
            "task_good_yield_ci95_lower": wilson_lower(len(batch_task_good), batch_total),
            "sandbox_build_yield": build_rate,
            "sandbox_build_yield_ci95_lower": wilson_lower(len(batch_built), len(batch_task_good)),
            "end_to_end_rate": e2e_rate,
            "end_to_end_ci95_lower": wilson_lower(len(batch_qualified), batch_total),
            "by_category": batch_categories,
            "category_mix": batch_mix,
            "passed": batch_passed,
        })
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
            "sandbox_artifacts_sha256": artifact_digests(
                Path(str(item.get("output", "")))
            ),
            "sandbox_evidence_sha256": evidence_artifact_digests(
                Path(str(item.get("output", "")))
            ),
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
        "version": "2.0",
        "kind": "agentic_rl_pretraining_materials",
        "evaluator_source_digest": history.get("config", {}).get("source_digest"),
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
        "category_mix": category_mix(results, policy),
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
        "rollout_provenance": rollout_provenance,
        "trajectory_schema_complete": bool(episodes) and complete_episodes == len(episodes),
        "user_simulator_calls": len(user_turns),
        "user_simulator_valid_calls": valid_user_turns,
        "user_simulator_protocol_rate": (
            valid_user_turns / len(user_turns) if user_turns else 0.0
        ),
        "user_simulator_outcomes": dict(user_outcomes),
        "runtime_integrity": runtime_integrity,
        "tool_and_reward_integrity": tool_and_reward_integrity,
        "data_governance": {
            "reports": len(governance_reports),
            "verified": governed_materials,
            "all_verified": (
                bool(qualified)
                and len(governance_reports) == len(qualified)
                and governed_materials == len(qualified)
            ),
        },
        "reward_counterfactuals": counterfactuals,
        "reward_false_positive_rate": false_positive_rate,
        "reward_false_negative_rate": false_negative_rate,
        "fresh_holdout_verified": holdout_freshness["verified"],
        "fresh_holdout_evidence": holdout_freshness,
        "holdout_batches": batch_measurements,
        "material_manifest_items": len(material_items),
    }

    gates = {
        "materialized_sample_size": (
            len(task_documents) >= policy["min_tasks"] * policy["min_holdout_batches"]
            and len(task_documents) == len(generated)
        ),
        "independent_holdout_batches": (
            len(batch_measurements) >= policy["min_holdout_batches"]
            and all(item["passed"] for item in batch_measurements)
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
        "category_mix": measurements["category_mix"]["passed"],
        "exact_deduplication": exact_duplicate_rate == 0,
        "near_deduplication": measurements["near_duplicate_rate"] <= policy["max_near_duplicate_rate"],
        "rollout_coverage": rollout_coverage and len(episodes) >= policy["min_total_episodes"],
        "rollout_provenance": rollout_provenance,
        "trajectory_schema": measurements["trajectory_schema_complete"],
        "user_simulator_protocol": (
            bool(user_turns)
            and measurements["user_simulator_protocol_rate"]
            >= policy["min_user_simulator_protocol_rate"]
        ),
        "user_simulator_outcome_coverage": (
            len(user_outcomes) >= policy["min_user_outcome_categories"]
            and bool(set(user_outcomes) & {"goal_satisfied", "user_acceptance"})
            and bool(set(user_outcomes) & interactive_outcomes)
        ),
        "trajectory_diversity": successes > 0 and successes < len(episodes),
        "environment_integrity": (
            measurements["environment_error_rate"] <= policy["max_environment_error_rate"]
            and fallbacks == 0
        ),
        "runtime_state_integrity": runtime_integrity,
        "tool_and_reward_integrity": tool_and_reward_integrity,
        "data_governance": measurements["data_governance"]["all_verified"],
        "material_identity": (
            len(material_items) == len(qualified)
            and len(material_fingerprints) == len(set(material_fingerprints))
            and isinstance(material_manifest.get("evaluator_source_digest"), str)
            and bool(material_manifest["evaluator_source_digest"])
            and all(item.get("sandbox_artifacts_sha256") for item in material_items)
            and all(item.get("sandbox_evidence_sha256") for item in material_items)
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
        "min_holdout_batches": 3,
        "min_task_yield": 0.90,
        "min_task_yield_ci95_lower": 0.85,
        "min_build_yield": 0.90,
        "min_build_yield_ci95_lower": 0.85,
        "min_end_to_end_rate": 0.85,
        "min_end_to_end_ci95_lower": 0.80,
        "min_category_rate": 0.75,
        "max_near_duplicate_rate": 0.05,
        "min_episodes_per_qualified_sandbox": 10,
        "min_total_episodes": 7500,
        "max_environment_error_rate": 0.001,
        "max_reward_false_positive_rate": 0.005,
        "max_reward_false_negative_rate": 0.02,
        "min_user_simulator_protocol_rate": 0.995,
        "min_user_outcome_categories": 3,
        "min_category_shares": {
            "direct_response": 0.10,
            "simple_agentic": 0.20,
            "multi_step_agentic": 0.35,
        },
        "max_category_shares": {"direct_response": 0.30},
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


def attach_bundle_verification(
    report: dict[str, Any], verification: Mapping[str, Any]
) -> dict[str, Any]:
    report["materials_bundle_verification"] = dict(verification)
    report["gates"]["portable_materials_bundle"] = (
        verification.get("verified") is True
        and verification.get("source_dataset_sha256")
        == report.get("materials_manifest", {}).get("dataset_sha256")
    )
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
    parser.add_argument("--bundle-output", type=Path)
    args = parser.parse_args()
    report = certify(load(args.history.resolve()), default_policy())
    from verify_training_materials import verify
    report = attach_artifact_verification(
        report, verify(report["materials_manifest"], args.project.resolve())
    )
    if report["certified"]:
        from export_training_materials import export_bundle, verify_bundle
        bundle_root = (
            args.bundle_output.resolve() if args.bundle_output
            else args.history.resolve().parent / "training_materials_bundle"
        )
        try:
            if bundle_root.is_dir() and any(bundle_root.iterdir()):
                bundle_verification = verify_bundle(bundle_root)
            else:
                bundle_verification = export_bundle(
                    report, bundle_root, args.project.resolve()
                )
        except Exception as exc:
            bundle_verification = {
                "verified": False,
                "failed_gates": ["bundle_export"],
                "error_type": type(exc).__name__,
            }
        report = attach_bundle_verification(report, bundle_verification)
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

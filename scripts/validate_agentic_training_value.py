#!/usr/bin/env python3
"""Deterministic gate for whether a sandbox teaches its declared tool policy."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping


PLACEHOLDERS = {"fixture-value", "example", "placeholder", "todo", "unknown", "test"}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def import_app(root: Path):
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("agentic_value_app", root / "app.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import generated app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_app()


def reward_of(run: Mapping[str, Any]) -> float | None:
    for event in reversed(run.get("history", [])):
        body = event.get("body") if isinstance(event, Mapping) else None
        if event.get("operation") == "reward" and isinstance(body, Mapping):
            reward = body.get("reward")
            if isinstance(reward, (int, float)) and not isinstance(reward, bool):
                return float(reward)
    return None


def has_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in PLACEHOLDERS or "fixture-value" in value.casefold()
    if isinstance(value, list):
        return any(has_placeholder(item) for item in value)
    if isinstance(value, Mapping):
        return any(has_placeholder(item) for item in value.values())
    return False


def empty_critical_collection(value: Any) -> bool:
    if isinstance(value, list):
        return not value or any(empty_critical_collection(item) for item in value)
    if isinstance(value, Mapping):
        return any(empty_critical_collection(item) for item in value.values())
    return False


def meaningful_result(body: Any) -> bool:
    if not isinstance(body, Mapping):
        return body not in (None, "", [], {})
    if isinstance(body.get("count"), int):
        return body["count"] > 0
    for key in ("records", "items", "results", "rows", "data"):
        if key in body:
            return isinstance(body[key], (list, Mapping)) and bool(body[key])
    ignored = {"status", "accepted", "request_id", "episode_id"}
    return any(key not in ignored and value not in (None, "", [], {}) for key, value in body.items())


def corrupt(value: Any) -> Any:
    if isinstance(value, str):
        return "__envfactory_unknown__"
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 987654
    if isinstance(value, list):
        return ["__envfactory_unknown__"]
    if isinstance(value, Mapping):
        changed = dict(value)
        if changed:
            key = next(iter(changed))
            changed[key] = corrupt(changed[key])
        return changed
    return "__envfactory_unknown__"


def scenario_without_assertions(scenario: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(scenario))
    result["assertions"] = []
    return result


def dependency_swap_positions(
    steps: list[dict[str, Any]], capability_dag: Mapping[str, Any]
) -> tuple[int, int] | None:
    """Return one actual producer/consumer pair to reverse.

    Multi-step routes may have independent prerequisite tools. Swapping two
    such producers is valid and must not be reported as an order-sensitivity
    failure. Only reverse a pair connected by a declared DAG edge.
    """
    positions: dict[str, int] = {}
    for index, step in enumerate(steps):
        if step.get("operation") == "tool_call" and isinstance(step.get("tool_name"), str):
            positions.setdefault(str(step["tool_name"]), index)
    for edge in capability_dag.get("edges", []):
        if not isinstance(edge, Mapping):
            continue
        source, target = edge.get("from_tool"), edge.get("to_tool")
        if source in positions and target in positions and positions[source] < positions[target]:
            return positions[source], positions[target]
    return None


def validate(root: Path) -> dict[str, Any]:
    task = load(root / "task.json")
    contract = task.get("training_contract") if isinstance(task.get("training_contract"), Mapping) else {}
    category = contract.get("category", task.get("training_category", "multi_step_agentic"))
    tool_required = category != "direct_response"
    dependency_required = bool(contract.get("dependency", {}).get("required", category == "multi_step_agentic"))
    scenarios = task.get("acceptance_contract", {}).get("executable_scenarios", [])
    by_kind = {
        item.get("kind"): item for item in scenarios
        if isinstance(item, Mapping) and isinstance(item.get("kind"), str)
    }
    failures: list[dict[str, Any]] = []
    evidence: dict[str, Any] = {"counterfactuals": {}}
    required_scenarios = contract.get("required_scenarios", ["goal_success", "goal_failure"])
    for kind in required_scenarios:
        if kind not in by_kind:
            failures.append({"gate": "scenario_coverage", "message": f"missing {kind}"})
    if failures:
        return {"curriculum_training_ready": False, "agentic_training_ready": False, "training_category": category, "agentic_eligible": tool_required, "failed_gates": ["scenario_coverage"], "evidence": evidence, "failures": failures}

    success_scenario = scenario_without_assertions(by_kind["goal_success"])
    tool_steps = [step for step in success_scenario.get("steps", []) if isinstance(step, Mapping) and step.get("operation") == "tool_call"]
    if tool_required and not tool_steps:
        failures.append({"gate": "agentic_path", "message": "goal_success has no tool calls"})
    if not tool_required and tool_steps:
        failures.append({"gate": "direct_response_path", "message": "direct_response goal_success must not call tools"})
    for index, step in enumerate(tool_steps):
        arguments = step.get("arguments")
        schema = next((item.get("function", {}).get("parameters", {}) for item in task.get("tools", [])
                       if item.get("function", {}).get("name") == step.get("tool_name")), {})
        if not isinstance(arguments, Mapping) or not set(schema.get("required", [])) <= set(arguments):
            failures.append({"gate": "semantic_fixture", "message": f"tool step {index} has no arguments"})
        elif has_placeholder(arguments):
            failures.append({"gate": "semantic_fixture", "message": f"tool step {index} uses placeholder arguments"})
        elif empty_critical_collection(arguments):
            failures.append({"gate": "semantic_fixture", "message": f"tool step {index} uses an empty collection"})

    os.environ.setdefault("SANDBOX_TRAINER_API_KEY", "envfactory-agentic-value-key")
    os.environ.setdefault("SANDBOX_EVALUATOR_MOCK", "1")
    app = import_app(root)
    from sandbox_runtime import AcceptanceScenarioRunner

    runner = AcceptanceScenarioRunner(
        app.handle,
        trainer_headers={"Authorization": f"Bearer {os.environ['SANDBOX_TRAINER_API_KEY']}"},
        business_snapshot=getattr(app, "business_snapshot", None),
        mutate_business_state=getattr(app, "mutate_business_state", None),
    )

    def execute(name: str, scenario: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        try:
            result = runner.run(scenario)
            evidence["counterfactuals"][name] = {"reward": reward_of(result), "status": "completed"}
            return result, None
        except Exception as exc:  # rejected counterfactuals are valid evidence
            evidence["counterfactuals"][name] = {"reward": None, "status": "rejected", "message": str(exc)}
            # Only an actual 4xx response is evidence of environment rejection.
            # A runner error, missing capture or server failure is not a proof.
            details = getattr(exc, "details", None)
            status = details.get("actual_status") if isinstance(details, Mapping) else None
            if not isinstance(status, int) or not 400 <= status < 500:
                failures.append({"gate": "counterfactual_execution", "message": f"{name}: {exc}"})
            return None, str(exc)

    success_run, success_error = execute("goal_success", success_scenario)
    failure_run, _ = execute("goal_failure", scenario_without_assertions(by_kind["goal_failure"]))
    success_reward = reward_of(success_run or {})
    failure_reward = reward_of(failure_run or {})
    if success_error or success_reward is None or success_reward < 0.6:
        failures.append({"gate": "success_trajectory", "message": "valid goal trajectory must produce reward >= 0.6", "reward": success_reward})
    if failure_reward is None or failure_reward > 0.2:
        failures.append({"gate": "failure_trajectory", "message": "failure trajectory must produce reward <= 0.2", "reward": failure_reward})

    if success_run and tool_required:
        tool_bodies = [item.get("body") for item in success_run.get("history", []) if item.get("operation") == "tool_call"]
        if not tool_bodies or not all(meaningful_result(body) for body in tool_bodies):
            failures.append({"gate": "meaningful_tool_results", "message": "every success-path tool call must return meaningful business evidence"})

    final_step = next((step for step in success_scenario.get("steps", []) if step.get("operation") == "agent_response"), None)
    if final_step and tool_required:
        no_tools = {
            "scenario_id": "agentic_no_tools",
            "steps": [
                {"operation": "reset", "body": {"episode_id": "agentic-no-tools", "seed": 1701}},
                copy.deepcopy(final_step),
                {"step_id": "reward", "operation": "reward"},
            ],
            "assertions": [],
        }
        no_tool_run, _ = execute("no_tools", no_tools)
        no_tool_reward = reward_of(no_tool_run or {})
        if no_tool_reward is None or no_tool_reward > 0.2:
            failures.append({"gate": "no_tool_reward_hacking", "message": "answering without tools must produce reward <= 0.2", "reward": no_tool_reward})

    for tool_index, _ in enumerate(tool_steps if tool_required else []):
        corrupted = copy.deepcopy(success_scenario)
        candidate_steps = [
            step for step in corrupted["steps"]
            if step.get("operation") == "tool_call"
        ]
        candidate = candidate_steps[tool_index]
        arguments = dict(candidate.get("arguments", {}))
        if arguments:
            key = next(iter(arguments))
            arguments[key] = corrupt(arguments[key])
            candidate["arguments"] = arguments
            name = f"corrupted_arguments_{tool_index + 1}"
            corrupted_run, corrupted_error = execute(name, corrupted)
            corrupted_reward = reward_of(corrupted_run or {})
            if corrupted_error is None and (corrupted_reward is None or corrupted_reward > 0.2):
                failures.append({
                    "gate": "argument_sensitivity",
                    "message": f"corrupting tool step {tool_index} must be rejected or reward <= 0.2",
                    "reward": corrupted_reward,
                })

    if dependency_required and len(tool_steps) >= 2:
        # Bind known baseline values before removing/reordering producers so
        # the counterfactual reaches the environment rather than failing in $ref.
        bound_success = copy.deepcopy(success_scenario)
        if success_run:
            for step in bound_success["steps"]:
                if step.get("operation") == "tool_call":
                    step["arguments"] = runner._resolve(step.get("arguments", {}), success_run.get("variables", {}))
        for tool_index in range(len(tool_steps)):
            skipped = copy.deepcopy(bound_success)
            positions = [
                index for index, step in enumerate(skipped["steps"])
                if step.get("operation") == "tool_call"
            ]
            del skipped["steps"][positions[tool_index]]
            skipped_run, skipped_error = execute(f"skipped_tool_{tool_index + 1}", skipped)
            skipped_reward = reward_of(skipped_run or {})
            if skipped_error is None and (skipped_reward is None or skipped_reward > 0.2):
                failures.append({
                    "gate": "step_necessity",
                    "message": f"skipping tool step {tool_index} must be rejected or reward <= 0.2",
                    "reward": skipped_reward,
                })
        reordered = copy.deepcopy(bound_success)
        dag = task.get("task_spec", {}).get("capability_dag", {})
        pair = dependency_swap_positions(reordered["steps"], dag if isinstance(dag, Mapping) else {})
        if pair is None:
            failures.append({
                "gate": "dependency_contract",
                "message": "dependency-required task has no ordered tool pair in capability_dag",
            })
        else:
            first_pos, second_pos = pair
            reordered["steps"][first_pos], reordered["steps"][second_pos] = (
                reordered["steps"][second_pos], reordered["steps"][first_pos]
            )
            reordered_run, reordered_error = execute("reordered_tools", reordered)
            reordered_reward = reward_of(reordered_run or {})
            if reordered_error is None and (reordered_reward is None or reordered_reward > 0.2):
                failures.append({
                    "gate": "order_sensitivity",
                    "message": "reordering dependent tool calls must be rejected or reward <= 0.2",
                    "reward": reordered_reward,
                })

    if task.get("noise_tools") and "noise_selection" in by_kind:
        noise_run, _ = execute("noise_selection", scenario_without_assertions(by_kind["noise_selection"]))
        noise_reward = reward_of(noise_run or {})
        if noise_reward is None or noise_reward > 0:
            failures.append({"gate": "noise_selection", "message": "noise tool trajectory must produce reward <= 0", "reward": noise_reward})

    failed_gates = sorted({item["gate"] for item in failures})
    return {
        "curriculum_training_ready": not failures,
        "agentic_training_ready": not failures,
        "training_category": category,
        "sandbox_profile": contract.get("sandbox_profile", category),
        "agentic_eligible": tool_required,
        "hard_gates_passed": not failures,
        "failed_gates": failed_gates,
        "evidence": evidence,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="验证沙箱是否能训练真实 Agentic 行为")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    report = validate(root)
    output = args.output or root / "agentic_training_value.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["curriculum_training_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

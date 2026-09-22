#!/usr/bin/env python3
"""Fast hard gate for generated Agentic-RL environments."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


FORBIDDEN_PUBLIC_KEYS = {"ground_truth", "expected_tool_call", "hidden_state", "api_key", "trainer_token", "password"}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: canonical(item) for key, item in value.items() if key not in {"request_id", "timestamp", "duration_ms", "created_at", "trace_hash"}}
    if isinstance(value, list):
        return [canonical(item) for item in value]
    return value


def forbidden_paths(value: Any, path: str = "$") -> list[str]:
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}"
            if key.casefold() in FORBIDDEN_PUBLIC_KEYS or any(token in key.casefold() for token in ("secret", "credential")):
                found.append(child)
            found.extend(forbidden_paths(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(forbidden_paths(item, f"{path}[{index}]"))
    return found


def reward_from_run(run: dict[str, Any]) -> float | None:
    for item in reversed(run.get("history", [])):
        if item.get("operation") == "reward" and isinstance(item.get("body"), dict):
            value = item["body"].get("reward")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    return None


def import_app(root: Path):
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("generated_sandbox_app", root / "app.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import generated app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "create_app"):
        raise RuntimeError("generated app.py must expose create_app()")
    return module.create_app()


def validate(root: Path) -> dict[str, Any]:
    task = load(root / "task.json")
    scenarios = task.get("acceptance_contract", {}).get("executable_scenarios", [])
    failures: list[dict[str, Any]] = []
    evidence: dict[str, Any] = {}
    kinds = {item.get("kind") for item in scenarios if isinstance(item, dict)}
    for required in ("goal_success", "goal_failure"):
        if required not in kinds:
            failures.append({"gate": "scenario_coverage", "message": f"missing {required} scenario"})
    if task.get("noise_tools") and "noise_selection" not in kinds:
        failures.append({"gate": "noise_tool_behavior", "message": "missing noise_selection scenario"})
    app = import_app(root)
    from sandbox_runtime import AcceptanceScenarioRunner  # imported from generated sandbox
    token = os.environ.get("SANDBOX_TRAINER_API_KEY", "envfactory-readiness-key")
    os.environ["SANDBOX_TRAINER_API_KEY"] = token
    runner = AcceptanceScenarioRunner(app.handle, trainer_headers={"Authorization": f"Bearer {token}"})
    runs: dict[str, list[dict[str, Any]]] = {}
    for scenario in scenarios:
        if not isinstance(scenario, dict) or scenario.get("kind") not in {"goal_success", "goal_failure", "noise_selection", "counterfactual"}:
            continue
        try:
            result = runner.run(scenario)
            runs.setdefault(scenario["kind"], []).append(result)
        except Exception as exc:  # noqa: BLE001 - evidence gate
            failures.append({"gate": "scenario_execution", "scenario_id": scenario.get("scenario_id"), "message": str(exc)})
    success = reward_from_run(runs.get("goal_success", [{}])[0]) if runs.get("goal_success") else None
    failure = reward_from_run(runs.get("goal_failure", [{}])[0]) if runs.get("goal_failure") else None
    evidence["rewards"] = {"success": success, "failure": failure}
    if success is None or failure is None:
        failures.append({"gate": "reward_separation", "message": "success/failure scenarios must call reward"})
    elif success <= failure or success - failure < 0.1:
        failures.append({"gate": "reward_separation", "message": "success reward is not sufficiently above failure reward", "success": success, "failure": failure})
    auth = {"Authorization": f"Bearer {token}"}
    app.handle("POST", "/v1/reset", {"episode_id": "readiness-leak", "seed": 17}, auth)
    status, observation, _ = app.handle("GET", "/v1/observation", headers=auth)
    leaks = forbidden_paths(observation)
    evidence["observation_scan"] = {"status": status, "forbidden_paths": leaks}
    if status != 200 or leaks:
        failures.append({"gate": "hidden_state_leakage", "message": "public observation leaks forbidden fields", "paths": leaks})
    if runs.get("goal_success"):
        scenario = next(item for item in scenarios if item.get("kind") == "goal_success")
        try:
            first = canonical(runner.run(scenario))
            second = canonical(runner.run(scenario))
            evidence["determinism"] = first == second
            if first != second:
                failures.append({"gate": "determinism", "message": "same structured trajectory is not deterministic"})
        except Exception as exc:
            failures.append({"gate": "determinism", "message": str(exc)})
    hard_gates = sorted({failure["gate"] for failure in failures})
    return {
        "training_ready": not failures,
        "hard_gates_passed": not failures,
        "failed_gates": hard_gates,
        "evidence": evidence,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.root.resolve())
    output = args.output or args.root / "training_readiness.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["training_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate the stable module DAG used to build a task sandbox.

The platform architecture is owned by EnvFactory.  A code model should fill
task-specific behavior, not rediscover the same architecture for every task.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("BUILD_CONTRACT.json must be an object")
    return value


def build_plan(contract: dict[str, Any]) -> dict[str, Any]:
    tools = contract.get("tools", [])
    tool_names = [
        item.get("function", {}).get("name")
        for item in tools
        if isinstance(item, dict) and isinstance(item.get("function"), dict)
    ]
    manifest = contract.get("artifacts", {}).get("data_manifest", {})
    tables = [
        item.get("table_name")
        for item in manifest.get("tables", [])
        if isinstance(item, dict) and item.get("table_name")
    ] if isinstance(manifest, dict) else []
    metrics = [
        item.get("id") for item in contract.get("metrics", [])
        if isinstance(item, dict) and item.get("id")
    ]
    return {
        "version": "1.0",
        "authority": "env_factory_outer_workflow",
        "nodes": [
            {
                "id": "data_layer",
                "goal": "Load the declared business data into isolated, deterministic episodes using the supplied runtime primitives.",
                "depends_on": [],
                "inputs": ["BUILD_CONTRACT.json", "data/business_data", "sandbox_runtime.py"],
                "outputs": ["data_layer.py", "tests/data_layer"],
                "validation": ["python3 -m pytest -q tests/data_layer"],
                "scope": {"tables": tables},
            },
            {
                "id": "tool_registry",
                "goal": "Implement only the declared task tool handlers; schema validation, tracing and mutation hooks remain generic.",
                "depends_on": ["data_layer"],
                "inputs": ["BUILD_CONTRACT.json.tools", "data_layer.py"],
                "outputs": ["task_impl.py business handlers", "tools.json", "tests/tools"],
                "validation": ["python3 -m pytest -q tests/tools"],
                "scope": {"tools": tool_names},
            },
            {
                "id": "user_and_reward",
                "goal": "Implement contract-driven user simulation and reward evaluation without hard-coded metric IDs or session selection.",
                "depends_on": ["data_layer", "tool_registry"],
                "inputs": ["BUILD_CONTRACT.json.metrics", "data/user_simulation", "runtime_llm.py"],
                "outputs": ["task_impl.py user renderer and custom metric scores", "tests/user_simulator", "tests/reward"],
                "validation": ["python3 -m pytest -q tests/user_simulator tests/reward"],
                "scope": {"metrics": metrics},
            },
            {
                "id": "trainer_api",
                "goal": "Wire the supplied runtime primitives and task components to the exact declared HTTP interface.",
                "depends_on": ["data_layer", "tool_registry", "user_and_reward"],
                "inputs": ["BUILD_CONTRACT.json.requirements.runtime_interface", "sandbox_runtime.py"],
                "outputs": ["app.py", "tests/trainer_api"],
                "validation": ["python3 -m pytest -q tests/trainer_api"],
            },
            {
                "id": "delivery",
                "goal": "Package and verify the completed sandbox; do not redesign already validated modules.",
                "depends_on": ["trainer_api"],
                "inputs": ["all production modules", "acceptance_contract"],
                "outputs": ["acceptance.sh", "acceptance_result.json", "Dockerfile", "docker_build.sh", "docker_run.sh", "requirements-dev.txt", "IMPLEMENTATION_REPORT.md"],
                "validation": ["python3 -m pytest -q", "bash ./acceptance.sh"],
            },
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = build_plan(_load(args.contract))
    args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

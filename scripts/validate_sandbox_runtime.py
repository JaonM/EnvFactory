#!/usr/bin/env python3
"""Static guard against task-specific demo implementations.

This is intentionally a narrow structural gate.  Business handlers may be
task-specific, but evaluator selection, reward aggregation, and user-script
execution must be driven by BUILD_CONTRACT and runtime artifacts rather than
by fixed metric ids, one session, or a fixed turn count.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    source_path = root / "app.py"
    contract_path = root / "BUILD_CONTRACT.json"
    if not source_path.is_file() or not contract_path.is_file():
        fail("runtime genericity check requires app.py and BUILD_CONTRACT.json")

    # Generated sandboxes are intentionally modular.  Inspect the production
    # runtime as a whole instead of requiring every implementation detail to
    # live in app.py (an earlier check incorrectly rejected good modular
    # implementations).
    production_names = (
        "app.py", "task_impl.py", "tool_registry.py", "reward_evaluator.py", "user_simulator.py"
    )
    sources = {
        name: (root / name).read_text(encoding="utf-8")
        for name in production_names if (root / name).is_file()
    }
    source = "\n".join(sources.values())
    shared_source_path = root / "sandbox_runtime.py"
    shared_source = shared_source_path.read_text(encoding="utf-8") if shared_source_path.is_file() else ""
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("task_spec"):
        import importlib.util
        template_path = Path(__file__).with_name("generate_sandbox_scaffold.py")
        module_spec = importlib.util.spec_from_file_location("scaffold_authority", template_path)
        template = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(template)
        if source_path.read_text(encoding="utf-8") != template.APP_SOURCE:
            fail("platform app.py differs from the EnvFactory scaffold; regenerate it before validation")
        platform = Path(__file__).resolve().parents[1] / "src" / "env_factory"
        for name in ("sandbox_runtime.py", "runtime_llm.py"):
            if not (root / name).is_file() or (root / name).read_bytes() != (platform / name).read_bytes():
                fail(f"platform runtime has changed: {name}")
    interface = contract.get("requirements", {}).get("runtime_interface", {})
    shared_runtime = interface.get("shared_runtime", {}) if isinstance(interface, dict) else {}
    required_components = shared_runtime.get("required_components", []) if isinstance(shared_runtime, dict) else []
    if isinstance(required_components, list):
        missing_components = [
            component for component in required_components
            if isinstance(component, str) and component not in source and component not in shared_source
        ]
        if missing_components:
            fail(f"runtime does not provide required shared components: {missing_components}")
    metrics = contract.get("metrics", [])
    if not isinstance(metrics, list) or not metrics:
        fail("BUILD_CONTRACT.metrics must be a non-empty list")

    # The runtime must load the metric/evaluator definitions instead of
    # re-encoding the generated metric ids in Python.
    if not re.search(
        r"(?:CONTRACT|contract|self\.contract)\s*(?:\[\s*['\"]metrics['\"]\s*\]|\.get\(\s*['\"]metrics['\"])",
        source,
    ):
        fail("reward evaluator must iterate CONTRACT.metrics")
    if "DeclarativeMetricEvaluator" not in source and not re.search(r"metric\s*(?:\.|\[\s*['\"]evaluator['\"]\s*\])", source):
        fail("reward evaluator must consume metric.evaluator from the contract")
    evaluator_boundary = (
        ("RuntimeLLMClient" in source and "json_chat" in source)
        or ("ContractEvaluatorRuntime" in source and "RuntimeLLMClient" in shared_source and "json_chat" in shared_source)
    )
    if not evaluator_boundary:
        fail("runtime must use RuntimeLLMClient for external evaluator/user simulation")

    metric_ids = [item.get("id") for item in metrics if isinstance(item, dict)]
    for metric_id in metric_ids:
        if isinstance(metric_id, str) and metric_id and metric_id in source:
            fail(f"runtime hard-codes generated metric id: {metric_id}")

    # A simulator may choose a deterministic seeded branch, but it cannot be
    # a replay of the first session with a fixed turn cutoff.
    if re.search(r"sessions\)\.glob\([^\n]+\)\[0\]", source) or re.search(r"glob\([^\n]+\)\)\[0\]", source):
        fail("User Simulator must select a profile/script branch, not always use the first session")
    if re.search(r"turns\s*>=\s*\d+", source) or re.search(r"turn_count\s*>=\s*\d+", source):
        fail("User Simulator termination must consume script/LLM should_end, not a fixed turn cutoff")
    if "ContractUserSimulator" not in source and ("class UserSimulator" not in source or not all(
        token in source for token in ("profiles", "scripts", "sessions")
    )):
        fail("User Simulator must load profiles, scripts, and sessions from its manifest")
    if re.search(r"ContractUserSimulator\s*\([^)]*renderer\s*=", source, re.S):
        fail("ContractUserSimulator must use its default external-LLM renderer; task-specific renderer injection is forbidden")

    # The generic runtime must expose a contract-driven tool registry and the
    # reward endpoint, while keeping reward calculation out of tool execution.
    shared_composition = all(token in source for token in ("ContractToolRegistry", "ContractRewardAggregator", "SandboxApplication"))
    legacy_composition = "class ToolRegistry" in source and "def execute" in source and "class RewardEvaluator" in source
    if not shared_composition and not legacy_composition:
        fail("runtime must contain a tool registry, tool executor, and reward endpoint")
    execute_start = source.find("def execute")
    reward_start = source.find("class RewardEvaluator")
    if execute_start >= 0 and reward_start > execute_start:
        execute_source = source[execute_start:reward_start]
        if "reward(" in execute_source or "_process_scores" in execute_source:
            fail("tool execution must not calculate reward")

    print("runtime genericity: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

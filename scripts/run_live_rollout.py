"""Bounded real-model trajectories; Agent sees public inputs, never acceptance answers."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

from env_factory.llm import LLMClient
from env_factory.runtime_llm import RuntimeLLMConfig
from env_factory.sandbox_runtime import BusinessGoalEvaluator


def episode(app, task, client, seed, max_steps):
    headers = {"Authorization": "Bearer " + os.environ["SANDBOX_TRAINER_API_KEY"],
               "X-Episode-ID": f"live-{seed}"}
    trace, usage = [], []

    def request(method, path, body=None):
        status, value, _ = app.handle(method, path, body, headers)
        trace.append({"method": method, "path": path, "status": status, "body": body, "result": value})
        if status >= 500 or (status >= 400 and not path.startswith("/v1/tools/")):
            raise RuntimeError(f"runtime endpoint failed: {path} status={status}")
        return status, value

    request("POST", "/v1/reset", {"episode_id": headers["X-Episode-ID"], "seed": seed})
    baseline = app.business_snapshot()
    initial_reward = request("GET", "/v1/reward")[1]["reward"]
    tools = request("GET", "/v1/tools")[1]["tools"]
    observation = request("GET", "/v1/observation")[1]
    names = {tool["function"]["name"] for tool in tools}
    public_input = task.get("public_input") if isinstance(task.get("public_input"), dict) else {}
    initial_message = public_input.get("initial_user_message")
    if not isinstance(initial_message, str) or not initial_message.strip():
        initial_message = task["task"]
    materials = public_input.get("materials")
    if isinstance(materials, list) and materials:
        initial_message = initial_message.strip() + "\n\nPublic materials:\n" + json.dumps(
            materials, ensure_ascii=False
        )
    conversation = [{"role": "user", "content": initial_message}]
    messages = [{"role": "system", "content": (
        'Complete the user task using only available tools and public observations. Return JSON only: '
        '{"kind":"tool","name":"...","arguments":{...}} or '
        '{"kind":"respond","content":"..."}. Respond to ask the user a question or deliver your answer. '
        'Never invent a tool result. Tool schemas: ' + json.dumps(tools, ensure_ascii=False))}, *conversation,
        {"role": "user", "content": json.dumps({"observation": observation}, ensure_ascii=False)}]
    termination = "step_budget"
    protocol_errors = 0
    for _ in range(max_steps):
        response = client.chat(messages, response_format={"type": "json_object"})
        usage.append(dict(response.usage or {}))
        messages.append({"role": "assistant", "content": response.content})
        try:
            action = json.loads(response.content)
            if not isinstance(action, dict):
                raise ValueError("action must be an object")
            if action.get("kind") == "tool":
                if action.get("name") not in names or not isinstance(action.get("arguments"), dict):
                    raise ValueError("unknown tool or invalid arguments")
                status, result = request("POST", "/v1/tools/" + action["name"], action["arguments"])
                messages.append({"role": "user", "content": json.dumps({"tool_result": result, "status": status}, ensure_ascii=False)})
            elif action.get("kind") == "respond" and isinstance(action.get("content"), str) and action["content"].strip():
                request("POST", "/v1/agent_response", {"content": action["content"]})
                conversation.append({"role": "assistant", "content": action["content"]})
                user = request("POST", "/v1/user_simulator", {"messages": conversation})[1]
                conversation.append({"role": "user", "content": user["user_query"]})
                messages.append({"role": "user", "content": user["user_query"]})
                if user.get("should_end"):
                    termination = user.get("termination_reason", "user_ended")
                    break
            else:
                raise ValueError("expected tool or respond action")
        except (ValueError, TypeError) as exc:
            protocol_errors += 1
            messages.append({"role": "user", "content": f"Invalid action: {exc}. Return a corrected JSON action."})
            if protocol_errors >= 3:
                termination = "agent_protocol_error"
                break
    final_reward = request("GET", "/v1/reward")[1]["reward"]
    repeated_reward = request("GET", "/v1/reward")[1]["reward"]
    replay = request("GET", "/v1/replay")[1]
    state = app.business_snapshot()
    goal = task.get("task_spec", {}).get("goal_contract", {})
    state_success = None
    if goal.get("row_predicates"):
        state_success = (BusinessGoalEvaluator.evaluate(goal["row_predicates"], state)
                         and BusinessGoalEvaluator.preserves_unrelated(goal, baseline, state)
                         and (not goal.get("requires_state_change") or baseline != state))
    fallbacks = [event for event in replay.get("events", []) if event.get("payload", {}).get("used_fallback")]
    issues = []
    if initial_reward >= 0.8:
        issues.append("high_reward_without_actions")
    if final_reward != repeated_reward:
        issues.append("unstable_reward_read")
    if state_success is False and final_reward >= 0.8:
        issues.append("reward_state_disagreement")
    if fallbacks:
        issues.append("runtime_llm_fallback")
    success = termination == "completed" and final_reward >= 0.8 and state_success is not False and not issues
    return {"seed": seed, "agent_success": success, "termination": termination,
            "initial_reward": initial_reward, "final_reward": final_reward,
            "state_goal_satisfied": state_success, "issues": issues, "usage": usage,
            "protocol_errors": protocol_errors, "trajectory": trace, "replay": replay,
            "initial_state": baseline, "final_state": state}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=20)
    args = parser.parse_args()
    if args.episodes <= 0 or args.max_steps <= 0:
        parser.error("episodes and steps must be positive")
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    client = LLMClient.from_env("LLM", timeout=float(os.getenv("LLM_TIMEOUT", "60")))
    os.environ["SANDBOX_EVALUATOR_MOCK"] = "0"
    runtime = RuntimeLLMConfig.from_env()
    os.environ["SANDBOX_MUTATION_MODE"] = "disabled"
    os.environ.setdefault("SANDBOX_TRAINER_API_KEY", "local-rollout-trainer")
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("rollout_app", root / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    task = json.loads((root / "task.json").read_text())
    report = {"mode": "live_rollout", "agent_model": client.model,
              "runtime_model": runtime.model, "episodes": [],
              "same_model_bias_possible": client.model == runtime.model,
              "live_rollout_verified": False, "passed": False}
    from loop_experiment import write_json
    with tempfile.TemporaryDirectory(prefix="envfactory-rollout-") as temporary:
        app = module.create_app(db_path=Path(temporary) / "episodes.sqlite3")
        for seed in range(args.episodes):
            try:
                result = episode(app, task, client, seed, args.max_steps)
            except Exception as exc:
                # Error messages from providers may contain credentials; record type only.
                result = {"seed": seed, "agent_success": False, "issues": ["execution_error"], "error_type": type(exc).__name__}
            report["episodes"].append(result)
            write_json(args.output, report)
    report["live_rollout_verified"] = all("execution_error" not in item["issues"] and "runtime_llm_fallback" not in item["issues"] for item in report["episodes"])
    report["agent_success_rate"] = sum(item["agent_success"] for item in report["episodes"]) / args.episodes
    report["environment_checks_passed"] = not any(item["issues"] for item in report["episodes"])
    report["quality_score"] = round(
        (0.3 if report["live_rollout_verified"] else 0.0)
        + (0.3 if report["environment_checks_passed"] else 0.0)
        + (0.4 if report["agent_success_rate"] > 0 else 0.0),
        2,
    )
    report["passed"] = report["live_rollout_verified"] and report["environment_checks_passed"] and report["agent_success_rate"] > 0
    if any("execution_error" in item["issues"] for item in report["episodes"]):
        report["failure_owner"] = "infrastructure"
    elif not report["environment_checks_passed"] or not report["live_rollout_verified"]:
        report["failure_owner"] = "environment"
    elif report["agent_success_rate"] == 0:
        report["failure_owner"] = "agent"
    else:
        report["failure_owner"] = None
    report["conclusion"] = "live_success_witness" if report["passed"] else "issues_or_insufficient_success_evidence"
    write_json(args.output, report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

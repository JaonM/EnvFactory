#!/usr/bin/env python3
"""Iterative sandbox build loop for the selected high-value tasks."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_sandbox import evidence_fingerprint


DEFAULT_TASK_IDS = (45, 78, 92, 175)
REQUIRED_MODEL = "gpt-5.6-luna"


def next_round(root: Path) -> int:
    values = []
    for path in root.glob("round-*"):
        try:
            values.append(int(path.name.split("-", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(values, default=0) + 1


def build_command(
    project: Path, task_id: int, output: Path, max_attempts: int, *, resume: bool = False
) -> list[str]:
    command = [
        "bash", str(project / "scripts/develop_sandbox_with_agent.sh"),
        "--input", str(project / f"output/task_artifacts/task-{task_id}/task.json"),
        "--output", str(output), "--agent", "codex", "--review-agent", "codex",
        "--model", REQUIRED_MODEL, "--review-model", REQUIRED_MODEL,
        "--runtime", "none", "--max-attempts", str(max_attempts), "--foreground",
        "--skip-auto-score",
    ]
    if resume:
        command.append("--resume")
    return command


def run_one(
    project: Path,
    round_root: Path,
    task_id: int,
    max_attempts: int,
    seed: Path | None = None,
) -> dict[str, Any]:
    output = round_root / f"task-{task_id}"
    if seed is not None and seed.is_dir():
        shutil.copytree(seed, output, dirs_exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    log = output / "loop_build.log"
    command = build_command(project, task_id, output, max_attempts, resume=seed is not None)
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=project, stdout=handle, stderr=subprocess.STDOUT, text=True)
    score_path = output / "sandbox_score.json"
    score_command = [
        sys.executable, str(project / "scripts/score_sandbox.py"), str(output),
        "--project", str(project), "--threshold", "8", "--output", str(score_path), "--execute",
    ]
    scored = subprocess.run(score_command, cwd=project, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        score = json.loads(score_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        score = {"score": 0.0, "passed": False, "error": scored.stdout[-4000:]}
    return {
        "task_id": f"task-{task_id}", "output": str(output),
        "build_exit_code": completed.returncode, "score_exit_code": scored.returncode,
        "score": score,
    }


def reusable_result(history: dict[str, Any], task_id: int) -> dict[str, Any] | None:
    wanted = f"task-{task_id}"
    for round_report in reversed(history.get("rounds", [])):
        for item in round_report.get("tasks", []):
            score = item.get("score", {})
            # A sandbox may have been rescored after fixing the scorer or the
            # evidence lifecycle. Prefer that authoritative on-disk result to
            # the immutable historical snapshot.
            try:
                score_path = Path(item["output"]) / "sandbox_score.json"
                current_score = json.loads(score_path.read_text(encoding="utf-8"))
                if isinstance(current_score, dict):
                    score = current_score
            except (KeyError, OSError, json.JSONDecodeError, TypeError):
                pass
            if (
                item.get("task_id") == wanted
                and score.get("passed") is True
                and score.get("score", 0) >= 8
                and score.get("model") == REQUIRED_MODEL
                and score.get("review_model") == REQUIRED_MODEL
                and score.get("executed") is True
                and score.get("evidence_fingerprint") == evidence_fingerprint(Path(item["output"]), Path(__file__).resolve().parents[1])
            ):
                return {
                    **item,
                    "score": score,
                    "reused": True,
                    "reused_from": round_report.get("round"),
                }
    return None


def latest_seed(history: dict[str, Any], task_id: int) -> Path | None:
    """Return the newest prior implementation for incremental refinement."""
    wanted = f"task-{task_id}"
    for round_report in reversed(history.get("rounds", [])):
        for item in round_report.get("tasks", []):
            if item.get("task_id") != wanted or not item.get("output"):
                continue
            candidate = Path(item["output"])
            if candidate.is_dir() and (candidate / "task_impl.py").is_file():
                return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="循环构建并按 Agentic 训练价值评分沙箱")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("output/sandbox_loop"))
    parser.add_argument("--max-rounds", type=int, default=50)
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--task-ids", default=",".join(map(str, DEFAULT_TASK_IDS)))
    args = parser.parse_args()
    if not 1 <= args.max_rounds <= 50:
        parser.error("--max-rounds 必须位于 [1, 50]")
    task_ids = tuple(int(item) for item in args.task_ids.split(",") if item.strip())
    if not task_ids:
        parser.error("--task-ids 至少需要一个任务 ID")
    project = args.project.resolve()
    root = args.output if args.output.is_absolute() else project / args.output
    root.mkdir(parents=True, exist_ok=True)
    start = next_round(root)
    history_path = root / "history.json"
    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        history = {"model": REQUIRED_MODEL, "task_ids": [f"task-{item}" for item in task_ids], "rounds": []}

    completed_rounds = len(history.get("rounds", []))
    remaining_rounds = max(0, 50 - completed_rounds)
    if remaining_rounds == 0:
        print("sandbox loop already reached the global 50-round limit", file=sys.stderr)
        return 1
    rounds_to_run = min(args.max_rounds, remaining_rounds)

    for offset in range(rounds_to_run):
        round_number = start + offset
        round_root = root / f"round-{round_number:02d}"
        round_root.mkdir(parents=True, exist_ok=False)
        reused = {task_id: reusable_result(history, task_id) for task_id in task_ids}
        pending = [task_id for task_id in task_ids if reused[task_id] is None]
        seeds = {task_id: latest_seed(history, task_id) for task_id in pending}
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
            futures = {
                task_id: executor.submit(
                    run_one, project, round_root, task_id, args.max_attempts, seeds[task_id]
                )
                for task_id in pending
            }
            built = {task_id: future.result() for task_id, future in futures.items()}
        results = [reused[task_id] or built[task_id] for task_id in task_ids]
        passed = all(
            item["score"].get("score", 0) >= 8
            and item["score"].get("passed") is True
            and item["score"].get("model") == REQUIRED_MODEL
            and item["score"].get("review_model") == REQUIRED_MODEL
            for item in results
        )
        report = {
            "round": round_number, "model": REQUIRED_MODEL,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "passed": passed, "tasks": results,
        }
        (round_root / "round_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        history["rounds"].append(report)
        history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"round": round_number, "passed": passed, "scores": {item["task_id"]: item["score"].get("score", 0) for item in results}}, ensure_ascii=False))
        if passed:
            return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

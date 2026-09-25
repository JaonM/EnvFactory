import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


loop = load("loop_experiment")
rollout = load("run_live_rollout")


class ExperimentTest(unittest.TestCase):
    def test_zero_round_limit_is_unbounded(self):
        self.assertEqual(list(itertools.islice(loop.round_numbers(0), 25))[-1], 25)
        self.assertEqual(list(loop.round_numbers(3)), [1, 2, 3])

    def test_denominator_includes_generation_failure_and_threshold_is_inclusive(self):
        report = loop.summarize([
            {"task_path": "a", "score": 9, "passed": True, "category": "direct_response"},
            {"task_path": "b", "score": 8, "passed": True, "category": "simple_agentic"},
            loop.failure("generation", "missing"),
        ], 8)
        self.assertEqual(report["requested"], 3)
        self.assertEqual(report["generated"], 2)
        self.assertEqual(report["qualified"], 2)
        self.assertEqual(report["end_to_end_rate"], 2 / 3)
        self.assertFalse(report["all_passed"])
        self.assertFalse(report["live_rollout_verified"])

    def test_timeout_and_nonzero_exit_are_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = loop.run_process([sys.executable, "-c", "import time; time.sleep(10)"], root, root / "timeout.log", .05)
            self.assertEqual(result["exit_code"], 124)
            self.assertTrue(result["timed_out"])
            result = loop.run_process([sys.executable, "-c", "raise SystemExit(7)"], root, root / "exit.log", 5)
            self.assertEqual(result["exit_code"], 7)
            self.assertFalse(loop.ACTIVE_PROCESSES)

    def test_pause_persists_only_active_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "runtime_state.json"
            previous_path = loop.BUDGET_STATE_PATH
            previous_started = loop.ACTIVE_RUN_STARTED
            try:
                loop.BUDGET_STATE_PATH = state_path
                loop.ACTIVE_RUN_STARTED = 100.0
                with patch.object(loop.time, "time", return_value=112.5):
                    loop.finish_active_budget("paused")
                state = json.loads(state_path.read_text())
                self.assertEqual(state["active_seconds"], 12.5)
                self.assertEqual(state["status"], "paused")
                self.assertNotIn("run_started_at", state)
            finally:
                loop.BUDGET_STATE_PATH = previous_path
                loop.ACTIVE_RUN_STARTED = previous_started

    def test_promotion_uses_yield_targets_instead_of_all_samples(self):
        passing = {
            "task_path": "task", "score": 9, "passed": True,
            "category": "simple_agentic",
            "task_score": {"eligible": True, "score": 9},
            "sandbox_score": {"passed": True, "score": 9},
            "live_rollout_verified": True,
        }
        results = [dict(passing) for _ in range(8)] + [
            {"task_path": "bad", "score": 0, "passed": False,
            "category": "simple_agentic", "failure_class": "build",
            "failure_code": "BUILD_BUSINESS",
             "task_score": {"eligible": True, "score": 9},
             "sandbox_score": {"passed": False, "score": 0}},
            loop.failure("generation", "missing", category="simple_agentic"),
        ]
        summary = loop.summarize(results, 8, targets={
            "task_yield": .9, "build_yield": .8, "end_to_end_rate": .7,
            "qualified_mean": 8.5, "category_rate": .6,
        })
        self.assertTrue(summary["target_met"])
        self.assertFalse(summary["all_passed"])
        self.assertEqual(summary["task_good_yield"], .9)
        self.assertAlmostEqual(summary["sandbox_build_yield"], 8 / 9)

    def test_rollout_failure_does_not_reduce_build_yield(self):
        result = {
            "task_path": "task",
            "score": 7.2,
            "passed": False,
            "failure_class": "live_rollout",
            "task_score": {"eligible": True, "score": 9},
            "sandbox_score": {"passed": True, "score": 8.5},
        }
        summary = loop.summarize([result], 8)
        self.assertEqual(summary["sandbox_build_yield"], 1)
        self.assertEqual(summary["end_to_end_rate"], 0)

    def test_all_infrastructure_failures_do_not_form_a_quality_round(self):
        results = [loop.failure("infrastructure", "network unavailable") for _ in range(5)]
        summary = loop.summarize(results, 8, targets={
            "task_yield": .85, "build_yield": .8, "end_to_end_rate": .7,
            "qualified_mean": 8.5, "category_rate": .6,
        })
        self.assertFalse(summary["valid_quality_round"])
        self.assertFalse(summary["target_met"])

    def test_resume_keeps_completed_jobs_and_restarts_incomplete_in_new_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            completed = {"passed": True, "score": 9, "task_path": "one"}
            report = {"round": 1, "state": "running", "jobs": [
                {"id": 1, "state": "complete", "result": completed},
                {"id": 2, "state": "running", "attempt": 1, "task_path": "two"}]}
            config = {"threshold": 8, "max_concurrency": 1, "build_mode": "clean", "generate_count": 0}
            with patch.object(loop, "build_one", return_value={"passed": True, "score": 9, "task_path": "two"}) as build:
                result = loop.run_round(ROOT, root, config, report)
            self.assertEqual(build.call_count, 1)
            self.assertTrue(str(build.call_args.args[2]).endswith("sample-002/attempt-2"))
            self.assertTrue(result["summary"]["all_passed"])
            disk = json.loads((root / "round-01/round_report.json").read_text())
            self.assertEqual(disk["jobs"][0]["result"], completed)

    def test_interrupted_generation_is_not_silently_resampled(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {"generate_count": 5, "threshold": 8, "max_concurrency": 1}
            with patch.object(loop, "run_process") as process:
                report = loop.run_round(ROOT, Path(tmp), config, {"round": 1, "state": "running", "generation_started": True})
            process.assert_not_called()
            self.assertEqual(report["summary"]["requested"], 5)
            self.assertEqual(report["summary"]["failures"], {"generation": 5})
            self.assertEqual(report["summary"]["failure_codes"], {"INFRA": 5})

    def test_generation_receives_reproducible_sampling_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "generate_count": 1,
                "threshold": 8,
                "max_concurrency": 1,
                "generation_timeout": 10,
                "experiment_seed": 700,
                "generation_hops": 4,
                "route_attempts": 2,
                "training_mix": "direct_response=0.2,simple_agentic=0.3,multi_step_agentic=0.5",
            }
            with patch.object(
                loop,
                "run_process",
                return_value={"exit_code": 1, "timed_out": False, "seconds": 0.1},
            ) as process:
                loop.run_round(ROOT, Path(tmp), config, {"round": 3, "state": "running"})
            command = process.call_args.args[0]
            self.assertEqual(command[command.index("--seed") + 1], "702")
            self.assertEqual(command[command.index("--hops") + 1], "4")
            self.assertEqual(command[command.index("--route-attempts") + 1], "2")
            self.assertEqual(
                command[command.index("--training-mix") + 1],
                config["training_mix"],
            )

    def test_generation_manifests_preserve_failed_route_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "round-01/generation/task_artifacts/task-9"
            task_dir.mkdir(parents=True)
            manifest = {
                "batch_index": 1, "training_category": "multi_step_agentic",
                "sample_seed": 99, "status": "failed",
            }
            (task_dir / "sample_manifest.json").write_text(json.dumps(manifest))
            (task_dir / "failure.json").write_text(json.dumps({
                "failure_class": "GEN_SCHEMA", "message": "bad schema",
            }))
            config = {"generate_count": 1, "threshold": 8, "max_concurrency": 1}
            report = loop.run_round(
                ROOT, root, config,
                {"round": 1, "state": "running", "generation_started": True},
            )
            result = report["jobs"][0]["result"]
            self.assertEqual(result["failure_code"], "GEN_SCHEMA")
            self.assertEqual(result["category"], "multi_step_agentic")
            self.assertEqual(result["sample_seed"], 99)

    def test_one_job_exception_does_not_abort_other_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {"generate_count": 0, "task_paths": ["one", "two"], "threshold": 8, "max_concurrency": 1, "build_mode": "clean"}
            with patch.object(loop, "build_one", side_effect=[ValueError("bad input"), {"passed": True, "score": 9}]):
                report = loop.run_round(ROOT, Path(tmp), config, {"round": 1, "state": "running"})
            self.assertEqual(report["summary"]["requested"], 2)
            self.assertEqual(report["summary"]["failures"], {"infrastructure": 1})

    def test_buildability_failure_is_not_mislabeled_as_luna_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_path = root / "task.json"
            task_path.write_text("{}")
            output = root / "sandbox"
            quality = SimpleNamespace(to_dict=lambda: {
                "eligible": True, "score": 9, "training_category": "simple_agentic",
            })
            def failed_build(*args, **kwargs):
                output.mkdir(exist_ok=True)
                (output / "buildability.json").write_text(json.dumps({
                    "buildable": False,
                    "issues": [{"code": "EXTERNAL_CAPABILITY_UNAVAILABLE"}],
                }))
                return {"exit_code": 4, "timed_out": False, "seconds": .1}
            config = {"threshold": 8, "max_attempts": 2, "build_timeout": 10,
                      "build_mode": "clean"}
            with (
                patch("env_factory.task_quality.score_file", return_value=quality),
                patch.object(loop, "run_process", side_effect=failed_build),
            ):
                result = loop.build_one(ROOT, task_path, output, config)
            self.assertEqual(result["failure_class"], "task_quality")
            self.assertEqual(result["failure_code"], "EXTERNAL_CAPABILITY_UNAVAILABLE")

    def test_holdout_requires_fresh_tasks_and_two_of_three_successes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = []
            for index in range(30):
                task = root / f"task-{index}.json"
                task.write_text(json.dumps({"task": index}))
                results.append({
                    "task_path": str(task), "sample_seed": 10_000 + index,
                    "score": 9, "passed": True, "category": "multi_step_agentic",
                    "task_score": {"eligible": True, "score": 9},
                    "sandbox_score": {"passed": True, "score": 9},
                    "live_rollout": {
                        "agent_success_rate": 2 / 3,
                        "all_episodes_environment_clean": True,
                        "all_episodes_fallback_free": True,
                    },
                })
            summary = loop.summarize_holdout(
                results, 8, expected_count=30, end_to_end_target=.7,
                rollout_success_target=2 / 3, previous_seeds={1, 2, 3},
            )
            self.assertTrue(summary["target_met"])
            results[0]["live_rollout"]["agent_success_rate"] = 1 / 3
            self.assertFalse(loop.summarize_holdout(
                results, 8, expected_count=30, end_to_end_target=.7,
                rollout_success_target=2 / 3,
            )["qualified_rollout_floor_met"])
            results[0]["sample_seed"] = 1
            self.assertFalse(loop.summarize_holdout(
                results, 8, expected_count=30, end_to_end_target=.7,
                rollout_success_target=2 / 3, previous_seeds={1},
            )["fresh_tasks_verified"])


class FakeApp:
    def __init__(self, *, correct=True, fallback=False):
        self.correct, self.fallback = correct, fallback
        self.changed = False

    def business_snapshot(self):
        return {"items": [{"id": 1, "done": self.changed}]}

    def handle(self, method, path, body=None, headers=None):
        if path == "/v1/reset":
            self.changed = False
            value = {}
        elif path == "/v1/tools":
            value = {"tools": [{"type": "function", "function": {"name": "finish", "parameters": {"type": "object"}}}]}
        elif path == "/v1/tools/finish":
            self.changed = self.correct
            value = {"updated": 1}
        elif path == "/v1/reward":
            value = {"reward": 1 if getattr(self, "answered", False) else 0}
        elif path == "/v1/agent_response":
            self.answered = True
            value = {"accepted": True}
        elif path == "/v1/user_simulator":
            value = {"user_query": "accepted", "should_end": True, "termination_reason": "completed"}
        elif path == "/v1/replay":
            value = {"events": [{"payload": {"used_fallback": self.fallback}}]}
        else:
            value = {}
        return 200, value, {}


class RolloutTest(unittest.TestCase):
    def test_release_gate_requires_two_of_three_clean_successes(self):
        episodes = [
            {"agent_success": True, "issues": []},
            {"agent_success": True, "issues": []},
            {"agent_success": False, "issues": []},
        ]
        report = rollout.summarize_episodes(episodes, 2 / 3)
        self.assertTrue(report["passed"])
        self.assertEqual(report["quality_score"], 1.0)
        episodes[1]["agent_success"] = False
        report = rollout.summarize_episodes(episodes, 2 / 3)
        self.assertFalse(report["passed"])
        self.assertEqual(report["failure_owner"], "agent")
        episodes[1]["issues"] = ["runtime_llm_fallback"]
        report = rollout.summarize_episodes(episodes, 0)
        self.assertFalse(report["all_episodes_fallback_free"])
        self.assertEqual(report["failure_owner"], "environment")

    def run_episode(self, app):
        task = {"task": "finish item", "acceptance_contract": {"secret": "do not show"},
                "public_input": {"initial_user_message": "finish the supplied item", "materials": [{
                    "name": "item", "mime_type": "application/json", "content": '{"id": 1}'
                }]},
                "task_spec": {"goal_contract": {"row_predicates": [{"table": "items", "where": {"id": 1},
                       "values": {"done": True}, "count": 1}], "requires_state_change": True}}}
        calls = []
        actions = iter([{"kind": "tool", "name": "finish", "arguments": {}}, {"kind": "respond", "content": "done"}])
        def chat(messages, **kwargs):
            calls.append(json.dumps(messages))
            return SimpleNamespace(content=json.dumps(next(actions)), usage={"total_tokens": 10})
        with patch.dict(os.environ, {"SANDBOX_TRAINER_API_KEY": "test"}):
            result = rollout.episode(app, task, SimpleNamespace(chat=chat), 0, 4)
        self.assertTrue(all("do not show" not in call for call in calls))
        self.assertTrue(any("Public materials" in call and "application/json" in call for call in calls))
        return result

    def test_live_agent_uses_only_public_inputs_and_records_usage(self):
        result = self.run_episode(FakeApp())
        self.assertTrue(result["agent_success"])
        self.assertEqual(sum(item["total_tokens"] for item in result["usage"]), 20)

    def test_high_reward_with_wrong_state_is_rejected(self):
        result = self.run_episode(FakeApp(correct=False))
        self.assertFalse(result["agent_success"])
        self.assertIn("reward_state_disagreement", result["issues"])

    def test_runtime_fallback_is_not_live_success(self):
        result = self.run_episode(FakeApp(fallback=True))
        self.assertFalse(result["agent_success"])
        self.assertIn("runtime_llm_fallback", result["issues"])

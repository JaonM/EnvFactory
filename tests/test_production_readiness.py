import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts/certify_training_materials.py"
    spec = importlib.util.spec_from_file_location("certify_training_materials", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


certifier = load_script()


class ProductionReadinessTest(unittest.TestCase):
    def make_history(self, root: Path, *, count: int = 300):
        evidence = root / "evidence"
        evidence.mkdir()
        (evidence / "training_readiness.json").write_text(json.dumps({
            "training_ready": True,
            "evidence": {
                "determinism": True,
                "observation_scan": {"forbidden_paths": []},
                "runtime_state": {
                    "reset_reproducible": True,
                    "episode_isolation": True,
                    "replay_consistent": True,
                },
            },
        }))
        (evidence / "agentic_training_value.json").write_text(json.dumps({
            "curriculum_training_ready": True,
            "evidence": {"counterfactuals": {
                "goal_success": {"reward": 1.0},
                "goal_failure": {"reward": 0.0},
                "no_tools": {"reward": 0.0},
            }},
        }))
        jobs = []
        for index in range(count):
            task = root / f"task-{index}.json"
            # A digest-like description avoids intentionally triggering the
            # semantic near-duplicate detector in this passing fixture.
            description = __import__("hashlib").sha256(f"task-{index}".encode()).hexdigest()
            task.write_text(json.dumps({"task": description}))
            episodes = [
                {
                    "seed": episode, "agent_success": episode < 8,
                    "termination": "completed" if episode < 8 else "step_budget",
                    "initial_reward": 0.0, "final_reward": 1.0 if episode < 8 else 0.0,
                    "issues": [], "usage": [], "initial_state": {}, "final_state": {},
                    "replay": {"events": []},
                    "trajectory": [
                        {
                            "method": "GET", "path": "/v1/reward", "status": 200,
                            "body": None, "result": {"reward": 0.0},
                        },
                        {
                            "method": "POST", "path": "/v1/user_simulator", "status": 200,
                            "body": {"messages": []}, "result": {
                                "user_query": "accepted", "should_end": True,
                                "outcome_category": "user_acceptance", "reason_code": "accepted",
                            },
                        },
                    ],
                }
                for episode in range(10)
            ]
            result = {
                "task_path": str(task), "output": str(evidence),
                "category": ("simple_agentic" if index % 2 else "multi_step_agentic"),
                "task_score": {"eligible": True, "score": 9},
                "sandbox_score": {"passed": True, "score": 9},
                "score": 9, "passed": True,
                "live_rollout": {"episodes": episodes},
            }
            jobs.append({"id": index + 1, "state": "complete", "result": result})
        return {"holdout": {"jobs": jobs, "summary": {"fresh_tasks_verified": True}}}

    def test_wilson_bound_accounts_for_sample_size(self):
        self.assertLess(certifier.wilson_lower(9, 10), .9)
        self.assertGreater(certifier.wilson_lower(900, 1000), .87)

    def test_production_profile_can_pass_complete_pretraining_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            report = certifier.certify(history, certifier.default_policy())
            self.assertTrue(report["certified"], report["failed_gates"])
            self.assertEqual(report["scope"], "pre_training_material_readiness")
            self.assertIn("rl_training_convergence", report["does_not_certify"])
            self.assertEqual(report["measurements"]["episodes"], 3000)

    def test_pilot_sized_holdout_cannot_claim_production_certification(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory), count=30)
            report = certifier.certify(history, certifier.default_policy())
            self.assertFalse(report["certified"])
            self.assertIn("materialized_sample_size", report["failed_gates"])
            self.assertIn("rollout_coverage", report["failed_gates"])

    def test_any_fallback_breaks_environment_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            history["holdout"]["jobs"][0]["result"]["live_rollout"]["episodes"][0][
                "issues"
            ] = ["runtime_llm_fallback"]
            report = certifier.certify(history, certifier.default_policy())
            self.assertFalse(report["gates"]["environment_integrity"])
            self.assertEqual(report["measurements"]["llm_fallbacks"], 1)

    def test_invalid_user_simulator_protocol_breaks_certification(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            for job in history["holdout"]["jobs"][:16]:
                turns = job["result"]["live_rollout"]["episodes"][0]["trajectory"]
                turns[1]["result"].pop("outcome_category")
            report = certifier.certify(history, certifier.default_policy())
            self.assertFalse(report["gates"]["user_simulator_protocol"])


if __name__ == "__main__":
    unittest.main()

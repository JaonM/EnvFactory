import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name="certify_training_materials"):
    path = ROOT / f"scripts/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


certifier = load_script()
verifier = load_script("verify_training_materials")


class ProductionReadinessTest(unittest.TestCase):
    def make_history(self, root: Path, *, count: int = 300, batches: int = 3):
        evidence = root / "evidence"
        evidence.mkdir()
        (evidence / "app.py").write_text("# immutable sandbox\n")
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
        (evidence / "agentic_training_value_live.json").write_text(json.dumps({
            "curriculum_training_ready": True,
            "validation_mode": "live_evaluator",
            "evidence": {"counterfactuals": {
                "goal_success": {"reward": 1.0},
                "goal_failure": {"reward": 0.0},
                "no_tools": {"reward": 0.0},
            }},
        }))
        (evidence / "data_governance.json").write_text(json.dumps({
            "version": "1.0",
            "eligible_for_external_model_processing": True,
            "data_origin": {
                "origin": "model_generated_synthetic",
                "contains_real_user_data": False,
                "intended_use": "agentic_rl_training_material",
            },
            "providers": {
                "agent": {
                    "host": "agent.example",
                    "model": "policy-model",
                    "identity_sha256": "a" * 64,
                },
                "user_simulator_and_reward": {
                    "host": "runtime.example",
                    "model": "simulator-model",
                    "identity_sha256": "b" * 64,
                },
            },
            "outbound_surfaces": {
                name: sorted(values)
                for name, values in certifier.REQUIRED_OUTBOUND_SURFACES.items()
            },
            "credential_findings": [],
            "pii_findings": [],
        }))
        holdouts = []
        for batch in range(batches):
            jobs = []
            for index in range(count):
                global_index = batch * count + index
                task = root / f"task-{global_index}.json"
                # A digest-like description avoids intentionally triggering the
                # semantic near-duplicate detector in this passing fixture.
                description = __import__("hashlib").sha256(
                    f"task-{global_index}".encode()
                ).hexdigest()
                task.write_text(json.dumps({"task": description}))
                episodes = [
                {
                    "schema_version": "2.0", "seed": episode, "agent_success": episode < 8,
                    "termination": "completed" if episode < 8 else "step_budget",
                    "initial_reward": 0.0, "final_reward": 1.0 if episode < 8 else 0.0,
                    "issues": [], "usage": [], "initial_state": {}, "final_state": {},
                    "replay": {"events": []},
                    "transitions": [{
                        "step": 0,
                        "agent_input": [{"role": "user", "content": "do it"}],
                        "assistant_output": '{"kind":"respond","content":"done"}',
                        "observation": {},
                        "action": {"kind": "respond", "content": "done"},
                        "result": {"status": 200},
                        "next_observation": {},
                        "reward": 1.0 if episode < 8 else 0.0,
                        "terminated": episode < 8,
                        "truncated": episode >= 8,
                    }],
                    "trajectory": [
                        {
                            "method": "GET", "path": "/v1/reward", "status": 200,
                            "body": None, "result": {"reward": 0.0},
                        },
                        {
                            "method": "POST", "path": "/v1/user_simulator", "status": 200,
                            "body": {"messages": []}, "result": {
                                "user_query": "accepted", "should_end": True,
                                "termination_reason": "completed",
                                "match_status": (
                                    "unmatched" if episode % 3 == 2 else "matched"
                                ),
                                "outcome_category": (
                                    "user_acceptance" if episode % 3 == 0
                                    else "information_required" if episode % 3 == 1
                                    else "agent_off_topic"
                                ),
                                "reason_code": "fixture_outcome",
                            },
                        },
                    ],
                }
                for episode in range(10)
                ]
                result = {
                    "task_path": str(task), "output": str(evidence),
                    "sample_seed": global_index + 1000,
                    "category": (
                        "direct_response" if index % 10 < 2
                        else "simple_agentic" if index % 10 < 5
                        else "multi_step_agentic"
                    ),
                    "task_score": {"eligible": True, "score": 9},
                    "sandbox_score": {
                        "passed": True, "score": 9,
                        "evidence_fingerprint": f"sandbox-{global_index}",
                    },
                    "score": 9, "passed": True,
                    "live_rollout": {
                        "schema_version": "2.0",
                        "task_sha256": __import__("hashlib").sha256(task.read_bytes()).hexdigest(),
                        "sandbox_artifacts_digest": certifier.portable_artifact_digest(evidence),
                        "agent_model": "policy-model",
                        "runtime_model": "simulator-model",
                        "agent_provider_sha256": "a" * 64,
                        "runtime_provider_sha256": "b" * 64,
                        "episodes": episodes,
                    },
                }
                jobs.append({"id": index + 1, "state": "complete", "result": result})
            holdouts.append({
                "holdout_batch": batch + 1,
                "jobs": jobs,
                "summary": {"fresh_tasks_verified": True},
            })
        return {
            "config": {"source_digest": "evaluator-source-v1"},
            "holdout": holdouts[0], "holdouts": holdouts,
        }

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
            self.assertEqual(report["measurements"]["episodes"], 9000)
            self.assertTrue(report["gates"]["rollout_provenance"])
            self.assertTrue(report["gates"]["category_mix"])
            self.assertTrue(report["gates"]["user_simulator_outcome_coverage"])
            self.assertTrue(report["gates"]["data_governance"])
            self.assertEqual(len(report["materials_manifest"]["items"]), 900)
            self.assertEqual(report["materials_manifest"]["version"], "2.0")
            self.assertEqual(
                report["materials_manifest"]["evaluator_source_digest"],
                "evaluator-source-v1",
            )
            self.assertEqual(len(report["materials_manifest"]["dataset_sha256"]), 64)

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
            jobs = [job for batch in history["holdouts"] for job in batch["jobs"]]
            for job in jobs[:46]:
                turns = job["result"]["live_rollout"]["episodes"][0]["trajectory"]
                turns[1]["result"].pop("outcome_category")
            report = certifier.certify(history, certifier.default_policy())
            self.assertFalse(report["gates"]["user_simulator_protocol"])

    def test_rollout_must_be_bound_to_the_exact_task_and_sandbox(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            result = history["holdouts"][0]["jobs"][0]["result"]
            result["live_rollout"]["task_sha256"] = "0" * 64
            report = certifier.certify(history, certifier.default_policy())
            self.assertFalse(report["gates"]["rollout_provenance"])

    def test_mock_reward_counterfactuals_cannot_certify_production(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = self.make_history(root)
            live_report = root / "evidence/agentic_training_value_live.json"
            value = json.loads(live_report.read_text())
            value["validation_mode"] = "offline_mock"
            live_report.write_text(json.dumps(value))
            report = certifier.certify(history, certifier.default_policy())
            self.assertFalse(report["gates"]["tool_and_reward_integrity"])

    def test_credential_finding_breaks_data_governance_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = self.make_history(root)
            path = root / "evidence/data_governance.json"
            value = json.loads(path.read_text())
            value["credential_findings"] = [{
                "kind": "assigned_secret", "path": "$.data/config.txt",
            }]
            path.write_text(json.dumps(value))
            report = certifier.certify(history, certifier.default_policy())
            self.assertFalse(report["gates"]["data_governance"])
            self.assertIn("data_governance", report["failed_gates"])

    def test_category_mix_prevents_single_route_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            for batch in history["holdouts"]:
                for job in batch["jobs"]:
                    job["result"]["category"] = "multi_step_agentic"
            report = certifier.certify(history, certifier.default_policy())
            self.assertFalse(report["gates"]["category_mix"])
            self.assertFalse(report["gates"]["independent_holdout_batches"])

    def test_user_simulator_needs_multiple_real_outcome_categories(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            for batch in history["holdouts"]:
                for job in batch["jobs"]:
                    for episode in job["result"]["live_rollout"]["episodes"]:
                        result = episode["trajectory"][1]["result"]
                        result.update(
                            match_status="matched",
                            outcome_category="user_acceptance",
                            reason_code="accepted",
                        )
            report = certifier.certify(history, certifier.default_policy())
            self.assertFalse(report["gates"]["user_simulator_outcome_coverage"])

    def test_holdout_summary_cannot_hide_reused_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            first = history["holdouts"][0]["jobs"][0]["result"]
            second = history["holdouts"][1]["jobs"][0]["result"]
            second["sample_seed"] = first["sample_seed"]
            report = certifier.certify(history, certifier.default_policy())
            self.assertFalse(report["gates"]["fresh_holdout"])
            self.assertFalse(report["measurements"]["fresh_holdout_evidence"]["verified"])

    def test_transition_terminal_flags_must_match_episode_termination(self):
        with tempfile.TemporaryDirectory() as directory:
            history = self.make_history(Path(directory))
            episode = history["holdouts"][0]["jobs"][0]["result"]["live_rollout"]["episodes"][0]
            episode["transitions"][-1]["terminated"] = False
            report = certifier.certify(history, certifier.default_policy())
            self.assertFalse(report["gates"]["trajectory_schema"])

    def test_material_manifest_detects_post_certification_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sandbox = root / "sandbox"
            sandbox.mkdir()
            task = sandbox / "task.json"
            task.write_text('{"task":"immutable"}')
            rollout = {"episodes": [{"agent_success": True}]}
            (sandbox / "live_rollout.json").write_text(json.dumps(rollout))
            item = {
                "task_path": str(task),
                "task_sha256": __import__("hashlib").sha256(task.read_bytes()).hexdigest(),
                "sandbox_root": str(sandbox),
                "sandbox_evidence_fingerprint": "sandbox-proof",
                "rollout_sha256": verifier.digest_json(rollout),
                "episode_count": 1,
            }
            manifest = {"version": "1.0", "kind": "agentic_rl_pretraining_materials", "items": [item]}
            manifest["dataset_sha256"] = verifier.digest_json(manifest)
            fingerprint = lambda sandbox_root, project: "sandbox-proof"
            self.assertTrue(verifier.verify(manifest, ROOT, fingerprint=fingerprint)["verified"])
            task.write_text('{"task":"tampered"}')
            report = verifier.verify(manifest, ROOT, fingerprint=fingerprint)
            self.assertFalse(report["verified"])
            self.assertIn("task_digest", report["failed_gates"])

    def test_failed_artifact_verification_revokes_certification(self):
        report = {"certified": True, "gates": {}, "failed_gates": []}
        certifier.attach_artifact_verification(report, {
            "verified": False, "failed_gates": ["task_digest"],
        })
        self.assertFalse(report["certified"])
        self.assertIn("material_artifacts_immutable", report["failed_gates"])

    def test_portable_bundle_is_a_final_certification_gate(self):
        report = {
            "certified": True,
            "gates": {"base": True},
            "failed_gates": [],
            "materials_manifest": {"dataset_sha256": "dataset"},
        }
        certifier.attach_bundle_verification(report, {
            "verified": True, "source_dataset_sha256": "different",
        })
        self.assertFalse(report["certified"])
        self.assertIn("portable_materials_bundle", report["failed_gates"])
        certifier.attach_bundle_verification(report, {
            "verified": True, "source_dataset_sha256": "dataset",
        })
        self.assertTrue(report["certified"])

    def test_v2_manifest_tracks_sandbox_files_independently_of_evaluator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "task.json"
            app = root / "app.py"
            rollout = {"episodes": []}
            task.write_text('{"task":"immutable"}')
            app.write_text("# v1\n")
            (root / "live_rollout.json").write_text(json.dumps(rollout))
            (root / "acceptance_result.json").write_text('{"business_acceptance":"passed"}')
            item = {
                "task_path": str(task),
                "task_sha256": __import__("hashlib").sha256(task.read_bytes()).hexdigest(),
                "sandbox_root": str(root),
                "sandbox_evidence_fingerprint": "historical-evaluator-proof",
                "sandbox_artifacts_sha256": verifier.portable_artifact_digests(root),
                "sandbox_evidence_sha256": verifier.evidence_artifact_digests(root),
                "rollout_sha256": verifier.digest_json(rollout),
                "episode_count": 0,
            }
            manifest = {
                "version": "2.0", "kind": "agentic_rl_pretraining_materials",
                "evaluator_source_digest": "old-evaluator", "items": [item],
            }
            manifest["dataset_sha256"] = verifier.digest_json(manifest)
            self.assertTrue(verifier.verify(manifest, ROOT)["verified"])
            app.write_text("# changed\n")
            report = verifier.verify(manifest, ROOT)
            self.assertFalse(report["verified"])
            self.assertIn("sandbox_digest", report["failed_gates"])

    def test_v2_manifest_detects_changed_certification_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "task.json"
            rollout = {"episodes": []}
            task.write_text('{"task":"immutable"}')
            (root / "app.py").write_text("# app\n")
            (root / "acceptance_result.json").write_text('{"business_acceptance":"passed"}')
            (root / "live_rollout.json").write_text(json.dumps(rollout))
            item = {
                "task_path": str(task),
                "task_sha256": __import__("hashlib").sha256(task.read_bytes()).hexdigest(),
                "sandbox_root": str(root),
                "sandbox_evidence_fingerprint": "sandbox-proof",
                "sandbox_artifacts_sha256": verifier.portable_artifact_digests(root),
                "sandbox_evidence_sha256": verifier.evidence_artifact_digests(root),
                "rollout_sha256": verifier.digest_json(rollout),
                "episode_count": 0,
            }
            manifest = {
                "version": "2.0", "kind": "agentic_rl_pretraining_materials",
                "evaluator_source_digest": "evaluator", "items": [item],
            }
            manifest["dataset_sha256"] = verifier.digest_json(manifest)
            (root / "acceptance_result.json").write_text('{"business_acceptance":"failed"}')
            report = verifier.verify(manifest, ROOT)
            self.assertFalse(report["verified"])
            self.assertIn("sandbox_evidence_digest", report["failed_gates"])

    def test_v2_manifest_detects_added_executable_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "task.json"
            rollout = {"episodes": []}
            task.write_text('{"task":"immutable"}')
            (root / "app.py").write_text("# app\n")
            (root / "live_rollout.json").write_text(json.dumps(rollout))
            (root / "acceptance_result.json").write_text('{"business_acceptance":"passed"}')
            item = {
                "task_path": str(task),
                "task_sha256": __import__("hashlib").sha256(task.read_bytes()).hexdigest(),
                "sandbox_root": str(root),
                "sandbox_evidence_fingerprint": "sandbox-proof",
                "sandbox_artifacts_sha256": verifier.portable_artifact_digests(root),
                "sandbox_evidence_sha256": verifier.evidence_artifact_digests(root),
                "rollout_sha256": verifier.digest_json(rollout),
                "episode_count": 0,
            }
            manifest = {
                "version": "2.0", "kind": "agentic_rl_pretraining_materials",
                "evaluator_source_digest": "evaluator", "items": [item],
            }
            manifest["dataset_sha256"] = verifier.digest_json(manifest)
            (root / "helper.py").write_text("# injected\n")
            report = verifier.verify(manifest, ROOT)
            self.assertFalse(report["verified"])
            self.assertIn("sandbox_digest", report["failed_gates"])


if __name__ == "__main__":
    unittest.main()

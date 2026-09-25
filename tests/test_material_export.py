import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from env_factory.material_artifacts import (
    digest_json,
    evidence_artifact_digests,
    portable_artifact_digests,
)
from env_factory.execution_provenance import collect_execution_provenance


ROOT = Path(__file__).resolve().parents[1]


def load_exporter():
    spec = importlib.util.spec_from_file_location(
        "export_training_materials", ROOT / "scripts/export_training_materials.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exporter = load_exporter()


class MaterialExportTest(unittest.TestCase):
    def source(self, root: Path):
        sandbox = root / "sandbox"
        sandbox.mkdir()
        task = sandbox / "task.json"
        task.write_text('{"task":"use the tool"}')
        (sandbox / "app.py").write_text("# portable runtime\n")
        (sandbox / "acceptance_result.json").write_text(
            '{"business_acceptance":"passed"}'
        )
        transition = {
            "step": 0,
            "agent_input": [{"role": "user", "content": "use the tool"}],
            "assistant_output": '{"kind":"respond","content":"done"}',
            "observation": {},
            "action": {"kind": "respond", "content": "done"},
            "result": {"status": 200},
            "next_observation": {},
            "reward": 1.0,
            "terminated": True,
            "truncated": False,
            "trainer_metadata": {
                "user_simulator": {"outcome_category": "user_acceptance"},
            },
        }
        rollout = {
            "schema_version": "2.0",
            "material_visibility_version": "1.0",
            "agent_model": "policy",
            "runtime_model": "simulator",
            "episodes": [{
                "schema_version": "2.0", "seed": 1, "agent_success": True,
                "termination": "completed", "transitions": [transition],
                "initial_reward": 0.0, "final_reward": 1.0,
                "trajectory": [{
                    "method": "GET", "path": "/v1/reward", "status": 200,
                    "result": {"reward": 1.0},
                }],
                "replay": {"events": []},
                "initial_state": {}, "final_state": {},
                "usage": [{}], "issues": [],
            }],
        }
        (sandbox / "live_rollout.json").write_text(json.dumps(rollout))
        item = {
            "task_path": str(task),
            "task_sha256": hashlib.sha256(task.read_bytes()).hexdigest(),
            "sandbox_root": str(sandbox),
            "sandbox_evidence_fingerprint": "f" * 64,
            "sandbox_artifacts_sha256": portable_artifact_digests(sandbox),
            "sandbox_evidence_sha256": evidence_artifact_digests(sandbox),
            "category": "simple_agentic",
            "score": 9.0,
            "rollout_sha256": digest_json(rollout),
            "episode_count": 1,
            "successful_episodes": 1,
        }
        manifest = {
            "version": "2.0",
            "kind": "agentic_rl_pretraining_materials",
            "evaluator_source_digest": "source",
            "execution_provenance": collect_execution_provenance(ROOT),
            "items": [item],
        }
        manifest["dataset_sha256"] = digest_json(manifest)
        certification = {
            "certification": "production_prepared_for_agentic_rl",
            "scope": "pre_training_material_readiness",
            "certified": True,
            "does_not_certify": [
                "rl_training_convergence",
                "post_training_policy_improvement",
                "cross_model_generalization",
            ],
            "policy": {"score_threshold": 8.0},
            "measurements": {"training_ready": 1},
            "gates": {"fixture": True},
            "failed_gates": [],
            "material_verification": {"verified": True},
            "materials_manifest": manifest,
        }
        return certification

    def test_export_is_portable_and_detects_post_export_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            bundle = root / "bundle"
            report = exporter.export_bundle(certification, bundle, ROOT)
            self.assertTrue(report["verified"], report)
            self.assertEqual(report["items"], 1)
            self.assertEqual(report["transitions"], 1)
            record = json.loads((bundle / "transitions.jsonl").read_text())
            self.assertEqual(record["transition"]["reward"], 1.0)
            self.assertEqual(record["episode_final_reward"], 1.0)
            self.assertEqual(record["agent_usage"], {})
            self.assertNotIn("trainer_metadata", record["transition"])
            card = json.loads((bundle / "dataset_card.json").read_text())
            self.assertEqual(card["composition"]["items"], 1)
            self.assertEqual(card["composition"]["transitions"], 1)
            self.assertEqual(
                card["distribution_status"],
                "internal_only_until_legal_and_security_review",
            )
            portable = json.loads((bundle / "certification.json").read_text())
            self.assertNotIn("materials_manifest", portable)
            self.assertTrue(portable["certified"])
            self.assertEqual(
                card["build_environment"], portable["execution_provenance"]
            )
            self.assertEqual(
                json.loads((bundle / "bundle_manifest.json").read_text())[
                    "execution_provenance_sha256"
                ],
                digest_json(card["build_environment"]),
            )
            copied_app = next((bundle / "environments").glob("*/app.py"))
            copied_app.write_text("# tampered\n")
            changed = exporter.verify_bundle(bundle)
            self.assertFalse(changed["verified"])
            self.assertIn("bundle_files", changed["failed_gates"])

    def test_bundle_verifier_rejects_semantically_rewritten_transition_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            bundle = root / "bundle"
            exporter.export_bundle(certification, bundle, ROOT)
            record = json.loads((bundle / "transitions.jsonl").read_text())
            record["transition"]["terminated"] = False
            (bundle / "transitions.jsonl").write_text(json.dumps(record) + "\n")
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files_sha256"]["transitions.jsonl"] = exporter.file_sha256(
                bundle / "transitions.jsonl"
            )
            unsigned = {key: value for key, value in manifest.items() if key != "bundle_sha256"}
            manifest["bundle_sha256"] = digest_json(unsigned)
            manifest_path.write_text(json.dumps(manifest))
            report = exporter.verify_bundle(bundle)
            self.assertFalse(report["verified"])
            self.assertIn("transition_projection", report["failed_gates"])

    def test_bundle_verifier_rejects_rehashed_misleading_dataset_card(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            bundle = root / "bundle"
            exporter.export_bundle(certification, bundle, ROOT)
            card_path = bundle / "dataset_card.json"
            card = json.loads(card_path.read_text())
            card["composition"]["transitions"] = 999
            card["license_status"] = "public_domain"
            card_path.write_text(json.dumps(card))
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files_sha256"]["dataset_card.json"] = exporter.file_sha256(card_path)
            unsigned = {key: value for key, value in manifest.items() if key != "bundle_sha256"}
            manifest["bundle_sha256"] = digest_json(unsigned)
            manifest_path.write_text(json.dumps(manifest))
            report = exporter.verify_bundle(bundle)
            self.assertFalse(report["verified"])
            self.assertIn("dataset_card", report["failed_gates"])

    def test_bundle_verifier_rejects_rehashed_environment_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            bundle = root / "bundle"
            exporter.export_bundle(certification, bundle, ROOT)
            card_path = bundle / "dataset_card.json"
            card = json.loads(card_path.read_text())
            card["build_environment"]["python"]["version"] = "0.0"
            card_path.write_text(json.dumps(card))
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files_sha256"]["dataset_card.json"] = exporter.file_sha256(
                card_path
            )
            unsigned = {
                key: value for key, value in manifest.items()
                if key != "bundle_sha256"
            }
            manifest["bundle_sha256"] = digest_json(unsigned)
            manifest_path.write_text(json.dumps(manifest))
            report = exporter.verify_bundle(bundle)
            self.assertFalse(report["verified"])
            self.assertIn("dataset_card", report["failed_gates"])

    def test_uncertified_report_cannot_be_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            certification["certified"] = False
            with self.assertRaisesRegex(ValueError, "certified"):
                exporter.export_bundle(certification, root / "bundle", ROOT)

    def test_export_rejects_action_that_does_not_match_raw_model_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            item = certification["materials_manifest"]["items"][0]
            rollout = json.loads(
                (Path(item["sandbox_root"]) / "live_rollout.json").read_text()
            )
            rollout["episodes"][0]["transitions"][0]["action"] = {
                "kind": "respond", "content": "fabricated",
            }
            with self.assertRaisesRegex(ValueError, "parsed_action_mismatch"):
                exporter._transition_records("a" * 16 + "-" + "b" * 16, item, rollout)


if __name__ == "__main__":
    unittest.main()

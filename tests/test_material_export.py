import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import subprocess

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
    def generation_provenance(self):
        return {
            "version": "1.0",
            "task_id": "task-1",
            "batch_index": 1,
            "run_seed": 100,
            "sample_seed": 101,
            "task_sha256": "PLACEHOLDER",
            "training_category": "simple_agentic",
            "resolved_task_intent": "tool_execution",
            "hops": 3,
            "generator_provider": {
                "host": "generator.example",
                "model": "generator-model",
                "identity_sha256": "c" * 64,
            },
            "generation_settings": {
                "route_attempt_limit": 3,
                "timeout_seconds": 60.0,
                "network_retries": 2,
            },
            "successful_attempt": 1,
            "attempts": [{
                "attempt": 1,
                "seed": 101,
                "status": "completed",
                "llm_trace": {
                    "version": "1.0",
                    "responses": 2,
                    "models": {"generator-model": 2},
                    "finish_reasons": {"stop": 2},
                    "usage": {"total_tokens": 42},
                    "response_id_sha256": ["d" * 64, "e" * 64],
                },
            }],
        }

    def signing_keys(self, root: Path) -> tuple[Path, Path]:
        private = root / "private.pem"
        public = root / "public.pem"
        subprocess.run(
            ["openssl", "genpkey", "-algorithm", "ED25519", "-out", str(private)],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["openssl", "pkey", "-in", str(private), "-pubout", "-out", str(public)],
            check=True, capture_output=True,
        )
        return private, public

    def source(self, root: Path):
        sandbox = root / "sandbox"
        sandbox.mkdir()
        task = sandbox / "task.json"
        task.write_text(json.dumps({
            "task": "use the tool",
            "requirements": {"runtime_interface": {
                "protocol": "http", "base_path": "/v1", "version": "1.0",
            }},
        }))
        (sandbox / "app.py").write_text("# portable runtime\n")
        (sandbox / "Dockerfile").write_text(
            "FROM python@sha256:" + "a" * 64 + "\nUSER sandbox\n"
        )
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
            "generation_provenance": self.generation_provenance(),
        }
        item["generation_provenance"]["task_sha256"] = item["task_sha256"]
        manifest = {
            "version": "4.0",
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
                "rl_training_execution",
                "downstream_training_system_compatibility",
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
            self.assertTrue(report["production_contract_ready"])
            record = json.loads((bundle / "transitions.jsonl").read_text())
            self.assertEqual(record["transition"]["reward"], 1.0)
            self.assertEqual(record["episode_final_reward"], 1.0)
            self.assertEqual(record["agent_usage"], {})
            manifest = json.loads((bundle / "bundle_manifest.json").read_text())
            self.assertEqual(
                record["task_family_id"], manifest["items"][0]["task_family_id"]
            )
            self.assertNotIn("trainer_metadata", record["transition"])
            card = json.loads((bundle / "dataset_card.json").read_text())
            self.assertEqual(card["composition"]["items"], 1)
            self.assertEqual(card["composition"]["transitions"], 1)
            self.assertEqual(
                card["composition"]["task_generation"]["configured_models"],
                {"generator-model": 1},
            )
            self.assertEqual(
                card["composition"]["task_generation"]["actual_response_models"],
                {"generator-model": 2},
            )
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
            contract = json.loads((bundle / "consumer_contract.json").read_text())
            self.assertEqual(contract["bundle_version"], "8.0")
            self.assertEqual(
                contract["records"]["policy_transition_fields"],
                exporter.consumer_contract()["records"]["policy_transition_fields"],
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

    def test_bundle_verifier_rejects_rehashed_generation_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            bundle = root / "bundle"
            exporter.export_bundle(certification, bundle, ROOT)
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["items"][0]["generation_provenance"]["sample_seed"] = 999
            unsigned = {
                key: value for key, value in manifest.items()
                if key != "bundle_sha256"
            }
            manifest["bundle_sha256"] = digest_json(unsigned)
            manifest_path.write_text(json.dumps(manifest))
            report = exporter.verify_bundle(bundle)
            self.assertFalse(report["verified"])
            self.assertIn("generation_provenance", report["failed_gates"])

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

    def test_bundle_verifier_rejects_rehashed_consumer_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            bundle = root / "bundle"
            exporter.export_bundle(certification, bundle, ROOT)
            contract_path = bundle / "consumer_contract.json"
            contract = json.loads(contract_path.read_text())
            contract["visibility"]["trainer_only"] = "transitions.jsonl"
            contract_path.write_text(json.dumps(contract))
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files_sha256"]["consumer_contract.json"] = exporter.file_sha256(
                contract_path
            )
            unsigned = {
                key: value for key, value in manifest.items()
                if key != "bundle_sha256"
            }
            manifest["bundle_sha256"] = digest_json(unsigned)
            manifest_path.write_text(json.dumps(manifest))
            report = exporter.verify_bundle(bundle)
            self.assertFalse(report["verified"])
            self.assertFalse(report["production_contract_ready"])
            self.assertIn("consumer_contract", report["failed_gates"])

    def test_legacy_v3_bundle_keeps_integrity_status_but_not_production_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            bundle = root / "bundle"
            exporter.export_bundle(certification, bundle, ROOT)
            card_path = bundle / "dataset_card.json"
            card = json.loads(card_path.read_text())
            card["version"] = "1.0"
            card.pop("consumer_contract")
            card_path.write_text(json.dumps(card))
            contract_path = bundle / "consumer_contract.json"
            contract_path.unlink()
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["version"] = "3.0"
            manifest.pop("consumer_contract_file")
            manifest["items"][0].pop("split")
            manifest["items"][0].pop("task_family_id")
            manifest["items"][0].pop("generation_provenance")
            transition_path = bundle / "transitions.jsonl"
            record = json.loads(transition_path.read_text())
            record.pop("split")
            record.pop("task_family_id")
            record.pop("generation_model")
            record.pop("generation_provider_identity_sha256")
            record.pop("generation_sample_seed")
            transition_path.write_text(json.dumps(record) + "\n")
            manifest["files_sha256"].pop("consumer_contract.json")
            manifest["files_sha256"]["transitions.jsonl"] = exporter.file_sha256(
                transition_path
            )
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
            self.assertTrue(report["verified"], report)
            self.assertFalse(report["production_contract_ready"])

    def test_legacy_v4_bundle_keeps_contract_but_has_no_trusted_attestation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            bundle = root / "bundle"
            exporter.export_bundle(certification, bundle, ROOT)
            contract_path = bundle / "consumer_contract.json"
            contract = exporter.consumer_contract("4.0")
            contract_path.write_text(json.dumps(contract))
            card_path = bundle / "dataset_card.json"
            card = json.loads(card_path.read_text())
            card["version"] = "1.1"
            card_path.write_text(json.dumps(card))
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["version"] = "4.0"
            manifest.pop("attestation")
            manifest["items"][0].pop("split")
            manifest["items"][0].pop("task_family_id")
            manifest["items"][0].pop("generation_provenance")
            transition_path = bundle / "transitions.jsonl"
            record = json.loads(transition_path.read_text())
            record.pop("split")
            record.pop("task_family_id")
            record.pop("generation_model")
            record.pop("generation_provider_identity_sha256")
            record.pop("generation_sample_seed")
            transition_path.write_text(json.dumps(record) + "\n")
            manifest["files_sha256"]["consumer_contract.json"] = exporter.file_sha256(
                contract_path
            )
            manifest["files_sha256"]["dataset_card.json"] = exporter.file_sha256(
                card_path
            )
            manifest["files_sha256"]["transitions.jsonl"] = exporter.file_sha256(
                transition_path
            )
            unsigned = {
                key: value for key, value in manifest.items()
                if key != "bundle_sha256"
            }
            manifest["bundle_sha256"] = digest_json(unsigned)
            manifest_path.write_text(json.dumps(manifest))
            report = exporter.verify_bundle(bundle)
            self.assertTrue(report["verified"], report)
            self.assertTrue(report["production_contract_ready"])
            self.assertFalse(report["trusted_attestation"])

    def test_rehashed_bundle_cannot_remove_environment_rebuild_entrypoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            bundle = root / "bundle"
            exporter.export_bundle(certification, bundle, ROOT)
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            item = manifest["items"][0]
            relative = f"{item['environment_path']}/Dockerfile"
            (bundle / relative).unlink()
            item["files_sha256"].pop("Dockerfile")
            manifest["files_sha256"].pop(relative)
            unsigned = {
                key: value for key, value in manifest.items()
                if key != "bundle_sha256"
            }
            manifest["bundle_sha256"] = digest_json(unsigned)
            manifest_path.write_text(json.dumps(manifest))
            report = exporter.verify_bundle(bundle)
            self.assertFalse(report["verified"])
            self.assertIn(
                "environment_reconstruction_contract", report["failed_gates"]
            )

    def test_signed_bundle_requires_the_trusted_key_and_detects_reserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private, public = self.signing_keys(root)
            certification = self.source(root)
            bundle = root / "bundle"
            report = exporter.export_bundle(
                certification, bundle, ROOT,
                signing_private_key=private, trusted_public_key=public,
            )
            self.assertTrue(report["verified"], report)
            self.assertTrue(report["trusted_attestation"])
            without_trust = exporter.verify_bundle(bundle)
            self.assertTrue(without_trust["verified"])
            self.assertFalse(without_trust["trusted_attestation"])

            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest_path.write_text(json.dumps(manifest, separators=(",", ":")))
            changed = exporter.verify_bundle(bundle, trusted_public_key=public)
            self.assertFalse(changed["verified"])
            self.assertFalse(changed["trusted_attestation"])
            self.assertIn("trusted_attestation", changed["failed_gates"])

    def test_rehashed_item_and_records_cannot_change_content_based_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            bundle = root / "bundle"
            exporter.export_bundle(certification, bundle, ROOT)
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            item = manifest["items"][0]
            self.assertEqual(item["split"], "train")
            item["split"] = "test"

            transition_path = bundle / "transitions.jsonl"
            record = json.loads(transition_path.read_text())
            record["split"] = "test"
            transition_path.write_text(json.dumps(record) + "\n")
            manifest["files_sha256"]["transitions.jsonl"] = exporter.file_sha256(
                transition_path
            )

            card_path = bundle / "dataset_card.json"
            card = json.loads(card_path.read_text())
            card["composition"]["splits"] = {
                "train": 0, "validation": 0, "test": 1,
            }
            card["composition"]["category_splits"]["simple_agentic"] = {
                "train": 0, "validation": 0, "test": 1,
            }
            card_path.write_text(json.dumps(card))
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
            self.assertIn("dataset_split_assignment", report["failed_gates"])

    def test_rehashed_bundle_cannot_forge_task_family_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            bundle = root / "bundle"
            exporter.export_bundle(certification, bundle, ROOT)
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["items"][0]["task_family_id"] = "0" * 64
            transition_path = bundle / "transitions.jsonl"
            record = json.loads(transition_path.read_text())
            record["task_family_id"] = "0" * 64
            transition_path.write_text(json.dumps(record) + "\n")
            manifest["files_sha256"]["transitions.jsonl"] = exporter.file_sha256(
                transition_path
            )
            unsigned = {
                key: value for key, value in manifest.items()
                if key != "bundle_sha256"
            }
            manifest["bundle_sha256"] = digest_json(unsigned)
            manifest_path.write_text(json.dumps(manifest))
            report = exporter.verify_bundle(bundle)
            self.assertFalse(report["verified"])
            self.assertIn("task_family_identity", report["failed_gates"])

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

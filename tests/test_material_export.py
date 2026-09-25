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
        }
        rollout = {
            "schema_version": "2.0",
            "agent_model": "policy",
            "runtime_model": "simulator",
            "episodes": [{
                "schema_version": "2.0", "seed": 1, "agent_success": True,
                "termination": "completed", "transitions": [transition],
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
            "items": [item],
        }
        manifest["dataset_sha256"] = digest_json(manifest)
        certification = {
            "certified": True,
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
            copied_app = next((bundle / "environments").glob("*/app.py"))
            copied_app.write_text("# tampered\n")
            changed = exporter.verify_bundle(bundle)
            self.assertFalse(changed["verified"])
            self.assertIn("bundle_files", changed["failed_gates"])

    def test_uncertified_report_cannot_be_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            certification = self.source(root)
            certification["certified"] = False
            with self.assertRaisesRegex(ValueError, "certified"):
                exporter.export_bundle(certification, root / "bundle", ROOT)


if __name__ == "__main__":
    unittest.main()

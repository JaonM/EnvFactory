import json
from pathlib import Path
import tempfile
import unittest

from env_factory.runtime_provenance import valid_container_rollout_execution


class RuntimeProvenanceTest(unittest.TestCase):
    def evidence(self, root: Path):
        image_id = "sha256:" + "a" * 64
        (root / "docker_image_metadata.json").write_text(json.dumps({
            "version": "3.0",
            "image_id": image_id,
            "runtime_user": "sandbox",
            "smoke_test": {
                "passed": True,
                "read_only_root": True,
                "cap_drop": "ALL",
                "no_new_privileges": True,
                "non_root_user": True,
            },
        }))
        return {
            "runtime_execution": {
                "version": "1.0",
                "mode": "docker_http",
                "container_image_id": image_id,
                "transport": "loopback_http",
                "read_only_root": True,
                "cap_drop": "ALL",
                "no_new_privileges": True,
                "non_root_user": True,
            },
        }

    def test_rollout_is_bound_to_validated_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rollout = self.evidence(root)
            self.assertTrue(valid_container_rollout_execution(rollout, root))
            rollout["runtime_execution"]["container_image_id"] = "sha256:" + "b" * 64
            self.assertFalse(valid_container_rollout_execution(rollout, root))

    def test_in_process_rollout_is_not_production_container_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.evidence(root)
            rollout = {"runtime_execution": {
                "version": "1.0", "mode": "in_process", "container_image_id": None,
            }}
            self.assertFalse(valid_container_rollout_execution(rollout, root))


if __name__ == "__main__":
    unittest.main()

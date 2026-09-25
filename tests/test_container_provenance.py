import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import importlib.util

from env_factory.container_provenance import verify_container_provenance


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "resolve_container_image", ROOT / "scripts/resolve_container_image.py"
)
resolver = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(resolver)


class ContainerProvenanceTest(unittest.TestCase):
    def test_manifest_resolution_selects_only_the_requested_platform(self):
        manifest = [
            {"Descriptor": {
                "digest": "sha256:" + "a" * 64,
                "platform": {"os": "linux", "architecture": "amd64"},
            }},
            {"Descriptor": {
                "digest": "sha256:" + "b" * 64,
                "platform": {"os": "linux", "architecture": "arm64"},
            }},
        ]
        self.assertEqual(
            resolver.resolve_reference(
                manifest, "registry.example/python:3.14", "linux", "aarch64"
            ),
            "registry.example/python@sha256:" + "b" * 64,
        )
        self.assertIsNone(
            resolver.resolve_reference(
                manifest, "registry.example/python:3.14", "windows", "arm64"
            )
        )

    def fixture(self, root: Path):
        image = "registry.example/python@sha256:" + "a" * 64
        dockerfile = root / "Dockerfile"
        requirements = root / "requirements-dev.txt"
        dockerfile.write_text(f"FROM {image}\nUSER sandbox\n")
        requirements.write_text("pluggy==1.6.0\npytest==9.1.1\n")
        metadata = {
            "version": "2.0",
            "base_image": image,
            "image_id": "sha256:" + "b" * 64,
            "runtime_user": "sandbox",
            "platform": {"os": "linux", "architecture": "amd64"},
            "dockerfile_sha256": hashlib.sha256(dockerfile.read_bytes()).hexdigest(),
            "requirements_sha256": hashlib.sha256(requirements.read_bytes()).hexdigest(),
            "smoke_test": {
                "passed": True,
                "network": "none",
                "read_only_root": True,
                "cap_drop": "ALL",
                "no_new_privileges": True,
                "non_root_user": True,
            },
        }
        (root / "docker_image_metadata.json").write_text(json.dumps(metadata))
        return metadata

    def test_verified_provenance_binds_pinned_inputs_and_security_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            self.assertTrue(verify_container_provenance(root)["verified"])

    def test_floating_base_image_and_dependency_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = self.fixture(root)
            (root / "Dockerfile").write_text("FROM python:3.14-slim\nUSER sandbox\n")
            (root / "requirements-dev.txt").write_text("pytest>=8,<10\n")
            metadata["dockerfile_sha256"] = hashlib.sha256(
                (root / "Dockerfile").read_bytes()
            ).hexdigest()
            metadata["requirements_sha256"] = hashlib.sha256(
                (root / "requirements-dev.txt").read_bytes()
            ).hexdigest()
            (root / "docker_image_metadata.json").write_text(json.dumps(metadata))
            report = verify_container_provenance(root)
            self.assertFalse(report["verified"])
            self.assertIn("dockerfile_base_image", report["failed_gates"])
            self.assertIn("dependency_pins", report["failed_gates"])

    def test_metadata_cannot_hide_changed_inputs_or_root_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = self.fixture(root)
            (root / "requirements-dev.txt").write_text("pytest==9.1.0\n")
            metadata["runtime_user"] = "root"
            (root / "docker_image_metadata.json").write_text(json.dumps(metadata))
            report = verify_container_provenance(root)
            self.assertIn("requirements_digest", report["failed_gates"])
            self.assertIn("non_root_user", report["failed_gates"])


if __name__ == "__main__":
    unittest.main()

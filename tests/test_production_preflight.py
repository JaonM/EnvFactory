from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from env_factory.production_preflight import (
    run_production_preflight,
    valid_production_preflight,
)


class Usage:
    free = 20 * 1024**3


class ProductionPreflightTest(unittest.TestCase):
    def project(self, root: Path) -> Path:
        for relative in (
            "pyproject.toml", "uv.lock", "scripts/loop_experiment.py",
            "scripts/develop_sandbox_with_agent.sh",
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n")
        return root

    @staticmethod
    def runner(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, "linux arm64\n", "")

    def test_complete_nonsecret_preflight_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.project(Path(directory))
            private = root / "private.pem"
            public = root / "public.pem"
            private.write_text("private")
            public.write_text("public")
            with patch(
                "env_factory.production_preflight.private_key_public_identity",
                return_value="a" * 64,
            ), patch(
                "env_factory.production_preflight.public_key_identity",
                return_value="a" * 64,
            ):
                report = run_production_preflight(
                    root, root,
                    signing_private_key=private,
                    trusted_public_key=public,
                    environment={
                        "LLM_API_KEY": "secret-value",
                        "LLM_MODEL": "generator-model",
                        "LLM_BASE_URL": "https://agent.example/v1",
                        "SANDBOX_LLM_MODEL": "runtime-model",
                        "SANDBOX_LLM_BASE_URL": "https://runtime.example/v1",
                    },
                    runner=self.runner,
                    which=lambda name: f"/usr/bin/{name}",
                    disk_usage=lambda path: Usage(),
                )
            self.assertTrue(report["ready"])
            self.assertTrue(valid_production_preflight(report))
            self.assertNotIn("secret-value", str(report))
            model = next(
                item for item in report["checks"]
                if item["name"] == "model_configuration"
            )
            self.assertEqual(model["evidence"]["agent_host"], "agent.example")
            self.assertTrue(valid_production_preflight(
                report,
                expected_generation_provider=model["evidence"][
                    "generation_provider"
                ],
                expected_agent_provider=model["evidence"]["agent_provider"],
                expected_runtime_provider=model["evidence"]["runtime_provider"],
                expected_signing_key_identity="a" * 64,
            ))
            mismatched = dict(model["evidence"]["runtime_provider"])
            mismatched["model"] = "different"
            self.assertFalse(valid_production_preflight(
                report,
                expected_agent_provider=model["evidence"]["agent_provider"],
                expected_runtime_provider=mismatched,
            ))
            self.assertFalse(valid_production_preflight(
                report,
                expected_signing_key_identity="b" * 64,
            ))

    def test_unavailable_docker_and_bad_runtime_limits_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.project(Path(directory))
            report = run_production_preflight(
                root, root,
                signing_private_key=root / "missing-private.pem",
                trusted_public_key=root / "missing-public.pem",
                environment={
                    "LLM_API_KEY": "key", "LLM_MODEL": "model",
                    "SANDBOX_LLM_TIMEOUT_SECONDS": "invalid",
                    "SANDBOX_LLM_MAX_RETRIES": "99",
                },
                which=lambda name: None if name == "docker" else f"/usr/bin/{name}",
                disk_usage=lambda path: type("Low", (), {"free": 1})(),
            )
            self.assertFalse(report["ready"])
            self.assertFalse(valid_production_preflight(report))
            self.assertEqual(
                set(report["failed_checks"]),
                {
                    "docker_daemon", "required_executables",
                    "model_runtime_limits", "bundle_signing_identity",
                    "workspace_capacity",
                },
            )

    def test_preflight_validator_rejects_forged_or_incomplete_evidence(self):
        self.assertFalse(valid_production_preflight({"ready": True}))
        report = {
            "version": "1.0",
            "scope": "production_pre_training_material_experiment",
            "network_probe_performed": False,
            "ready": True,
            "failed_checks": [],
            "checks": [],
        }
        self.assertFalse(valid_production_preflight(report))


if __name__ == "__main__":
    unittest.main()

import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "assess_task_buildability", ROOT / "scripts/assess_task_buildability.py"
)
buildability = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(buildability)


class BuildabilityTest(unittest.TestCase):
    def write_task(self, root: Path, mode: str = "stateless") -> None:
        (root / "task.json").write_text(json.dumps({
            "task": "整理公开输入",
            "environment_plan": {"mode": mode},
            "task_spec": {},
            "tools": [], "noise_tools": [], "tool_implementations": [],
            "metric_implementations": [], "artifacts": {},
        }), encoding="utf-8")

    def test_external_task_without_provider_is_owned_by_task_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_task(root, "external_capability")
            quality = SimpleNamespace(to_dict=lambda: {"eligible": True, "score": 9.0})
            with (
                patch.object(buildability, "score_file", return_value=quality),
                patch.object(buildability, "validate_task_spec"),
                patch.dict(os.environ, {}, clear=True),
            ):
                report = buildability.assess(root)
            self.assertFalse(report["buildable"])
            self.assertEqual(report["failure_owner"], "task_generation")
            self.assertIn("EXTERNAL_CAPABILITY_UNAVAILABLE", {
                item["code"] for item in report["issues"]
            })

    def test_valid_platform_contract_reaches_builder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_task(root)
            quality = SimpleNamespace(to_dict=lambda: {"eligible": True, "score": 9.0})
            with (
                patch.object(buildability, "score_file", return_value=quality),
                patch.object(buildability, "validate_task_spec"),
            ):
                report = buildability.assess(root)
            self.assertTrue(report["buildable"])
            self.assertEqual(report["issues"], [])

    def test_project_relative_manifest_is_resolved_from_self_contained_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_task(root, "reference_data")
            schema = root / "schemas" / "items.json"
            schema.parent.mkdir()
            schema.write_text(json.dumps({
                "columns": [{"name": "id", "type": "INTEGER", "nullable": False}],
                "primary_key": ["id"],
            }), encoding="utf-8")
            rows = root / "rows" / "items.jsonl"
            rows.parent.mkdir()
            rows.write_text('{"id":1}\n', encoding="utf-8")
            task = json.loads((root / "task.json").read_text(encoding="utf-8"))
            task["artifacts"] = {"data_manifest": {
                "root": "output/task_artifacts/task-42",
                "tables": [{"table_name": "items", "schema_file": "schemas/items.json",
                            "rows_file": "rows/items.jsonl"}],
            }}
            (root / "task.json").write_text(json.dumps(task), encoding="utf-8")
            quality = SimpleNamespace(to_dict=lambda: {"eligible": True, "score": 9.0})
            with (
                patch.object(buildability, "score_file", return_value=quality),
                patch.object(buildability, "validate_task_spec"),
            ):
                report = buildability.assess(root)
            self.assertTrue(report["buildable"])
            self.assertEqual(report["issues"], [])

    def test_invalid_business_foreign_key_fails_before_builder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_task(root, "reference_data")
            (root / "schemas").mkdir()
            (root / "rows").mkdir()
            (root / "schemas/users.json").write_text(json.dumps({
                "columns": [{"name": "id", "type": "INTEGER", "nullable": False}],
                "primary_key": ["id"],
            }), encoding="utf-8")
            (root / "rows/users.jsonl").write_text('{"id":1}\n', encoding="utf-8")
            (root / "schemas/orders.json").write_text(json.dumps({
                "columns": [
                    {"name": "id", "type": "INTEGER", "nullable": False},
                    {"name": "user_id", "type": "INTEGER", "nullable": False},
                ],
                "primary_key": ["id"],
                "foreign_keys": [{"column": "user_id", "ref_table": "users", "ref_column": "id"}],
            }), encoding="utf-8")
            (root / "rows/orders.jsonl").write_text('{"id":1,"user_id":2}\n', encoding="utf-8")
            task = json.loads((root / "task.json").read_text(encoding="utf-8"))
            task["artifacts"] = {"data_manifest": {"root": ".", "tables": [
                {"table_name": "users", "schema_file": "schemas/users.json", "rows_file": "rows/users.jsonl"},
                {"table_name": "orders", "schema_file": "schemas/orders.json", "rows_file": "rows/orders.jsonl"},
            ]}}
            (root / "task.json").write_text(json.dumps(task), encoding="utf-8")
            quality = SimpleNamespace(to_dict=lambda: {"eligible": True, "score": 9.0})
            with (
                patch.object(buildability, "score_file", return_value=quality),
                patch.object(buildability, "validate_task_spec"),
            ):
                report = buildability.assess(root)
            self.assertFalse(report["buildable"])
            self.assertIn("BUSINESS_DATA_INVALID", {item["code"] for item in report["issues"]})


if __name__ == "__main__":
    unittest.main()

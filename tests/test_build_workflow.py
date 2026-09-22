import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


def load_script(name):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BuildWorkflowTest(unittest.TestCase):
    def test_sandbox_builder_help_exposes_model_selection(self):
        completed = subprocess.run(
            ["bash", str(ROOT / "scripts" / "develop_sandbox_with_agent.sh"), "--help"],
            text=True, capture_output=True,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertIn("--model NAME", completed.stdout)
        self.assertIn("--review-model NAME", completed.stdout)

    def test_scaffold_generator_creates_compilable_thin_composition(self):
        generator = load_script("generate_sandbox_scaffold.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "BUILD_CONTRACT.json").write_text("{}", encoding="utf-8")
            generator.generate(root)
            app = (root / "app.py").read_text(encoding="utf-8")
            implementation = (root / "task_impl.py").read_text(encoding="utf-8")
            compile(app, "app.py", "exec")
            compile(implementation, "task_impl.py", "exec")
            self.assertIn("SandboxApplication", app)
            self.assertIn("class TaskHooks", implementation)
    def test_readiness_helpers_remove_dynamic_fields_and_find_leaks(self):
        readiness = load_script("validate_training_readiness.py")
        value = {"request_id": "a", "nested": {"timestamp": 1, "value": 2}}
        self.assertEqual(readiness.canonical(value), {"nested": {"value": 2}})
        self.assertEqual(
            readiness.forbidden_paths({"public": {"ground_truth": "secret"}}),
            ["$.public.ground_truth"],
        )

    def test_development_plan_is_deterministic_and_valid(self):
        generator = load_script("generate_development_plan.py")
        validator = load_script("validate_development_plan.py")
        contract = {
            "tools": [{"function": {"name": "lookup"}}],
            "metrics": [{"id": "success"}],
            "artifacts": {"data_manifest": {"tables": [{"table_name": "items"}]}},
        }
        first = generator.build_plan(contract)
        second = generator.build_plan(contract)
        self.assertEqual(first, second)
        self.assertEqual(validator.validate(first), [])
        self.assertEqual(first["authority"], "env_factory_outer_workflow")
        self.assertEqual(first["nodes"][0]["scope"]["tables"], ["items"])

    def test_runtime_validator_accepts_modular_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = {"metrics": [{"id": "goal"}]}
            (root / "BUILD_CONTRACT.json").write_text(json.dumps(contract), encoding="utf-8")
            (root / "app.py").write_text(
                "from tool_registry import ToolRegistry\nfrom reward_evaluator import RewardEvaluator\n",
                encoding="utf-8",
            )
            (root / "tool_registry.py").write_text(
                "class ToolRegistry:\n    def execute(self):\n        return None\n",
                encoding="utf-8",
            )
            (root / "reward_evaluator.py").write_text(
                "class RewardEvaluator:\n"
                "    def __init__(self, contract):\n"
                "        self.metrics = contract.get('metrics', [])\n"
                "    def evaluate(self, metric):\n"
                "        evaluator = metric['evaluator']\n"
                "        return evaluator\n"
                "from runtime_llm import RuntimeLLMClient\n"
                "def call(client): return client.json_chat([])\n",
                encoding="utf-8",
            )
            (root / "user_simulator.py").write_text(
                "class UserSimulator:\n"
                "    def __init__(self):\n"
                "        self.profiles = []; self.scripts = []; self.sessions = []\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                ["python3", str(ROOT / "scripts" / "validate_sandbox_runtime.py"), "--root", str(root)],
                text=True, capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()

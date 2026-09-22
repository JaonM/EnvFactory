import json
import os
import tempfile
import unittest
from pathlib import Path

from env_factory.sandbox_runtime import (
    ContractRewardAggregator,
    ContractToolRegistry,
    ContractUserSimulator,
    DeclarativeToolCompiler,
    DeclarativeMetricEvaluator,
    AcceptanceScenarioRunner,
    EpisodeStore,
    EvaluatorMock,
    SandboxError,
    SandboxApplication,
    ManifestDataStore,
    require_trainer,
    validate_json_schema,
)


class SandboxRuntimeTest(unittest.TestCase):
    def test_manifest_data_store_supports_stateless_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EpisodeStore(Path(directory) / "episodes.sqlite3")
            data = ManifestDataStore(
                {"environment_mode": "stateless", "tables": []}, directory, store
            )
            data.reset()
            self.assertEqual(data.baseline, {})
            self.assertEqual(data.snapshot_hash(), __import__("hashlib").sha256(b"{}").hexdigest())

    def test_contract_user_simulator_is_seeded_and_falls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EpisodeStore(Path(directory) / "episodes.sqlite3")
            episode = store.reset(episode_id="user", seed=3)
            simulator = ContractUserSimulator(store, [{
                "turns": [{"role": "user", "content": "第一问"}, {"role": "assistant", "content": "答"}, {"role": "user", "content": "结束"}],
                "user_end_flags": [False, True],
            }], renderer=lambda _: (_ for _ in ()).throw(RuntimeError("timeout")))
            simulator.reset(episode)
            first = simulator.turn([{"role": "assistant", "content": "你好"}])
            second = simulator.turn([{"role": "assistant", "content": "继续"}])
            self.assertEqual(first, {"user_query": "第一问", "should_end": False, "attachments": []})
            self.assertTrue(second["should_end"])
    def test_acceptance_scenario_runner_captures_and_reuses_values(self):
        calls = []
        def call(method, path, body, headers):
            calls.append((method, path, body, headers))
            if path == "/v1/reset":
                return 200, {"episode_id": "ep-1"}, {}
            return 200, {"records": [{"id": "1"}], "count": 1}, {}
        runner = AcceptanceScenarioRunner(call, trainer_headers={"Authorization": "Bearer x"})
        result = runner.run({
            "scenario_id": "query",
            "steps": [
                {"operation": "reset", "body": {"seed": 7}, "capture": {"episode": "$.episode_id"}},
                {"operation": "tool_call", "tool_name": "lookup", "arguments": {"episode": {"$ref": "episode"}}},
            ],
            "assertions": [{"path": "$.count", "operator": "eq", "expected": 1}],
        })
        self.assertEqual(result["variables"]["episode"], "ep-1")
        self.assertEqual(calls[1][2]["episode"], "ep-1")

    def test_acceptance_runner_submits_agent_response(self):
        calls = []
        def call(method, path, body, headers):
            calls.append((method, path, body, headers))
            return 200, {"accepted": True}, {}
        runner = AcceptanceScenarioRunner(call, trainer_headers={"Authorization": "Bearer x"})
        runner.run({
            "scenario_id": "answer",
            "steps": [{"operation": "agent_response", "content": "最终回答"}],
            "assertions": [{"path": "$.accepted", "operator": "eq", "expected": True}],
        })
        self.assertEqual(calls[0][0:3], ("POST", "/v1/agent_response", {"content": "最终回答"}))

    def test_acceptance_runner_supports_snapshots_mutation_and_step_assertions(self):
        state = {"status": "pending"}
        runner = AcceptanceScenarioRunner(
            lambda *args: (200, {}, {}),
            business_snapshot=lambda: dict(state),
            mutate_business_state=lambda mutation: state.update(mutation.get("changes", {})) or dict(state),
        )
        result = runner.run({
            "scenario_id": "counterfactual",
            "steps": [
                {"step_id": "before", "operation": "business_snapshot", "capture": {"before_status": "$.status"}},
                {"step_id": "mutated", "operation": "mutate_business_state", "mutation": {"changes": {"status": "failed"}}},
            ],
            "assertions": [
                {"source": "step:mutated", "path": "$.status", "operator": "changed", "expected": {"$ref": "before_status"}}
            ],
        })
        self.assertEqual(result["step_results"]["mutated"]["status"], "failed")
    def test_declarative_metric_evaluator_uses_runtime_context(self):
        evaluator = DeclarativeMetricEvaluator()
        specs = [
            {
                "metric_id": "enough_rows", "source": "business_state",
                "path": "$.results", "operator": "count_gte", "expected": 2,
                "score_mapping": {"pass": 1, "fail": 0},
            },
            {
                "metric_id": "no_noise", "source": "trajectory",
                "path": "$.noise_calls", "operator": "eq", "expected": 0,
                "score_mapping": {"pass": 0, "fail": -1},
            },
        ]
        scores = evaluator.evaluate_all(specs, {
            "business_state": {"results": [1, 2, 3]},
            "trajectory": {"noise_calls": 0},
        })
        self.assertEqual(scores, {"enough_rows": 1.0, "no_noise": 0.0})

    def test_declarative_metric_evaluator_detects_noise_tool_calls(self):
        evaluator = DeclarativeMetricEvaluator()
        spec = {
            "metric_id": "no_noise", "source": "trajectory", "path": "$.events",
            "operator": "none_tool_calls", "expected": ["weather"],
            "score_mapping": {"pass": 0, "fail": -1},
        }
        clean = evaluator.evaluate(spec, {"trajectory": {"events": []}})
        noisy = evaluator.evaluate(spec, {"trajectory": {"events": [
            {"event": "tool_call", "payload": {"tool_name": "weather"}}
        ]}})
        self.assertEqual((clean, noisy), (0.0, -1.0))
    def test_declarative_tool_compiler_filters_projects_and_orders(self):
        class FakeData:
            def table(self, name):
                self.name = name
                return [
                    {"id": "2", "kind": "cdn", "enabled": False, "internal": "x"},
                    {"id": "1", "kind": "cdn", "enabled": True, "internal": "y"},
                    {"id": "3", "kind": "storage", "enabled": True, "internal": "z"},
                ]

        compiler = DeclarativeToolCompiler(FakeData())
        handler = compiler.compile({
            "tool_name": "query",
            "operation": "select",
            "table": "services",
            "filters": [
                {"argument": "kind", "column": "kind", "operator": "eq"},
                {"argument": "enabled", "column": "enabled", "operator": "eq"},
            ],
            "projection": ["id", "kind"],
            "order_by": ["id"],
            "result_field": "services",
        })
        self.assertEqual(handler({"kind": "cdn", "enabled": True}), {
            "services": [{"id": "1", "kind": "cdn"}], "count": 1
        })

    def test_manifest_store_write_operations(self):
        class MemoryStore:
            def __init__(self): self.rows = [{"id": "1", "value": "a"}]
            def table(self, name): return [dict(row) for row in self.rows]
            def insert(self, name, row): self.rows.append(dict(row)); return dict(row)
            def update(self, name, selector, changes):
                count = 0
                for row in self.rows:
                    if all(row.get(k) == v for k, v in selector.items()): row.update(changes); count += 1
                return count
            def delete(self, name, selector):
                before = len(self.rows); self.rows = [r for r in self.rows if not all(r.get(k) == v for k, v in selector.items())]; return before - len(self.rows)
        data = MemoryStore()
        compiler = DeclarativeToolCompiler(data)
        update = compiler.compile({"operation": "update", "table": "items", "result_field": "records", "filters": [], "projection": [], "order_by": [], "selector": {"id": "id"}, "changes": {"value": "value"}})
        self.assertEqual(update({"id": "1", "value": "b"})["updated_count"], 1)
        delete = compiler.compile({"operation": "delete", "table": "items", "result_field": "records", "filters": [], "projection": [], "order_by": [], "selector": {"id": "id"}})
        self.assertEqual(delete({"id": "1"})["deleted_count"], 1)

    def test_manifest_data_store_resets_isolated_business_copies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "schemas").mkdir()
            (root / "rows").mkdir()
            schema = {
                "columns": [{"name": "id"}, {"name": "value"}],
                "primary_key": ["id"],
                "foreign_keys": [],
            }
            (root / "schemas" / "items.json").write_text(json.dumps(schema), encoding="utf-8")
            (root / "rows" / "items.jsonl").write_text('{"id":"1","value":"a"}\n', encoding="utf-8")
            manifest = {"tables": [{
                "table_name": "items",
                "schema_file": "schemas/items.json",
                "rows_file": "rows/items.jsonl",
            }]}
            episodes = EpisodeStore(root / "episodes.sqlite3")
            data = ManifestDataStore(manifest, root, episodes)
            episodes.reset(episode_id="a", seed=1, data_hash=data.data_hash)
            data.reset()
            data.replace_table("items", [{"id": "1", "value": "changed"}])
            self.assertEqual(data.select("items", id="1")[0]["value"], "changed")
            episodes.reset(episode_id="b", seed=1, data_hash=data.data_hash)
            data.reset()
            self.assertEqual(data.select("items", id="1")[0]["value"], "a")

    def test_shared_application_routes_and_authenticates(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = os.environ.get("SANDBOX_TRAINER_API_KEY")
            os.environ["SANDBOX_TRAINER_API_KEY"] = "trainer-key"
            try:
                store = EpisodeStore(Path(directory) / "episodes.sqlite3")
                tools = [{
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                            "required": ["name"],
                            "additionalProperties": False,
                        },
                    },
                }]
                registry = ContractToolRegistry(tools, {"lookup": lambda args: {"value": args["name"]}})
                app = SandboxApplication(
                    episode_store=store,
                    tool_registry=registry,
                    observation=lambda: {"public": True},
                    reward=lambda: {"reward": 1.0},
                    user_turn=lambda messages: {"user_query": "ok", "should_end": False},
                    data_hash="data-v1",
                )
                status, body, _ = app.handle("GET", "/health")
                self.assertEqual((status, body["status"]), (200, "ok"))
                status, _, _ = app.handle("GET", "/v1/observation")
                self.assertEqual(status, 401)
                auth = {"Authorization": "Bearer trainer-key"}
                status, reset, _ = app.handle("POST", "/v1/reset", {"seed": 7}, auth)
                self.assertEqual(reset["seed"], 7)
                status, result, _ = app.handle("POST", "/v1/tools/lookup", {"name": "x"})
                self.assertEqual((status, result), (200, {"value": "x"}))
                status, accepted, _ = app.handle(
                    "POST", "/v1/agent_response", {"content": "最终回答"}, auth
                )
                self.assertEqual((status, accepted), (200, {"accepted": True}))
                self.assertEqual(store.get_state("final_agent_response"), "最终回答")
                status, reward, _ = app.handle("GET", "/v1/reward", headers=auth)
                self.assertEqual(reward["reward"], 1.0)
            finally:
                if previous is None:
                    os.environ.pop("SANDBOX_TRAINER_API_KEY", None)
                else:
                    os.environ["SANDBOX_TRAINER_API_KEY"] = previous

    def test_contract_tool_registry_validates_dispatches_and_records(self):
        events = []

        def record(*args, **kwargs):
            events.append((args, kwargs))

        tools = [{
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                    "additionalProperties": False,
                },
            },
        }]
        registry = ContractToolRegistry(
            tools, {"lookup": lambda args: {"value": args["name"]}}, event_recorder=record
        )
        self.assertEqual(registry.execute("lookup", {"name": "x"}), {"value": "x"})
        self.assertEqual(len(events), 1)
        with self.assertRaises(SandboxError):
            registry.execute("lookup", {})
        with self.assertRaises(SandboxError):
            registry.execute("lookup", {"name": "x", "extra": True})

    def test_noise_tool_needs_no_task_handler_and_has_no_business_effect(self):
        events = []
        tools = [{
            "type": "function",
            "function": {
                "name": "weather",
                "parameters": {
                    "type": "object", "properties": {}, "required": [],
                    "additionalProperties": False,
                },
            },
        }]
        registry = ContractToolRegistry(
            tools,
            {},
            noise_tools=[{"name": "weather", "category": "unrelated"}],
            event_recorder=lambda *args, **kwargs: events.append((args, kwargs)),
        )
        result = registry.execute("weather", {})
        self.assertEqual(result["category"], "unrelated")
        self.assertIn("without changing", result["message"])
        self.assertTrue(events[0][0][1]["noise"])

    def test_reward_aggregator_uses_contract_weights_and_ranges(self):
        aggregator = ContractRewardAggregator([
            {"id": "success", "weight": 0.8, "score_range": [0, 1]},
            {"id": "process", "weight": 0.2, "score_range": [0, 1]},
            {"id": "penalty", "weight": 1.0, "score_range": [-1, 0]},
        ])
        result = aggregator.aggregate({"success": 1, "process": 0.5, "penalty": -0.25})
        self.assertAlmostEqual(result["reward"], 0.65)
        self.assertEqual(set(result["components"]), {"success", "process", "penalty"})
        with self.assertRaises(SandboxError):
            aggregator.aggregate({"success": 2})

    def test_json_schema_validator_handles_nested_values(self):
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"enabled": {"type": "boolean"}},
                        "required": ["enabled"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["items"],
            "additionalProperties": False,
        }
        validate_json_schema(schema, {"items": [{"enabled": True}]})
        with self.assertRaises(SandboxError):
            validate_json_schema(schema, {"items": [{"enabled": "yes"}]})

    def test_episode_reset_idempotency_and_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EpisodeStore(Path(directory) / "episodes.sqlite3")
            episode = store.reset(episode_id="ep-1", seed=7, data_hash="data-v1")
            first = store.event("tool_call", {"name": "query"}, {"ok": True}, idem_key="req-1")
            second = store.event("tool_call", {"name": "query"}, {"ok": False}, idem_key="req-1")
            self.assertEqual(first, second)
            replay = store.replay()
            self.assertEqual(replay["episode_id"], episode.episode_id)
            self.assertEqual(replay["seed"], 7)
            self.assertEqual(len(replay["events"]), 1)
            self.assertTrue(replay["trace_hash"])
            store.set_state("user_state", {"node_id": "start", "known_facts": ["a"]})
            self.assertEqual(store.get_state("user_state")["node_id"], "start")
            store.reset(episode_id="ep-2", seed=7, data_hash="data-v1")
            self.assertEqual(store.get_state("user_state"), None)

    def test_trainer_authentication(self):
        previous = os.environ.get("SANDBOX_TRAINER_API_KEY")
        os.environ["SANDBOX_TRAINER_API_KEY"] = "test-secret"
        try:
            require_trainer("Bearer test-secret")
            with self.assertRaises(SandboxError) as context:
                require_trainer("Bearer wrong")
            self.assertEqual(context.exception.status, 401)
        finally:
            if previous is None:
                os.environ.pop("SANDBOX_TRAINER_API_KEY", None)
            else:
                os.environ["SANDBOX_TRAINER_API_KEY"] = previous

    def test_evaluator_mock_records_calls(self):
        mock = EvaluatorMock(lambda request: {"match": request["actual"] == request["expected"]})
        self.assertEqual(mock.evaluate({"actual": "a", "expected": "a"}), {"match": True})
        self.assertEqual(len(mock.calls), 1)
        self.assertTrue(mock.calls[0]["request_hash"])


if __name__ == "__main__":
    unittest.main()

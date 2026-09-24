import json
import os
import tempfile
import unittest
from pathlib import Path

from env_factory.sandbox_runtime import (
    ContractRewardAggregator,
    ContractRewardGate,
    ContractEvaluatorRuntime,
    ContractToolRegistry,
    ContractUserSimulator,
    DeclarativeToolCompiler,
    DeclarativeMetricEvaluator,
    AcceptanceScenarioRunner,
    EpisodeStore,
    ExternalCapabilityClient,
    DataManifestValidator,
    EvaluatorMock,
    SandboxError,
    SandboxApplication,
    ManifestDataStore,
    require_trainer,
    validate_json_schema,
)


class SandboxRuntimeTest(unittest.TestCase):
    def test_external_capability_client_uses_training_fixture_or_explicit_gap(self):
        previous = os.environ.get("SANDBOX_EXTERNAL_FIXTURES")
        try:
            os.environ.pop("SANDBOX_EXTERNAL_FIXTURES", None)
            self.assertEqual(ExternalCapabilityClient().query("prices", {})["status"], "data_unavailable")
            with tempfile.TemporaryDirectory() as directory:
                fixture = Path(directory) / "external.json"
                fixture.write_text(json.dumps({"prices": {"status": "ok", "items": [1]}}), encoding="utf-8")
                os.environ["SANDBOX_EXTERNAL_FIXTURES"] = str(fixture)
                self.assertEqual(ExternalCapabilityClient().query("prices", {})["items"], [1])
        finally:
            if previous is None:
                os.environ.pop("SANDBOX_EXTERNAL_FIXTURES", None)
            else:
                os.environ["SANDBOX_EXTERNAL_FIXTURES"] = previous

    def test_contract_evaluator_runtime_caches_and_traces_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EpisodeStore(Path(directory) / "episodes.sqlite3")
            store.reset(episode_id="eval", seed=1)
            evaluator = ContractEvaluatorRuntime(store)
            metric = {"id": "quality", "evaluator": {"kind": "external_llm_judge"}}
            first = evaluator.json_judge(metric, {"answer": "x"}, fallback={"label": "fail"})
            second = evaluator.json_judge(metric, {"answer": "x"}, fallback={"label": "different"})
            self.assertEqual(first, {"label": "fail"})
            self.assertEqual(second, first)
            events = store.replay()["events"]
            self.assertEqual([item["event"] for item in events], ["evaluator_call"])

    def test_manifest_data_store_supports_stateless_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EpisodeStore(Path(directory) / "episodes.sqlite3")
            data = ManifestDataStore(
                {"environment_mode": "stateless", "tables": []}, directory, store
            )
            data.reset()
            self.assertEqual(data.baseline, {})
            self.assertEqual(data.snapshot_hash(), __import__("hashlib").sha256(b"{}").hexdigest())

    def test_contract_user_simulator_failure_does_not_accept_completion_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EpisodeStore(Path(directory) / "episodes.sqlite3")
            episode = store.reset(episode_id="user", seed=3)
            simulator = ContractUserSimulator(
                store,
                profiles=[{"profile_id": "p1"}],
                scripts=[{
                    "script_id": "s1", "initial_state": "start", "variables": {},
                    "recovery_policy": {"max_recoveries": 2, "user_behavior": "请重新回答。"},
                    "states": [
                        {"state_id": "start", "user_behavior": "第一问", "terminal": False},
                        {"state_id": "done", "user_behavior": "结束", "terminal": True},
                    ],
                    "transitions": [{
                        "transition_id": "finish", "from_state": "start", "to_state": "done",
                        "condition": "continue", "should_end": True, "updates": {},
                    }],
                }],
                renderer=lambda _: (_ for _ in ()).throw(RuntimeError("timeout")),
            )
            simulator.reset(episode)
            first = simulator.turn([{"role": "assistant", "content": "任务已完成，结果如下。"}])
            state = store.get_state(ContractUserSimulator.STATE_KEY)
            self.assertEqual(first["match_status"], "unmatched")
            self.assertEqual(first["outcome_category"], "unrecognized")
            self.assertFalse(first["should_end"])
            self.assertEqual(first["reason_code"], "simulator_unavailable")
            self.assertEqual(state["state_id"], "start")

    def test_contract_user_simulator_fallback_does_not_advance_on_irrelevant_reply(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EpisodeStore(Path(directory) / "episodes.sqlite3")
            episode = store.reset(episode_id="irrelevant", seed=3)
            simulator = ContractUserSimulator(
                store,
                profiles=[{"profile_id": "p1"}],
                scripts=[{
                    "script_id": "s1", "initial_state": "start", "variables": {},
                    "recovery_policy": {"max_recoveries": 2, "user_behavior": "请重新回答。"},
                    "states": [
                        {"state_id": "start", "user_behavior": "提出任务", "terminal": False},
                        {"state_id": "review", "user_behavior": "检查结果", "terminal": False},
                        {"state_id": "done", "user_behavior": "结束", "terminal": True},
                    ],
                    "transitions": [
                        {"transition_id": "satisfied", "outcome_category": "goal_satisfied", "from_state": "start", "to_state": "review", "condition": "任务完成", "should_end": False, "updates": {}},
                        {"transition_id": "accepted", "outcome_category": "user_acceptance", "from_state": "review", "to_state": "done", "condition": "接受结果", "should_end": True, "updates": {}},
                    ],
                }],
                renderer=lambda _: (_ for _ in ()).throw(RuntimeError("offline")),
            )
            simulator.reset(episode)
            result = simulator.turn([{"role": "assistant", "content": "今天天气不错。"}])
            state = store.get_state(ContractUserSimulator.STATE_KEY)
            self.assertEqual(result["match_status"], "unmatched")
            self.assertEqual(state["state_id"], "start")
            self.assertEqual(state["recovery_count"], 1)

    def test_contract_user_simulator_tracks_declared_fsm_state(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EpisodeStore(Path(directory) / "episodes.sqlite3")
            episode = store.reset(episode_id="fsm", seed=1)
            scripts = [{
                "script_id": "s1", "initial_state": "start", "variables": {"known": False},
                "recovery_policy": {"max_recoveries": 2, "user_behavior": "请重新回答。"},
                "states": [{"state_id": "start", "user_behavior": "开始", "terminal": False}, {"state_id": "done", "user_behavior": "结束", "terminal": True}],
                "transitions": [{
                    "transition_id": "t1", "from_state": "start", "to_state": "done",
                    "condition": "agent completes", "should_end": True, "updates": {"known": True},
                }],
            }]
            simulator = ContractUserSimulator(
                store, profiles=[{"profile_id": "p1"}], scripts=scripts,
                renderer=lambda _: {"user_query": "结束", "transition_id": "t1"},
            )
            simulator.reset(episode)
            result = simulator.turn([{"role": "assistant", "content": "完成"}])
            state = store.get_state(ContractUserSimulator.STATE_KEY)
            self.assertTrue(result["should_end"])
            self.assertEqual(state["state_id"], "done")
            self.assertTrue(state["variables"]["known"])

    def test_contract_user_simulator_does_not_advance_on_unmatched_dialogue(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EpisodeStore(Path(directory) / "episodes.sqlite3")
            episode = store.reset(episode_id="recovery", seed=1)
            script = {
                "script_id": "s1", "initial_state": "start", "variables": {},
                "recovery_policy": {"max_recoveries": 2, "user_behavior": "请重新回答。"},
                "states": [
                    {"state_id": "start", "user_behavior": "开始", "terminal": False},
                    {"state_id": "done", "user_behavior": "结束", "terminal": True},
                ],
                "transitions": [{
                    "transition_id": "finish", "from_state": "start", "to_state": "done",
                    "condition": "完成", "should_end": True, "updates": {},
                }],
            }
            simulator = ContractUserSimulator(
                store, profiles=[{"profile_id": "p1"}], scripts=[script],
                renderer=lambda _: {
                    "user_query": "这没有回答我的问题。", "match_status": "ambiguous",
                    "outcome_category": "unrecognized",
                    "reason_code": "off_topic_response",
                },
            )
            simulator.reset(episode)
            result = simulator.turn([{"role": "assistant", "content": "无关回复"}])
            state = store.get_state(ContractUserSimulator.STATE_KEY)
            self.assertEqual(state["state_id"], "start")
            self.assertEqual(state["recovery_count"], 1)
            self.assertFalse(result["should_end"])
            self.assertEqual(result["match_status"], "ambiguous")
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

    def test_manifest_validator_accepts_descriptive_foreign_key_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "schemas").mkdir()
            (root / "rows").mkdir()
            parent_schema = {
                "columns": [{"name": "id"}],
                "primary_key": ["id"],
                "foreign_keys": [],
            }
            child_schema = {
                "columns": [{"name": "id"}, {"name": "parent_id"}],
                "primary_key": ["id"],
                "foreign_keys": [{
                    "column": "parent_id",
                    "references_table": "parents",
                    "references_column": "id",
                }],
            }
            (root / "schemas" / "parents.json").write_text(json.dumps(parent_schema), encoding="utf-8")
            (root / "schemas" / "children.json").write_text(json.dumps(child_schema), encoding="utf-8")
            (root / "rows" / "parents.jsonl").write_text('{"id":"p1"}\n', encoding="utf-8")
            (root / "rows" / "children.jsonl").write_text(
                '{"id":"c1","parent_id":"p1"}\n', encoding="utf-8"
            )
            manifest = {"tables": [
                {
                    "table_name": "parents",
                    "schema_file": "schemas/parents.json",
                    "rows_file": "rows/parents.jsonl",
                },
                {
                    "table_name": "children",
                    "schema_file": "schemas/children.json",
                    "rows_file": "rows/children.jsonl",
                },
            ]}
            validated = DataManifestValidator.validate(manifest, root)
            self.assertEqual(validated["tables"]["children"][0]["parent_id"], "p1")

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
                status, result, tool_headers = app.handle("POST", "/v1/tools/lookup", {"name": "x"})
                self.assertEqual((status, result), (200, {"value": "x"}))
                self.assertTrue(tool_headers["X-Tool-Call-ID"].startswith("call-"))
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
        self.assertIn("tool_call_id", events[0][0][1])
        self.assertIn("duration_ms", events[0][0][1])
        with self.assertRaises(SandboxError):
            registry.execute("lookup", {})
        with self.assertRaises(SandboxError):
            registry.execute("lookup", {"name": "x", "extra": True})

    def test_tool_idempotency_prevents_reexecuting_business_handler(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EpisodeStore(Path(directory) / "episodes.sqlite3")
            store.reset(episode_id="idem", seed=1)
            calls = []
            tools = [{"type": "function", "function": {
                "name": "write", "parameters": {
                    "type": "object", "properties": {"value": {"type": "string"}},
                    "required": ["value"], "additionalProperties": False,
                }
            }}]
            registry = ContractToolRegistry(
                tools, {"write": lambda args: calls.append(dict(args)) or {"saved": args["value"]}},
                event_recorder=store.event,
            )
            first = registry.execute("write", {"value": "a"}, idem_key="same")
            second = registry.execute("write", {"value": "a"}, idem_key="same")
            with self.assertRaisesRegex(SandboxError, "another request"):
                registry.execute("write", {"value": "different"}, idem_key="same")
            self.assertEqual(first, second)
            self.assertEqual(calls, [{"value": "a"}])

    def test_shared_runtime_owns_all_declared_mutation_seams(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "update",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            },
        }]
        calls = []
        registry = ContractToolRegistry(tools, {"update": lambda args: calls.append(args) or {"value": args.get("value")}})
        previous = os.environ.get("SANDBOX_MUTATION_MODE")
        previous_key = os.environ.get("SANDBOX_TRAINER_API_KEY")
        try:
            os.environ["SANDBOX_MUTATION_MODE"] = "skip_business_write"
            self.assertEqual(registry.execute("update", {"value": "x"})["mutation"], "skip_business_write")
            self.assertEqual(calls, [])
            os.environ["SANDBOX_MUTATION_MODE"] = "ignore_tool_arguments"
            self.assertEqual(registry.execute("update", {"unexpected": True}), {"value": None})

            with tempfile.TemporaryDirectory() as directory:
                store = EpisodeStore(Path(directory) / "episodes.sqlite3")
                app = SandboxApplication(
                    episode_store=store, tool_registry=registry,
                    observation=lambda: {}, reward=lambda: {"reward": 1.0},
                    user_turn=lambda messages: {"user_query": "ok", "should_end": False},
                )
                os.environ["SANDBOX_MUTATION_MODE"] = "bypass_trainer_auth"
                self.assertEqual(app.handle("POST", "/v1/reset", {})[0], 200)
                os.environ["SANDBOX_MUTATION_MODE"] = "constant_reward"
                os.environ["SANDBOX_TRAINER_API_KEY"] = "mutation-key"
                status, reward, _ = app.handle(
                    "GET", "/v1/reward", headers={"Authorization": "Bearer mutation-key"}
                )
                self.assertEqual(status, 200)
                self.assertEqual(
                    reward,
                    {"reward": 0.0, "raw_reward": 0.0, "components": {"__mutation__": 0.0}},
                )
        finally:
            if previous is None:
                os.environ.pop("SANDBOX_MUTATION_MODE", None)
            else:
                os.environ["SANDBOX_MUTATION_MODE"] = previous
            if previous_key is None:
                os.environ.pop("SANDBOX_TRAINER_API_KEY", None)
            else:
                os.environ["SANDBOX_TRAINER_API_KEY"] = previous_key

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
        self.assertNotIn("category", result)
        self.assertEqual(result, {"records": [], "count": 0})
        self.assertNotIn("message", result)
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

    def test_reward_aggregator_rejects_missing_declared_scores(self):
        aggregator = ContractRewardAggregator([
            {"id": "success", "weight": 1.0, "score_range": [0, 1]},
        ])
        with self.assertRaisesRegex(SandboxError, "declared metrics have no runtime score"):
            aggregator.aggregate({})

    def test_reward_gate_blocks_outcome_without_required_tool_progress(self):
        metrics = [
            {"id": "process", "category": "process", "score_range": [0, 1]},
            {"id": "outcome", "category": "outcome", "score_range": [0, 1]},
        ]
        gate = ContractRewardGate({
            "training_contract": {"category": "simple_agentic"},
            "capability_dag": {"nodes": ["lookup"]},
        }, metrics)
        blocked = gate.apply(
            {"process": 0.0, "outcome": 1.0},
            {"trajectory": {"events": []}},
        )
        self.assertEqual(blocked, {"process": 0.0, "outcome": 0.0})
        allowed = gate.apply(
            {"process": 1.0, "outcome": 1.0},
            {"trajectory": {"events": [{
                "event": "tool_call", "payload": {"tool_name": "lookup", "noise": False},
            }]}},
        )
        self.assertEqual(allowed["outcome"], 1.0)

    def test_reward_gate_enforces_multi_step_dependency_order(self):
        metrics = [{"id": "outcome", "category": "outcome", "score_range": [0, 1]}]
        gate = ContractRewardGate({
            "training_contract": {"category": "multi_step_agentic"},
            "capability_dag": {
                "nodes": ["lookup", "update"],
                "edges": [{"from_tool": "lookup", "to_tool": "update", "via": "record_id"}],
            },
        }, metrics)
        def context(names):
            return {"trajectory": {"events": [
                {"event": "tool_call", "payload": {"tool_name": name, "noise": False}}
                for name in names
            ]}}
        self.assertEqual(gate.apply({"outcome": 1.0}, context(["update", "lookup"]))["outcome"], 0.0)
        self.assertEqual(gate.apply({"outcome": 1.0}, context(["lookup", "update"]))["outcome"], 1.0)

    def test_reward_gate_leaves_unnecessary_tool_cost_to_penalty_metric(self):
        gate = ContractRewardGate({
            "training_contract": {"category": "direct_response"},
            "capability_dag": {"nodes": [], "edges": []},
        }, [{"id": "outcome", "category": "outcome", "score_range": [0, 1]}])
        result = gate.apply({"outcome": 1.0}, {"trajectory": {"events": [{
            "event": "tool_call", "payload": {"tool_name": "roll_virtual_die", "noise": True},
        }]}})
        self.assertEqual(result["outcome"], 1.0)

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

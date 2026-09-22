import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from env_factory.task_pipeline import PipelineGenerationError, TaskGenerationPipeline


class StageRetryTest(unittest.TestCase):
    def test_task_description_shape_error_is_retried_with_feedback(self):
        class FakeLLM:
            def __init__(self):
                self.calls = []

            def complete(self, prompt, **kwargs):
                self.calls.append((prompt, kwargs))
                if len(self.calls) == 1:
                    return SimpleNamespace(content=json.dumps({"output": "错误包装", "type": "text"}))
                return SimpleNamespace(content=json.dumps({
                    "task": "查询指定对象的信息",
                    "task_intent": "query",
                    "goal": "完成查询",
                    "context": [],
                    "expected_result": "结构化结果",
                    "complexity": "simple",
                    "requirements": {"input_modalities": ["text"]},
                }))

        llm = FakeLLM()
        pipeline = TaskGenerationPipeline(llm, retries=2)
        result = pipeline._call(
            "task_description",
            "生成任务描述",
            {"output": {"task": "string", "complexity": "simple"}},
        )
        self.assertEqual(result["task"], "查询指定对象的信息")
        self.assertEqual(len(llm.calls), 2)
        self.assertIn("previous_validation_error", llm.calls[1][0])

    def test_user_dialogue_shape_error_is_retried_with_feedback(self):
        class FakeLLM:
            def __init__(self):
                self.calls = []

            def complete(self, prompt, **kwargs):
                self.calls.append((prompt, kwargs))
                if len(self.calls) == 1:
                    return SimpleNamespace(content=json.dumps({"message": "继续说明"}))
                return SimpleNamespace(content=json.dumps({"message": "现在可以结束了", "should_end": True}))

        llm = FakeLLM()
        pipeline = TaskGenerationPipeline(llm, retries=2)
        result = pipeline._call(
            "dialogue_sessions.script-1-session-1.user.1",
            "模拟用户",
            {"output": {"message": "string", "should_end": False}},
        )
        self.assertTrue(result["should_end"])
        self.assertEqual(len(llm.calls), 2)
        self.assertIn("should_end must be boolean", llm.calls[1][0])

    def test_table_data_json_error_is_retried_with_compact_rows_contract(self):
        class FakeLLM:
            def __init__(self):
                self.calls = []

            def complete(self, prompt, **kwargs):
                self.calls.append((prompt, kwargs))
                if len(self.calls) == 1:
                    return SimpleNamespace(content='{"rows":[{"name":"未闭合}')
                return SimpleNamespace(content=json.dumps({"rows": [{"name": "工坊"}]}))

        llm = FakeLLM()
        pipeline = TaskGenerationPipeline(llm, retries=2)
        result = pipeline._call(
            "environment_table_data.workshop",
            "生成表数据",
            {"table": {"columns": [{"name": "name"}]}, "output": {"rows": []}},
        )
        self.assertEqual(result["rows"][0]["name"], "工坊")
        self.assertEqual(len(llm.calls), 2)
        self.assertIn("rows 数组", llm.calls[1][1]["system_prompt"])


class BusinessDataArtifactsTest(unittest.TestCase):
    def test_stateless_manifest_has_no_business_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = TaskGenerationPipeline._materialize_business_data(
                [], "# 无业务数据", Path(directory), environment_mode="stateless"
            )
            self.assertEqual(manifest["environment_mode"], "stateless")
            self.assertEqual(manifest["tables"], [])

    def test_materialize_business_data_writes_schema_rows_and_document(self):
        tables = [{
            "table_name": "clothing_items",
            "description": "衣物",
            "columns": [{"name": "item_id", "type": "string", "description": "标识"}],
            "primary_key": ["item_id"],
            "foreign_keys": [],
            "rows": [{"item_id": "item-1"}],
        }]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = TaskGenerationPipeline._materialize_business_data(tables, "# 数据说明", root)

            self.assertEqual(manifest["tables"][0]["schema_file"], "schemas/clothing_items.json")
            self.assertEqual(manifest["tables"][0]["rows_file"], "rows/clothing_items.jsonl")
            self.assertEqual(json.loads((root / "schemas/clothing_items.json").read_text())["table_name"], "clothing_items")
            self.assertEqual((root / "rows/clothing_items.jsonl").read_text().strip(), '{"item_id":"item-1"}')
            self.assertEqual((root / "data_document.md").read_text(), "# 数据说明\n")


class AgentActionContractTest(unittest.TestCase):
    def test_extract_without_persistence_is_forced_to_stateless(self):
        plan = TaskGenerationPipeline._resolve_environment_plan(
            {"mode": "stateful", "requires_business_data": True, "requires_persistence": True, "reason": "模型过度设计"},
            task_description={"task": "从对话中提取植物信息", "goal": "返回结构化结果", "expected_result": "JSON"},
            task_intent="extract",
        )
        self.assertEqual(plan["mode"], "stateless")
        self.assertFalse(plan["requires_business_data"])

    def test_explicit_persistence_keeps_extract_task_stateful(self):
        plan = TaskGenerationPipeline._resolve_environment_plan(
            {"mode": "stateful", "requires_business_data": True, "requires_persistence": True, "reason": "任务明确要求保存"},
            task_description={"task": "提取信息并写入数据库", "goal": "保存记录", "expected_result": "新记录"},
            task_intent="extract",
        )
        self.assertEqual(plan["mode"], "stateful")
    def test_capability_plan_separates_tools_from_reasoning_and_response(self):
        actions = [
            {"name": "查询库存"},
            {"name": "比较候选项"},
            {"name": "生成最终回答"},
        ]
        capabilities = [
            {"action_name": "查询库存", "kind": "environment_operation", "requires_tool": True, "reason": "读取私有库存"},
            {"action_name": "比较候选项", "kind": "agent_reasoning", "requires_tool": False, "reason": "根据查询结果推理"},
            {"action_name": "生成最终回答", "kind": "agent_response", "requires_tool": False, "reason": "由 Agent 回复"},
        ]
        TaskGenerationPipeline._validate_capability_plan(capabilities, actions)
        with self.assertRaises(PipelineGenerationError):
            TaskGenerationPipeline._validate_capability_plan(capabilities[:2], actions)

    def test_actions_use_semantic_atomic_fields(self):
        action = {
            "name": "record_clothing_items",
            "description": "记录用户提供的衣物信息",
            "atomicity_rationale": "该动作只有一个记录边界，不能继续拆成独立的 Agent 决策",
            "inputs": [{"name": "items", "description": "用户提供的衣物列表"}],
            "outputs": [{"name": "recorded_items", "description": "已保存的衣物记录"}],
            "preconditions": ["items 非空"],
            "effects": ["持久化衣物业务记录"],
        }
        TaskGenerationPipeline._validate_actions([action])

        invalid = dict(action)
        invalid["inputs"] = [{"name": "items"}]
        with self.assertRaises(PipelineGenerationError):
            TaskGenerationPipeline._validate_actions([invalid])


class OpenAIToolArtifactTest(unittest.TestCase):
    def test_noise_audit_identifies_materially_useful_tool(self):
        unsafe = TaskGenerationPipeline._validate_noise_tool_audit(
            [{"name": "route_lookup", "is_safe_noise": False, "reason": "可提供排查证据"}],
            noise_names={"route_lookup"},
        )
        self.assertEqual(unsafe, {"route_lookup"})

    def test_noise_audit_accepts_safe_noise_tool(self):
        unsafe = TaskGenerationPipeline._validate_noise_tool_audit(
            [{"name": "roll_virtual_die", "is_safe_noise": True, "reason": "与任务无关"}],
            noise_names={"roll_virtual_die"},
        )
        self.assertEqual(unsafe, set())

    def test_fallback_noise_tool_avoids_business_tool_name_collision(self):
        tool, metadata = TaskGenerationPipeline._fallback_noise_tool(
            occupied_names={"roll_virtual_die"}
        )
        self.assertEqual(tool["function"]["name"], "roll_virtual_die_2")
        self.assertEqual(metadata["name"], "roll_virtual_die_2")
        TaskGenerationPipeline._validate_tools([tool])

    def test_placeholder_tool_name_is_rejected(self):
        tool = {
            "type": "function",
            "function": {
                "name": "noise_tool_1",
                "description": "占位工具",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }
        with self.assertRaisesRegex(PipelineGenerationError, "placeholder name"):
            TaskGenerationPipeline._validate_tools([tool])

    def test_missing_nested_descriptions_are_filled_deterministically(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "query_matching_treatments",
                "description": "查询匹配治疗方案",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "patient_profile": {
                            "type": "object",
                            "properties": {
                                "genomic_alterations": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                        },
                    },
                    "required": ["patient_profile"],
                },
            },
        }]
        TaskGenerationPipeline._fill_tool_schema_descriptions(tools)
        TaskGenerationPipeline._validate_tools(tools)
        items = tools[0]["function"]["parameters"]["properties"]["patient_profile"]["properties"]["genomic_alterations"]["items"]
        self.assertTrue(items["description"])

    def test_declarative_tool_implementation_validation(self):
        tools = [{"function": {"name": "lookup", "parameters": {
            "properties": {"kind": {"type": "string"}}
        }}}]
        tables = [{"table_name": "items", "columns": [
            {"name": "id"}, {"name": "kind"}
        ]}]
        specs = [{
            "tool_name": "lookup", "operation": "select", "table": "items",
            "filters": [{"argument": "kind", "column": "kind", "operator": "eq"}],
            "projection": ["id", "kind"], "order_by": ["id"], "result_field": "records",
        }]
        TaskGenerationPipeline._validate_tool_implementations(specs, tools=tools, tables=tables)
        with self.assertRaises(PipelineGenerationError):
            TaskGenerationPipeline._validate_tool_implementations(
                [dict(specs[0], table="missing")], tools=tools, tables=tables
            )

    def test_materialize_tools_writes_only_standard_tool_array(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "record_clothing_items",
                "description": "记录衣物信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "items": {"type": "array", "description": "衣物列表"},
                    },
                    "required": ["items"],
                },
            },
        }]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = TaskGenerationPipeline._materialize_tools(tools, root)
            self.assertEqual(manifest["file"], "tools.json")
            self.assertEqual(json.loads((root / "tools.json").read_text()), tools)

    def test_runtime_interface_lists_tools_and_reward_function(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "query_recipe",
                "description": "查询食谱",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dish": {"type": "string", "description": "菜名"},
                    },
                    "required": ["dish"],
                },
            },
        }]
        formula = {"type": "separate_sign_weighted_sum", "score_range": [-1, 1]}
        interface = TaskGenerationPipeline._build_runtime_interface(tools, formula)
        TaskGenerationPipeline._validate_runtime_interface(interface, tools, formula)
        tool_endpoint = next(item for item in interface["endpoints"] if item["name"] == "query_recipe")
        self.assertEqual(tool_endpoint["method"], "POST")
        self.assertEqual(tool_endpoint["path"], "/v1/tools/{tool_name}")
        self.assertEqual(tool_endpoint["request_schema"], tools[0]["function"]["parameters"])
        self.assertIn("不包含 observation 或 reward", tool_endpoint["response_schema"]["description"])
        user_endpoint = next(item for item in interface["endpoints"] if item["name"] == "user_simulator")
        self.assertEqual(user_endpoint["path"], "/v1/user_simulator")
        self.assertEqual(user_endpoint["access"], "rl_trainer_only")
        self.assertEqual(user_endpoint["response_schema"]["required"], ["user_query", "should_end"])
        self.assertEqual(user_endpoint["request_schema"]["required"], ["messages"])
        self.assertIn("Trainer", user_endpoint["request_schema"]["description"])
        response_endpoint = next(item for item in interface["endpoints"] if item["name"] == "agent_response")
        self.assertEqual(response_endpoint["path"], "/v1/agent_response")
        self.assertEqual(response_endpoint["request_schema"]["required"], ["content"])
        self.assertEqual(interface["reward_functions"], [{"name": "reward", "endpoint": "/v1/reward"}])
        reward_endpoint = next(item for item in interface["endpoints"] if item["name"] == "reward")
        self.assertEqual(reward_endpoint["access"], "rl_trainer_only")
        replay_endpoint = next(item for item in interface["endpoints"] if item["name"] == "replay")
        self.assertEqual(replay_endpoint["access"], "rl_trainer_only")
        self.assertEqual(interface["security"]["trainer"]["environment_variable"], "SANDBOX_TRAINER_API_KEY")
        self.assertTrue(interface["episode"]["deterministic_replay"])
        self.assertEqual(interface["llm_runtime"]["api_key_environment_variable"], "SANDBOX_LLM_API_KEY")
        self.assertEqual(interface["evaluator_runtime"]["mock_mode_environment_variable"], "SANDBOX_EVALUATOR_MOCK")
        self.assertIn("ContractToolRegistry", interface["shared_runtime"]["required_components"])
        self.assertIn("ContractRewardAggregator", interface["shared_runtime"]["required_components"])


class RewardContractTest(unittest.TestCase):
    def test_noise_penalty_is_compiled_as_deterministic_rule(self):
        metrics = [{
            "id": "penalty_noise_old", "category": "penalty", "type": "model-based",
            "condition": "noise", "weight": 1,
        }]
        TaskGenerationPipeline._ensure_deterministic_noise_penalty(
            metrics, noise_names=["weather"]
        )
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0]["id"], "penalty_noise_tool_usage")
        self.assertEqual(metrics[0]["type"], "rule-based")

    def test_stateless_metrics_reject_process_and_business_state(self):
        process = [{"category": "process", "evaluator": {"kind": "hybrid_tool_call"}}]
        with self.assertRaises(PipelineGenerationError):
            TaskGenerationPipeline._validate_metrics_for_environment(
                process, environment_mode="stateless"
            )
        business = [{
            "category": "outcome",
            "evaluation_inputs": ["business_data"],
            "evaluator": {"kind": "external_llm_judge"},
        }]
        with self.assertRaises(PipelineGenerationError):
            TaskGenerationPipeline._validate_metrics_for_environment(
                business, environment_mode="stateless"
            )

    def test_uncompiled_rule_metric_is_promoted_to_external_judge(self):
        metrics = [{
            "id": "grounded", "category": "outcome", "type": "rule-based",
            "scope": "terminal", "rubric": "回答与业务数据一致", "condition": "grounded",
            "evaluator": {"kind": "document_rule", "source": "runtime_rule", "assertion": "grounded", "score_mapping": {"true": 1, "false": 0}},
            "score_range": [0, 1], "weight": 1,
        }]
        TaskGenerationPipeline._promote_unimplemented_rule_metrics(metrics, [])
        self.assertEqual(metrics[0]["type"], "model-based")
        self.assertEqual(metrics[0]["evaluator"]["kind"], "external_llm_judge")
        self.assertTrue(metrics[0]["evaluation_inputs"])
        self.assertTrue(metrics[0]["criteria"])

    def test_task_readiness_requires_noise_and_business_trajectories(self):
        kwargs = {
            "capability_plan": [{"action_name": "查询", "requires_tool": True}],
            "tool_bindings": [{"tool_name": "query", "action_name": "查询"}],
            "noise_tools": [{"name": "weather"}],
            "metrics": [{
                "id": "penalty_noise_tool_usage", "category": "penalty",
                "type": "rule-based",
            }],
            "metric_implementations": [{
                "metric_id": "penalty_noise_tool_usage", "source": "trajectory",
                "path": "$.events", "operator": "none_tool_calls",
                "expected": ["weather"], "score_mapping": {"pass": 0, "fail": -1},
            }],
            "business_scenarios": [
                {"kind": "goal_success"}, {"kind": "goal_failure"}, {"kind": "noise_selection"},
            ],
        }
        TaskGenerationPipeline._validate_task_readiness(**kwargs)
        with self.assertRaises(PipelineGenerationError):
            TaskGenerationPipeline._validate_task_readiness(**(kwargs | {"noise_tools": []}))

    def test_complexity_is_derived_from_executable_structure(self):
        self.assertEqual(TaskGenerationPipeline._derive_complexity(business_tool_count=1, metric_count=4, key_step_count=1), "simple")
        self.assertEqual(TaskGenerationPipeline._derive_complexity(business_tool_count=2, metric_count=6, key_step_count=2), "standard")
        self.assertEqual(TaskGenerationPipeline._derive_complexity(business_tool_count=5, metric_count=9, key_step_count=5), "complex")

    def test_training_profile_distinguishes_abstention_from_tool_use(self):
        self.assertEqual(
            TaskGenerationPipeline._derive_training_profile(
                business_tool_count=0, noise_tool_count=1, key_step_count=0
            ),
            "tool_abstention",
        )
        self.assertEqual(
            TaskGenerationPipeline._derive_training_profile(
                business_tool_count=2, noise_tool_count=1, key_step_count=2
            ),
            "multi_step_tool_use",
        )

    def test_executable_scenarios_require_success_failure_and_noise(self):
        tools = [{"function": {"name": "lookup", "parameters": {"properties": {}}}},
                 {"function": {"name": "weather", "parameters": {"properties": {}}}}]
        scenarios = [
            {"scenario_id": "ok", "kind": "goal_success", "steps": [{"operation": "tool_call", "tool_name": "lookup", "arguments": {}}], "assertions": [{"path": "$.reward", "operator": "gte", "expected": 0.8}]},
            {"scenario_id": "bad", "kind": "goal_failure", "steps": [{"operation": "reward"}], "assertions": [{"path": "$.reward", "operator": "lte", "expected": 0.2}]},
            {"scenario_id": "noise", "kind": "noise_selection", "steps": [{"operation": "tool_call", "tool_name": "weather", "arguments": {}}], "assertions": [{"path": "$.reward", "operator": "lte", "expected": 0}]},
        ]
        TaskGenerationPipeline._validate_executable_scenarios(
            scenarios, tools=tools, noise_tools=[{"name": "weather"}]
        )
        with self.assertRaises(PipelineGenerationError):
            TaskGenerationPipeline._validate_executable_scenarios(
                scenarios[:2], tools=tools, noise_tools=[{"name": "weather"}]
            )

    def test_executable_scenario_identity_is_normalized(self):
        scenarios = TaskGenerationPipeline._normalize_executable_scenarios([
            {"scenario_id": "runtime_reset_replay", "steps": []},
            {"scenario_id": "happy_success", "kind": "success", "steps": [], "assertions": []},
            {"scenario_id": "bad_failure", "kind": "failure", "steps": [], "assertions": []},
            {"scenario_id": "noise_case", "kind": "noise", "steps": [], "assertions": []},
            {"scenario_id": "noise_case", "kind": "noise-selection", "steps": [], "assertions": []},
        ])
        self.assertEqual(
            [item["kind"] for item in scenarios],
            ["goal_success", "goal_failure", "noise_selection", "noise_selection"],
        )
        self.assertEqual(len({item["scenario_id"] for item in scenarios}), 4)

    def test_deterministic_business_scenario_baseline_is_complete(self):
        tools = [
            {"type": "function", "function": {"name": "query", "description": "查询", "parameters": {"type": "object", "properties": {}, "required": []}}},
            {"type": "function", "function": {"name": "weather", "description": "天气", "parameters": {"type": "object", "properties": {}, "required": []}}},
        ]
        scenarios = TaskGenerationPipeline._build_business_scenario_baseline(
            task_description={"expected_result": "给出结论和依据"},
            tools=tools,
            noise_tools=[{"name": "weather", "category": "unrelated"}],
        )
        TaskGenerationPipeline._validate_executable_scenarios(
            scenarios, tools=tools, noise_tools=[{"name": "weather"}]
        )
        success = next(item for item in scenarios if item["kind"] == "goal_success")
        self.assertIn("agent_response", [step["operation"] for step in success["steps"]])

    def test_declarative_metric_implementation_validation(self):
        metrics = [{"id": "done", "type": "rule-based", "score_range": [0, 1]}]
        specs = [{
            "metric_id": "done", "source": "business_state", "path": "$.done",
            "operator": "eq", "expected": True,
            "score_mapping": {"pass": 1, "fail": 0},
        }]
        TaskGenerationPipeline._validate_metric_implementations(specs, metrics)
        with self.assertRaises(PipelineGenerationError):
            TaskGenerationPipeline._validate_metric_implementations(
                [dict(specs[0], operator="python_eval")], metrics
            )

    def test_raw_final_agent_response_cannot_use_synthetic_object_path(self):
        metrics = [{"id": "count", "type": "rule-based", "score_range": [0, 1]}]
        specs = [{
            "metric_id": "count", "source": "final_agent_response",
            "path": "$.tea_varieties", "operator": "count_gte", "expected": 5,
            "score_mapping": {"pass": 1, "fail": 0},
        }]
        with self.assertRaisesRegex(PipelineGenerationError, "raw final_agent_response"):
            TaskGenerationPipeline._validate_metric_implementations(specs, metrics)

    def test_document_rules_are_promoted_before_dsl_compilation(self):
        metrics = [{
            "id": "count", "category": "outcome", "type": "rule-based",
            "scope": "terminal", "condition": "至少五种茶", "rubric": "至少五种茶",
            "evaluator": {
                "kind": "document_rule", "source": "runtime_rule",
                "assertion": "count_tea_varieties(final_agent_response) >= 5",
                "score_mapping": {"true": 1, "false": 0},
            },
            "score_range": [0, 1], "weight": 1,
        }]
        TaskGenerationPipeline._promote_unstructured_document_rules(metrics)
        self.assertEqual(metrics[0]["type"], "model-based")
        self.assertEqual(metrics[0]["evaluator"]["kind"], "external_llm_judge")
        self.assertNotIn("condition", metrics[0])

    def test_noise_penalty_removes_duplicate_nested_noise_reference(self):
        metrics = [{
            "id": "penalty_irrelevant_tool_use", "category": "penalty",
            "condition": "avoid irrelevant tools", "rubric": "avoid tools",
            "evaluator": {
                "kind": "trajectory_rule",
                "assertion": "tool_call.name != 'weather'",
            },
        }]
        TaskGenerationPipeline._ensure_deterministic_noise_penalty(
            metrics, noise_names=["weather"]
        )
        self.assertEqual([item["id"] for item in metrics], ["penalty_noise_tool_usage"])

    def test_acceptance_penalty_cases_use_zero_for_success(self):
        contract = TaskGenerationPipeline._build_acceptance_contract(
            task_description={"task": "test"},
            data_manifest={}, data_tables=[], actions=[], tools=[], key_steps=[],
            metrics=[{
                "id": "penalty", "category": "penalty",
                "evaluator": {"score_mapping": {"pass": 0, "fail": -1}},
            }],
            reward_formula={},
        )
        cases = {item["case_id"]: item for item in contract["reward_cases"]}
        self.assertEqual(cases["penalty.pass"]["expected_score"], 0)
        self.assertEqual(cases["penalty.fail"]["expected_score"], -1)

    def test_acceptance_rejects_reversed_penalty_case(self):
        metrics = [{
            "id": "penalty", "category": "penalty",
            "evaluator": {"score_mapping": {"pass": 0, "fail": -1}},
        }]
        contract = TaskGenerationPipeline._build_acceptance_contract(
            task_description={"task": "test"}, data_manifest={}, data_tables=[],
            actions=[], tools=[], key_steps=[], metrics=metrics, reward_formula={},
        )
        contract["reward_cases"][0]["expected_score"] = -1
        with self.assertRaisesRegex(PipelineGenerationError, "reversed expected score"):
            TaskGenerationPipeline._validate_acceptance_contract(
                contract, actions=[], tools=[], metrics=metrics
            )

    def test_acceptance_contract_merge_preserves_required_scenario_shape(self):
        baseline = {
            "authority": "env_factory_outer_workflow",
            "scenarios": [{"scenario_id": "s1", "kind": "lifecycle", "steps": ["reset"]}],
            "tool_cases": [], "argument_probes": [], "executable_scenarios": [],
            "reward_cases": [], "fixtures": {}, "invariants": ["x"],
            "mutations": [], "mutation_tests": ["x"],
        }
        candidate = {"scenarios": [{"scenario_id": "s1", "kind": "", "steps": None}]}
        merged = TaskGenerationPipeline._merge_acceptance_contract(candidate, baseline)
        self.assertEqual(merged["scenarios"], baseline["scenarios"])

    def test_metric_type_is_recovered_from_executable_evaluator(self):
        metrics = [{
            "id": "outcome",
            "type": "business-rule",
            "evaluator": {"kind": "business_state_rule"},
        }]
        normalized = TaskGenerationPipeline._normalize_metric_types(metrics)
        self.assertEqual(normalized[0]["type"], "rule-based")

    def test_rule_assertion_is_recovered_from_metric_condition(self):
        metrics = [{
            "id": "business_complete",
            "type": "rule-based",
            "condition": "completed_items >= requested_items",
            "evaluator": {
                "kind": "business_state_rule",
                "source": "runtime_rule",
                "score_mapping": {"true": 1, "false": 0},
            },
        }]
        normalized = TaskGenerationPipeline._normalize_metric_evaluators(metrics)
        self.assertEqual(
            normalized[0]["evaluator"]["assertion"],
            "completed_items >= requested_items",
        )

    def test_rule_condition_is_recovered_from_evaluator_assertion(self):
        metrics = [{
            "id": "business_complete",
            "type": "rule-based",
            "evaluator": {
                "kind": "business_state_rule",
                "source": "runtime_rule",
                "assertion": "completed == true",
                "score_mapping": {"true": 1, "false": 0},
            },
        }]
        normalized = TaskGenerationPipeline._normalize_metric_evaluators(metrics)
        self.assertEqual(normalized[0]["condition"], "completed == true")

    def test_metric_weights_are_normalized_by_sign_group(self):
        metrics = [
            {"category": "process", "weight": 2.0},
            {"category": "outcome", "weight": 3.0},
            {"category": "penalty", "weight": 4.0},
        ]
        normalized = TaskGenerationPipeline._normalize_metric_weights(metrics)
        self.assertAlmostEqual(normalized[0]["weight"], 0.4)
        self.assertAlmostEqual(normalized[1]["weight"], 0.6)
        self.assertAlmostEqual(normalized[2]["weight"], 1.0)

    def test_process_outcome_penalty_and_normalized_formula(self):
        metrics = [
            {
                "id": "process",
                "category": "process",
                "type": "hybrid",
                "scope": "step",
                "target_action": "query_recipe",
                "evaluation_inputs": ["recent_conversation", "tool_call", "tool_arguments"],
                "criteria": ["动作和参数符合上下文"],
                "condition": "llm_expected_tool_call_exact_match",
                "evaluator": {"kind": "hybrid_tool_call", "source": "external_llm", "comparison": "exact_tool_name_and_canonical_arguments", "score_mapping": {"match": 1, "mismatch": 0}},
                "score_range": [0, 1],
                "weight": 0.25,
                "rubric": "过程动作正确",
            },
            {
                "id": "outcome",
                "category": "outcome",
                "type": "rule-based",
                "scope": "terminal",
                "condition": "task_completed == true",
                "evaluator": {"kind": "business_state_rule", "source": "runtime_rule", "assertion": "task_completed == true", "score_mapping": {"true": 1, "false": 0}},
                "score_range": [0, 1],
                "weight": 0.75,
                "rubric": "任务完成",
            },
            {
                "id": "penalty",
                "category": "penalty",
                "type": "hybrid",
                "scope": "step",
                "condition": "irrelevant_output == true",
                "evaluation_inputs": ["recent_conversation", "agent_message"],
                "criteria": ["是否偏离用户诉求"],
                "evaluator": {"kind": "hybrid_outcome", "source": "external_llm", "rule": {"kind": "trajectory_rule", "assertion": "irrelevant_output == true"}, "external_llm": {"criteria": ["是否偏离用户诉求"]}, "score_mapping": {"violation": -1, "clear": 0}},
                "score_range": [-1, 0],
                "weight": 1.0,
                "rubric": "偏离用户诉求惩罚",
            },
        ]
        TaskGenerationPipeline._validate_metrics(metrics)
        TaskGenerationPipeline._validate_reward_formula(
            {"type": "separate_sign_weighted_sum", "formula": "R = clip(pos + neg, -1, 1)", "positive_weight_sum": 1, "negative_weight_sum": 1, "score_range": [-1, 1]},
            metrics,
        )


if __name__ == "__main__":
    unittest.main()

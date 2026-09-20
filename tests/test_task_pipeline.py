import json
import tempfile
import unittest
from pathlib import Path

from env_factory.task_pipeline import PipelineGenerationError, TaskGenerationPipeline


class BusinessDataArtifactsTest(unittest.TestCase):
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


class RewardContractTest(unittest.TestCase):
    def test_missing_semantic_criteria_can_fall_back_to_rubric(self):
        metrics = [{"id": "process", "type": "hybrid", "rubric": "检查关键动作是否正确"}]
        normalized = TaskGenerationPipeline._normalize_metric_evaluation_fields(metrics)
        self.assertEqual(normalized[0]["criteria"], ["检查关键动作是否正确"])

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

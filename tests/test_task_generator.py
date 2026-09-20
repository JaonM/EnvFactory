import json
import unittest
from unittest.mock import patch

from env_factory import (
    SceneNode,
    TaskGenerator,
    TaskType,
)
from env_factory.task_pipeline import TaskGenerationPipeline


class FakeStore:
    def random_scene_event_path(self, hops):
        self.hops = hops
        if hops == 0:
            return (SceneNode("买衣服", ("购买服装",)),)
        return (
            SceneNode("买衣服", ("购买服装",)),
            SceneNode("尺寸", ("尺码",)),
        )


class FakeLLM:
    def complete(self, prompt, **kwargs):
        if prompt.startswith("{"):
            payload = json.loads(prompt)
            self.keywords = payload["keywords"]
            self.task_type = payload["task_type"]
        if "输出 JSON 对象" in kwargs.get("system_prompt", ""):
            class Response:
                content = '{"goal":"购买合适的衣服","actions":["选择尺码"]}'

            return Response()

        class Response:
            content = (
                '{"task":"帮我买一件合适尺码的衣服"}'
                if "任务描述" not in prompt
                else (
                    '[{"type":"state","field":"order_status","description":"当前订单状态","value":"pending","visibility":"observable"},'
                    '{"type":"action","field":"place_order","description":"提交订单","value":{"params":{}},"visibility":"observable"},'
                    '{"type":"transition_rule","field":"submit_order","description":"提交订单后的状态变化","value":{"when":"place_order is called","effect":"order_status becomes submitted"},"visibility":"hidden"},'
                    '{"type":"termination","field":"success","description":"任务成功条件","value":["order_status == submitted"],"visibility":"hidden"}]'
                    if "奖励规则" not in prompt
                    else '[{"id":"check_order","type":"rule-based","scope":"terminal","condition":"order_status == submitted","reward":1,"penalty":0,"once":true,"rubric":"检查工具调用结果","weight":1}]'
                )
            )

        return Response()


class TaskGeneratorTest(unittest.TestCase):
    @staticmethod
    def _pipeline_artifacts():
        return {
            "task": "帮我买一件合适尺码的衣服",
            "requirements": {"input_modalities": ["text"]},
            "complexity": "standard",
            "environment": [
                {"type": "state", "field": "order_status", "description": "订单状态", "value": "pending"},
                {"type": "action", "field": "place_order", "description": "提交订单", "value": "place_order"},
            ],
            "metrics": [{
                "id": "check_order", "type": "rule-based", "scope": "terminal",
                "condition": "order_status == submitted", "reward": 1.0,
                "penalty": 0.0, "once": True, "rubric": "检查工具调用结果", "weight": 1.0,
            }],
            "data_models": [], "media_fixtures": [], "truth_bindings": [],
            "user_profiles": [], "user_scripts": [], "dialogue_sessions": [],
            "actions": [], "tools": [], "tool_bindings": [],
            "observation_schema": {}, "reward_formula": {}, "termination": [],
        }
    def test_task_type_accepts_comma_separated_values(self):
        with patch("env_factory.task_generator.random.choice", return_value=TaskType.RESEARCH):
            self.assertEqual(
                TaskGenerator._select_task_type("QA, Event, Research"),
                TaskType.RESEARCH,
            )

    def test_zero_hops_uses_one_scene(self):
        store = FakeStore()
        llm = FakeLLM()
        captured = {}
        def generate(**kwargs):
            captured.update(kwargs)
            return self._pipeline_artifacts()
        with patch("env_factory.task_generator.random.choice", side_effect=lambda items: items[0]):
            with patch.object(TaskGenerationPipeline, "generate", side_effect=generate):
                task = TaskGenerator(store, llm).generate(0, TaskType.EVENT)

        self.assertEqual(store.hops, 0)
        self.assertEqual(captured["keywords"], ["买衣服"])
        self.assertEqual(task.desc, "帮我买一件合适尺码的衣服")

    def test_generate_uses_path_keywords(self):
        store = FakeStore()
        llm = FakeLLM()
        captured = {}
        def generate(**kwargs):
            captured.update(kwargs)
            return self._pipeline_artifacts()
        with (
            patch("env_factory.task_generator.random.randint", return_value=2),
            patch("env_factory.task_generator.random.choice", side_effect=lambda items: items[0]),
        ):
            with patch.object(TaskGenerationPipeline, "generate", side_effect=generate):
                task = TaskGenerator(store, llm).generate(3, TaskType.EVENT)

        self.assertEqual(store.hops, 2)
        self.assertEqual(captured["keywords"], ["买衣服", "尺寸"])
        self.assertEqual(captured["task_type"], "Event")
        self.assertEqual(task.desc, "帮我买一件合适尺码的衣服")
        self.assertEqual(task.task_type, TaskType.EVENT)
        self.assertIn("action", {item["type"] for item in task.env})
        self.assertEqual(
            task.metrics,
            [{
                "id": "check_order",
                "type": "rule-based",
                "scope": "terminal",
                "condition": "order_status == submitted",
                "reward": 1.0,
                "penalty": 0.0,
                "once": True,
                "rubric": "检查工具调用结果",
                "weight": 1.0,
            }],
        )


if __name__ == "__main__":
    unittest.main()

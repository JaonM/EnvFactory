import json
import unittest
from unittest.mock import patch

from env_factory import (
    SceneNode,
    TaskEnvironmentGenerator,
    TaskGenerator,
    TaskMetricsGenerator,
    TaskType,
)


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
    def test_environment_parser_skips_empty_optional_records(self):
        environment = TaskEnvironmentGenerator._parse_environment(
            json.dumps([
                {"type": "user_profile", "field": "age", "description": "用户年龄", "value": "", "visibility": "hidden"},
                {"type": "state", "field": "status", "description": "当前状态", "value": "pending", "visibility": "observable"},
                {"type": "action", "field": "submit", "description": "提交操作", "value": {"params": {}}, "visibility": "observable"},
                {"type": "transition_rule", "field": "submit_rule", "description": "提交后的状态变化", "value": {"when": "submit", "effect": "status=submitted"}, "visibility": "hidden"},
                {"type": "termination", "field": "success", "description": "成功条件", "value": ["status == submitted"], "visibility": "hidden"},
            ], ensure_ascii=False)
        )

        self.assertEqual(len(environment), 4)

    def test_environment_has_no_visibility_and_only_profile_is_observable(self):
        environment = TaskEnvironmentGenerator._parse_environment(
            json.dumps([
                {"type": "user_profile", "field": "interest", "description": "用户兴趣", "value": "服装", "visibility": "hidden"},
                {"type": "task_info", "field": "goal", "description": "任务目标", "value": "完成任务", "visibility": "observable"},
                {"type": "state", "field": "status", "description": "当前状态", "value": "pending", "visibility": "observable"},
                {"type": "action", "field": "submit", "description": "提交", "value": {"params": {}}, "visibility": "observable"},
                {"type": "transition_rule", "field": "submit_rule", "description": "状态变化", "value": {"when": "submit", "effect": "done"}, "visibility": "observable"},
                {"type": "termination", "field": "success", "description": "成功条件", "value": ["status == done"], "visibility": "observable"},
            ], ensure_ascii=False)
        )

        observable, hidden = TaskEnvironmentGenerator.split_observation(environment)
        self.assertEqual([item["type"] for item in observable], ["user_profile"])
        self.assertEqual(len(hidden), 5)
        self.assertTrue(all("visibility" not in item for item in environment))

    def test_metrics_parser_normalizes_positive_penalty(self):
        metric = TaskMetricsGenerator._parse_metric(
            {
                "id": "safety",
                "type": "rule-based",
                "scope": "step",
                "condition": "safety_violation == false",
                "reward": 0.1,
                "penalty": 1.0,
                "rubric": "遵守安全规则",
                "weight": 0.5,
            },
            0,
        )

        self.assertEqual(metric["penalty"], -1.0)

    def test_task_type_accepts_comma_separated_values(self):
        with patch("env_factory.task_generator.random.choice", return_value=TaskType.RESEARCH):
            self.assertEqual(
                TaskGenerator._select_task_type("QA, Event, Research"),
                TaskType.RESEARCH,
            )

    def test_zero_hops_uses_one_scene(self):
        store = FakeStore()
        llm = FakeLLM()
        with patch("env_factory.task_generator.random.choice", side_effect=lambda items: items[0]):
            task = TaskGenerator(store, llm).generate(0, TaskType.EVENT)

        self.assertEqual(store.hops, 0)
        self.assertEqual(llm.keywords, ["买衣服"])
        self.assertEqual(task.desc, "帮我买一件合适尺码的衣服")

    def test_hierarchy_child_limit_is_configurable(self):
        generator = TaskGenerator(FakeStore(), FakeLLM(), hierarchy_child_limit=3)
        self.assertEqual(generator.hierarchy_child_limit, 3)
        with self.assertRaises(ValueError):
            TaskGenerator(FakeStore(), FakeLLM(), hierarchy_child_limit=0)

    def test_generate_uses_path_keywords(self):
        store = FakeStore()
        llm = FakeLLM()
        with (
            patch("env_factory.task_generator.random.randint", return_value=2),
            patch("env_factory.task_generator.random.choice", side_effect=lambda items: items[0]),
        ):
            task = TaskGenerator(store, llm).generate(3, TaskType.EVENT)

        self.assertEqual(store.hops, 2)
        self.assertEqual(llm.keywords, ["买衣服", "尺寸"])
        self.assertEqual(llm.task_type, "Event")
        self.assertEqual(task.desc, "帮我买一件合适尺码的衣服")
        self.assertEqual(task.task_type, TaskType.EVENT)
        self.assertEqual(task.env[0]["type"], "state")
        self.assertIn("action", {item["type"] for item in task.env})
        observable, hidden = TaskEnvironmentGenerator.split_observation(task.env)
        self.assertEqual(observable, [])
        self.assertEqual(len(hidden), len(task.env))
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

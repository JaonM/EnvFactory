import json
import unittest
from unittest.mock import patch

from env_factory import SceneNode, TaskEnvironmentGenerator, TaskGenerator, TaskType


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
    def test_zero_hops_uses_one_scene(self):
        store = FakeStore()
        llm = FakeLLM()
        with patch("env_factory.task_generator.random.choice", side_effect=lambda items: items[0]):
            task = TaskGenerator(store, llm).generate(0, TaskType.EVENT, "complete")

        self.assertEqual(store.hops, 0)
        self.assertEqual(llm.keywords, ["买衣服"])
        self.assertEqual(task.desc, "帮我买一件合适尺码的衣服")

    def test_generate_uses_path_keywords(self):
        store = FakeStore()
        llm = FakeLLM()
        with (
            patch("env_factory.task_generator.random.randint", return_value=2),
            patch("env_factory.task_generator.random.choice", side_effect=lambda items: items[0]),
        ):
            task = TaskGenerator(store, llm).generate(3, TaskType.EVENT, "complete")

        self.assertEqual(store.hops, 2)
        self.assertEqual(llm.keywords, ["买衣服", "尺寸"])
        self.assertEqual(llm.task_type, "Event")
        self.assertEqual(task.desc, "帮我买一件合适尺码的衣服")
        self.assertEqual(task.task_type, TaskType.EVENT)
        self.assertEqual(task.environment_mode.value, "complete")
        self.assertEqual(task.env[0]["type"], "state")
        self.assertIn("action", {item["type"] for item in task.env})
        observable, hidden = TaskEnvironmentGenerator.split_observation(task.env)
        self.assertEqual(observable[0]["visibility"], "observable")
        self.assertEqual(hidden[0]["visibility"], "hidden")
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

import json
import unittest
from unittest.mock import patch

from env_factory import SceneNode, TaskGenerator, TaskType


class FakeStore:
    def random_scene_event_path(self, hops):
        self.hops = hops
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

        class Response:
            content = (
                '{"task":"帮我买一件合适尺码的衣服"}'
                if "任务描述" not in prompt
                else (
                    '[{"type":"用户画像","field":"年龄","value":"30岁"}]'
                    if "奖励规则" not in prompt
                    else '[{"type":"rule-based","rubric":"检查工具调用结果","weight":1}]'
                )
            )

        return Response()


class TaskGeneratorTest(unittest.TestCase):
    def test_generate_uses_path_keywords(self):
        store = FakeStore()
        llm = FakeLLM()
        with patch("env_factory.task_generator.random.randint", return_value=2):
            task = TaskGenerator(store, llm).generate(3, TaskType.EVENT)

        self.assertEqual(store.hops, 2)
        self.assertEqual(llm.keywords, ["买衣服", "购买服装", "尺寸", "尺码"])
        self.assertEqual(llm.task_type, "Event")
        self.assertEqual(task.desc, "帮我买一件合适尺码的衣服")
        self.assertEqual(task.env, [{"type": "用户画像", "field": "年龄", "value": "30岁"}])
        self.assertEqual(
            task.metrics,
            [{"type": "rule-based", "rubric": "检查工具调用结果", "weight": 1.0}],
        )


if __name__ == "__main__":
    unittest.main()

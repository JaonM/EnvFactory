"""Generate long-horizon task descriptions from Scene graph paths."""

import json
import math
import random
import re
from typing import Any

from .graph_builder import Neo4jGraphStore
from .knowledge_graph import SceneNode, TaskType
from .llm import LLMClient
from .task import Task, TaskEnvironmentMode


class TaskGenerationError(ValueError):
    """Raised when a graph path or LLM task response is invalid."""


def _parse_json(content: str) -> Any:
    """Parse a JSON response, including an optional Markdown code fence."""

    normalized = content.strip()
    match = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```", normalized, re.DOTALL | re.IGNORECASE
    )
    return json.loads(match.group(1) if match else normalized)


def _render_prompt(template: str, **values: object) -> str:
    for key, value in values.items():
        replacement = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        template = template.replace("{{" + key + "}}", replacement)
    return template


class TaskMetricsGenerator:
    """Generate model-based and rule-based reward rubrics for a task."""

    PROMPT = """# 背景
我现需要以 Agentic RL 的方式训练个人 AI 分身模型的长程任务完成能力，现已构建任务描述和任务环境，请你听从指令。
个人 AI 分身的动作可以有以下选择：
- 对用户进行澄清追问，每次追问需要控制问题个数，不要出现夺命连环问的情况
- 回答用户
- 调用一个或同时调用多个工具

# 指令
参考“分析逻辑”针对“任务描述”生成 rubrics 奖励规则，按照“输出格式”进行输出。

# 任务环境
{{env}}

# 分析逻辑
1. 基于任务环境作为输入，假设你作为个人AI 助理模型，下一个动作是什么
2. 以你的下一个动作为标准答案，设计 rubrics 的奖励规则和规则权重（权重不能为负），注意区分奖励规则属于 model-based 还是 rule-based。
3. rule-based 奖励规则定义为你可以使用确定性规则进行评判；model-based 奖励规则定义为你需要使用模型根据语义信息进行评判。

# 任务描述
{{task_desc}}

# 输出格式
JSON格式输出，例如：[{"type":"rule-based/model-based","rubric":"奖励规则描述","weight:":"规则权重"}]

只输出合法 JSON，不要解释。"""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def generate(self, task_desc: str, environment: list[dict[str, str]]) -> list[dict[str, object]]:
        if not task_desc.strip():
            raise ValueError("task_desc must not be empty")
        prompt = _render_prompt(self.PROMPT, env=environment, task_desc=task_desc.strip())
        response = self.llm.complete(
            prompt,
            thinking=False,
            temperature=0.3,
            max_tokens=1_000,
        )
        try:
            payload = _parse_json(response.content)
            if not isinstance(payload, list):
                raise TypeError
            metrics = []
            for item in payload:
                if not isinstance(item, dict):
                    raise TypeError
                rubric_type = item.get("type")
                rubric = item.get("rubric")
                weight = item.get("weight", item.get("weight:"))
                if rubric_type not in {"rule-based", "model-based"}:
                    raise ValueError("unsupported rubric type")
                if not isinstance(rubric, str) or not rubric.strip():
                    raise TypeError
                weight = float(weight)
                if not math.isfinite(weight) or weight < 0:
                    raise ValueError("rubric weight must not be negative")
                metrics.append({"type": rubric_type, "rubric": rubric.strip(), "weight": weight})
            return metrics
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise TaskGenerationError("invalid task metrics response") from exc


class TaskEnvironmentGenerator:
    """Generate the user profile and task information required by a task."""

    COMPLETE_PROMPT = """# 背景
我现需要以 Agentic RL 的方式训练个人 AI 分身模型的长程任务完成能力，现已获得任务描述，请你听从指令。

# 指令
请你根据“任务描述”，模拟生成完成任务所需要的**完整**任务环境信息，包含「用户画像」和「任务信息」两个维度，按照“输出格式”进行输出。

# 任务描述
{{task_desc}}

# 输出格式
JSON格式输出，例如：[{"type":"用户画像/任务信息","field":"字段名称","value:":"字段值"}]

只输出合法 JSON，不要解释。"""

    INCOMPLETE_PROMPT = """# 背景
我现需要以 Agentic RL 的方式训练个人 AI 分身模型的长程任务完成能力，现已获得任务描述，请你听从指令。

# 指令
请你根据“任务描述”，模拟生成完成任务所需要的任务环境信息，任务信息可以不完整，即随机删减部分信息，删减力度随机，包含「用户画像」和「任务信息」两个维度，按照“输出格式”进行输出。

# 任务描述
{{task_desc}}

# 输出格式
JSON格式输出，例如：[{"type":"用户画像/任务信息","field":"字段名称","value:":"字段值"}]

只输出合法 JSON，不要解释。"""

    def __init__(self, llm: LLMClient, *, complete_probability: float = 0.5) -> None:
        if not 0 <= complete_probability <= 1:
            raise ValueError("complete_probability must be between 0 and 1")
        self.llm = llm
        self.complete_probability = complete_probability

    def generate(
        self,
        task_desc: str,
        mode: TaskEnvironmentMode | str = TaskEnvironmentMode.RANDOM,
    ) -> list[dict[str, str]]:
        if not task_desc.strip():
            raise ValueError("task_desc must not be empty")
        selected_mode = self._select_mode(mode)
        prompt = _render_prompt(
            self.COMPLETE_PROMPT if selected_mode is TaskEnvironmentMode.COMPLETE
            else self.INCOMPLETE_PROMPT,
            task_desc=task_desc.strip(),
        )
        response = self.llm.complete(
            prompt,
            thinking=False,
            temperature=0.7,
            max_tokens=1_500,
        )
        try:
            payload = _parse_json(response.content)
            if not isinstance(payload, list):
                raise TypeError
            environment = []
            for item in payload:
                if not isinstance(item, dict):
                    raise TypeError
                if not all(isinstance(item.get(field), str) and item[field].strip() for field in ("type", "field", "value")):
                    raise TypeError
                environment.append({field: item[field].strip() for field in ("type", "field", "value")})
            return environment
        except (json.JSONDecodeError, TypeError) as exc:
            raise TaskGenerationError("invalid task environment response") from exc

    def _select_mode(self, mode: TaskEnvironmentMode | str) -> TaskEnvironmentMode:
        try:
            selected = mode if isinstance(mode, TaskEnvironmentMode) else TaskEnvironmentMode(mode)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in TaskEnvironmentMode)
            raise ValueError(f"unsupported environment mode {mode!r}; expected one of: {allowed}") from exc
        if selected is TaskEnvironmentMode.RANDOM:
            return (
                TaskEnvironmentMode.COMPLETE
                if random.random() < self.complete_probability
                else TaskEnvironmentMode.INCOMPLETE
            )
        return selected


class TaskGenerator:
    """Sample event-element paths and turn their keywords into tasks."""

    SYSTEM_PROMPT = """# 背景
我现需要以 Agentic RL 的方式训练个人 AI 分身模型的长程任务完成能力，现已获得任务描述，请你听从指令。

# 指令
请你根据“路径关键词”结合“任务类型”生成一句口语化的任务描述，关键词越多，任务描述粒度越细，按照“输出格式”进行返回。

# 路径关键词
{{keywords}}

# 任务类型
{{task_type}}

# 任务类型说明
1.QA：问答任务
2.Event：事件办理任务，通常是 Long-Horizon 任务
3.Coding：编程任务
4.Chat：聊天式任务
5.Research：研究性任务，如 DeepResearch

# 输出格式
JSON格式输出，例如：{"task":"任务描述"}

只输出合法 JSON，不要解释。"""

    def __init__(self, store: Neo4jGraphStore, llm: LLMClient) -> None:
        self.store = store
        self.llm = llm
        self.environment_generator = TaskEnvironmentGenerator(llm)
        self.metrics_generator = TaskMetricsGenerator(llm)

    def generate(
        self,
        hops: int = 3,
        task_type: TaskType | str | None = None,
        environment_mode: TaskEnvironmentMode | str = TaskEnvironmentMode.RANDOM,
    ) -> Task:
        """Generate one task using a random path length between 1 and ``hops``."""

        if hops <= 0:
            raise ValueError("hops must be greater than zero")
        selected_hops = random.randint(1, hops)
        path = self.store.random_scene_event_path(selected_hops)
        if not path:
            raise TaskGenerationError(f"no same-event-element path found for {hops} hops")
        selected_type = self._select_task_type(task_type)
        keywords = self._keywords(path)
        response = self.llm.complete(
            json.dumps(
                {"keywords": keywords, "task_type": selected_type.value},
                ensure_ascii=False,
            ),
            system_prompt=self.SYSTEM_PROMPT,
            thinking=False,
            temperature=0.7,
            max_tokens=300,
            response_format="json_object",
        )
        try:
            payload = _parse_json(response.content)
            task = payload["task"]
            if not isinstance(task, str) or not task.strip():
                raise TypeError
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise TaskGenerationError("invalid task generation response") from exc
        task_desc = task.strip()
        environment = self.environment_generator.generate(task_desc, environment_mode)
        metrics = self.metrics_generator.generate(task_desc, environment)
        return Task(desc=task_desc, env=environment, metrics=metrics)

    @staticmethod
    def _select_task_type(task_type: TaskType | str | None) -> TaskType:
        if task_type is None:
            return random.choice(tuple(TaskType))
        if isinstance(task_type, TaskType):
            return task_type
        try:
            return TaskType(task_type)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in TaskType)
            raise ValueError(
                f"unsupported task_type {task_type!r}; expected one of: {allowed}"
            ) from exc

    @staticmethod
    def _keywords(path: tuple[SceneNode, ...]) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                word.strip()
                for scene in path
                for word in (scene.name, *scene.words)
                if word.strip()
            )
        )

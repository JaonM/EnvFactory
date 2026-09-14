"""Generate long-horizon task descriptions from Scene graph paths."""

import json
import logging
import math
import random
import re
import time
from typing import Any

from .graph_builder import Neo4jGraphStore
from .knowledge_graph import SceneNode, TaskType
from .llm import LLMClient
from .task import Task, TaskEnvironmentMode

logger = logging.getLogger(__name__)


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

    SCOPES = {"step", "state", "terminal", "trajectory"}

    PROMPT = """# 背景
我现需要以 Agentic RL 的方式训练个人 AI 分身模型的长程任务完成能力，现已构建任务描述和任务环境，请你听从指令。
个人 AI 分身的单个动作可以有以下选择：
A 对用户进行澄清追问，每次追问需要控制问题个数，不要出现夺命连环问的情况
B 回答用户
C 调用一个或同时调用多个工具

# 指令
参考“分析逻辑”针对“任务描述”和结构化任务环境生成可用于 Agentic RL 的奖励规则，按照“输出格式”进行输出。

# 任务环境
{{env}}

# 分析逻辑
1. 区分 step、state、terminal、trajectory 四种评估时机。
2. rule-based 必须提供可由 state/action/事件确定性判断的 condition；model-based 必须提供 criteria 和 score_range。
3. 每条规则必须能引用任务环境中的状态、动作或交互轨迹，不得使用无法执行的空泛描述。
4. reward 和 penalty 必须是有限数字；安全违规的 penalty 应高于普通过程奖励，最终成功奖励应高于单步奖励。
5. 不泄露任何“背景”中的信息
5. 动作轨迹需要动作A/B结尾，结尾之前可以合理选择A/B/C进行搭配组成动作轨迹

# 任务描述
{{task_desc}}

# 输出格式
JSON 格式输出，例如：[{"id":"safe_action","type":"rule-based","scope":"step","condition":"action.type == 'send_reminder' and safety_violation == false","reward":0.1,"penalty":0,"once":false,"weight":0.2},{"id":"task_success","type":"rule-based","scope":"terminal","condition":"task_success == true","reward":1.0,"penalty":0,"once":true,"weight":0.5},{"id":"communication_quality","type":"model-based","scope":"trajectory","criteria":["是否根据用户反馈调整计划"],"score_range":[0,1],"weight":0.3}]

只输出合法 JSON，不要解释。"""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def generate(self, task_desc: str, environment: list[dict[str, Any]]) -> list[dict[str, object]]:
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
            metrics = [self._parse_metric(item) for item in payload]
            return metrics
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise TaskGenerationError("invalid task metrics response") from exc

    @staticmethod
    def _parse_metric(item: Any) -> dict[str, object]:
        if not isinstance(item, dict):
            raise TypeError
        metric_id = item.get("id")
        rubric_type = item.get("type")
        scope = item.get("scope")
        rubric = item.get("rubric")
        if not isinstance(metric_id, str) or not metric_id.strip():
            raise TypeError
        if rubric_type not in {"rule-based", "model-based"}:
            raise ValueError("unsupported rubric type")
        if scope not in TaskMetricsGenerator.SCOPES:
            raise ValueError("unsupported metric scope")
        if not isinstance(rubric, str) or not rubric.strip():
            raise TypeError
        weight = float(item.get("weight", item.get("weight:")))
        reward = float(item.get("reward", 0))
        penalty = float(item.get("penalty", 0))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("rubric weight must not be negative")
        if not math.isfinite(reward) or not math.isfinite(penalty) or penalty > 0:
            raise ValueError("invalid metric reward or penalty")
        if rubric_type == "rule-based":
            condition = item.get("condition")
            if not isinstance(condition, str) or not condition.strip():
                raise TypeError
        else:
            criteria = item.get("criteria")
            score_range = item.get("score_range")
            if not isinstance(criteria, list) or not criteria or not all(
                isinstance(value, str) and value.strip() for value in criteria
            ):
                raise TypeError
            if (
                not isinstance(score_range, list)
                or len(score_range) != 2
                or not all(isinstance(value, (int, float)) for value in score_range)
                or score_range[0] > score_range[1]
            ):
                raise TypeError
        return {
            **item,
            "id": metric_id.strip(),
            "type": rubric_type,
            "scope": scope,
            "rubric": rubric.strip(),
            "reward": reward,
            "penalty": penalty,
            "weight": weight,
            "once": bool(item.get("once", False)),
        }


class TaskEnvironmentGenerator:
    """Generate a structured, executable task environment."""

    ENVIRONMENT_TYPES = {
        "user_profile", "task_info", "state", "hidden_state",
        "action", "transition_rule", "termination",
    }
    REQUIRED_TYPES = {"state", "action", "transition_rule", "termination"}
    VISIBILITIES = {"observable", "hidden"}

    COMPLETE_PROMPT = """# 背景
我现需要以 Agentic RL 的方式训练个人 AI 分身模型的长程任务完成能力，现已获得任务描述，请你听从指令。

# 指令
请你根据“任务描述”，生成一个可被 AI Agent 执行和评估的结构化任务环境。每条记录必须包含英文 type、field、description、value、visibility 字段。visibility=observable 表示初始暴露给 Agent，visibility=hidden 表示初始不暴露、需要通过用户交互、工具调用或环境事件获取。环境必须包含 user_profile、task_info、state、hidden_state、action、transition_rule、termination 七类信息，不泄露任何“背景”中的信息。

# 任务描述
{{task_desc}}

# 输出格式
JSON 格式输出，每条记录包含 type、field、description、value、visibility；例如：[{"type":"state","field":"status","description":"当前订单状态","value":"pending","visibility":"observable"},{"type":"hidden_state","field":"user_preference","description":"用户偏好的沟通方式","value":"concise","visibility":"hidden"},{"type":"action","field":"submit","description":"提交订单","value":{"params":{}},"visibility":"observable"},{"type":"transition_rule","field":"submit_rule","description":"提交订单后的状态变化","value":{"when":"submit is called","effect":"status becomes submitted"},"visibility":"hidden"},{"type":"termination","field":"success","description":"任务成功条件","value":["status == submitted"],"visibility":"hidden"}]

state 是当前状态，hidden_state 是 Agent 初始不可见但环境内部使用的状态，action 是 Agent 可执行的动作及参数，transition_rule 定义动作或时间如何改变状态，termination 定义可判定的 success/failure 条件。hidden、transition_rule 和 termination 记录不应直接暴露给 Agent。value 可以是数字、布尔值、字符串、数组或对象，不能使用空泛描述。最多 30 条，确保 JSON 完整。

只输出合法 JSON，不要解释。"""

    INCOMPLETE_PROMPT = """# 背景
我现需要以 Agentic RL 的方式训练个人 AI 分身模型的长程任务完成能力，现已获得任务描述，请你听从指令。

# 指令
请你根据“任务描述”，生成一个可被 AI Agent 执行和评估的结构化任务环境。每条记录必须包含英文 type、field、description、value、visibility 字段。visibility=observable 表示初始暴露给 Agent，visibility=hidden 表示初始不暴露、需要通过用户交互、工具调用或环境事件获取。用户画像、任务信息和部分 state 可以不完整，即随机隐藏部分非关键观测信息，但必须保留 action、transition_rule 和 termination 结构，不泄露任何“背景”中的信息。

# 任务描述
{{task_desc}}

# 输出格式
JSON 格式输出，格式同完整环境：每条记录包含英文 type、field、description、value、visibility；type 必须是 user_profile、task_info、state、hidden_state、action、transition_rule、termination 之一。

不完整模式只能隐藏部分 user_profile、task_info 或 observable state，不得删除全部 state、action、transition_rule、termination。description 必须清楚说明 field 的业务含义；value 可以是数字、布尔值、字符串、数组或对象，不能使用空泛描述。最多 30 条，确保 JSON 完整。

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
    ) -> list[dict[str, Any]]:
        return self.generate_with_mode(task_desc, mode)[0]

    def generate_with_mode(
        self,
        task_desc: str,
        mode: TaskEnvironmentMode | str = TaskEnvironmentMode.RANDOM,
    ) -> tuple[list[dict[str, Any]], TaskEnvironmentMode]:
        if not task_desc.strip():
            raise ValueError("task_desc must not be empty")
        selected_mode = self._select_mode(mode)
        prompt = _render_prompt(
            self.COMPLETE_PROMPT if selected_mode is TaskEnvironmentMode.COMPLETE
            else self.INCOMPLETE_PROMPT,
            task_desc=task_desc.strip(),
        )
        try:
            return self._parse_environment(self._complete(prompt, 5_000)), selected_mode
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("任务环境响应无效，重试一次：%s", exc)
            try:
                return self._parse_environment(self._complete(prompt, 10_000)), selected_mode
            except (json.JSONDecodeError, TypeError, ValueError) as retry_error:
                logger.warning("任务环境重试失败，跳过当前任务：%s", retry_error)
                raise TaskGenerationError("invalid executable task environment response") from retry_error

    def _complete(self, prompt: str, max_tokens: int) -> str:
        response = self.llm.complete(
            prompt,
            thinking=False,
            temperature=0.7,
            max_tokens=max_tokens,
        )
        logger.debug(
            "任务环境 LLM 响应：finish_reason=%s，content_chars=%d",
            getattr(response, "finish_reason", None), len(response.content),
        )
        return response.content

    @staticmethod
    def _parse_environment(content: str) -> list[dict[str, Any]]:
        payload = _parse_json(content)
        if not isinstance(payload, list):
            raise TypeError
        environment: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                raise TypeError
            item_type = item.get("type")
            field = item.get("field")
            description = item.get("description")
            if item_type not in TaskEnvironmentGenerator.ENVIRONMENT_TYPES:
                raise ValueError("unsupported environment item type")
            visibility = item.get("visibility")
            if visibility not in TaskEnvironmentGenerator.VISIBILITIES:
                raise ValueError("environment item visibility must be observable or hidden")
            if (
                not isinstance(field, str) or not field.strip()
                or not isinstance(description, str) or not description.strip()
                or "value" not in item
            ):
                raise TypeError
            value = item["value"]
            if value is None or value == "" or value == [] or value == {}:
                raise TypeError
            environment.append({
                **item,
                "type": item_type,
                "field": field.strip(),
                "description": description.strip(),
                "visibility": visibility,
            })
        present_types = {item["type"] for item in environment}
        if not TaskEnvironmentGenerator.REQUIRED_TYPES <= present_types:
            missing = sorted(TaskEnvironmentGenerator.REQUIRED_TYPES - present_types)
            raise TypeError(f"missing executable environment types: {missing}")
        return environment

    @staticmethod
    def split_observation(
        environment: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split the generated environment into Agent-visible and hidden data."""

        observable = [item for item in environment if item["visibility"] == "observable"]
        hidden = [item for item in environment if item["visibility"] == "hidden"]
        logger.debug(
            "任务环境拆分完成：可观测记录=%d，隐藏记录=%d",
            len(observable), len(hidden),
        )
        return observable, hidden

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

    INTENT_PROMPT = """你是任务意图规划助手。请根据路径关键词和任务类型，提炼一个真实用户任务的意图结构。
输出 JSON 对象，包含 goal、context、constraints、actions、expected_result 字段。
不要生成最终任务描述，不要解释；没有必要的信息可以为空数组或空字符串。"""

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

# 表达风格
{{style}}

# 任务复杂度
{{complexity}}

# 任务意图
{{intent}}

# 生成要求
1. 任务必须有明确目标，不能只是罗列关键词。
2. 将关键词自然融入任务，不要机械复述。
3. 任务描述需要简洁明确。
4. 任务要像真实用户提出的请求，符合日常交流习惯。
5. 不要使用“请根据以下关键词”“基于上述关键词”等模板化开头。
6. 不要输出标题、解释或列表，只返回 JSON。

# 输出格式
JSON格式输出，例如：{"task":"任务描述"}

只输出合法 JSON，不要解释。"""
    STYLES = (
        "直接请求",
        "带背景说明",
        "带约束条件",
        "遇到问题寻求建议",
        "多步骤委托",
        "临时起意的生活请求",
        "专业人士委托",
    )
    COMPLEXITIES = ("简单", "标准", "详细")

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
        task_style: str | None = None,
    ) -> Task:
        """Generate one task using a random path length between 0 and ``hops``."""

        if hops < 0:
            raise ValueError("hops must not be negative")
        started = time.perf_counter()
        selected_hops = random.randint(0, hops)
        path = self.store.random_scene_event_path(selected_hops)
        if not path and selected_hops > 0:
            # Sparse graphs may not contain a path at the initially selected
            # depth. Try shorter paths so batch generation remains productive.
            for fallback_hops in range(selected_hops - 1, -1, -1):
                path = self.store.random_scene_event_path(fallback_hops, attempts=3)
                if path:
                    logger.info(
                        "路径跳数降级：请求=%d，实际=%d",
                        selected_hops, fallback_hops,
                    )
                    selected_hops = fallback_hops
                    break
        if not path:
            logger.warning("任务路径抽取失败：请求跳数=%d，实际跳数=%d", hops, selected_hops)
            raise TaskGenerationError(f"no Scene node or path found for {selected_hops} hops")
        keywords = self._keywords(path)
        logger.info(
            "任务路径抽取完成：请求跳数=%d，实际跳数=%d，节点数=%d，关键词数=%d，关键词=%s，耗时=%.2fs",
            hops,
            selected_hops,
            len(path),
            len(keywords),
            keywords,
            time.perf_counter() - started,
        )
        selected_type = self._select_task_type(task_type)
        selected_style = task_style or random.choice(self.STYLES)
        if selected_style not in self.STYLES:
            raise ValueError(f"unsupported task_style: {selected_style}")
        complexity = random.choice(self.COMPLEXITIES)
        stage_started = time.perf_counter()
        request = {
            "keywords": keywords,
            "task_type": selected_type.value,
            "style": selected_style,
            "complexity": complexity,
        }
        intent_response = self.llm.complete(
            json.dumps(request, ensure_ascii=False),
            system_prompt=self.INTENT_PROMPT,
            thinking=False,
            temperature=0.5,
            max_tokens=600,
            response_format="json_object",
        )
        try:
            intent = _parse_json(intent_response.content)
            if not isinstance(intent, dict) or not isinstance(intent.get("goal"), str):
                raise TypeError
        except (json.JSONDecodeError, TypeError) as exc:
            raise TaskGenerationError("invalid task intent response") from exc
        logger.info("任务意图生成完成：耗时=%.2fs", time.perf_counter() - stage_started)
        stage_started = time.perf_counter()
        description_prompt = json.dumps({**request, "intent": intent}, ensure_ascii=False)
        description_system_prompt = _render_prompt(
            self.SYSTEM_PROMPT,
            style=selected_style,
            complexity=complexity,
            intent=intent,
        )
        task_desc = self._generate_description(
            description_prompt,
            description_system_prompt,
        )
        logger.info("任务描述生成完成：耗时=%.2fs", time.perf_counter() - stage_started)
        stage_started = time.perf_counter()
        environment, selected_environment_mode = self.environment_generator.generate_with_mode(
            task_desc, environment_mode
        )
        logger.info("任务环境生成完成：耗时=%.2fs", time.perf_counter() - stage_started)
        stage_started = time.perf_counter()
        metrics = self.metrics_generator.generate(task_desc, environment)
        logger.info("观测指标生成完成：耗时=%.2fs", time.perf_counter() - stage_started)
        logger.info("任务生成完成：总耗时=%.2fs", time.perf_counter() - started)
        return Task(
            desc=task_desc,
            env=environment,
            metrics=metrics,
            task_type=selected_type,
            environment_mode=selected_environment_mode,
        )

    def _generate_description(self, prompt: str, system_prompt: str) -> str:
        for attempt, max_tokens in enumerate((300, 600), start=1):
            response = self.llm.complete(
                prompt,
                system_prompt=system_prompt
                + (
                    "\n请只输出完整、简短、合法的 JSON，task 只包含一句话。"
                    if attempt > 1
                    else ""
                ),
                thinking=False,
                temperature=0.7 if attempt == 1 else 0.3,
                max_tokens=max_tokens,
                response_format="json_object",
            )
            try:
                payload = _parse_json(response.content)
                task = payload["task"]
                if not isinstance(task, str) or not task.strip():
                    raise TypeError
                return task.strip()
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                if attempt == 1:
                    logger.warning("任务描述响应格式异常，重试一次：%s", exc)
                else:
                    raise TaskGenerationError("invalid task generation response") from exc
        raise AssertionError("unreachable")

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
        keywords = []
        for scene in path:
            candidates = [word.strip() for word in (scene.name, *scene.words) if word.strip()]
            if candidates:
                keywords.append(random.choice(candidates))
        return tuple(dict.fromkeys(keywords))

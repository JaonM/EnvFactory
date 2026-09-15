"""Generate long-horizon task descriptions from Scene graph paths."""

import json
import logging
import math
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .graph_builder import Neo4jGraphStore
from .knowledge_graph import SceneNode, TaskType
from .llm import LLMClient
from .task import Task

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
    METRIC_LIMITS = {"simple": 4, "standard": 6, "complex": 8}

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

# 任务描述
{{task_desc}}

# 任务复杂度
{{complexity}}

# 复杂度特征
{{complexity_features}}

# 输出格式
JSON 格式输出，每条记录还必须包含简短的 rubric 字段，例如：[{"id":"safe_action","type":"rule-based","scope":"step","condition":"action.type == 'send_reminder' and safety_violation == false","reward":0.1,"penalty":0,"once":false,"rubric":"安全发送提醒","weight":0.2},{"id":"task_success","type":"rule-based","scope":"terminal","condition":"task_success == true","reward":1.0,"penalty":0,"once":true,"rubric":"任务成功完成","weight":0.5},{"id":"communication_quality","type":"model-based","scope":"trajectory","criteria":["是否根据用户反馈调整计划"],"score_range":[0,1],"rubric":"交互质量","weight":0.3}]

最多生成 {{max_metrics}} 条规则，优先保留安全、过程进展、最终成功和轨迹质量规则；每条 rubric 和 condition/criteria 都要简短；只输出 JSON 数组，不要使用 metrics 包装对象。

只输出合法 JSON，不要解释。"""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def generate(
        self,
        task_desc: str,
        environment: list[dict[str, Any]],
        complexity: str = "standard",
        complexity_features: dict[str, Any] | None = None,
    ) -> list[dict[str, object]]:
        if not task_desc.strip():
            raise ValueError("task_desc must not be empty")
        max_metrics = self._metric_limit(complexity)
        prompt = _render_prompt(
            self.PROMPT,
            env=environment,
            task_desc=task_desc.strip(),
            complexity=complexity,
            complexity_features=complexity_features or {},
            max_metrics=max_metrics,
        )
        retry_suffix = f"\n只输出最多 {max_metrics} 条简短规则，确保 JSON 在长度限制内完整。"
        for attempt, max_tokens in enumerate((1_000, 2_000), start=1):
            response = self.llm.complete(
                prompt + (retry_suffix if attempt > 1 else ""),
                thinking=False,
                temperature=0.3 if attempt == 1 else 0.1,
                max_tokens=max_tokens,
            )
            finish_reason = getattr(response, "finish_reason", None)
            logger.debug(
                "观测指标 LLM 响应：尝试=%d，finish_reason=%s，content_chars=%d",
                attempt, finish_reason, len(response.content),
            )
            try:
                payload = _parse_json(response.content)
                if not isinstance(payload, list):
                    raise TypeError("metrics response must be a JSON array")
                if len(payload) > max_metrics:
                    raise ValueError(
                        f"metrics count {len(payload)} exceeds limit {max_metrics}"
                    )
                return [
                    self._parse_metric(item, index)
                    for index, item in enumerate(payload)
                ]
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                if attempt == 1:
                    logger.warning(
                        "观测指标响应无效，重试一次：%s，finish_reason=%s，content_chars=%d",
                        exc, finish_reason, len(response.content),
                    )
                    continue
                raise TaskGenerationError(
                    f"invalid task metrics response: {exc}"
                ) from exc
        raise AssertionError("unreachable")

    @classmethod
    def _metric_limit(cls, complexity: str) -> int:
        try:
            return cls.METRIC_LIMITS[complexity]
        except KeyError as exc:
            raise ValueError(
                f"unsupported complexity {complexity!r}; expected one of: {', '.join(cls.METRIC_LIMITS)}"
            ) from exc

    @staticmethod
    def _parse_metric(item: Any, index: int) -> dict[str, object]:
        if not isinstance(item, dict):
            raise TypeError(f"metric[{index}] must be an object")
        metric_id = item.get("id")
        rubric_type = item.get("type")
        scope = item.get("scope")
        rubric = item.get("rubric")
        if not isinstance(metric_id, str) or not metric_id.strip():
            raise TypeError(f"metric[{index}].id must be a non-empty string")
        if rubric_type not in {"rule-based", "model-based"}:
            raise ValueError(f"metric[{index}].type must be rule-based or model-based")
        if scope not in TaskMetricsGenerator.SCOPES:
            raise ValueError(f"metric[{index}].scope must be one of {sorted(TaskMetricsGenerator.SCOPES)}")
        if not isinstance(rubric, str) or not rubric.strip():
            raise TypeError(f"metric[{index}].rubric must be a non-empty string")
        try:
            weight = float(item.get("weight", item.get("weight:")))
            reward = float(item.get("reward", 0))
            penalty = float(item.get("penalty", 0))
        except (TypeError, ValueError) as exc:
            raise TypeError(f"metric[{index}] reward, penalty, and weight must be numbers") from exc
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"metric[{index}].weight must be finite and non-negative")
        if not math.isfinite(reward):
            raise ValueError(f"metric[{index}].reward must be finite")
        if not math.isfinite(penalty):
            raise ValueError(f"metric[{index}].penalty must be finite")
        if penalty > 0:
            logger.warning(
                "规范化正惩罚值：metric=%s，penalty=%s，将转换为 %s",
                metric_id,
                penalty,
                -penalty,
            )
            penalty = -penalty
        if rubric_type == "rule-based":
            condition = item.get("condition")
            if not isinstance(condition, str) or not condition.strip():
                raise TypeError(f"metric[{index}].condition must be a non-empty string")
        else:
            criteria = item.get("criteria")
            score_range = item.get("score_range")
            if not isinstance(criteria, list) or not criteria or not all(
                isinstance(value, str) and value.strip() for value in criteria
            ):
                raise TypeError(f"metric[{index}].criteria must be a non-empty string array")
            if (
                not isinstance(score_range, list)
                or len(score_range) != 2
                or not all(isinstance(value, (int, float)) for value in score_range)
                or score_range[0] > score_range[1]
            ):
                raise TypeError(f"metric[{index}].score_range must be a numeric two-item array")
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
    COMPLETE_PROMPT = """# 背景
我现需要以 Agentic RL 的方式训练个人 AI 分身模型的长程任务完成能力，现已获得任务描述，请你听从指令。

# 指令
请你根据“任务描述”和“用户画像候选下位词”，生成一个可被 AI Agent 执行和评估的结构化任务环境。每条记录必须包含英文 type、field、description、value 字段，不要输出 visibility 字段。只有 user_profile 类型的信息允许在初始观测中暴露给 Agent；task_info、state、hidden_state、action、transition_rule、termination 均不得向 Agent 暴露。环境必须包含 user_profile、task_info、state、hidden_state、action、transition_rule、termination 七类信息，不泄露任何“背景”中的信息。

# 任务描述
{{task_desc}}

# 用户画像候选下位词
{{profile_terms}}

# 任务复杂度特征
{{complexity_features}}

# 输出格式
JSON 格式输出，每条记录只包含 type、field、description、value。例如：[{"type":"user_profile","field":"interest","description":"用户兴趣","value":"服装"},{"type":"task_info","field":"goal","description":"任务目标","value":"提交订单"},{"type":"state","field":"status","description":"当前订单状态","value":"pending"},{"type":"action","field":"submit","description":"提交订单","value":"submit_order"},{"type":"transition_rule","field":"submit_rule","description":"提交后的状态变化","value":{"when":"submit is called","effect":"status becomes submitted"}},{"type":"termination","field":"success","description":"成功条件","value":["status == submitted"]}]

user_profile 只能基于候选下位词生成兴趣、知识背景或偏好，不得从任务描述臆造姓名、年龄、住址等个人事实；没有足够候选信息时省略具体个人事实。state 是当前状态，hidden_state 是 Agent 初始不可见但环境内部使用的状态，action 是 Agent 可执行的动作名称；action 记录不要生成 params、参数 schema 或参数字段，动作参数由沙箱工具定义和 Agent 调用时校验。transition_rule 定义动作或时间如何改变状态，termination 定义可判定的 success/failure 条件。hidden、transition_rule 和 termination 记录不应直接暴露给 Agent。value 可以是数字、布尔值、字符串、数组或对象，不能使用空泛描述；未知信息直接省略，不得输出 null、空字符串、空数组或空对象。最多 30 条，确保 JSON 完整。

只输出合法 JSON，不要解释。"""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def generate(
        self,
        task_desc: str,
        complexity_features: dict[str, Any] | None = None,
        profile_terms: tuple[str, ...] = (),
    ) -> list[dict[str, Any]]:
        if not task_desc.strip():
            raise ValueError("task_desc must not be empty")
        prompt = _render_prompt(
            self.COMPLETE_PROMPT,
            task_desc=task_desc.strip(),
            complexity_features=complexity_features or {},
            profile_terms=profile_terms,
        )
        try:
            return self._parse_environment(self._complete(prompt, 5_000))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("任务环境响应无效，重试一次：%s", exc)
            try:
                return self._parse_environment(self._complete(prompt, 10_000))
            except (json.JSONDecodeError, TypeError, ValueError) as retry_error:
                logger.warning("任务环境重试失败，跳过当前任务：%s", retry_error)
                raise TaskGenerationError(
                    f"invalid executable task environment response: {retry_error}"
                ) from retry_error

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
            raise TypeError("environment response must be a JSON array")
        environment: list[dict[str, Any]] = []
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise TypeError(f"environment[{index}] must be an object")
            item_type = item.get("type")
            field = item.get("field")
            description = item.get("description")
            if item_type not in TaskEnvironmentGenerator.ENVIRONMENT_TYPES:
                raise ValueError(f"environment[{index}].type is invalid: {item_type!r}")
            if (
                not isinstance(field, str) or not field.strip()
                or not isinstance(description, str) or not description.strip()
                or "value" not in item
            ):
                raise TypeError(
                    f"environment[{index}] requires non-empty field/description and value"
                )
            value = item["value"]
            if value is None or value == "" or value == [] or value == {}:
                logger.warning(
                    "忽略空环境记录：index=%d，type=%s，field=%s",
                    index,
                    item_type,
                    field,
                )
                continue
            normalized_item = {
                **item,
                "type": item_type,
                "field": field.strip(),
                "description": description.strip(),
            }
            # Visibility is derived from the type by split_observation and is
            # intentionally not persisted in the task environment payload.
            normalized_item.pop("visibility", None)
            environment.append(normalized_item)
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

        observable = [item for item in environment if item["type"] == "user_profile"]
        hidden = [item for item in environment if item["type"] != "user_profile"]
        logger.debug(
            "任务环境拆分完成：可观测记录=%d，隐藏记录=%d",
            len(observable), len(hidden),
        )
        return observable, hidden

class TaskGenerator:
    """Sample event-element paths and turn their keywords into tasks."""

    INTENT_PROMPT = """你是任务意图规划助手。请根据路径关键词和任务类型，提炼一个真实用户任务的意图结构。
输出 JSON 对象，包含 goal、context、constraints、actions、expected_result、complexity_features 字段。
complexity_features 包含 action_count、constraint_count、requires_clarification、requires_multi_turn、requires_tool、estimated_tool_calls、requires_scheduling、time_horizon、uncertainty；这些字段只描述任务事实，不要直接判断复杂度等级。
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

# 复杂度特征
{{complexity_features}}

# 任务意图
{{intent}}

# 开头表达偏好
{{opening_preference}}

# 表达变化参数
{{variation_profile}}

# 生成要求
1. 任务必须有明确目标，不能只是罗列关键词。
2. 将关键词自然融入任务，不要机械复述。
3. 任务描述需要简洁明确。
4. 任务要像真实用户提出的请求，符合日常交流习惯。
5. 不要使用“请根据以下关键词”“基于上述关键词”等模板化开头。
6. 除非表达风格明确要求交代背景，否则不要以“我”“我想”“我需要”“最近我”等第一人称开头；优先使用直接请求、疑问句、场景描述或无主语祈使句。
7. 只保留对完成任务有帮助的信息，不要为了增加长度机械扩写关键词。
8. 不要输出标题、解释或列表，只返回 JSON。

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
        "日常对话",
    )
    REQUEST_FORMS = (
        "直接交代需求",
        "用疑问句提出请求",
        "先说明场景再提出目标",
        "描述遇到的问题并寻求处理方案",
        "提出一个带顺带事项的委托",
    )
    TONES = ("自然随和", "简洁明确", "礼貌客气", "比较着急但不过度夸张", "带一点犹豫，希望获得建议")
    STRUCTURES = ("一句话说明目标", "背景加目标", "目标加关键限制", "主任务加一项合理的后续要求")
    MOTIVATIONS = (
        "临时生活需求", "工作或学习中的实际委托", "遇到问题后的求助",
        "为后续安排提前做准备", "出于兴趣进行探索", "不强调动机，直接提出需求",
    )
    LENGTHS = ("短句", "中等长度", "中等长度", "较完整但避免冗长")
    def __init__(
        self,
        store: Neo4jGraphStore,
        llm: LLMClient,
        *,
        hierarchy_child_limit: int = 10,
        hierarchy_workers: int = 4,
    ) -> None:
        if hierarchy_child_limit <= 0 or hierarchy_workers <= 0:
            raise ValueError("hierarchy_child_limit and hierarchy_workers must be greater than zero")
        self.store = store
        self.llm = llm
        self.hierarchy_child_limit = hierarchy_child_limit
        self.hierarchy_workers = hierarchy_workers
        self.environment_generator = TaskEnvironmentGenerator(llm)
        self.metrics_generator = TaskMetricsGenerator(llm)

    def generate(
        self,
        hops: int = 3,
        task_type: TaskType | str | None = None,
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
        graph_features = self._graph_features(path, keywords, selected_hops)
        variation_profile = self._variation_profile()
        stage_started = time.perf_counter()
        request = {
            "keywords": keywords,
            "task_type": selected_type.value,
            "style": selected_style,
            "opening_preference": self._opening_preference(),
            "variation_profile": variation_profile,
            "graph_features": graph_features,
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
        complexity_features = self._merge_complexity_features(graph_features, intent)
        complexity = self._classify_complexity(complexity_features)
        request.update({
            "complexity": complexity,
            "complexity_features": complexity_features,
        })
        logger.info(
            "任务复杂度识别完成：等级=%s，得分=%d，特征=%s",
            complexity,
            self._complexity_score(complexity_features),
            complexity_features,
        )
        logger.info("任务意图生成完成：耗时=%.2fs", time.perf_counter() - stage_started)
        stage_started = time.perf_counter()
        description_prompt = json.dumps({**request, "intent": intent}, ensure_ascii=False)
        description_system_prompt = _render_prompt(
            self.SYSTEM_PROMPT,
            style=selected_style,
            opening_preference=request["opening_preference"],
            variation_profile=variation_profile,
            complexity=complexity,
            complexity_features=complexity_features,
            intent=intent,
        )
        task_desc = self._generate_description(
            description_prompt,
            description_system_prompt,
        )
        logger.info("任务描述生成完成：耗时=%.2fs", time.perf_counter() - stage_started)
        stage_started = time.perf_counter()
        profile_terms = self._profile_terms(keywords)
        logger.info(
            "用户画像候选词抽取完成：关键词数=%d，下位词数=%d，下位词=%s",
            len(keywords),
            len(profile_terms),
            profile_terms,
        )
        environment = self.environment_generator.generate(
            task_desc,
            complexity_features,
            profile_terms,
        )
        logger.info("任务环境生成完成：耗时=%.2fs", time.perf_counter() - stage_started)
        stage_started = time.perf_counter()
        metrics = self.metrics_generator.generate(
            task_desc,
            environment,
            complexity,
            complexity_features,
        )
        logger.info("观测指标生成完成：耗时=%.2fs", time.perf_counter() - stage_started)
        logger.info("任务生成完成：总耗时=%.2fs", time.perf_counter() - started)
        return Task(
            desc=task_desc,
            env=environment,
            metrics=metrics,
            task_type=selected_type,
            complexity=complexity,
            complexity_features=complexity_features,
        )

    @staticmethod
    def _opening_preference() -> str:
        """Prefer non-first-person openings while retaining limited variety."""

        if random.random() < 0.8:
            return (
                "本次优先避免第一人称开头。请使用直接请求、疑问句、场景描述或无主语表达，"
                "例如“帮我整理……”或“如何分析……”。"
            )
        return "允许自然使用少量第一人称，但不要形成固定模板或连续重复。"

    @classmethod
    def _variation_profile(cls) -> str:
        return (
            f"请求方式：{random.choice(cls.REQUEST_FORMS)}；"
            f"语气：{random.choice(cls.TONES)}；"
            f"组织结构：{random.choice(cls.STRUCTURES)}；"
            f"使用场景：{random.choice(cls.MOTIVATIONS)}；"
            f"长度：{random.choice(cls.LENGTHS)}。"
        )

    def _profile_terms(self, keywords: tuple[str, ...]) -> tuple[str, ...]:
        if not keywords:
            return ()
        get_children = getattr(self.store, "get_hierarchy_children", None)
        if not callable(get_children):
            logger.warning("图谱存储不支持下位词查询，用户画像候选词为空")
            return ()
        selected_count = random.randint(1, len(keywords))
        selected = random.sample(keywords, selected_count)
        with ThreadPoolExecutor(max_workers=min(self.hierarchy_workers, selected_count)) as executor:
            futures = {
                executor.submit(
                    get_children,
                    keyword,
                    limit=self.hierarchy_child_limit,
                ): keyword
                for keyword in selected
            }
            terms = []
            for future in futures:
                keyword = futures[future]
                try:
                    children = future.result()
                except Exception as exc:
                    logger.warning("下位词查询失败，跳过关键词：关键词=%s，原因=%s", keyword, exc)
                    continue
                terms.extend(child.name for child in children)
                for child in children:
                    terms.extend(child.words)
        return tuple(dict.fromkeys(term.strip() for term in terms if term.strip()))

    @staticmethod
    def _graph_features(
        path: tuple[SceneNode, ...],
        keywords: tuple[str, ...],
        hops: int,
    ) -> dict[str, object]:
        return {
            "hops": hops,
            "node_count": len(path),
            "keyword_count": len(keywords),
            "alias_count": sum(len(node.words) for node in path),
        }

    @staticmethod
    def _merge_complexity_features(
        graph_features: dict[str, object],
        intent: dict[str, object],
    ) -> dict[str, object]:
        intent_features = intent.get("complexity_features")
        if not isinstance(intent_features, dict):
            intent_features = {}
        actions = intent.get("actions")
        constraints = intent.get("constraints")
        features = {
            **graph_features,
            "action_count": len(actions) if isinstance(actions, list) else 0,
            "constraint_count": len(constraints) if isinstance(constraints, list) else 0,
            "requires_clarification": False,
            "requires_multi_turn": False,
            "requires_tool": False,
            "estimated_tool_calls": 0,
            "requires_scheduling": False,
            "time_horizon": "short",
            "uncertainty": "low",
        }
        for key in tuple(features):
            if key in graph_features:
                continue
            if key in intent_features:
                features[key] = intent_features[key]
        return features

    @staticmethod
    def _complexity_score(features: dict[str, object]) -> int:
        def number(name: str) -> float:
            value = features.get(name, 0)
            return float(value) if isinstance(value, (int, float)) else 0.0

        def flag(name: str) -> int:
            return int(features.get(name) is True)

        return round(
            number("hops") * 2
            + number("node_count")
            + number("keyword_count") * 0.5
            + number("action_count") * 2
            + number("constraint_count")
            + number("estimated_tool_calls") * 2
            + flag("requires_clarification") * 2
            + flag("requires_multi_turn") * 3
            + flag("requires_scheduling") * 2
        )

    @classmethod
    def _classify_complexity(cls, features: dict[str, object]) -> str:
        score = cls._complexity_score(features)
        if score <= 5:
            return "simple"
        if score <= 12:
            return "standard"
        return "complex"

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
            if not isinstance(task_type, str):
                raise ValueError
            values = tuple(value.strip() for value in task_type.split(","))
            if not values or any(not value for value in values):
                raise ValueError
            selected_types = tuple(dict.fromkeys(TaskType(value) for value in values))
            return random.choice(selected_types)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in TaskType)
            raise ValueError(f"unsupported task_type {task_type!r}; expected comma-separated values from: {allowed}") from exc

    @staticmethod
    def _keywords(path: tuple[SceneNode, ...]) -> tuple[str, ...]:
        keywords = []
        for scene in path:
            candidates = [word.strip() for word in (scene.name, *scene.words) if word.strip()]
            if candidates:
                keywords.append(random.choice(candidates))
        return tuple(dict.fromkeys(keywords))

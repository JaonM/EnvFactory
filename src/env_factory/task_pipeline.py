"""External-LLM task generation pipeline.

Each stage has an independent prompt, JSON envelope, validation and retry
boundary.  The final task is assembled from the validated stage artifacts;
the LLM never writes the final task contract in one call.
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .llm import LLMClient

logger = logging.getLogger(__name__)


class PipelineGenerationError(ValueError):
    """Raised when one pipeline stage cannot produce a valid artifact."""


def _json(content: str) -> Any:
    text = content.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.S | re.I)
    return json.loads(match.group(1) if match else text)


class TaskGenerationPipeline:
    """Generate a complete task through independently validated stages."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        script_count: int = 3,
        sessions_per_script: int = 2,
        minimum_dialogue_turns: int = 4,
        maximum_dialogue_turns: int = 12,
        retries: int = 3,
        noise_tool_max: int = 3,
    ) -> None:
        if (
            script_count <= 0
            or sessions_per_script < 2
            or minimum_dialogue_turns < 2
            or maximum_dialogue_turns < minimum_dialogue_turns
            or retries <= 0
            or noise_tool_max < 0
        ):
            raise ValueError("invalid pipeline counts or retries")
        self.llm = llm
        self.script_count = script_count
        self.sessions_per_script = sessions_per_script
        self.minimum_dialogue_turns = minimum_dialogue_turns
        self.maximum_dialogue_turns = maximum_dialogue_turns
        self.retries = retries
        self.noise_tool_max = noise_tool_max

    def _call(self, stage: str, system: str, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        logger.info("pipeline stage started: stage=%s", stage)
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                effective_system = system
                if stage == "agent_actions" or stage.startswith("agent_actions."):
                    effective_system += (
                        "\n本阶段的输入隔离规则优先级最高：只允许使用 payload 中的 task_description、"
                        "business_model 和 dialogue_sessions。business_model 只有实体、表、字段、关系和约束定义，"
                        "不得索取、猜测或使用业务数据行、数据文档、隐藏真值、内部记录、具体字段值或数据库 ID；"
                        "如果旧提示提到完整业务环境，以本条输入隔离规则和实际 payload 为准。"
                    )
                effective_payload = payload
                if last_error is not None:
                    effective_system += (
                        f"\n上一轮输出未通过本阶段结构校验，具体错误是：{last_error}。"
                        "请修复该错误，严格按照 output 示例返回完整 JSON 对象；不要返回 output/type 包装对象，"
                        "不要省略必需字段。"
                    )
                    if stage.startswith("environment_table_data."):
                        effective_system += (
                            "本阶段只允许返回一个顶层 rows 数组；不要返回 table、columns、schema、"
                            "markdown 或解释文字。每行必须是 JSON object，所有字符串中的双引号、反斜杠和换行"
                            "必须正确转义；保持字段简短，避免长篇文本。"
                        )
                    effective_payload = dict(payload)
                    effective_payload["previous_validation_error"] = str(last_error)
                response = self.llm.complete(
                    json.dumps(effective_payload, ensure_ascii=False, indent=2),
                    system_prompt=effective_system + "\n只输出合法 JSON 对象，不要解释。",
                    thinking=False,
                    temperature=0.2 if attempt > 1 else 0.5,
                    max_tokens=(
                        8_000 if stage.startswith("observations_rewards")
                        else 8_000 if stage.startswith("environment_table_data.")
                        else 16_000
                    ),
                    response_format="json_object",
                )
                if getattr(response, "finish_reason", None) in {"length", "max_tokens"}:
                    raise ValueError("LLM response was truncated by the token limit")
                value = _json(response.content)
                if not isinstance(value, dict):
                    raise TypeError("stage result must be a JSON object")
                # Some OpenAI-compatible providers wrap structured output in
                # a single `output` object. Normalize that envelope once so
                # every pipeline stage can consume the same contract.
                if set(value) == {"output"} and isinstance(value.get("output"), dict):
                    value = value["output"]
                if stage == "task_description":
                    task_value = value.get("task")
                    if not isinstance(task_value, str) or not task_value.strip():
                        raise ValueError(
                            "task_description response must contain a non-empty task string"
                        )
                    complexity = value.get("complexity")
                    if complexity not in {"simple", "standard", "complex"}:
                        raise ValueError(
                            "task_description.complexity must be simple, standard or complex"
                        )
                    expected_intent = payload.get("task_intent")
                    if expected_intent is not None and value.get("task_intent") != expected_intent:
                        raise ValueError(
                            f"task_description.task_intent must be {expected_intent!r}"
                        )
                if stage.startswith("environment_table_data."):
                    rows = value.get("rows")
                    if not isinstance(rows, list) or not rows:
                        raise ValueError(
                            f"{stage} response must contain a non-empty rows array"
                        )
                if stage.startswith("dialogue_sessions."):
                    content = value.get("message") or value.get("content")
                    if not isinstance(content, str) or not content.strip():
                        raise ValueError(
                            f"{stage} response must contain a non-empty message/content string"
                        )
                    if ".user." in stage and not isinstance(value.get("should_end"), bool):
                        raise ValueError(
                            f"{stage} response should_end must be boolean"
                        )
                logger.info(
                    "pipeline stage completed: stage=%s attempt=%d duration_ms=%.1f result_keys=%s",
                    stage,
                    attempt,
                    (time.perf_counter() - started) * 1000,
                    sorted(value.keys()),
                )
                return value
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "pipeline stage attempt failed: stage=%s attempt=%d/%d duration_ms=%.1f error=%s",
                    stage,
                    attempt,
                    self.retries,
                    (time.perf_counter() - started) * 1000,
                    exc,
                )
        logger.error(
            "pipeline stage failed: stage=%s attempts=%d duration_ms=%.1f error=%s",
            stage,
            self.retries,
            (time.perf_counter() - started) * 1000,
            last_error,
        )
        raise PipelineGenerationError(f"stage {stage} failed: {last_error}") from last_error

    def _simulate_dialogue_session(
        self,
        *,
        description: dict[str, Any],
        environment: dict[str, Any],
        profile: Any,
        script: dict[str, Any],
        script_id: str,
        session_id: str,
    ) -> dict[str, Any]:
        """Run one seeded-style conversation using separate User and Agent prompts."""
        started = time.perf_counter()
        logger.info("dialogue session started: session=%s script=%s", session_id, script_id)
        turns: list[dict[str, str]] = []
        termination_reason: str | None = None
        user_end_flags: list[bool] = []

        while len(turns) < self.maximum_dialogue_turns:
            user_result = self._call(
                f"dialogue_sessions.{session_id}.user.{len(turns) + 1}",
                "你是 User LLM，只能按照用户画像和用户剧本扮演用户。用户剧本是嵌套决策树：根据当前对话和用户画像，"
                "选择一个 condition 与当前情况匹配的分支，沿 next 进入下一个节点；每轮只选择一个分支，不要把多个分支"
                "拼接成一条不符合树结构的行为。围绕当前任务进行口语化提问、补充、澄清、接受、拒绝、纠正、犹豫、沉默"
                "或表达不确定。不要替 Agent 完成任务，不要查看或猜测业务数据中的隐藏真值。消费剧本分支中的 should_end 标志；只有剧本判断用户应停止提问且达到最少对话轮数时才将 should_end 设为 true，否则必须为 false。",
                {"task_description": description, "user_profile": profile, "user_script": script,
                 "conversation": turns, "minimum_turns": self.minimum_dialogue_turns,
                 "maximum_turns": self.maximum_dialogue_turns,
                 "output": {"message": "string", "should_end": False}},
            )
            # OpenAI-compatible chat providers commonly call the generated
            # text `content`; the pipeline contract uses `message`. Accept
            # both and normalize to the stored conversation format.
            message = user_result.get("message") or user_result.get("content")
            if not isinstance(message, str) or not message.strip():
                raise PipelineGenerationError(f"{session_id} user response must contain message/content")
            should_end = user_result.get("should_end")
            if not isinstance(should_end, bool):
                raise PipelineGenerationError(f"{session_id} user response should_end must be boolean")
            turns.append({"role": "user", "content": message.strip()})
            user_end_flags.append(should_end)
            if len(turns) >= self.minimum_dialogue_turns and should_end:
                termination_reason = "user_script_end"
                break
            if len(turns) >= self.maximum_dialogue_turns:
                termination_reason = "max_turns_reached"
                break

            agent_result = self._call(
                f"dialogue_sessions.{session_id}.agent.{len(turns) + 1}",
                "你是 Agent LLM。根据任务描述、公开业务环境数据和当前对话回答用户。不要读取或推断任何用户画像信息；只根据用户在当前对话中明确表达的内容、任务描述和公开业务环境数据作答。不要编造环境中不存在的数据，也不要把用户的话当作业务真值。回答应推进任务、提出必要澄清或给出基于环境数据的结果。不要输出对话终止信号，是否结束只由 User Simulator 根据用户剧本决定。",
                {"task_description": description, "environment": environment, "conversation": turns,
                 "minimum_turns": self.minimum_dialogue_turns,
                 "maximum_turns": self.maximum_dialogue_turns,
                 "output": {"message": "string"}},
            )
            message = agent_result.get("message") or agent_result.get("content")
            if not isinstance(message, str) or not message.strip():
                raise PipelineGenerationError(f"{session_id} agent response must contain message/content")
            turns.append({"role": "agent", "content": message.strip()})
            if len(turns) >= self.maximum_dialogue_turns:
                termination_reason = "max_turns_reached"
                break

        result = {
            "script_id": script_id,
            "session_id": session_id,
            "profile_id": profile.get("profile_id") if isinstance(profile, dict) else None,
            "turns": turns,
            "turn_count": len(turns),
            "user_end_flags": user_end_flags,
            "termination_reason": termination_reason or "max_turns_reached",
        }
        logger.info(
            "dialogue session completed: session=%s script=%s turns=%d termination=%s duration_ms=%.1f",
            session_id,
            script_id,
            len(turns),
            result["termination_reason"],
            (time.perf_counter() - started) * 1000,
        )
        return result

    @staticmethod
    def _list(value: Any, name: str, *, minimum: int = 1) -> list[Any]:
        if not isinstance(value, list) or len(value) < minimum:
            raise PipelineGenerationError(f"{name} must be a list with at least {minimum} item(s)")
        return value

    def generate(
        self,
        *,
        keywords: list[str],
        task_type: str,
        style: str,
        task_intent: str = "query",
        graph_context: dict[str, Any],
        artifact_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        pipeline_started = time.perf_counter()
        logger.info(
            "task pipeline started: task_type=%s style=%s keywords=%d scripts=%d sessions_per_script=%d",
            task_type,
            style,
            len(keywords),
            self.script_count,
            self.sessions_per_script,
        )
        intent_rules = {
            "query": "目标是查询、查找或整理已有信息，不生成策划方案",
            "explain": "目标是解释概念、原理或现象，不把任务改写为策划方案",
            "compare": "目标是比较两个或多个对象、方案或属性，必须体现比较维度",
            "recommend": "目标是基于条件筛选并推荐选项，必须体现选择依据",
            "diagnose": "目标是定位问题、分析原因或提出排查路径",
            "modify": "目标是修改、校正或更新已有业务数据或内容",
            "execute": "目标是执行明确操作并产生可验证结果",
            "plan": "目标是规划、安排或策划方案，允许生成执行计划",
            "summarize": "目标是汇总已有信息并形成简洁摘要，不扩展为策划任务",
            "create": "目标是从输入和业务数据创建新的内容、记录或交付物",
            "extract": "目标是从非结构化或多字段信息中提取指定事实或字段",
            "classify": "目标是按照明确标准对对象、记录或内容进行分类",
            "validate": "目标是验证内容、记录或结果是否满足明确规则",
            "audit": "目标是审查完整性、风险、合规性或异常，并输出问题清单",
            "calculate": "目标是基于业务数据执行计算并给出可复核结果",
            "estimate": "目标是基于已有信息进行估算、预测或区间判断，并说明依据",
            "schedule": "目标是根据时间、资源或约束安排具体日程，不生成泛化策划方案",
            "monitor": "目标是查看、跟踪或汇报对象的当前状态和变化",
            "troubleshoot": "目标是根据症状、日志或记录定位故障并提出处理步骤",
            "transform": "目标是将已有内容转换为指定格式、结构、语言或表达方式",
            "decide": "目标是基于约束和证据支持一个明确选择或决策",
            "simulate": "目标是基于业务数据模拟场景、过程或结果，并输出假设条件",
        }
        if task_intent not in intent_rules:
            raise PipelineGenerationError(f"unsupported task_intent: {task_intent}")
        # 1. Task description.
        description = self._call(
            "task_description",
            f"根据主题和关键词生成真实用户任务。任务意图已经固定为 {task_intent}：{intent_rules[task_intent]}。必须严格遵循该意图，不得用其他意图替换它。必须明确目标、上下文、约束、预期结果和复杂度事实。只有任务确实需要图片、音频、视频或文件输入时，requirements.input_modalities 才能包含对应媒体类型；否则只使用 text 或 structured_data。",
            {"keywords": keywords, "task_type": task_type, "style": style, "task_intent": task_intent,
             "graph_context": graph_context,
             "output": {"task": "string", "task_intent": task_intent, "goal": "string", "context": [], "expected_result": "string", "complexity": "simple|standard|complex", "requirements": {}}},
        )
        task_desc = description.get("task")
        if not isinstance(task_desc, str) or not task_desc.strip():
            raise PipelineGenerationError("task_description.task must be non-empty")
        returned_intent = description.get("task_intent")
        if returned_intent is not None and returned_intent != task_intent:
            raise PipelineGenerationError(
                f"task_description.task_intent must be {task_intent!r}, got {returned_intent!r}"
            )
        requirements = description.get("requirements")
        if not isinstance(requirements, dict):
            requirements = {"input_modalities": ["text"]}
        input_modalities = requirements.get("input_modalities", ["text"])
        media_required = isinstance(input_modalities, list) and any(
            item in {"image", "audio", "video", "file"} for item in input_modalities
        )

        environment_candidate = self._call(
            "environment_plan",
            "判断任务运行时真正需要的环境模式。stateless 表示只处理用户提供的文本或结构化输入，不需要预置业务数据或持久化；reference_data 表示需要只读业务资料；stateful 表示任务明确要求创建、修改、审批、排程、交易或持久化业务状态；external_capability 表示核心依赖搜索、天气、计算或其他外部能力。不要因为后续对话可能提出扩展请求而选择 stateful；只依据 task_description 中明确的任务目标和预期结果。",
            {
                "task_description": description,
                "task_intent": task_intent,
                "output": {"mode": "stateless|reference_data|stateful|external_capability", "requires_business_data": False, "requires_persistence": False, "reason": "string"},
            },
        )
        environment_plan = self._resolve_environment_plan(
            environment_candidate, task_description=description, task_intent=task_intent
        )

        materialized_dir = Path(artifact_dir) if artifact_dir is not None else Path("output/task_artifacts") / f"task-{uuid.uuid4().hex}"
        if environment_plan["mode"] in {"stateless", "external_capability"}:
            entity_plan = {"entities": []}
            table_definitions: list[dict[str, Any]] = []
            data_tables: list[dict[str, Any]] = []
            records: list[Any] = []
            data_document = "# 无业务数据环境\n\n该任务不预置业务实体、数据表或持久化业务状态。\n"
            data_manifest = self._materialize_business_data(
                [], data_document, materialized_dir, environment_mode=environment_plan["mode"]
            )
        else:
            # 2a. Analyze business entities. This stage does not generate state or perform web search.
            entity_plan = self._call(
            "environment_entities",
            "根据任务描述分析完成任务所需的最小必要业务实体。只保留完成任务、支持 Agent 查询或修改、以及奖励评测真正需要持久化的业务事实；不需要独立查询、复用或更新的静态说明、标签和建议作为其他实体的字段或 JSON 保存。只有存在独立生命周期、独立查询/更新需求或明确业务关系时才拆分实体。输出实体、用途、必须保存的业务事实和实体关系，实体必须足够覆盖完整任务但遵循最小必要原则。",
            {"task_description": description, "task_type": task_type, "keywords": keywords,
             "output": {"entities": [{"entity_id": "string", "name": "string", "description": "string", "required_facts": [], "relationships": []}]}},
        )
            self._validate_entities(entity_plan.get("entities"))

        # 2b. Design atomic table schemas, without rows yet.
            table_design = self._call(
            "environment_table_design",
            "根据任务描述和最小必要业务实体设计可持久化的原子数据库表。对需要持久化的关系数据遵循第三范式（3NF）：每个表表达一个清晰主题，字段依赖候选键、依赖整个键且不通过非键字段传递依赖；使用主键、外键和必要的关联表表达关系。优先使用最少数量的表完整覆盖任务；只有确有独立生命周期、独立访问需求或必要的一对多/多对多业务关系时才拆表，否则将信息作为字段、枚举、JSON 或文本保存。每张表说明存在必要性，并声明主键、外键、字段类型、可见性、索引和约束。输出表定义。",
            {"task_description": description, "entities": entity_plan["entities"],
             "output": {"tables": [{"table_name": "string", "description": "string", "columns": [{"name": "string", "type": "string", "description": "string", "nullable": False}], "primary_key": ["id"], "foreign_keys": [], "indexes": [], "constraints": []}]}},
        )
            table_definitions = table_design.get("tables")
            self._validate_table_definitions(table_definitions)

        # 2c. Generate each table's rows independently so tables can be built
        # concurrently and retried without regenerating unrelated data.
            def generate_table(table: dict[str, Any]) -> dict[str, Any]:
                result = self._call(
                f"environment_table_data.{table['table_name']}",
                "根据已确认的单张表结构生成完整但紧凑、真实、可直接插入数据库的业务数据。本阶段只返回顶层 {\"rows\": [...]}；rows 必须非空，每行覆盖全部字段，主键唯一。不要返回 table、columns、schema、markdown 或解释文字；字符串中的双引号、反斜杠和换行必须按 JSON 规则转义，文本字段保持简短。",
                {"task_description": description, "entities": entity_plan["entities"], "table": table,
                 "output": {"rows": []}},
            )
                if not isinstance(result.get("rows"), list):
                    logger.error(
                    "environment table data invalid: table=%s result_keys=%s expected=table.rows",
                    table["table_name"],
                    sorted(result),
                )
                    raise PipelineGenerationError(
                    f"environment_table_data.{table['table_name']} must return table.rows; "
                    f"received keys={sorted(result)}"
                )
                return table | {"rows": result["rows"]}

            generated_tables: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=min(8, len(table_definitions))) as executor:
                futures = [executor.submit(generate_table, table) for table in table_definitions]
                for future in as_completed(futures):
                    generated_tables.append(future.result())
            generated_tables.sort(key=lambda table: str(table.get("table_name", "")))
            data_tables = generated_tables
            self._validate_data_tables(data_tables)

        # 2d. Check cross-table consistency and return actual business records.
            consistency = self._call(
            "environment_data_consistency",
            "检查并修正完整业务表数据，校验主键唯一、外键存在、字段类型、必填字段、业务关系和任务覆盖度。返回修正后的 data_tables 与 records。",
            {"task_description": description, "entities": entity_plan["entities"], "tables": data_tables,
             "output": {"data_tables": data_tables, "records": []}},
        )
            data_tables = consistency.get("data_tables")
            self._validate_data_tables(data_tables)
            records = consistency.get("records", [])
            if not isinstance(records, list):
                raise PipelineGenerationError("environment_data_consistency.records must be a list")

        # 2e. Write the persistence handoff document from final tables.
            document_prompt = "根据最终业务表和完整 rows 编写给 Code Agent 的 data_document。说明每张表的用途、字段、类型、可见性、主键、外键、索引、约束、初始化顺序、关系和持久化要求。返回非空、完整、可执行的 Markdown 文档，至少包含每张表的初始化说明和字段说明。文档只解释最终数据。"
            document_payload = {"task_description": description, "entities": entity_plan["entities"], "data_tables": data_tables,
                            "output": {"data_document": "# 业务数据说明\n"}}
            data_document: str | None = None
            document_error: PipelineGenerationError | None = None
            for document_attempt in range(1, self.retries + 1):
                repair_hint = ""
                if document_error is not None:
                    repair_hint = f"\n上一版文档未通过校验，必须修复以下错误后重新输出完整文档：{document_error}"
                document = self._call(
                "environment_data_document" if document_attempt == 1 else "environment_data_document.repair",
                document_prompt + repair_hint,
                document_payload,
            )
                candidate_document = document.get("data_document")
                if isinstance(candidate_document, str) and candidate_document.strip():
                    data_document = candidate_document
                    break
                document_error = PipelineGenerationError("environment_data_document requires a non-empty data_document")
                logger.warning(
                "environment data document validation failed: attempt=%d/%d error=%s",
                document_attempt,
                self.retries,
                document_error,
            )
            if data_document is None:
                raise document_error or PipelineGenerationError("environment_data_document requires a non-empty data_document")
            data_manifest = self._materialize_business_data(
                data_tables,
                data_document,
                materialized_dir,
            )

        media_generation = {"required": False, "language": "python", "code": "", "dependencies": [], "entrypoint": "", "output_dir": ""}
        if media_required:
            media_result = self._call(
                "environment_media_generation",
                "根据任务描述和最终业务数据编写用于生成任务相关媒体数据的 Python 程序。生成与任务直接相关的媒体文件和必要元数据，返回依赖、入口和输出目录。媒体识别和媒体评测不属于本阶段。",
                {"task_description": description, "data_tables": data_tables, "data_document": data_document,
                 "output": {"media_generation": media_generation | {"required": True}}},
            )
            media_generation = media_result.get("media_generation")
        if not media_required:
            media_generation = {"required": False, "language": "python", "code": "", "dependencies": [], "entrypoint": "", "output_dir": ""}
        elif (
            not isinstance(media_generation, dict)
            or media_generation.get("required") is not True
            or media_generation.get("language") != "python"
            or not isinstance(media_generation.get("code"), str)
            or not media_generation["code"].strip()
            or not isinstance(media_generation.get("dependencies", []), list)
            or not isinstance(media_generation.get("entrypoint"), str)
            or not isinstance(media_generation.get("output_dir"), str)
        ):
            raise PipelineGenerationError("media task requires executable Python media_generation code, dependencies, entrypoint and output_dir")
        environment = {"records": records, "data_tables": data_tables, "data_document": data_document,
                       "media_generation": media_generation}
        # Do not embed the generated rows (or the consistency-stage record
        # summary) in task.json. Downstream stages already received the full
        # in-memory data; the persisted task only needs the manifest.
        environment_summary = {"records": [], "data_manifest": data_manifest,
                               "media_generation": media_generation}

        # 3. User profiles and task-specific scripts are deliberately generated
        # by separate prompts. Profiles are free-form persona variation and do
        # not need to match the task; scripts are task-specific behavior plans
        # grounded in the generated business environment.
        profiles_result = self._call(
            "user_profiles",
            "自由设计多样、真实且彼此有明显差异的用户画像。画像独立于当前任务和业务环境，不要把任务关键词、业务实体或隐藏真值写入画像。每个画像使用结构化字段：profile_id、identity_summary、age_range、occupation_or_life_stage、location_context、education_background、domain_knowledge、goals_and_motivations、communication_style、language_habits、decision_style、risk_tolerance、patience_level、trust_level、information_disclosure_style、questioning_style、feedback_style、budget_or_resource_sensitivity、time_sensitivity、accessibility_needs、frustration_triggers、misconceptions_or_biases、known_facts、unknown_facts、behavior_tendencies。数组字段至少提供 2 项，枚举或等级字段要给出清晰值和简短解释；画像应能指导 User LLM 在对话中表现出不同的措辞、节奏、追问、犹豫、接受和拒绝行为。只输出画像对象。",
            {"count": self.script_count, "output": {"user_profiles": [{
                "profile_id": "profile-1",
                "identity_summary": "...",
                "age_range": "...",
                "occupation_or_life_stage": "...",
                "location_context": "...",
                "education_background": "...",
                "domain_knowledge": {"level": "...", "areas": [], "evidence": "..."},
                "goals_and_motivations": [],
                "communication_style": {"tone": "...", "verbosity": "...", "directness": "..."},
                "language_habits": [],
                "decision_style": {"pattern": "...", "needs": []},
                "risk_tolerance": "...",
                "patience_level": "...",
                "trust_level": "...",
                "information_disclosure_style": "...",
                "questioning_style": "...",
                "feedback_style": "...",
                "budget_or_resource_sensitivity": "...",
                "time_sensitivity": "...",
                "accessibility_needs": [],
                "frustration_triggers": [],
                "misconceptions_or_biases": [],
                "known_facts": [],
                "unknown_facts": [],
                "behavior_tendencies": []
            }]}},
        )
        profiles = self._list(profiles_result.get("user_profiles"), "user_profiles", minimum=self.script_count)
        normalized_profiles: list[dict[str, Any]] = []
        seen_profile_ids: set[str] = set()
        for index, raw_profile in enumerate(profiles[:self.script_count], start=1):
            if not isinstance(raw_profile, dict):
                raise PipelineGenerationError(f"user_profiles[{index - 1}] must be an object")
            profile = dict(raw_profile)
            profile_id = profile.get("profile_id")
            profile_id = profile_id.strip() if isinstance(profile_id, str) else ""
            if not profile_id or profile_id in seen_profile_ids:
                profile_id = f"profile-{index}"
                while profile_id in seen_profile_ids:
                    profile_id = f"profile-{index + len(seen_profile_ids)}"
            profile["profile_id"] = profile_id
            seen_profile_ids.add(profile_id)
            normalized_profiles.append(profile)
        profiles = normalized_profiles
        script_prompt = (
            "根据当前任务描述和完整业务环境数据设计多样的用户行为剧本。每个剧本必须是可执行的嵌套决策树，"
            "根节点和每个后续节点都包含 node_id、user_behavior、branches；每个 branches 元素包含 branch_id、"
            "condition、next，next 是下一个同结构节点。根据当前对话中 Agent 的回答选择满足 condition 的分支，"
            "不要把多个分支合并成一条线性行为。尽可能生成丰富但互不重复的分支，通常每个剧本生成 4-8 个有意义的"
            "分支和 2-4 层路径，覆盖信息补充、追问、接受、拒绝、纠正、犹豫、沉默、转移话题和结束等情况；"
                "分支必须围绕任务描述、实际业务实体、字段、公开记录和 Agent 可执行目标。每个分支必须增加布尔字段 should_end，"
                "表示该分支下用户是否停止提问；只有真正结束的分支为 true，其他分支为 false，并且每个剧本至少包含一个 true 分支。"
                "可引用公开业务事实，但不得泄露隐藏真值、评测规则或实现细节。每个剧本提供非空 script_id 字符串。"
        )
        script_payload = {
            "task_description": description,
            "environment": environment,
            "count": self.script_count,
            "output": {"user_scripts": [{
                "script_id": "script-1",
                "goal": "...",
                "tree": {"node_id": "root", "user_behavior": "...", "branches": [{
                    "branch_id": "branch-1",
                    "condition": "...",
                    "should_end": False,
                    "next": {"node_id": "node-1", "user_behavior": "...", "branches": []},
                }, {
                    "branch_id": "branch-2",
                    "condition": "用户表示问题已解决",
                    "should_end": True,
                    "next": {"node_id": "node-2", "user_behavior": "结束对话", "branches": []},
                }]},
            }]},
        }
        user_scripts: list[dict[str, Any]] | None = None
        script_error: PipelineGenerationError | None = None
        for script_attempt in range(1, self.retries + 1):
            repair_hint = ""
            if script_error is not None:
                repair_hint = (
                    f"\n上一版用户剧本未通过结构校验，必须修复以下错误并重新输出全部剧本：{script_error}。"
                    "不要只返回修复片段。"
                )
            scripts_result = self._call(
                "user_scripts" if script_attempt == 1 else "user_scripts.repair",
                script_prompt + repair_hint,
                script_payload,
            )
            try:
                raw_scripts = self._list(
                    scripts_result.get("user_scripts"),
                    "user_scripts",
                    minimum=self.script_count,
                )
                candidate_scripts: list[dict[str, Any]] = []
                seen_script_ids: set[str] = set()
                for index, raw_script in enumerate(raw_scripts[:self.script_count], start=1):
                    if not isinstance(raw_script, dict):
                        raise PipelineGenerationError(f"user_scripts[{index - 1}] must be an object")
                    normalized_script = dict(raw_script)
                    raw_id = normalized_script.get("script_id") or normalized_script.get("id")
                    script_id = raw_id.strip() if isinstance(raw_id, str) else ""
                    if not script_id or script_id in seen_script_ids:
                        original_id = script_id or "<missing>"
                        candidate_index = index
                        script_id = f"script-{candidate_index}"
                        while script_id in seen_script_ids:
                            candidate_index += 1
                            script_id = f"script-{candidate_index}"
                        logger.warning(
                            "normalized user script id: index=%d original=%s normalized=%s",
                            index - 1,
                            original_id,
                            script_id,
                        )
                    normalized_script["script_id"] = script_id
                    self._validate_user_script_tree(normalized_script, index - 1)
                    seen_script_ids.add(script_id)
                    candidate_scripts.append(normalized_script)
                user_scripts = candidate_scripts
                break
            except PipelineGenerationError as exc:
                script_error = exc
                logger.warning(
                    "user script validation failed: attempt=%d/%d error=%s",
                    script_attempt,
                    self.retries,
                    exc,
                )
        if user_scripts is None:
            raise script_error or PipelineGenerationError("user_scripts generation failed")

        # 4. Generate sessions by two independent role prompts: the User LLM
        # follows the script, while the Agent LLM answers from task data.
        script_ids = {str(item["script_id"]) for item in user_scripts}
        sessions: list[dict[str, Any]] = []
        for script in user_scripts[:self.script_count]:
            script_id = str(script.get("script_id") or script.get("id"))
            for session_index in range(self.sessions_per_script):
                # Each session combines one task-specific script with a
                # randomly selected persona, so the same script is exercised
                # with different communication styles and decision patterns.
                profile = random.choice(profiles)
                sessions.append(self._simulate_dialogue_session(
                    description=description,
                    environment=environment,
                    profile=profile,
                    script=script,
                    script_id=script_id,
                    session_id=f"{script_id}-session-{session_index + 1}",
                ))
        user_simulation_manifest = self._materialize_user_simulation(
            profiles, user_scripts, sessions, materialized_dir / "user_simulation"
        )

        session_counts: dict[str, int] = {}
        seen_session_ids: set[str] = set()
        for session in sessions:
            if not isinstance(session, dict) or not self.minimum_dialogue_turns <= len(session.get("turns", [])) <= self.maximum_dialogue_turns:
                raise PipelineGenerationError("every dialogue session must stay within the configured turn bounds")
            script_id = session.get("script_id")
            session_id = session.get("session_id")
            if script_id not in script_ids or not isinstance(session_id, str) or not session_id.strip():
                raise PipelineGenerationError("every dialogue session must reference a valid script_id and session_id")
            if session_id in seen_session_ids:
                raise PipelineGenerationError("dialogue session_id values must be unique")
            seen_session_ids.add(session_id)
            turns = session.get("turns")
            if not isinstance(turns, list) or any(
                not isinstance(turn, dict) or turn.get("role") not in {"agent", "user"}
                or not isinstance(turn.get("content"), str) or not turn["content"].strip()
                for turn in turns
            ) or any(turns[index].get("role") == turns[index + 1].get("role") for index in range(len(turns) - 1)):
                raise PipelineGenerationError("dialogue turns must be non-empty and strictly alternate agent/user")
            user_turn_count = sum(turn.get("role") == "user" for turn in turns)
            if not isinstance(session.get("user_end_flags"), list) or len(session["user_end_flags"]) != user_turn_count or any(
                not isinstance(flag, bool) for flag in session["user_end_flags"]
            ):
                raise PipelineGenerationError("dialogue session user_end_flags must match user turns")
            if session.get("termination_reason") not in {
                "user_script_end", "max_turns_reached",
            }:
                raise PipelineGenerationError("dialogue session requires a valid termination_reason")
            session_counts[script_id] = session_counts.get(script_id, 0) + 1
        if set(session_counts) != script_ids or any(
            session_counts.get(script_id, 0) < self.sessions_per_script for script_id in script_ids
        ):
            raise PipelineGenerationError("every user script must have the required number of sessions")

        # 5. Decompose Agent actions to irreducible Agent-visible operations.
        # Simulated dialogues reveal realistic interaction boundaries; they do
        # not define transitions or rewards.
        business_model = {
            "entities": entity_plan["entities"],
            "tables": [
                {
                    "table_name": table["table_name"],
                    "description": table["description"],
                    "columns": table["columns"],
                    "primary_key": table.get("primary_key", []),
                    "foreign_keys": table.get("foreign_keys", []),
                    "indexes": table.get("indexes", []),
                    "constraints": table.get("constraints", []),
                }
                for table in table_definitions
            ],
        }
        actions = self._call(
            "agent_actions",
            "根据任务描述、环境计划、业务模型和模拟对话反推 Agent 必须做的原子动作。模拟对话只能提供任务范围内的行为证据，不得把用户临时提出但 task_description 未要求的保存、写库、查询确认或其他扩展请求提升为正式动作。environment_plan.mode=stateless 时禁止生成任何读取或修改数据库、持久化状态或调用外部系统的动作。凡是仅根据用户输入和通用语言能力进行识别、比较、提取、格式化和最终表达的步骤都属于 Agent 自身推理或回答。每个输出动作必须不可再拆，并提供 atomicity_rationale、inputs、outputs、preconditions、effects。",
            {"task_description": description, "business_model": business_model,
             "environment_plan": environment_plan,
             "dialogue_sessions": sessions,
             "output": {"actions": [{"name": "string", "description": "string", "atomicity_rationale": "string", "inputs": [{"name": "string", "description": "string"}], "outputs": [{"name": "string", "description": "string"}], "preconditions": ["string"], "effects": ["string"]}]}},
        )
        action_list = self._list(actions.get("actions"), "agent_actions")
        self._validate_actions(action_list)

        # Classify semantic actions before tool generation.  Environment
        # operations may become tools; reasoning and response composition
        # remain the policy's responsibility and must not be outsourced.
        capability_payload = {
            "task_description": description,
            "business_model": business_model,
            "agent_actions": action_list,
            "output": {"capabilities": [{
                "action_name": "string",
                "kind": "environment_operation|agent_reasoning|agent_response",
                "requires_tool": True,
                "reason": "string",
            }]},
        }
        capability_plan: list[Any] | None = None
        capability_error: PipelineGenerationError | None = None
        for capability_attempt in range(1, self.retries + 1):
            repair_hint = ""
            if capability_error is not None:
                repair_hint = (
                    f"\n上一版能力规划未通过校验：{capability_error}。"
                    "必须逐一覆盖所有输入动作且不得增加动作；kind 与 requires_tool 必须一致。"
                )
            capability_result = self._call(
                "capability_plan" if capability_attempt == 1 else "capability_plan.repair",
                "将每个原子 Agent 动作分类为 environment_operation、agent_reasoning 或 agent_response。只有必须读取或修改沙箱私有业务状态、调用外部系统或使用确定性专用能力的动作才是 environment_operation 且 requires_tool=true。比较、分析、筛选、选择、解释、总结和生成最终自然语言回答通常属于 agent_reasoning 或 agent_response，requires_tool=false。必须逐一覆盖输入动作，action_name 必须原样引用，不得新增、删除或改名。" + repair_hint,
                capability_payload,
            )
            try:
                candidate_capabilities = self._list(capability_result.get("capabilities"), "capabilities")
                self._validate_capability_plan(candidate_capabilities, action_list)
                capability_plan = candidate_capabilities
                break
            except PipelineGenerationError as exc:
                capability_error = exc
                logger.warning("capability plan validation failed: attempt=%d/%d error=%s", capability_attempt, self.retries, exc)
        if capability_plan is None:
            raise capability_error or PipelineGenerationError("capability plan generation failed")
        if environment_plan["mode"] == "stateless":
            scoped_names = {
                item["action_name"] for item in capability_plan if not item["requires_tool"]
            }
            action_list = [item for item in action_list if item["name"] in scoped_names]
            capability_plan = [
                item for item in capability_plan if item["action_name"] in scoped_names
            ]
        tool_action_names = {
            item["action_name"] for item in capability_plan if item["requires_tool"]
        }
        tool_action_list = [item for item in action_list if item["name"] in tool_action_names]
        tool_actions = {"actions": tool_action_list}

        # 6. Generate strict OpenAI Function Tools from the atomic actions.
        noise_count = 0 if self.noise_tool_max == 0 else random.randint(min(2, self.noise_tool_max), self.noise_tool_max)
        noise_categories = (["related_irrelevant", "unrelated"] + [
            random.choice(("unrelated", "related_irrelevant")) for _ in range(max(0, noise_count - 2))
        ])[:noise_count]
        noise_tool_example = [
            {
                "type": "function",
                "function": {
                    "name": f"semantic_function_name_{index + 1}",
                    "description": "噪声工具示例描述",
                    "parameters": {
                        "type": "object",
                        "description": "噪声工具参数对象",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                },
            }
            for index in range(noise_count)
        ]
        noise_metadata_example = [
            {
                "name": f"semantic_function_name_{index + 1}",
                "category": category,
                "rationale": "说明该工具为什么属于指定噪声类别",
            }
            for index, category in enumerate(noise_categories)
        ]
        tool_prompt = "根据任务描述、模拟对话和已经确认可工具化的环境操作，生成标准 OpenAI Function Tool 定义。此阶段不得读取或参考完整业务环境数据、业务数据行、数据文档、隐藏真值、数据库记录或其原始字段值；工具定义只能从任务描述、对话中出现的公开信息和环境操作的业务语义推导。不得为比较、分析、选择、解释、总结或最终回答增加工具。工具设计要依据动作 inputs/outputs，确定自然的工具名、描述、参数名、类型、枚举值、必填性和参数说明；不要把业务数据中的具体记录、内部主键、数据库 ID、隐藏真值或用户确认 ID 写入工具定义。用户交互不生成 ask_user 工具，用户回复通过独立的 UserSimulator HTTP 接口获得。无 Agent 输入时使用 properties={}、required=[]。如果没有环境操作，tools 必须是空数组。输出标准工具 schema，并为顶层参数、嵌套对象属性和 array items schema 提供 description；数组对象同时定义 items.properties 和 items.required。每个业务工具在 tool_bindings 中只使用 tool_name 和 action_name；不要输出 trainer_action。严格按照 payload.noise_spec 的数量和顺序生成噪声工具。噪声工具必须是完整的标准 OpenAI Function Tool：unrelated 与当前任务完全无关；related_irrelevant 与任务主题相关但不影响任务目标完成。噪声工具不得绑定原子动作、不得修改任务关键业务数据、不得被奖励指标视为任务进展。每个工具必须根据其真实能力使用具体、可理解的 snake_case 名称；payload.output 中 semantic_function_name_N 只是结构占位符，严禁原样返回，也不得使用 noise_tool_N、tool_N、function_N 等占位名称。"
        tool_payload = {"task_description": description,
                        "dialogue_sessions": sessions,
                        "agent_actions": tool_actions,
                        "noise_spec": {"count": noise_count, "categories": noise_categories,
                                       "category_definitions": {
                                           "unrelated": "与当前任务完全无关",
                                           "related_irrelevant": "与当前任务相关但与任务目标完成无关",
                                       }},
                        "output": {"tools": [], "noise_tools": noise_tool_example,
                                   "noise_tool_metadata": noise_metadata_example,
                                   "tool_bindings": [{"tool_name": "string", "action_name": "string"}]}}
        tool_list: list[dict[str, Any]] | None = None
        tool_bindings: list[dict[str, Any]] = []
        noise_tool_metadata: list[dict[str, Any]] = []
        tool_error: PipelineGenerationError | None = None
        for tool_attempt in range(1, self.retries + 1):
            repair_hint = ""
            if tool_error is not None:
                repair_hint = (
                    f"\n上一版工具定义未通过校验，必须修复以下错误后重新输出完整结果：{tool_error}。"
                    "必须同时返回 tools、noise_tools、noise_tool_metadata 和 tool_bindings。"
                    "noise_tool_metadata 必须是与 noise_tools 数量相同的数组；每项必须包含对应工具 name、"
                    "与 noise_spec.categories 同位置一致的 category，以及非空 rationale。"
                )
            tools = self._call(
                "openai_tools" if tool_attempt == 1 else "openai_tools.repair",
                tool_prompt + repair_hint,
                tool_payload,
            )
            try:
                candidate_tools = self._list(tools.get("tools"), "tools", minimum=0)
                self._fill_tool_schema_descriptions(candidate_tools)
                self._validate_tools(candidate_tools)
                if any(item["function"]["name"] == "ask_user" for item in candidate_tools):
                    raise PipelineGenerationError("ask_user 不再是 LLM Tool，用户交互必须使用 UserSimulator 接口")
                candidate_noise_tools = self._list(tools.get("noise_tools"), "noise_tools", minimum=noise_count)
                if len(candidate_noise_tools) != noise_count:
                    raise PipelineGenerationError(
                        f"noise_tools must contain exactly {noise_count} item(s)"
                    )
                self._fill_tool_schema_descriptions(candidate_noise_tools)
                self._validate_tools(candidate_noise_tools)
                if any(item["function"]["name"] == "ask_user" for item in candidate_noise_tools):
                    raise PipelineGenerationError("noise_tools 不能定义 ask_user")
                primary_names = {item["function"]["name"] for item in candidate_tools}
                noise_names = {item["function"]["name"] for item in candidate_noise_tools}
                if primary_names & noise_names:
                    raise PipelineGenerationError("noise tool names must not duplicate task tool names")
                candidate_noise_metadata = tools.get("noise_tool_metadata", [])
                if not isinstance(candidate_noise_metadata, list) or len(candidate_noise_metadata) != noise_count:
                    raise PipelineGenerationError("noise_tool_metadata must match noise_tools")
                for index, metadata in enumerate(candidate_noise_metadata):
                    if not isinstance(metadata, dict) or metadata.get("name") not in noise_names:
                        raise PipelineGenerationError("noise metadata references an unknown tool")
                    if metadata.get("category") != noise_categories[index]:
                        raise PipelineGenerationError("noise metadata category does not match noise_spec")
                    if not isinstance(metadata.get("rationale"), str) or not metadata["rationale"].strip():
                        raise PipelineGenerationError("noise metadata requires rationale")
                if candidate_noise_tools:
                    noise_audit = self._call(
                        "noise_tool_audit",
                        "逐一审查候选噪声工具是否可能实质帮助完成任务。只要工具能提供原因分析证据、关键事实、比较依据、验证手段、排查路径所需资料或最终答案内容，就不是安全噪声，is_safe_noise=false。主题相关但确实不能推进任何任务目标时才可为 true。必须逐一覆盖输入工具并原样返回 name。",
                        {
                            "task_description": description,
                            "agent_actions": action_list,
                            "candidate_noise_tools": candidate_noise_tools,
                            "candidate_metadata": candidate_noise_metadata,
                            "output": {"decisions": [{
                                "name": "string", "is_safe_noise": True, "reason": "string"
                            }]},
                        },
                    )
                    unsafe_noise_names = self._validate_noise_tool_audit(
                        noise_audit.get("decisions"), noise_names=noise_names
                    )
                    if unsafe_noise_names:
                        logger.warning(
                            "discarding materially useful tools from noise set: %s",
                            sorted(unsafe_noise_names),
                        )
                        candidate_noise_tools = [
                            item for item in candidate_noise_tools
                            if item["function"]["name"] not in unsafe_noise_names
                        ]
                        candidate_noise_metadata = [
                            item for item in candidate_noise_metadata
                            if item["name"] not in unsafe_noise_names
                        ]
                        noise_names -= unsafe_noise_names
                    if not candidate_noise_tools:
                        fallback_tool, fallback_metadata = self._fallback_noise_tool(
                            occupied_names=primary_names
                        )
                        candidate_noise_tools = [fallback_tool]
                        candidate_noise_metadata = [fallback_metadata]
                        noise_names = {fallback_metadata["name"]}
                candidate_bindings = tools.get("tool_bindings", [])
                if not isinstance(candidate_bindings, list):
                    raise PipelineGenerationError("tool_bindings must be a list")
                normalized_bindings = []
                for binding in candidate_bindings:
                    if not isinstance(binding, dict):
                        raise PipelineGenerationError("each tool binding must be an object")
                    tool_name = binding.get("tool_name")
                    # Accept artifacts produced by the former contract, but
                    # persist only the single canonical action_name field.
                    action_name = binding.get("action_name") or binding.get("trainer_action")
                    if not all(isinstance(value, str) and value.strip() for value in (tool_name, action_name)):
                        raise PipelineGenerationError("each tool binding needs tool_name and action_name")
                    if tool_name not in primary_names:
                        raise PipelineGenerationError("tool binding references an unknown tool")
                    if action_name not in tool_action_names:
                        raise PipelineGenerationError("tool binding references an action that is not tool-eligible")
                    normalized_bindings.append({"tool_name": tool_name, "action_name": action_name})
                if {item["tool_name"] for item in normalized_bindings} != primary_names:
                    raise PipelineGenerationError("every business tool must have exactly one tool binding")
                if len({item["tool_name"] for item in normalized_bindings}) != len(normalized_bindings):
                    raise PipelineGenerationError("business tool bindings must not be duplicated")
                tool_list = candidate_tools + candidate_noise_tools
                tool_bindings = normalized_bindings
                noise_tool_metadata = candidate_noise_metadata
                break
            except PipelineGenerationError as exc:
                tool_error = exc
                logger.warning("openai tool schema validation failed: attempt=%d/%d error=%s", tool_attempt, self.retries, exc)
        if tool_list is None:
            raise tool_error or PipelineGenerationError("openai tool generation failed")
        tools_manifest = self._materialize_tools(tool_list, materialized_dir)

        implementation_result = self._call(
            "tool_implementation_specs",
            "为可以直接由数据表操作实现的业务工具生成声明式实现。operation 支持 select、aggregate_count、insert、update、delete；复杂计算、跨表业务决策、文档生成和噪声工具不要生成 spec。filters.argument 以及 selector/values/changes 的键必须来自工具 parameters，对应值必须是目标表字段。operator 只能是 eq、in、contains、gte、lte。只返回 specs 数组。",
            {
                "business_model": business_model,
                "tools": candidate_tools,
                "output": {"specs": [{
                    "tool_name": "string", "operation": "select", "table": "string",
                    "filters": [{"argument": "string", "column": "string", "operator": "eq"}],
                    "projection": ["string"], "order_by": ["string"], "result_field": "records",
                    "selector": {"tool_argument": "table_column"},
                    "values": {"tool_argument": "table_column"},
                    "changes": {"tool_argument": "table_column"},
                }]},
            },
        )
        tool_implementations = implementation_result.get("specs", [])
        if not isinstance(tool_implementations, list):
            tool_implementations = []
        try:
            self._validate_tool_implementations(
                tool_implementations, tools=candidate_tools, tables=business_model["tables"]
            )
        except PipelineGenerationError as exc:
            # Declarative compilation is an optimization.  Invalid optional
            # Specs must not discard an otherwise valid task; the sandbox
            # builder can still implement those business handlers explicitly.
            logger.warning("tool implementation specs ignored: %s", exc)
            tool_implementations = []

        # 7a. Identify the small set of goal-critical steps before creating
        # process metrics. Ordinary tools are not process-reward candidates.
        if not tool_action_list:
            key_steps: list[Any] = []
            logger.info("reward key-step generation skipped: no tool-eligible business actions")
        else:
            key_step_payload = {
                "task_description": description, "environment": environment,
                "agent_actions": tool_actions, "tools": candidate_tools,
                "output": {"key_steps": [{"step_id": "step-1", "action_name": "string", "rationale": "string", "required_for_goal": True, "dependencies": []}]},
            }
            key_steps = []
            key_step_error: PipelineGenerationError | None = None
            for key_step_attempt in range(1, self.retries + 1):
                repair_hint = ""
                if key_step_error is not None:
                    repair_hint = (
                        f"\n上一版关键步骤未通过校验：{key_step_error}。"
                        "action_name 只能原样引用 payload.agent_actions.actions 中的动作，不得引用噪声工具或新增动作。"
                    )
                key_steps_result = self._call(
                    "reward_key_steps" if key_step_attempt == 1 else "reward_key_steps.repair",
                    "先从任务目标和可工具化环境动作中梳理真正必需的关键步骤。只保留显著推进目标、改变关键业务数据或形成必要中间结果的动作；读取背景信息、噪声工具、可选探索和重复调用通常不是关键步骤。只输出关键步骤定义，不生成 metric 或 reward。" + repair_hint,
                    key_step_payload,
                )
                try:
                    candidate_key_steps = self._list(
                        key_steps_result.get("key_steps"), "key_steps", minimum=0
                    )
                    self._validate_key_steps(candidate_key_steps, tool_action_list)
                    key_steps = candidate_key_steps
                    break
                except PipelineGenerationError as exc:
                    key_step_error = exc
                    logger.warning(
                        "reward key-step validation failed: attempt=%d/%d error=%s",
                        key_step_attempt, self.retries, exc,
                    )
            else:
                raise key_step_error or PipelineGenerationError("reward key-step generation failed")

        # 7b. Design executable observations and rewards from key steps.
        reward_prompt = "根据任务、业务数据模型、原子动作和工具定义生成紧凑、可执行的 observation 与 reward 设计。只保留与任务目标完成强相关的关键过程指标和目标结果指标，不为普通动作机械创建指标；如果任务无需关键工具动作或可直接生成答案，process 指标可以为空；如果存在关键工具动作，每个关键动作或关键动作链都可以有对应过程指标，不限制过程指标数量。观测指标不得依赖任务生成阶段的 user_profiles、user_scripts 或 dialogue_sessions 等模拟产物；evaluation_inputs 只能引用沙箱运行时实际产生的 conversation、public_observation、available_tools、tool_call、tool_results、business_data、final_agent_response、terminal_observation 等输入。每个 metric 必须同时生成机器可执行的 evaluator 对象，不能只有自然语言 condition/criteria：evaluator.kind 必须声明评估器类型，evaluator.source 必须声明 runtime_rule 或 external_llm，evaluator.score_mapping 必须声明如何把评估结果映射到 score_range。关键过程指标必须是 hybrid，并使用 evaluator.kind=hybrid_tool_call、evaluator.source=external_llm、evaluator.comparison=exact_tool_name_and_canonical_arguments，以及 target_action、evaluation_inputs、criteria 和固定 condition=llm_expected_tool_call_exact_match。沙箱 Code Agent 根据这个指标在实现评估器时调用外部 LLM，结合当前运行时 Context、可用工具和 criteria 生成期望的工具名和参数真值；然后由规则引擎对 Agent 实际 tool_call 的工具名和规范化参数进行确定性精确比对。任务 JSON 不要嵌入 LLM prompt、output_schema 或嵌套 expected-call 配置；不要用 LLM 直接给最终过程分数，也不要把工具选择错误或参数错误设计成 penalty。结果指标只判断任务目标是否完成或关键业务数据是否达到目标，优先使用可量化的业务数据变化；rule-based 必须声明 evaluator.kind=business_state_rule、document_rule 或 trajectory_rule，明确 source_fields、assertion 和 score_mapping；model-based 必须声明 evaluator.kind=external_llm_judge、source=external_llm；hybrid 结果指标必须同时声明 evaluator.kind=hybrid_outcome、rule 和 external_llm 字段。惩罚指标只有在直接影响任务目标时才保留，用于偏离用户诉求、无效循环或业务数据偏离预期，不评价工具选择或参数错误。每个 metric 包含 id、category、type、scope、rubric、weight、score_range 和 evaluator；rule-based 或 hybrid 提供 condition，model-based 或 hybrid 提供 evaluation_inputs 和 criteria。process/outcome 分数范围为 [0,1]，penalty 分数范围为 [-1,0]；所有 process 与 outcome 指标的权重合计为 1，所有 penalty 指标的权重合计为 1，且 outcome 权重合计大于 process 权重合计。reward_formula 必须把所有 process/outcome 项放在同一个正反馈加权和中，把 penalty 项放在独立的负反馈加权和中，不得分别归一化 process 和 outcome；标准公式为 R = clip(sum(w_i*score_i, category in {process,outcome}) + sum(w_j*score_j, category == penalty), -1, 1)，该公式在分数范围和权重归一化成立时落在 [-1,1]。category 取 process、outcome、penalty。所有 rubric、criteria、assertion 和 description 必须是简短单句，禁止输出长篇解释或回显输入数据。"
        reward_prompt += " payload.key_steps 是上一阶段确认的关键步骤。process 指标只能评价这些 key_steps 中的 action_name；不得为非关键读取、噪声工具、可选探索或每个工具机械创建过程奖励。先使用 key_steps 判断是否确实需要过程奖励，再生成最少且必要的 process metrics。"
        reward_prompt += " 优先使用可执行的 runtime_rule 结果指标，并将多个关键动作合并为最少必要的过程指标。"
        if environment_plan["mode"] == "stateless":
            reward_prompt += " 当前任务是 stateless：不得生成 process 指标，不得以 business_data 或数据库变化作为成功条件。结果指标必须根据运行时 conversation 与 final_agent_response 评价任务完成度、忠实性和格式；噪声工具调用只能作为轨迹惩罚。"
        reward_environment = {
            "data_manifest": data_manifest,
            "tables": business_model["tables"] if "business_model" in locals() else [
                {
                    "table_name": table["table_name"], "description": table.get("description", ""),
                    "columns": table.get("columns", []), "primary_key": table.get("primary_key", []),
                    "foreign_keys": table.get("foreign_keys", []), "constraints": table.get("constraints", []),
                }
                for table in table_definitions
            ],
        }
        reward_payload = {"task_description": description, "environment": reward_environment,
                          "environment_plan": environment_plan,
                          "agent_actions": {"actions": action_list},
                          "tools": tool_list, "key_steps": key_steps, "output": {"observation_schema": {}, "metrics": [{
                              "id": "process_key_tool_action", "category": "process", "type": "hybrid", "scope": "step",
                              "target_action": "key_tool_action",
                              "evaluation_inputs": ["recent_conversation", "public_observation", "tool_call", "tool_arguments"],
                              "criteria": ["当前上下文是否需要执行该关键动作"],
                              "condition": "llm_expected_tool_call_exact_match",
                              "evaluator": {"kind": "hybrid_tool_call", "source": "external_llm", "comparison": "exact_tool_name_and_canonical_arguments", "score_mapping": {"match": 1, "mismatch": 0}},
                              "score_range": [0, 1],
                              "weight": 0.2, "rubric": "过程动作选择正确"}],
                              "reward_formula": {"type": "separate_sign_weighted_sum", "formula": "R = clip(sum(w_i * score_i for category in {process,outcome}) + sum(w_j * score_j for category == penalty), -1, 1)", "positive_weight_sum": 1, "negative_weight_sum": 1, "score_range": [-1, 1]},
                              }}
        rewards: dict[str, Any] | None = None
        reward_error: PipelineGenerationError | None = None
        for reward_attempt in range(1, self.retries + 1):
            repair_hint = ""
            if reward_error is not None:
                repair_hint = (
                    f"\n上一版 observation/reward 设计未通过校验，必须修复以下错误后重新输出完整结果：{reward_error}。"
                    "不要回显输入 payload，也不要返回 task_description、environment、agent_actions、tools 或 output 字段。"
                    "顶层只能返回 observation_schema、metrics、reward_formula；"
                    "metric.type 只能是精确字符串 rule-based、model-based 或 hybrid；"
                    "evaluator.kind 为 business_state_rule、document_rule 或 trajectory_rule 时 type=rule-based；"
                    "evaluator.kind 为 external_llm_judge 时 type=model-based；"
                    "evaluator.kind 为 hybrid_tool_call 或 hybrid_outcome 时 type=hybrid；"
                    "metrics 必须是非空数组，scope 只能是 step、state、terminal、trajectory，"
                    "category 只能是 process、outcome、penalty；process 指标只能对应 payload.key_steps 中的 action_name，且 type 必须为 hybrid；"
                    "每个关键 process 指标必须包含非空 target_action、evaluation_inputs、criteria，且 condition 必须严格为 llm_expected_tool_call_exact_match；"
                    "每个 metric 必须包含 evaluator 对象、kind、source 和 score_mapping；process evaluator 必须是 hybrid_tool_call/external_llm/exact_tool_name_and_canonical_arguments；"
                    "rule-based 结果或惩罚 evaluator 必须声明 business_state_rule、document_rule 或 trajectory_rule 及可执行 assertion；model-based 必须使用 external_llm_judge；"
                    "每个 rule-based 或 hybrid 指标必须补充非空 condition；每个 model-based 或 hybrid 指标必须补充非空 evaluation_inputs 和 criteria 数组，"
                    "不能返回 criteria 缺失、空数组或 null。"
                )
            candidate_rewards = self._call(
                "observations_rewards" if reward_attempt == 1 else "observations_rewards.repair",
                reward_prompt + repair_hint,
                reward_payload,
            )
            try:
                candidate_metrics = self._list(candidate_rewards.get("metrics"), "metrics")
                candidate_metrics = self._normalize_metric_types(candidate_metrics)
                candidate_metrics = self._normalize_metric_evaluators(candidate_metrics)
                candidate_metrics = self._normalize_metric_weights(candidate_metrics)
                candidate_rewards["metrics"] = candidate_metrics
                self._validate_metrics(candidate_metrics, key_steps=key_steps)
                self._validate_metrics_for_environment(
                    candidate_metrics, environment_mode=environment_plan["mode"]
                )
                # Build the formula from the validated metric weights instead
                # of trusting an independently generated expression. Positive
                # process+outcome weights sum to 1; penalty weights sum to 1.
                # Therefore the two signed contributions are already bounded
                # by [0, 1] and [-1, 0], and their sum is bounded by [-1, 1].
                candidate_rewards["reward_formula"] = self._canonical_reward_formula(candidate_metrics)
                self._validate_reward_formula(candidate_rewards["reward_formula"], candidate_metrics)
                if not isinstance(candidate_rewards.get("observation_schema", {}), dict):
                    raise PipelineGenerationError("observation_schema must be an object")
                rewards = candidate_rewards
                break
            except PipelineGenerationError as exc:
                reward_error = exc
                logger.warning("observation/reward validation failed: attempt=%d/%d error=%s", reward_attempt, self.retries, exc)
        if rewards is None:
            raise reward_error or PipelineGenerationError("observations_rewards generation failed")
        if noise_tool_metadata:
            self._ensure_deterministic_noise_penalty(
                rewards["metrics"],
                noise_names=[str(item["name"]) for item in noise_tool_metadata],
            )
            rewards["metrics"] = self._normalize_metric_weights(rewards["metrics"])
            self._validate_metrics(rewards["metrics"], key_steps=key_steps)
            self._validate_metrics_for_environment(
                rewards["metrics"], environment_mode=environment_plan["mode"]
            )
            rewards["reward_formula"] = self._canonical_reward_formula(rewards["metrics"])

        # final_agent_response is stored by the shared runtime as a raw string.  A
        # document_rule such as $.tea_varieties therefore invents structure
        # that does not exist at runtime.  Preserve the semantic criterion by
        # evaluating it with the external judge instead of compiling a bogus
        # JSON path.
        self._promote_unstructured_document_rules(rewards["metrics"])
        rewards["reward_formula"] = self._canonical_reward_formula(rewards["metrics"])

        metric_impl_result = self._call(
            "metric_implementation_specs",
            "为能够从运行时结构化上下文确定性判断的 rule-based 指标生成声明式实现。source 只能是 business_state、trajectory、final_agent_response、observation；path 使用简单 $.field.nested 路径；operator 只能是 eq、ne、gte、lte、contains、exists、count_gte、count_eq。expected 必须是可序列化常量。score_mapping 使用 pass/fail，并且分数必须位于对应 metric.score_range。model-based、hybrid 或无法可靠结构化判断的指标不要生成 spec。只返回 specs 数组。",
            {
                "metrics": rewards["metrics"], "business_model": business_model,
                "output": {"specs": [{
                    "metric_id": "string", "source": "business_state", "path": "$.field",
                    "operator": "eq", "expected": True,
                    "score_mapping": {"pass": 1, "fail": 0},
                }]},
            },
        )
        metric_implementations = metric_impl_result.get("specs", [])
        if not isinstance(metric_implementations, list):
            metric_implementations = []
        if noise_tool_metadata:
            metric_implementations = [
                item for item in metric_implementations
                if isinstance(item, dict) and item.get("metric_id") != "penalty_noise_tool_usage"
            ]
            metric_implementations.append({
                "metric_id": "penalty_noise_tool_usage",
                "source": "trajectory",
                "path": "$.events",
                "operator": "none_tool_calls",
                "expected": [str(item["name"]) for item in noise_tool_metadata],
                "score_mapping": {"pass": 0, "fail": -1},
            })
        try:
            self._validate_metric_implementations(metric_implementations, rewards["metrics"])
            if environment_plan["mode"] == "stateless" and any(
                isinstance(item, dict) and item.get("source") == "business_state"
                for item in metric_implementations
            ):
                raise PipelineGenerationError("stateless metric implementations cannot use business_state")
        except PipelineGenerationError as exc:
            logger.warning("metric implementation specs incomplete; promoting uncovered rules: %s", exc)
            metric_implementations = []
            # The noise predicate is EnvFactory-owned and remains executable
            # even when an unrelated LLM-generated metric spec is malformed.
            if noise_tool_metadata:
                metric_implementations.append({
                    "metric_id": "penalty_noise_tool_usage",
                    "source": "trajectory",
                    "path": "$.events",
                    "operator": "none_tool_calls",
                    "expected": [str(item["name"]) for item in noise_tool_metadata],
                    "score_mapping": {"pass": 0, "fail": -1},
                })
        self._promote_unimplemented_rule_metrics(
            rewards["metrics"], metric_implementations,
            environment_mode=environment_plan["mode"],
        )
        self._validate_metrics(rewards["metrics"], key_steps=key_steps)
        self._validate_metrics_for_environment(
            rewards["metrics"], environment_mode=environment_plan["mode"]
        )
        rewards["reward_formula"] = self._canonical_reward_formula(rewards["metrics"])

        acceptance_seed = self._build_acceptance_contract(
            task_description=description,
            data_manifest=data_manifest,
            data_tables=data_tables,
            actions=action_list,
            tools=tool_list,
            key_steps=key_steps,
            metrics=rewards["metrics"],
            reward_formula=rewards["reward_formula"],
        )
        acceptance_payload = {
            "task_description": description,
            "data_manifest": data_manifest,
            "business_tables": data_tables,
            "actions": action_list,
            "tools": tool_list,
            "key_steps": key_steps,
            "metrics": rewards["metrics"],
            "reward_formula": rewards["reward_formula"],
            "baseline_contract": acceptance_seed,
            "output": {"acceptance_contract": acceptance_seed},
        }
        acceptance_contract: dict[str, Any] | None = None
        acceptance_error: PipelineGenerationError | None = None
        for acceptance_attempt in range(1, self.retries + 1):
            repair_hint = ""
            if acceptance_error is not None:
                repair_hint = (
                    f"\n上一版验收契约未通过校验，必须修复以下错误并返回完整 acceptance_contract：{acceptance_error}。"
                    "每个 scenarios 项必须是 object，并且同时包含非空 scenario_id、非空 kind、steps 数组；"
                    "goal_critical 场景还必须引用已有 action_name。每个 tool_cases 项必须包含 case_id、已有 tool_name、"
                    "expected object；每个 reward_cases 项必须包含 case_id、已有 metric_id 和数值 expected_score。"
                    "不要返回缺字段、空字符串或 null。"
                )
            candidate = self._call(
                "acceptance_contract" if acceptance_attempt == 1 else "acceptance_contract.repair",
                "根据任务、完整业务数据、原子动作、工具和奖励指标生成 EnvFactory 独立业务验收契约。该契约供外层黑盒验收使用，不由 Code Agent 修改。必须覆盖真实业务成功路径、失败路径、前置条件、业务数据前后变化、隐藏真值不泄露、工具不返回 reward/observation、reset/episode 隔离、replay、幂等、反事实奖励和 mutation testing。不要只生成 HTTP smoke；每个关键步骤要给出可执行的场景和断言。不要生成新的业务语义，所有表、字段、工具、动作和指标必须来自输入。只输出 acceptance_contract 对象。" + repair_hint,
                acceptance_payload,
            )
            candidate_contract = candidate.get("acceptance_contract", candidate)
            candidate_contract = self._merge_acceptance_contract(candidate_contract, acceptance_seed)
            try:
                self._validate_acceptance_contract(candidate_contract, actions=action_list, tools=tool_list, metrics=rewards["metrics"])
                acceptance_contract = candidate_contract
                break
            except PipelineGenerationError as exc:
                acceptance_error = exc
                logger.warning("acceptance contract validation failed: attempt=%d/%d error=%s", acceptance_attempt, self.retries, exc)
        if acceptance_contract is None:
            # The baseline is generated deterministically from already
            # validated task artifacts and is itself an EnvFactory-owned
            # contract.  It is safer to preserve a valid acceptance contract
            # than to discard the whole task because an LLM omitted a
            # structural field after all repair attempts.
            self._validate_acceptance_contract(
                acceptance_seed, actions=action_list, tools=tool_list, metrics=rewards["metrics"]
            )
            logger.warning(
                "acceptance contract LLM output remained invalid after %d attempts; using validated baseline",
                self.retries,
            )
            acceptance_contract = acceptance_seed

        executable_payload = {
                "task_description": description,
                "business_tables": data_tables,
                "tools": tool_list,
                "noise_tools": noise_tool_metadata,
                "key_steps": key_steps,
                "metrics": rewards["metrics"],
                "output": {"scenarios": []},
        }
        business_scenarios: list[Any] | None = None
        executable_error: PipelineGenerationError | None = None
        success_fixture = self._select_success_response_fixture(
            task_description=description, sessions=sessions
        )
        try:
            fixture_result = self._call(
                "acceptance_success_fixture",
                "根据任务描述和结果指标生成一份真正完成任务的最终回答，用作成功验收 fixture。回答必须直接解决具体任务，包含指标要求的实质内容，不能只复述输出格式、提纲或评分标准。只返回 content 字段。",
                {
                    "task_description": description,
                    "metrics": [
                        {key: metric.get(key) for key in ("id", "rubric", "criteria")}
                        for metric in rewards["metrics"] if metric.get("category") == "outcome"
                    ],
                    "output": {"content": "完整成功回答"},
                },
            )
            candidate_fixture = fixture_result.get("content")
            if isinstance(candidate_fixture, str) and len(candidate_fixture.strip()) >= 100:
                success_fixture = candidate_fixture.strip()
        except PipelineGenerationError as exc:
            logger.warning("success response fixture generation failed; using dialogue-derived fallback: %s", exc)
        executable_baseline = self._build_business_scenario_baseline(
            task_description=description,
            tools=tool_list,
            noise_tools=noise_tool_metadata,
            success_content=success_fixture,
        )
        for executable_attempt in range(1, self.retries + 1):
            repair_hint = ""
            if executable_error is not None:
                repair_hint = (
                    f"\n上一版轨迹未通过校验：{executable_error}。必须返回完整 scenarios，"
                    "每个场景都要有非空 assertions；必须同时包含 goal_success、goal_failure，"
                    "存在噪声工具时还必须包含 noise_selection。"
                )
            executable_result = self._call(
                "acceptance_executable_scenarios" if executable_attempt == 1 else "acceptance_executable_scenarios.repair",
                "生成机器可执行的业务验收轨迹，不能输出自然语言步骤。至少包含 goal_success 和 goal_failure；存在噪声工具时增加 noise_selection。使用 operation=reset、tool_call、agent_response、observation、reward、replay、business_snapshot、mutate_business_state；goal_success 使用 agent_response.content 提交最终回答后再取 reward。每步可用 step_id、capture，后续参数可用 {$ref: 变量名}。assertions 使用 source=step:<step_id> 或 variables、JSON path 和有限 operator。成功轨迹只能使用业务工具推进目标，失败或噪声轨迹必须验证不会获得成功奖励。所有工具、参数、表、字段和值必须来自输入，不得使用 Python、SQL 或表达式字符串。" + repair_hint,
                executable_payload,
            )
            candidate_scenarios = self._normalize_executable_scenarios(
                executable_result.get("scenarios", [])
            )
            if not candidate_scenarios:
                candidate_scenarios = executable_baseline
            try:
                self._validate_executable_scenarios(
                    candidate_scenarios, tools=tool_list, noise_tools=noise_tool_metadata
                )
                business_scenarios = candidate_scenarios
                break
            except PipelineGenerationError as exc:
                executable_error = exc
                logger.warning("business executable scenarios invalid: attempt=%d/%d error=%s", executable_attempt, self.retries, exc)
        if business_scenarios is None:
            self._validate_executable_scenarios(
                executable_baseline, tools=tool_list, noise_tools=noise_tool_metadata
            )
            logger.warning(
                "business executable scenario proposals remained invalid; using deterministic baseline: %s",
                executable_error,
            )
            business_scenarios = executable_baseline
        acceptance_contract["executable_scenarios"] = [
            *acceptance_contract.get("executable_scenarios", []), *business_scenarios
        ]

        requirements = description.get("requirements")
        if not isinstance(requirements, dict):
            requirements = {"input_modalities": ["text"]}
        requirements = dict(requirements)
        requirements.setdefault("media_truth_mode", "programmatic")
        # The HTTP contract is derived after tools and rewards are validated.
        # Keeping it deterministic prevents a second LLM-generated interface
        # description from drifting away from the executable tool schema.
        requirements["runtime_interface"] = self._build_runtime_interface(
            tool_list, rewards.get("reward_formula", {})
        )
        self._validate_runtime_interface(
            requirements["runtime_interface"], tool_list, rewards.get("reward_formula", {})
        )
        declared_complexity = description.get("complexity")
        if declared_complexity not in {"simple", "standard", "complex"}:
            raise PipelineGenerationError("task_description.complexity must be simple, standard or complex")
        complexity = self._derive_complexity(
            business_tool_count=len(tool_bindings),
            metric_count=len(rewards["metrics"]),
            key_step_count=len(key_steps),
        )
        training_profile = self._derive_training_profile(
            business_tool_count=len(tool_bindings),
            noise_tool_count=len(noise_tool_metadata),
            key_step_count=len(key_steps),
        )
        readiness_warnings = []
        max_turn_sessions = sum(
            session.get("termination_reason") == "max_turns_reached" for session in sessions
        )
        if max_turn_sessions:
            readiness_warnings.append(
                f"{max_turn_sessions}/{len(sessions)} dialogue sessions reached the turn limit"
            )
        if complexity != declared_complexity:
            readiness_warnings.append(
                f"complexity normalized from {declared_complexity} to {complexity}"
            )
        self._validate_task_readiness(
            capability_plan=capability_plan,
            tool_bindings=tool_bindings,
            noise_tools=noise_tool_metadata,
            metrics=rewards["metrics"],
            metric_implementations=metric_implementations,
            business_scenarios=business_scenarios,
            require_noise=self.noise_tool_max > 0,
        )
        result = {
            "task": task_desc.strip(), "task_type": task_type, "task_intent": task_intent,
            "complexity": complexity,
            "requirements": requirements,
            "environment_plan": environment_plan,
            "environment": self._environment_records(environment_summary, action_list),
            "media_generation": media_generation,
            "data_manifest": data_manifest,
            "user_simulation_manifest": user_simulation_manifest,
            "tools_manifest": tools_manifest,
            "reward_key_steps": key_steps,
            "capability_plan": capability_plan,
            "actions": action_list, "tools": tool_list,
            "tool_bindings": tool_bindings,
            "tool_implementations": tool_implementations,
            "noise_tools": noise_tool_metadata,
            "observation_schema": rewards.get("observation_schema", {}), "metrics": rewards["metrics"],
            "metric_implementations": metric_implementations,
            "reward_formula": rewards.get("reward_formula", {}),
            "acceptance_contract": acceptance_contract,
            "task_readiness": {
                "ready": True,
                "training_profile": training_profile,
                "errors": [],
                "warnings": readiness_warnings,
            },
            "generation_pipeline": {"version": "1.0", "stages": [
                "task_description", "environment_plan", "environment_entities", "environment_table_design",
                "environment_table_data", "environment_data_consistency", "environment_data_document",
                "environment_media_generation", "user_profiles", "user_scripts", "dialogue_sessions",
                "agent_actions", "capability_plan", "openai_tools", "tool_implementation_specs", "reward_key_steps",
                "observations_rewards", "metric_implementation_specs", "acceptance_contract",
                "acceptance_executable_scenarios",
            ]},
        }
        logger.info(
            "task pipeline completed: task_type=%s complexity=%s actions=%d tools=%d metrics=%d duration_ms=%.1f artifact_dir=%s",
            task_type,
            complexity,
            len(action_list),
            len(tool_list),
            len(rewards["metrics"]),
            (time.perf_counter() - pipeline_started) * 1000,
            materialized_dir,
        )
        return result

    @staticmethod
    def _schema_fixture(schema: dict[str, Any]) -> Any:
        """Create a type-valid, data-independent tool request template."""
        if "default" in schema:
            return schema["default"]
        enum = schema.get("enum")
        if isinstance(enum, list) and enum:
            return enum[0]
        kind = schema.get("type")
        if kind == "object":
            properties = schema.get("properties", {})
            return {name: TaskGenerationPipeline._schema_fixture(value) for name, value in properties.items() if name in schema.get("required", [])}
        if kind == "array":
            return []
        if kind == "integer" or kind == "number":
            return 1
        if kind == "boolean":
            return True
        return "fixture-value"

    @classmethod
    def _build_acceptance_contract(
        cls,
        *,
        task_description: dict[str, Any],
        data_manifest: dict[str, Any],
        data_tables: list[dict[str, Any]],
        actions: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        key_steps: list[dict[str, Any]],
        metrics: list[dict[str, Any]],
        reward_formula: dict[str, Any],
    ) -> dict[str, Any]:
        tool_cases: list[dict[str, Any]] = []
        for tool in tools:
            function = tool["function"]
            parameters = function["parameters"]
            required = list(parameters.get("required", []))
            tool_cases.append({
                "case_id": f"{function['name']}.valid_shape",
                "tool_name": function["name"],
                "kind": "schema_and_business_smoke",
                "arguments_template": cls._schema_fixture(parameters),
                "expected": {"status_class": [200], "must_not_return": ["observation", "reward"]},
            })
            if required:
                tool_cases.append({
                    "case_id": f"{function['name']}.missing_required",
                    "tool_name": function["name"], "kind": "invalid_input",
                    "arguments": {}, "expected": {"status_class": [400, 422]},
                })
            if parameters.get("additionalProperties") is False:
                tool_cases.append({
                    "case_id": f"{function['name']}.unexpected_property",
                    "tool_name": function["name"], "kind": "invalid_input",
                    "arguments": {"__outer_invalid_property__": True}, "expected": {"status_class": [400, 422]},
                })
        table_names = [table.get("table_name") for table in data_tables]
        critical_fields = {
            table.get("table_name"): [column.get("name") for column in table.get("columns", [])]
            for table in data_tables
        }
        reward_cases = []
        for metric in metrics:
            mapping = metric.get("evaluator", {}).get("score_mapping", {})
            values = [value for value in mapping.values() if isinstance(value, (int, float)) and not isinstance(value, bool)]
            if not values:
                continue
            reward_cases.append({
                "case_id": f"{metric['id']}.pass",
                "metric_id": metric["id"],
                "expected_score": max(values),
                "fixture": "success",
            })
            reward_cases.append({
                "case_id": f"{metric['id']}.fail",
                "metric_id": metric["id"],
                "expected_score": min(values),
                "fixture": "failure",
            })
        argument_probes = [
            {"probe_id": f"{case['case_id']}.argument_sensitivity", "tool_name": case["tool_name"], "arguments": case["arguments_template"]}
            for case in tool_cases
            if case.get("kind") == "schema_and_business_smoke"
            and isinstance(case.get("arguments_template"), dict) and case["arguments_template"]
        ]
        executable_scenarios = [{
            "scenario_id": "runtime_reset_replay",
            "steps": [
                {"operation": "reset", "body": {"episode_id": "acceptance-episode", "seed": 17}, "expected_status": 200},
                {"operation": "observation", "expected_status": 200},
                {"operation": "replay", "expected_status": 200},
            ],
            "assertions": [{"path": "$.episode_id", "operator": "eq", "expected": "acceptance-episode"}],
        }, *[
            {
                "scenario_id": f"invalid_{case['case_id']}",
                "steps": [{"step_id": "invalid_call", "operation": "tool_call", "tool_name": case["tool_name"], "arguments": case.get("arguments", {}), "expected_status": case["expected"]["status_class"][0]}],
                "assertions": [{"source": "step:invalid_call", "path": "$.error", "operator": "exists", "expected": True}],
            }
            for case in tool_cases if case.get("kind") == "invalid_input"
        ]]
        return {
            "version": "1.0",
            "authority": "env_factory_outer_workflow",
            "task_goal": task_description.get("goal") or task_description.get("task"),
            "fixtures": {
                "data_manifest": data_manifest,
                "tables": table_names,
                "critical_fields": critical_fields,
                "initial_data_hash": __import__("hashlib").sha256(json.dumps(data_tables, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
            },
            "scenarios": [
                {"scenario_id": "reset_isolation", "kind": "lifecycle", "steps": ["reset(seed=17)", "write_episode_a", "reset(seed=17)", "assert_episode_a_not_visible"]},
                {"scenario_id": "replay_integrity", "kind": "replay", "steps": ["reset(seed=17)", "execute_tool_cases", "get_replay", "assert_trace_hash"]},
                *[{"scenario_id": f"key_step_{step['step_id']}", "kind": "goal_critical", "action_name": step["action_name"], "required_for_goal": step["required_for_goal"]} for step in key_steps],
            ],
            "tool_cases": tool_cases,
            "argument_probes": argument_probes,
            "executable_scenarios": executable_scenarios,
            "invariants": [
                "tool response never contains observation or reward",
                "failed tool call does not partially mutate business data",
                "hidden truth is absent from public observation",
                "reward is computed only by explicit reward endpoint",
            ],
            "mutations": [
                {"mutation_id": f"mutate_{table}", "table": table, "strategy": "change_one_critical_field", "fields": critical_fields.get(table, [])}
                for table in table_names
            ],
            "reward_cases": reward_cases,
            "reward_formula": reward_formula,
            "mutation_tests": [
                "replace_tool_result_with_constant",
                "skip_business_write",
                "return_constant_reward",
                "ignore_tool_arguments",
                "bypass_trainer_auth",
            ],
            "actions": [{"name": action["name"], "preconditions": action.get("preconditions", []), "effects": action.get("effects", [])} for action in actions],
        }

    @staticmethod
    def _merge_acceptance_contract(candidate: Any, baseline: dict[str, Any]) -> dict[str, Any]:
        """Merge an LLM proposal onto the deterministic valid contract.

        The baseline supplies structural completeness.  Candidate fields are
        retained, so the LLM may add richer business assertions without being
        allowed to remove required scenario/tool/reward cases accidentally.
        """
        if not isinstance(candidate, dict):
            return dict(baseline)
        merged = dict(baseline)
        for key, value in candidate.items():
            if key not in {"scenarios", "tool_cases", "reward_cases", "argument_probes", "executable_scenarios"}:
                if isinstance(merged.get(key), dict) and isinstance(value, dict):
                    merged[key] = {**merged[key], **value}
                else:
                    merged[key] = value

        identity = {
            "scenarios": "scenario_id",
            "tool_cases": "case_id",
            "reward_cases": "case_id",
        }

        def safe_overlay(base_item: Any, candidate_item: Any) -> dict[str, Any]:
            if not isinstance(base_item, dict):
                return dict(candidate_item) if isinstance(candidate_item, dict) else {}
            if not isinstance(candidate_item, dict):
                return dict(base_item)
            result = dict(base_item)
            for field, value in candidate_item.items():
                if value is None or value == "":
                    continue
                if field == "steps" and not isinstance(value, list):
                    continue
                if field == "expected" and not isinstance(value, dict):
                    continue
                result[field] = value
            return result

        for key, id_field in identity.items():
            base_items = baseline.get(key, [])
            candidate_items = candidate.get(key, [])
            if not isinstance(base_items, list):
                base_items = []
            if not isinstance(candidate_items, list):
                candidate_items = []
            by_id = {
                item.get(id_field): item
                for item in candidate_items
                if isinstance(item, dict) and isinstance(item.get(id_field), str) and item.get(id_field)
            }
            result: list[Any] = []
            used: set[int] = set()
            for index, base_item in enumerate(base_items):
                overlay = by_id.get(base_item.get(id_field)) if isinstance(base_item, dict) else None
                if overlay is not None:
                    for candidate_index, candidate_item in enumerate(candidate_items):
                        if candidate_item is overlay:
                            used.add(candidate_index)
                            break
                if overlay is None and index < len(candidate_items) and isinstance(candidate_items[index], dict):
                    overlay = candidate_items[index]
                    used.add(index)
                if isinstance(base_item, dict):
                    result.append(safe_overlay(base_item, overlay))
                else:
                    result.append(overlay or base_item)
            for index, item in enumerate(candidate_items):
                if index not in used and isinstance(item, dict) and item not in result:
                    result.append(item)
            merged[key] = result
        return merged

    @staticmethod
    def _validate_acceptance_contract(contract: Any, *, actions: list[dict[str, Any]], tools: list[dict[str, Any]], metrics: list[dict[str, Any]]) -> None:
        if not isinstance(contract, dict) or contract.get("authority") != "env_factory_outer_workflow":
            raise PipelineGenerationError("acceptance_contract must be EnvFactory-owned")
        for key in ("fixtures", "scenarios", "tool_cases", "argument_probes", "executable_scenarios", "invariants", "mutations", "reward_cases", "mutation_tests"):
            if not isinstance(contract.get(key), list | dict):
                raise PipelineGenerationError(f"acceptance_contract.{key} is invalid")
        if not contract["scenarios"] or (tools and not contract["tool_cases"]) or not contract["reward_cases"]:
            raise PipelineGenerationError("acceptance_contract requires scenarios, tool_cases and reward_cases")
        action_names = {action.get("name") for action in actions}
        tool_names = {tool.get("function", {}).get("name") for tool in tools}
        metric_by_id = {metric.get("id"): metric for metric in metrics}
        metric_ids = set(metric_by_id)
        for case in contract["tool_cases"]:
            if not isinstance(case, dict) or not case.get("case_id") or case.get("tool_name") not in tool_names or not isinstance(case.get("expected"), dict):
                raise PipelineGenerationError("acceptance_contract references unknown tool")
        for case in contract["reward_cases"]:
            if (
                not isinstance(case, dict) or not case.get("case_id")
                or case.get("metric_id") not in metric_ids
                or isinstance(case.get("expected_score"), bool)
                or not isinstance(case.get("expected_score"), (int, float))
            ):
                raise PipelineGenerationError("acceptance_contract references unknown metric")
            metric = metric_by_id[case["metric_id"]]
            mapping = metric.get("evaluator", {}).get("score_mapping", {})
            values = [
                value for value in mapping.values()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            ]
            expected = max(values) if str(case["case_id"]).endswith(".pass") else min(values)
            if values and case["expected_score"] != expected:
                raise PipelineGenerationError(
                    f"acceptance_contract reward case {case['case_id']} has reversed expected score"
                )
        for scenario in contract["scenarios"]:
            if not isinstance(scenario, dict) or not scenario.get("scenario_id") or not scenario.get("kind") or not isinstance(scenario.get("steps"), list):
                raise PipelineGenerationError("acceptance_contract scenario is incomplete")
            if scenario.get("kind") == "goal_critical" and scenario.get("action_name") not in action_names:
                raise PipelineGenerationError("acceptance_contract references unknown action")
        if not contract.get("invariants") or not contract.get("mutation_tests"):
            raise PipelineGenerationError("acceptance_contract requires invariants and mutation_tests")

    @staticmethod
    def _validate_executable_scenarios(
        scenarios: Any, *, tools: list[dict[str, Any]], noise_tools: list[dict[str, Any]]
    ) -> None:
        if not isinstance(scenarios, list) or not scenarios:
            raise PipelineGenerationError("executable business scenarios must be a non-empty list")
        tool_schemas = {
            tool["function"]["name"]: tool["function"]["parameters"] for tool in tools
        }
        noise_names = {item.get("name") for item in noise_tools if isinstance(item, dict)}
        allowed_operations = {
            "reset", "tool_call", "agent_response", "observation", "reward", "replay",
            "business_snapshot", "mutate_business_state",
        }
        allowed_operators = {
            "eq", "ne", "gte", "lte", "contains", "exists", "count_gte",
            "count_eq", "changed", "unchanged", "subset",
        }
        kinds: set[str] = set()
        ids: set[str] = set()
        for index, scenario in enumerate(scenarios):
            if not isinstance(scenario, dict) or not isinstance(scenario.get("scenario_id"), str):
                raise PipelineGenerationError(f"executable_scenarios[{index}] is invalid")
            if scenario["scenario_id"] in ids or scenario.get("kind") not in {"goal_success", "goal_failure", "noise_selection", "counterfactual"}:
                raise PipelineGenerationError(f"executable_scenarios[{index}] id/kind is invalid")
            ids.add(scenario["scenario_id"])
            kinds.add(scenario["kind"])
            steps = scenario.get("steps")
            if not isinstance(steps, list) or not steps:
                raise PipelineGenerationError(f"executable_scenarios[{index}] requires steps")
            captures: set[str] = set()
            for step in steps:
                if not isinstance(step, dict) or step.get("operation") not in allowed_operations:
                    raise PipelineGenerationError(f"executable_scenarios[{index}] has invalid operation")
                if step["operation"] == "tool_call":
                    name, arguments = step.get("tool_name"), step.get("arguments", {})
                    if name not in tool_schemas or not isinstance(arguments, dict):
                        raise PipelineGenerationError(f"executable_scenarios[{index}] references invalid tool")
                    if scenario["kind"] == "goal_success" and name in noise_names:
                        raise PipelineGenerationError("goal_success scenario cannot use noise tools")
                    schema = tool_schemas[name]
                    if set(arguments) - set(schema.get("properties", {})):
                        raise PipelineGenerationError(f"executable_scenarios[{index}] has unknown tool arguments")
                    if set(schema.get("required", [])) - set(arguments):
                        raise PipelineGenerationError(f"executable_scenarios[{index}] misses required tool arguments")
                if step["operation"] == "agent_response" and (
                    not isinstance(step.get("content"), str) or not step["content"].strip()
                ):
                    raise PipelineGenerationError(f"executable_scenarios[{index}] has invalid agent response")
                capture = step.get("capture", {})
                if capture and (not isinstance(capture, dict) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in capture.items())):
                    raise PipelineGenerationError(f"executable_scenarios[{index}] capture is invalid")
                captures.update(capture)
            assertions = scenario.get("assertions", [])
            if not isinstance(assertions, list) or not assertions:
                raise PipelineGenerationError(f"executable_scenarios[{index}] requires assertions")
            if any(not isinstance(item, dict) or item.get("operator") not in allowed_operators for item in assertions):
                raise PipelineGenerationError(f"executable_scenarios[{index}] assertion is invalid")
            reward_assertions = [
                item for item in assertions
                if isinstance(item, dict) and item.get("path") == "$.reward"
            ]
            if scenario["kind"] == "goal_success" and not any(
                item.get("operator") == "gte"
                and isinstance(item.get("expected"), (int, float))
                and not isinstance(item.get("expected"), bool)
                and float(item["expected"]) >= 0.5
                for item in reward_assertions
            ):
                raise PipelineGenerationError("goal_success must assert reward >= 0.5")
            if scenario["kind"] == "goal_failure" and not any(
                item.get("operator") == "lte"
                and isinstance(item.get("expected"), (int, float))
                and not isinstance(item.get("expected"), bool)
                and float(item["expected"]) <= 0.2
                for item in reward_assertions
            ):
                raise PipelineGenerationError("goal_failure must assert reward <= 0.2")
            if scenario["kind"] == "noise_selection" and not any(
                item.get("operator") == "lte"
                and isinstance(item.get("expected"), (int, float))
                and not isinstance(item.get("expected"), bool)
                and float(item["expected"]) <= 0
                for item in reward_assertions
            ):
                raise PipelineGenerationError("noise_selection must assert non-positive reward")
        if not {"goal_success", "goal_failure"} <= kinds:
            raise PipelineGenerationError("executable scenarios require goal_success and goal_failure")
        if noise_names and "noise_selection" not in kinds:
            raise PipelineGenerationError("tasks with noise tools require a noise_selection scenario")

    @classmethod
    def _build_business_scenario_baseline(
        cls,
        *,
        task_description: dict[str, Any],
        tools: list[dict[str, Any]],
        noise_tools: list[dict[str, Any]],
        success_content: str | None = None,
    ) -> list[dict[str, Any]]:
        noise_names = {
            item.get("name") for item in noise_tools if isinstance(item, dict)
        }
        business_tools = [
            tool for tool in tools if tool.get("function", {}).get("name") not in noise_names
        ]
        success_steps: list[dict[str, Any]] = [
            {"operation": "reset", "body": {"episode_id": "goal-success", "seed": 17}, "expected_status": 200}
        ]
        for index, tool in enumerate(business_tools, start=1):
            function = tool["function"]
            success_steps.append({
                "step_id": f"business_tool_{index}",
                "operation": "tool_call",
                "tool_name": function["name"],
                "arguments": cls._schema_fixture(function["parameters"]),
                "expected_status": 200,
            })
        success_steps.extend([
            {
                "step_id": "final_answer",
                "operation": "agent_response",
                "content": str(success_content or task_description.get("expected_result") or task_description.get("goal") or "已完成任务并给出依据。"),
                "expected_status": 200,
            },
            {"step_id": "success_reward", "operation": "reward", "expected_status": 200},
        ])
        scenarios: list[dict[str, Any]] = [{
            "scenario_id": "deterministic_goal_success",
            "kind": "goal_success",
            "steps": success_steps,
            "assertions": [
                {"source": "step:success_reward", "path": "$.reward", "operator": "gte", "expected": 0.6}
            ],
        }, {
            "scenario_id": "deterministic_goal_failure",
            "kind": "goal_failure",
            "steps": [
                {"operation": "reset", "body": {"episode_id": "goal-failure", "seed": 17}, "expected_status": 200},
                {"step_id": "failure_reward", "operation": "reward", "expected_status": 200},
            ],
            "assertions": [
                {"source": "step:failure_reward", "path": "$.reward", "operator": "lte", "expected": 0.2}
            ],
        }]
        if noise_tools:
            noise_name = str(noise_tools[0]["name"])
            noise_tool = next(tool for tool in tools if tool.get("function", {}).get("name") == noise_name)
            scenarios.append({
                "scenario_id": "deterministic_noise_selection",
                "kind": "noise_selection",
                "steps": [
                    {"operation": "reset", "body": {"episode_id": "noise-selection", "seed": 17}, "expected_status": 200},
                    {
                        "step_id": "noise_call", "operation": "tool_call", "tool_name": noise_name,
                        "arguments": cls._schema_fixture(noise_tool["function"]["parameters"]), "expected_status": 200,
                    },
                    {"step_id": "noise_reward", "operation": "reward", "expected_status": 200},
                ],
                "assertions": [
                    {"source": "step:noise_reward", "path": "$.reward", "operator": "lte", "expected": 0}
                ],
            })
        return scenarios

    @staticmethod
    def _select_success_response_fixture(
        *, task_description: dict[str, Any], sessions: list[dict[str, Any]]
    ) -> str:
        candidates = [
            str(turn.get("content", "")).strip()
            for session in sessions if isinstance(session, dict)
            for turn in session.get("turns", []) if isinstance(turn, dict) and turn.get("role") == "agent"
            if isinstance(turn.get("content"), str) and turn["content"].strip()
        ]
        if candidates:
            return max(candidates, key=len)
        return str(
            task_description.get("expected_result")
            or task_description.get("goal")
            or "已完成任务并给出依据。"
        )

    @staticmethod
    def _normalize_executable_scenarios(scenarios: Any) -> list[dict[str, Any]]:
        """Canonicalize business scenario identity without changing its semantics."""
        if not isinstance(scenarios, list):
            return []
        aliases = {
            "success": "goal_success",
            "successful": "goal_success",
            "goal-success": "goal_success",
            "failure": "goal_failure",
            "failed": "goal_failure",
            "goal-failure": "goal_failure",
            "noise": "noise_selection",
            "noise-selection": "noise_selection",
            "counter-factual": "counterfactual",
        }
        allowed = {"goal_success", "goal_failure", "noise_selection", "counterfactual"}
        normalized: list[dict[str, Any]] = []
        used_ids: set[str] = set()
        counts: dict[str, int] = {}
        for raw in scenarios:
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            raw_kind = item.get("kind")
            kind = aliases.get(str(raw_kind).strip().lower(), raw_kind)
            scenario_id = item.get("scenario_id")
            identity = str(scenario_id or "").lower()
            if kind not in allowed:
                if "success" in identity:
                    kind = "goal_success"
                elif "fail" in identity:
                    kind = "goal_failure"
                elif "noise" in identity:
                    kind = "noise_selection"
                elif "counter" in identity or "mutation" in identity:
                    kind = "counterfactual"
                else:
                    # Models sometimes echo lifecycle/replay entries from the
                    # baseline. They are already retained by the outer
                    # contract and are not business success/failure scenarios.
                    continue
            item["kind"] = kind
            counts[kind] = counts.get(kind, 0) + 1
            base_id = str(scenario_id).strip() if isinstance(scenario_id, str) else ""
            if not base_id:
                base_id = f"{kind}_{counts[kind]}"
            unique_id = base_id
            suffix = 2
            while unique_id in used_ids:
                unique_id = f"{base_id}_{suffix}"
                suffix += 1
            item["scenario_id"] = unique_id
            used_ids.add(unique_id)
            normalized.append(item)
        return normalized

    @staticmethod
    def _fill_tool_schema_descriptions(tools: list[Any]) -> None:
        """Fill cosmetic JSON Schema descriptions without another LLM call."""
        def fill_schema(schema: Any, label: str) -> None:
            if not isinstance(schema, dict):
                return
            if not isinstance(schema.get("description"), str) or not schema["description"].strip():
                schema["description"] = f"{label} 的输入值。"
            properties = schema.get("properties")
            if isinstance(properties, dict):
                for name, child in properties.items():
                    fill_schema(child, str(name))
            items = schema.get("items")
            if isinstance(items, dict):
                fill_schema(items, f"{label} 数组元素")

        for tool in tools:
            function = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(function, dict):
                continue
            name = function.get("name") if isinstance(function.get("name"), str) else "工具"
            if not isinstance(function.get("description"), str) or not function["description"].strip():
                function["description"] = f"执行 {name} 操作。"
            fill_schema(function.get("parameters"), f"{name} 参数")

    @staticmethod
    def _validate_noise_tool_audit(decisions: Any, *, noise_names: set[str]) -> set[str]:
        if not isinstance(decisions, list):
            raise PipelineGenerationError("noise tool audit must return decisions")
        seen: set[str] = set()
        unsafe: set[str] = set()
        for index, decision in enumerate(decisions):
            if not isinstance(decision, dict):
                raise PipelineGenerationError(f"noise audit decisions[{index}] must be an object")
            name = decision.get("name")
            if name not in noise_names or name in seen:
                raise PipelineGenerationError("noise audit references an unknown or duplicate tool")
            if not isinstance(decision.get("is_safe_noise"), bool):
                raise PipelineGenerationError("noise audit requires boolean is_safe_noise")
            if not isinstance(decision.get("reason"), str) or not decision["reason"].strip():
                raise PipelineGenerationError("noise audit requires reason")
            if not decision["is_safe_noise"]:
                unsafe.add(str(name))
            seen.add(str(name))
        if seen != noise_names:
            raise PipelineGenerationError(
                f"noise audit must cover every noise tool; missing={sorted(noise_names - seen)}"
            )
        return unsafe

    @staticmethod
    def _fallback_noise_tool(
        *, occupied_names: set[str]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        base = "roll_virtual_die"
        name = base
        suffix = 2
        while name in occupied_names:
            name = f"{base}_{suffix}"
            suffix += 1
        return ({
            "type": "function",
            "function": {
                "name": name,
                "description": "掷一个虚拟六面骰并返回随机点数；不读取或修改任务业务状态。",
                "parameters": {
                    "type": "object",
                    "description": "虚拟骰子参数。",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        }, {
            "name": name,
            "category": "unrelated",
            "rationale": "通用娱乐性随机工具，不提供任务事实、证据、比较依据或验证能力。",
        })

    @staticmethod
    def _validate_tools(tools: list[Any]) -> None:
        names: set[str] = set()
        for index, tool in enumerate(tools):
            function = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(tool, dict) or tool.get("type") != "function" or not isinstance(function, dict):
                raise PipelineGenerationError(f"tools[{index}] is not an OpenAI function tool")
            name = function.get("name")
            if not isinstance(name, str) or not name or name in names:
                raise PipelineGenerationError(f"tools[{index}].function.name is invalid or duplicated")
            if re.fullmatch(
                r"(?:noise_)?tool_?\d*|function_?\d*|semantic_function_name_?\d*",
                name,
                flags=re.IGNORECASE,
            ):
                raise PipelineGenerationError(
                    f"tool {name} uses a placeholder name; use a concrete semantic function name"
                )
            if not isinstance(function.get("description"), str) or not function["description"].strip():
                raise PipelineGenerationError(f"tool {name} has no description")
            parameters = function.get("parameters")
            if not isinstance(parameters, dict) or parameters.get("type") != "object":
                raise PipelineGenerationError(f"tool {name} parameters must be an object schema")
            required = parameters.get("required", [])
            properties = parameters.get("properties", {})
            if not isinstance(properties, dict) or not isinstance(required, list) or any(item not in properties for item in required):
                raise PipelineGenerationError(f"tool {name} has invalid properties/required")
            forbidden = {"acceptance_evidence_id", "acceptance_message_id", "ground_truth", "user_accepted"}
            if forbidden & set(properties):
                raise PipelineGenerationError(f"tool {name} exposes forbidden internal input")
            def check_properties(items: dict[str, Any], prefix: str) -> None:
                for property_name, schema in items.items():
                    if not isinstance(schema, dict) or not isinstance(schema.get("description"), str) or not schema["description"].strip():
                        raise PipelineGenerationError(f"tool {name}.{prefix}{property_name} requires a description")
                    if schema.get("type") == "object":
                        nested = schema.get("properties", {})
                        if not isinstance(nested, dict):
                            raise PipelineGenerationError(f"tool {name}.{prefix}{property_name} has invalid nested properties")
                        check_properties(nested, prefix + property_name + ".")
                    if schema.get("type") == "array" and isinstance(schema.get("items"), dict):
                        check_properties({"[]": schema["items"]}, prefix + property_name)
            check_properties(properties, "")
            names.add(name)

    @staticmethod
    def _validate_tool_implementations(
        specs: list[Any], *, tools: list[dict[str, Any]], tables: list[dict[str, Any]]
    ) -> None:
        tool_parameters = {
            tool["function"]["name"]: set(tool["function"]["parameters"].get("properties", {}))
            for tool in tools
        }
        table_columns = {
            table["table_name"]: {column["name"] for column in table.get("columns", [])}
            for table in tables
        }
        seen: set[str] = set()
        for index, spec in enumerate(specs):
            if not isinstance(spec, dict):
                raise PipelineGenerationError(f"tool_implementations[{index}] must be an object")
            name, table = spec.get("tool_name"), spec.get("table")
            if name not in tool_parameters or name in seen:
                raise PipelineGenerationError(f"tool_implementations[{index}] references an invalid tool")
            operation = spec.get("operation")
            if operation not in {"select", "aggregate_count", "insert", "update", "delete"} or table not in table_columns:
                raise PipelineGenerationError(f"tool_implementations[{index}] operation/table is invalid")
            if not isinstance(spec.get("result_field"), str) or not spec["result_field"]:
                raise PipelineGenerationError(f"tool_implementations[{index}] requires result_field")
            filters = spec.get("filters", [])
            if not isinstance(filters, list):
                raise PipelineGenerationError(f"tool_implementations[{index}].filters must be a list")
            for rule in filters:
                if (
                    not isinstance(rule, dict)
                    or rule.get("argument") not in tool_parameters[name]
                    or rule.get("column") not in table_columns[table]
                    or rule.get("operator") not in {"eq", "in", "contains", "gte", "lte"}
                ):
                    raise PipelineGenerationError(f"tool_implementations[{index}] has an invalid filter")
            for field in ("projection", "order_by"):
                values = spec.get(field, [])
                if not isinstance(values, list) or any(value not in table_columns[table] for value in values):
                    raise PipelineGenerationError(f"tool_implementations[{index}].{field} is invalid")
            for field in ("selector", "values", "changes"):
                mapping = spec.get(field, {})
                if not isinstance(mapping, dict) or any(
                    argument not in tool_parameters[name] or column not in table_columns[table]
                    for argument, column in mapping.items()
                ):
                    raise PipelineGenerationError(f"tool_implementations[{index}].{field} is invalid")
            if operation in {"update", "delete"} and not spec.get("selector"):
                raise PipelineGenerationError(f"tool_implementations[{index}] requires selector")
            if operation == "insert" and not spec.get("values"):
                raise PipelineGenerationError(f"tool_implementations[{index}] requires values")
            if operation == "update" and not spec.get("changes"):
                raise PipelineGenerationError(f"tool_implementations[{index}] requires changes")
            seen.add(name)

    @staticmethod
    def _build_runtime_interface(
        tools: list[dict[str, Any]], reward_formula: dict[str, Any]
    ) -> dict[str, Any]:
        """Build the HTTP contract consumed by sandbox builders and trainers.

        This is deliberately derived from validated artifacts rather than
        requested from the LLM.  Every generated LLM tool is listed with its
        exact request schema, and the single runtime reward function has an
        explicit endpoint.
        """
        tool_endpoints = []
        for tool in tools:
            function = tool["function"]
            tool_endpoints.append({
                "name": function["name"],
                "kind": "llm_tool",
                "method": "POST",
                "path": "/v1/tools/{tool_name}",
                "request_schema": function["parameters"],
                "response_schema": {
                    "type": "object",
                    "description": "本次工具调用的业务结果；不包含 observation 或 reward。",
                },
            })
        return {
            "protocol": "http",
            "base_path": "/v1",
            "version": "1.0",
            "shared_runtime": {
                "module": "sandbox_runtime.py",
                "version": "1.1",
                "required_components": [
                    "EpisodeStore",
                    "ManifestDataStore",
                    "DeclarativeToolCompiler",
                    "SandboxApplication",
                    "ContractToolRegistry",
                    "ContractRewardAggregator",
                    "DeclarativeMetricEvaluator",
                    "AcceptanceScenarioRunner",
                    "ContractUserSimulator",
                    "validate_json_schema",
                ],
            },
            "errors": {
                "content_type": "application/json",
                "schema": {
                    "type": "object",
                    "required": ["error"],
                    "error_required": ["code", "message", "request_id"],
                },
                "status_codes": {"invalid_argument": 400, "unauthorized": 401, "forbidden": 403, "not_found": 404, "conflict": 409, "server_error": 500},
            },
            "observability": {
                "request_id_header": "X-Request-ID",
                "required_fields": ["request_id", "episode_id", "tool_call_id", "timestamp", "duration_ms"],
                "credential_redaction": True,
                "structured_format": "json",
            },
            "security": {
                "trainer": {
                    "scheme": "bearer",
                    "environment_variable": "SANDBOX_TRAINER_API_KEY",
                    "required_for": ["reset", "observation", "user_simulator", "agent_response", "reward", "replay"],
                },
                "agent": {
                    "scheme": "none",
                    "allowed_kinds": ["llm_tool"],
                    "cannot_access": ["user_simulator", "agent_response", "reward", "replay"],
                },
            },
            "episode": {
                "isolation": "per_episode",
                "reset_accepts_seed": True,
                "deterministic_replay": True,
                "idempotency_header": "Idempotency-Key",
                "persist_trace": True,
            },
            "llm_runtime": {
                "provider": "openai_compatible_chat_completions",
                "api_key_environment_variable": "SANDBOX_LLM_API_KEY",
                "base_url_environment_variable": "SANDBOX_LLM_BASE_URL",
                "model_environment_variable": "SANDBOX_LLM_MODEL",
                "timeout_environment_variable": "SANDBOX_LLM_TIMEOUT_SECONDS",
                "max_retries_environment_variable": "SANDBOX_LLM_MAX_RETRIES",
                "never_persist_credentials": True,
            },
            "evaluator_runtime": {
                "external_llm": "SANDBOX_LLM_API_KEY",
                "mock_mode_environment_variable": "SANDBOX_EVALUATOR_MOCK",
                "record_call_trace": True,
                "cache_by_context_hash": True,
            },
            "mutation_testing": {
                "environment_variable": "SANDBOX_MUTATION_MODE",
                "modes": ["constant_tool_result", "skip_business_write", "constant_reward", "ignore_tool_arguments", "bypass_trainer_auth"],
                "production_default": "disabled",
                "must_be_observable_in_test_only": True,
            },
            "launcher": {
                "command": ["python", "app.py", "--port", "{port}"],
                "health_path": "/health",
            },
            "endpoints": [
                {"name": "health", "kind": "system", "method": "GET", "path": "/health"},
                {
                    "name": "reset", "kind": "system", "method": "POST", "path": "/v1/reset",
                    "access": "rl_trainer_only",
                    "request_schema": {
                        "type": "object", "description": "创建或重置独立 episode。",
                        "properties": {
                            "episode_id": {"type": "string", "description": "可选的 episode 标识。"},
                            "seed": {"type": "integer", "description": "可选的确定性随机种子。"},
                        }, "required": [], "additionalProperties": False,
                    },
                },
                {"name": "observation", "kind": "system", "access": "rl_trainer_only", "method": "GET", "path": "/v1/observation"},
                {"name": "tools", "kind": "system", "method": "GET", "path": "/v1/tools"},
                {
                    "name": "user_simulator",
                    "kind": "user_simulator",
                    "access": "rl_trainer_only",
                    "method": "POST",
                    "path": "/v1/user_simulator",
                    "request_schema": {
                        "type": "object",
                        "description": "RL Trainer 传入的完整对话上下文。该接口不暴露给待训练 Agent。",
                        "properties": {
                            "messages": {
                                "type": "array",
                                "description": "从当前会话开始的完整对话消息序列，按时间顺序排列。",
                                "items": {
                                    "type": "object",
                                    "description": "一条对话消息。",
                                    "properties": {
                                        "role": {"type": "string", "description": "消息角色，例如 user 或 assistant。"},
                                        "content": {"type": "string", "description": "消息文本内容。"},
                                    },
                                    "required": ["role", "content"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["messages"],
                        "additionalProperties": False,
                    },
                    "response_schema": {
                        "type": "object",
                        "description": "User Simulator 生成的用户查询和终止标志。",
                        "properties": {
                            "user_query": {"type": "string", "description": "模拟用户生成的自然语言查询。"},
                            "should_end": {"type": "boolean", "description": "用户是否停止继续提问。"},
                        },
                        "required": ["user_query", "should_end"],
                        "additionalProperties": False,
                    },
                },
                {
                    "name": "agent_response",
                    "kind": "agent_response",
                    "access": "rl_trainer_only",
                    "method": "POST",
                    "path": "/v1/agent_response",
                    "request_schema": {
                        "type": "object",
                        "description": "RL Trainer 提交待训练 Agent 的最终自然语言回答。",
                        "properties": {
                            "content": {"type": "string", "description": "Agent 的非空最终回答。"},
                        },
                        "required": ["content"],
                        "additionalProperties": False,
                    },
                    "response_schema": {
                        "type": "object",
                        "description": "确认最终回答已写入当前 episode。",
                    },
                },
                *tool_endpoints,
                {
                    "name": "reward",
                    "kind": "reward_function",
                    "access": "rl_trainer_only",
                    "method": "GET",
                    "path": "/v1/reward",
                    "formula": reward_formula,
                    "response_schema": {
                        "type": "object",
                        "description": "调用该接口时统一计算并返回当前会话奖励和各指标得分。",
                    },
                },
                {
                    "name": "replay",
                    "kind": "replay",
                    "access": "rl_trainer_only",
                    "method": "GET",
                    "path": "/v1/replay",
                    "response_schema": {
                        "type": "object",
                        "description": "返回当前 episode 的可回放轨迹、seed、版本和数据哈希。",
                    },
                },
            ],
            "llm_tools": [endpoint["name"] for endpoint in tool_endpoints],
            "reward_functions": [{"name": "reward", "endpoint": "/v1/reward"}],
        }

    @staticmethod
    def _validate_runtime_interface(
        interface: Any, tools: list[dict[str, Any]], reward_formula: dict[str, Any]
    ) -> None:
        if not isinstance(interface, dict) or interface.get("protocol") != "http":
            raise PipelineGenerationError("requirements.runtime_interface must declare http")
        shared_runtime = interface.get("shared_runtime")
        required_shared = {
            "EpisodeStore", "ManifestDataStore", "DeclarativeToolCompiler", "SandboxApplication", "ContractToolRegistry",
            "ContractRewardAggregator", "DeclarativeMetricEvaluator", "AcceptanceScenarioRunner", "ContractUserSimulator",
            "validate_json_schema"
        }
        if (
            not isinstance(shared_runtime, dict)
            or shared_runtime.get("module") != "sandbox_runtime.py"
            or not required_shared <= set(shared_runtime.get("required_components", []))
        ):
            raise PipelineGenerationError("runtime_interface shared runtime contract is incomplete")
        endpoints = interface.get("endpoints")
        if not isinstance(endpoints, list) or not endpoints:
            raise PipelineGenerationError("runtime_interface.endpoints must be non-empty")
        endpoint_keys = {(item.get("name"), item.get("method"), item.get("path"))
                         for item in endpoints if isinstance(item, dict)}
        for name, method, path in (
            ("health", "GET", "/health"),
            ("reset", "POST", "/v1/reset"),
            ("observation", "GET", "/v1/observation"),
            ("tools", "GET", "/v1/tools"),
            ("user_simulator", "POST", "/v1/user_simulator"),
            ("agent_response", "POST", "/v1/agent_response"),
            ("reward", "GET", "/v1/reward"),
            ("replay", "GET", "/v1/replay"),
        ):
            if (name, method, path) not in endpoint_keys:
                raise PipelineGenerationError(f"runtime_interface missing {method} {path}")
        expected_tools = {tool["function"]["name"]: tool["function"]["parameters"] for tool in tools}
        declared_tools = [
            item for item in endpoints
            if isinstance(item, dict) and item.get("kind") == "llm_tool"
        ]
        if len(declared_tools) != len(expected_tools):
            raise PipelineGenerationError("runtime_interface must list every LLM tool exactly once")
        for item in declared_tools:
            name = item.get("name")
            if name not in expected_tools or item.get("method") != "POST" or item.get("path") != "/v1/tools/{tool_name}":
                raise PipelineGenerationError(f"invalid runtime endpoint for tool {name!r}")
            if item.get("request_schema") != expected_tools[name]:
                raise PipelineGenerationError(f"runtime schema does not match tool {name}")
        if interface.get("llm_tools") != list(expected_tools):
            raise PipelineGenerationError("runtime_interface.llm_tools is inconsistent with tools")
        user_simulator_endpoint = next(
            item for item in endpoints if item.get("name") == "user_simulator"
        )
        if user_simulator_endpoint.get("kind") != "user_simulator" or user_simulator_endpoint.get("access") != "rl_trainer_only":
            raise PipelineGenerationError("runtime_interface user_simulator endpoint has invalid kind")
        expected_user_request = {
            "type": "object",
            "description": "RL Trainer 传入的完整对话上下文。该接口不暴露给待训练 Agent。",
            "properties": {
                "messages": {
                    "type": "array",
                    "description": "从当前会话开始的完整对话消息序列，按时间顺序排列。",
                    "items": {
                        "type": "object",
                        "description": "一条对话消息。",
                        "properties": {
                            "role": {"type": "string", "description": "消息角色，例如 user 或 assistant。"},
                            "content": {"type": "string", "description": "消息文本内容。"},
                        },
                        "required": ["role", "content"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["messages"],
            "additionalProperties": False,
        }
        if user_simulator_endpoint.get("request_schema") != expected_user_request:
            raise PipelineGenerationError("user_simulator request schema must use complete messages")
        reward_endpoint = next(item for item in endpoints if item.get("name") == "reward")
        if reward_endpoint.get("access") != "rl_trainer_only":
            raise PipelineGenerationError("reward endpoint must be Trainer-only")
        reward_endpoint = next(item for item in endpoints if item.get("name") == "reward")
        if reward_endpoint.get("formula") != reward_formula:
            raise PipelineGenerationError("runtime reward endpoint formula is inconsistent")
        replay_endpoint = next(item for item in endpoints if item.get("name") == "replay")
        if replay_endpoint.get("access") != "rl_trainer_only":
            raise PipelineGenerationError("runtime replay endpoint must be Trainer-only")
        security = interface.get("security")
        if not isinstance(security, dict):
            raise PipelineGenerationError("runtime_interface.security is required")
        trainer_security = security.get("trainer")
        if not isinstance(trainer_security, dict) or trainer_security.get("scheme") != "bearer" \
                or trainer_security.get("environment_variable") != "SANDBOX_TRAINER_API_KEY":
            raise PipelineGenerationError("runtime_interface trainer authentication is invalid")
        episode = interface.get("episode")
        if not isinstance(episode, dict) or episode.get("isolation") != "per_episode" \
                or episode.get("reset_accepts_seed") is not True \
                or episode.get("deterministic_replay") is not True:
            raise PipelineGenerationError("runtime_interface episode isolation/replay contract is invalid")
        llm_runtime = interface.get("llm_runtime")
        if not isinstance(llm_runtime, dict) or llm_runtime.get("api_key_environment_variable") != "SANDBOX_LLM_API_KEY":
            raise PipelineGenerationError("runtime_interface LLM credential boundary is invalid")
        evaluator_runtime = interface.get("evaluator_runtime")
        if not isinstance(evaluator_runtime, dict) or evaluator_runtime.get("mock_mode_environment_variable") != "SANDBOX_EVALUATOR_MOCK":
            raise PipelineGenerationError("runtime_interface evaluator mock contract is invalid")
        mutation_testing = interface.get("mutation_testing")
        if not isinstance(mutation_testing, dict) or mutation_testing.get("environment_variable") != "SANDBOX_MUTATION_MODE" \
                or not isinstance(mutation_testing.get("modes"), list) or not mutation_testing["modes"]:
            raise PipelineGenerationError("runtime_interface mutation testing contract is invalid")
        launcher = interface.get("launcher")
        if not isinstance(launcher, dict) or launcher.get("command") != ["python", "app.py", "--port", "{port}"]:
            raise PipelineGenerationError("runtime_interface launcher contract is invalid")
        errors = interface.get("errors")
        if not isinstance(errors, dict) or errors.get("content_type") != "application/json" \
                or not isinstance(errors.get("schema"), dict):
            raise PipelineGenerationError("runtime_interface error protocol is invalid")
        observability = interface.get("observability")
        if not isinstance(observability, dict) or observability.get("request_id_header") != "X-Request-ID" \
                or observability.get("credential_redaction") is not True:
            raise PipelineGenerationError("runtime_interface observability contract is invalid")

    @staticmethod
    def _validate_actions(actions: list[Any]) -> None:
        """Validate the action decomposition before tools are generated."""
        names: set[str] = set()
        for index, action in enumerate(actions):
            if not isinstance(action, dict):
                raise PipelineGenerationError(f"agent_actions[{index}] must be an object")
            name = action.get("name") or action.get("action")
            if not isinstance(name, str) or not name.strip() or name in names:
                raise PipelineGenerationError(f"agent_actions[{index}] has invalid or duplicate name")
            if not isinstance(action.get("atomicity_rationale"), str) or not action["atomicity_rationale"].strip():
                raise PipelineGenerationError(f"agent action {name} requires atomicity_rationale")
            for field in ("inputs", "outputs"):
                values = action.get(field)
                if not isinstance(values, list):
                    raise PipelineGenerationError(f"agent action {name}.{field} must be a list")
                for item_index, item in enumerate(values):
                    if not isinstance(item, dict):
                        raise PipelineGenerationError(f"agent action {name}.{field}[{item_index}] must be an object")
                    for item_field in ("name", "description"):
                        if not isinstance(item.get(item_field), str) or not item[item_field].strip():
                            raise PipelineGenerationError(
                                f"agent action {name}.{field}[{item_index}] requires {item_field}"
                            )
            for field in ("preconditions", "effects"):
                values = action.get(field)
                if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
                    raise PipelineGenerationError(f"agent action {name}.{field} must be a list of non-empty strings")
            names.add(name)

    @staticmethod
    def _validate_capability_plan(capabilities: list[Any], actions: list[Any]) -> None:
        action_names = {
            action.get("name") for action in actions
            if isinstance(action, dict) and isinstance(action.get("name"), str)
        }
        seen: set[str] = set()
        for index, capability in enumerate(capabilities):
            if not isinstance(capability, dict):
                raise PipelineGenerationError(f"capabilities[{index}] must be an object")
            action_name = capability.get("action_name")
            kind = capability.get("kind")
            requires_tool = capability.get("requires_tool")
            if action_name not in action_names or action_name in seen:
                raise PipelineGenerationError(f"capabilities[{index}] has unknown or duplicate action_name")
            if kind not in {"environment_operation", "agent_reasoning", "agent_response"}:
                raise PipelineGenerationError(f"capabilities[{index}] has invalid kind")
            if not isinstance(requires_tool, bool):
                raise PipelineGenerationError(f"capabilities[{index}] requires boolean requires_tool")
            if requires_tool != (kind == "environment_operation"):
                raise PipelineGenerationError(f"capabilities[{index}] kind/requires_tool conflict")
            if not isinstance(capability.get("reason"), str) or not capability["reason"].strip():
                raise PipelineGenerationError(f"capabilities[{index}] requires reason")
            seen.add(action_name)
        if seen != action_names:
            missing = sorted(action_names - seen)
            raise PipelineGenerationError(f"capability plan must cover every action; missing={missing}")

    @staticmethod
    def _resolve_environment_plan(
        candidate: Any, *, task_description: dict[str, Any], task_intent: str
    ) -> dict[str, Any]:
        modes = {"stateless", "reference_data", "stateful", "external_capability"}
        value = dict(candidate) if isinstance(candidate, dict) else {}
        mode = value.get("mode")
        if mode not in modes:
            raise PipelineGenerationError("environment_plan.mode is invalid")
        task_text = " ".join(str(task_description.get(key, "")) for key in ("task", "goal", "expected_result"))
        persistence_markers = (
            "写入", "保存", "持久化", "更新", "修改", "删除", "创建记录", "提交订单",
            "数据库", "审批", "排程", "预订", "write", "save", "persist", "update",
            "delete", "create record", "database", "approve", "schedule", "book",
        )
        external_markers = ("天气", "实时", "联网", "搜索", "汇率", "weather", "search", "exchange rate")
        explicitly_persistent = any(marker in task_text.lower() for marker in persistence_markers)
        explicitly_external = any(marker in task_text.lower() for marker in external_markers)
        stateless_intents = {"extract", "summarize", "classify", "transform", "explain"}
        if task_intent in stateless_intents and not explicitly_persistent and not explicitly_external:
            mode = "stateless"
        requires_business_data = mode in {"reference_data", "stateful"}
        requires_persistence = mode == "stateful"
        reason = value.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            reason = "由任务明确目标和预期结果推导运行时环境模式。"
        return {
            "mode": mode,
            "requires_business_data": requires_business_data,
            "requires_persistence": requires_persistence,
            "reason": reason.strip(),
        }

    @staticmethod
    def _derive_complexity(
        *, business_tool_count: int, metric_count: int, key_step_count: int
    ) -> str:
        if business_tool_count <= 1 and metric_count <= 4 and key_step_count <= 1:
            return "simple"
        if business_tool_count <= 4 and metric_count <= 8 and key_step_count <= 4:
            return "standard"
        return "complex"

    @staticmethod
    def _derive_training_profile(
        *, business_tool_count: int, noise_tool_count: int, key_step_count: int
    ) -> str:
        if business_tool_count == 0:
            return "tool_abstention" if noise_tool_count else "language_only"
        if key_step_count > 1 or business_tool_count > 1:
            return "multi_step_tool_use"
        return "single_step_tool_use"

    @staticmethod
    def _validate_task_readiness(
        *,
        capability_plan: list[Any],
        tool_bindings: list[Any],
        noise_tools: list[Any],
        metrics: list[Any],
        metric_implementations: list[Any],
        business_scenarios: list[Any],
        require_noise: bool = True,
    ) -> None:
        eligible = {
            item.get("action_name") for item in capability_plan
            if isinstance(item, dict) and item.get("requires_tool") is True
        }
        bound = {
            item.get("action_name") for item in tool_bindings if isinstance(item, dict)
        }
        if not bound <= eligible:
            raise PipelineGenerationError("task readiness: a tool delegates agent reasoning or response composition")
        if require_noise and not noise_tools:
            raise PipelineGenerationError("task readiness: tool-selection tasks require at least one noise tool")
        rule_ids = {
            item.get("id") for item in metrics
            if isinstance(item, dict) and item.get("type") == "rule-based"
        }
        implemented = {
            item.get("metric_id") for item in metric_implementations if isinstance(item, dict)
        }
        if not rule_ids <= implemented:
            raise PipelineGenerationError("task readiness: rule-based metrics lack executable implementations")
        kinds = {
            item.get("kind") for item in business_scenarios if isinstance(item, dict)
        }
        if not {"goal_success", "goal_failure"} <= kinds:
            raise PipelineGenerationError("task readiness: success and failure trajectories are required")
        if noise_tools and "noise_selection" not in kinds:
            raise PipelineGenerationError("task readiness: a noise-selection trajectory is required")
        TaskGenerationPipeline._validate_metric_runtime_semantics(
            metrics=metrics,
            metric_implementations=metric_implementations,
            noise_tools=noise_tools,
        )

    @staticmethod
    def _validate_user_script_tree(script: dict[str, Any], index: int) -> None:
        """Validate the branching behavior tree consumed by User LLM."""
        tree = script.get("tree")
        if not isinstance(tree, dict):
            raise PipelineGenerationError(f"user_scripts[{index}] requires a tree object")
        seen_nodes: set[str] = set()
        seen_branches: set[str] = set()
        branch_count = 0
        has_end_signal = False

        def visit(node: Any, path: str, depth: int = 0) -> None:
            nonlocal branch_count, has_end_signal
            if depth > 8:
                raise PipelineGenerationError(f"user_scripts[{index}] tree exceeds maximum depth 8")
            if not isinstance(node, dict):
                raise PipelineGenerationError(f"user_scripts[{index}] tree node {path} must be an object")
            node_id = node.get("node_id")
            behavior = node.get("user_behavior")
            branches = node.get("branches")
            if not isinstance(node_id, str) or not node_id.strip() or node_id in seen_nodes:
                raise PipelineGenerationError(f"user_scripts[{index}] tree node {path} requires a unique node_id")
            if not isinstance(behavior, str) or not behavior.strip():
                raise PipelineGenerationError(f"user_scripts[{index}] tree node {node_id} requires user_behavior")
            if not isinstance(branches, list):
                raise PipelineGenerationError(f"user_scripts[{index}] tree node {node_id} requires branches array")
            seen_nodes.add(node_id)
            for branch_index, branch in enumerate(branches):
                if not isinstance(branch, dict):
                    raise PipelineGenerationError(f"user_scripts[{index}] branch {path}.{branch_index} must be an object")
                branch_id = branch.get("branch_id")
                condition = branch.get("condition")
                should_end = branch.get("should_end")
                next_node = branch.get("next")
                if not isinstance(branch_id, str) or not branch_id.strip() or branch_id in seen_branches:
                    raise PipelineGenerationError(f"user_scripts[{index}] branch {path}.{branch_index} requires a unique branch_id")
                if not isinstance(condition, str) or not condition.strip():
                    raise PipelineGenerationError(f"user_scripts[{index}] branch {branch_id} requires condition")
                if not isinstance(should_end, bool):
                    raise PipelineGenerationError(f"user_scripts[{index}] branch {branch_id} requires boolean should_end")
                has_end_signal = has_end_signal or should_end
                if not isinstance(next_node, dict):
                    raise PipelineGenerationError(f"user_scripts[{index}] branch {branch_id} requires next node")
                seen_branches.add(branch_id)
                branch_count += 1
                visit(next_node, f"{path}.{branch_index}", depth + 1)

        visit(tree, "root")
        if branch_count < 2:
            raise PipelineGenerationError(f"user_scripts[{index}] tree must contain at least two branches")
        if not has_end_signal:
            raise PipelineGenerationError(f"user_scripts[{index}] tree must contain a should_end=true branch")

    @staticmethod
    def _normalize_metric_weights(metrics: list[Any]) -> list[Any]:
        """Normalize positive and negative metric weights independently.

        LLMs often emit proportional weights such as 0.4/0.5/1.0 instead of
        already-normalized weights. Since the ratio is the meaningful part,
        normalize each sign group before strict contract validation.
        """
        positive = 0.0
        negative = 0.0
        for metric in metrics:
            if not isinstance(metric, dict) or not isinstance(metric.get("weight"), (int, float)):
                continue
            category = metric.get("category")
            if category in {"process", "outcome"} and metric["weight"] >= 0:
                positive += float(metric["weight"])
            elif category == "penalty" and metric["weight"] >= 0:
                negative += float(metric["weight"])
        if positive > 0 and not math.isclose(positive, 1.0, abs_tol=1e-6):
            logger.info("normalized positive metric weights: original_sum=%s", positive)
            for metric in metrics:
                if isinstance(metric, dict) and metric.get("category") in {"process", "outcome"} and isinstance(metric.get("weight"), (int, float)):
                    metric["weight"] = float(metric["weight"]) / positive
        if negative > 0 and not math.isclose(negative, 1.0, abs_tol=1e-6):
            logger.info("normalized penalty metric weights: original_sum=%s", negative)
            for metric in metrics:
                if isinstance(metric, dict) and metric.get("category") == "penalty" and isinstance(metric.get("weight"), (int, float)):
                    metric["weight"] = float(metric["weight"]) / negative
        return metrics

    @staticmethod
    def _normalize_metric_types(metrics: list[Any]) -> list[Any]:
        """Derive the canonical metric type from an unambiguous evaluator."""
        evaluator_to_type = {
            "business_state_rule": "rule-based",
            "document_rule": "rule-based",
            "trajectory_rule": "rule-based",
            "external_llm_judge": "model-based",
            "hybrid_tool_call": "hybrid",
            "hybrid_outcome": "hybrid",
        }
        for metric in metrics:
            if not isinstance(metric, dict) or metric.get("type") in {"rule-based", "model-based", "hybrid"}:
                continue
            evaluator = metric.get("evaluator")
            kind = evaluator.get("kind") if isinstance(evaluator, dict) else None
            canonical = evaluator_to_type.get(kind)
            if canonical:
                logger.info(
                    "normalized metric type from evaluator: metric=%s original=%r normalized=%s",
                    metric.get("id"), metric.get("type"), canonical,
                )
                metric["type"] = canonical
        return metrics

    @staticmethod
    def _normalize_metric_evaluators(metrics: list[Any]) -> list[Any]:
        """Canonicalize duplicated rule text emitted at either contract level."""
        rule_kinds = {"business_state_rule", "document_rule", "trajectory_rule"}
        for metric in metrics:
            if not isinstance(metric, dict):
                continue
            evaluator = metric.get("evaluator")
            if not isinstance(evaluator, dict):
                continue
            condition = metric.get("condition")
            assertion = evaluator.get("assertion")
            if evaluator.get("kind") in rule_kinds:
                if not isinstance(assertion, str) or not assertion.strip():
                    if isinstance(condition, str) and condition.strip():
                        evaluator["assertion"] = condition.strip()
                        logger.info(
                            "normalized rule assertion from condition: metric=%s",
                            metric.get("id"),
                        )
                elif not isinstance(condition, str) or not condition.strip():
                    metric["condition"] = assertion.strip()
            elif evaluator.get("kind") == "hybrid_outcome":
                rule = evaluator.get("rule")
                if isinstance(rule, dict):
                    rule_assertion = rule.get("assertion")
                    if (not isinstance(rule_assertion, str) or not rule_assertion.strip()) and isinstance(condition, str) and condition.strip():
                        rule["assertion"] = condition.strip()
        return metrics

    @staticmethod
    def _ensure_deterministic_noise_penalty(
        metrics: list[Any], *, noise_names: list[str]
    ) -> None:
        retained: list[Any] = []
        for metric in metrics:
            if not isinstance(metric, dict):
                retained.append(metric)
                continue
            text = json.dumps(metric, ensure_ascii=False, sort_keys=True).lower()
            if metric.get("category") == "penalty" and (
                "noise" in text
                or "噪声" in text
                or "tool_call" in text
                or "tool use" in text
                or "工具调用" in text
                or "工具使用" in text
                or any(name.lower() in text for name in noise_names)
            ):
                continue
            retained.append(metric)
        retained.append({
            "id": "penalty_noise_tool_usage",
            "category": "penalty",
            "type": "rule-based",
            "scope": "trajectory",
            "condition": "trajectory contains no calls to declared noise tools",
            "evaluator": {
                "kind": "trajectory_rule",
                "source": "runtime_rule",
                "assertion": "no declared noise tool appears in trajectory events",
                "score_mapping": {"pass": 0, "fail": -1},
            },
            "score_range": [-1, 0],
            "weight": 1,
            "rubric": "调用任一声明的噪声工具时施加确定性惩罚。",
        })
        metrics[:] = retained

    @staticmethod
    def _promote_unstructured_document_rules(metrics: list[Any]) -> None:
        """Use a semantic judge for facts that only exist inside document text."""
        for metric in metrics:
            if not isinstance(metric, dict) or metric.get("type") != "rule-based":
                continue
            evaluator = metric.get("evaluator")
            if not isinstance(evaluator, dict) or evaluator.get("kind") != "document_rule":
                continue
            criterion = evaluator.get("assertion") or metric.get("rubric") or metric.get("condition")
            mapping = evaluator.get("score_mapping", {"pass": 1, "fail": 0})
            metric["type"] = "model-based"
            metric["evaluation_inputs"] = ["conversation", "final_agent_response"]
            metric["criteria"] = [str(criterion or "判断最终文档是否满足该指标。")]
            metric["evaluator"] = {
                "kind": "external_llm_judge",
                "source": "external_llm",
                "score_mapping": dict(mapping) if isinstance(mapping, dict) else {"pass": 1, "fail": 0},
            }
            metric.pop("condition", None)
            logger.info(
                "promoted document rule over raw final_agent_response to external judge: metric=%s",
                metric.get("id"),
            )

    @staticmethod
    def _validate_metric_runtime_semantics(
        *, metrics: list[Any], metric_implementations: list[Any], noise_tools: list[Any]
    ) -> None:
        """Reject contracts whose valid-looking DSL would produce reversed rewards."""
        metric_by_id = {
            item.get("id"): item for item in metrics
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
        noise_names = {
            item.get("name") for item in noise_tools
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
        noise_penalties = 0
        for index, spec in enumerate(metric_implementations):
            if not isinstance(spec, dict):
                continue
            metric = metric_by_id.get(spec.get("metric_id"), {})
            mapping = spec.get("score_mapping", {})
            if metric.get("category") == "penalty":
                if mapping.get("pass") != 0 or not isinstance(mapping.get("fail"), (int, float)) or mapping.get("fail") >= 0:
                    raise PipelineGenerationError(
                        f"task readiness: penalty metric {spec.get('metric_id')} has reversed score mapping"
                    )
            elif isinstance(mapping, dict) and mapping.get("pass", 0) < mapping.get("fail", 0):
                raise PipelineGenerationError(
                    f"task readiness: positive metric {spec.get('metric_id')} has reversed score mapping"
                )
            if spec.get("source") == "final_agent_response" and spec.get("path") not in {"$", ""}:
                raise PipelineGenerationError(
                    f"task readiness: metric_implementations[{index}] addresses fields on raw final_agent_response"
                )
            expected = spec.get("expected")
            targets_noise = (
                spec.get("operator") == "none_tool_calls"
                or expected in noise_names
                or (isinstance(expected, list) and bool(set(expected) & noise_names))
            )
            if metric.get("category") == "penalty" and targets_noise:
                noise_penalties += 1
        if noise_names and noise_penalties != 1:
            raise PipelineGenerationError(
                f"task readiness: expected exactly one executable noise penalty, found {noise_penalties}"
            )

    @staticmethod
    def _validate_key_steps(key_steps: list[Any], actions: list[Any]) -> None:
        action_names = {
            str(action.get("name") or action.get("action"))
            for action in actions
            if isinstance(action, dict)
        }
        step_ids: set[str] = set()
        for index, step in enumerate(key_steps):
            if not isinstance(step, dict):
                raise PipelineGenerationError(f"key_steps[{index}] must be an object")
            step_id = step.get("step_id")
            action_name = step.get("action_name")
            if not isinstance(step_id, str) or not step_id.strip() or step_id in step_ids:
                raise PipelineGenerationError(f"key_steps[{index}] has invalid or duplicate step_id")
            if not isinstance(action_name, str) or action_name not in action_names:
                raise PipelineGenerationError(f"key_steps[{index}] references an unknown action")
            if not isinstance(step.get("rationale"), str) or not step["rationale"].strip():
                raise PipelineGenerationError(f"key_steps[{index}] requires rationale")
            if not isinstance(step.get("required_for_goal"), bool):
                raise PipelineGenerationError(f"key_steps[{index}] requires boolean required_for_goal")
            if not isinstance(step.get("dependencies", []), list):
                raise PipelineGenerationError(f"key_steps[{index}].dependencies must be a list")
            step_ids.add(step_id)

    @staticmethod
    def _validate_metrics(metrics: list[Any], *, key_steps: list[Any] | None = None) -> None:
        ids: set[str] = set()
        category_weights = {"process": 0.0, "outcome": 0.0, "penalty": 0.0}
        key_action_names = {
            str(step.get("action_name"))
            for step in (key_steps or [])
            if isinstance(step, dict) and isinstance(step.get("action_name"), str)
        }
        for index, metric in enumerate(metrics):
            if not isinstance(metric, dict):
                raise PipelineGenerationError(f"metrics[{index}] must be an object")
            metric_id = metric.get("id")
            if not isinstance(metric_id, str) or not metric_id.strip() or metric_id in ids:
                raise PipelineGenerationError(f"metrics[{index}] has invalid or duplicate id")
            category = metric.get("category")
            if category not in category_weights:
                raise PipelineGenerationError(
                    f"metrics[{index}] has invalid category: {category!r}; "
                    "expected one of 'process', 'outcome', 'penalty'"
                )
            metric_type = metric.get("type")
            if metric_type not in {"rule-based", "model-based", "hybrid"}:
                raise PipelineGenerationError(f"metrics[{index}] has invalid type")
            if metric.get("scope") not in {"step", "state", "terminal", "trajectory"}:
                raise PipelineGenerationError(f"metrics[{index}] has invalid scope")
            if not isinstance(metric.get("rubric"), str) or not metric["rubric"].strip():
                raise PipelineGenerationError(f"metrics[{index}] requires rubric")
            weight = metric.get("weight")
            if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
                raise PipelineGenerationError(f"metrics[{index}] weight must be a finite non-negative number")
            score_range = metric.get("score_range")
            expected_range = [-1, 0] if category == "penalty" else [0, 1]
            if score_range != expected_range:
                raise PipelineGenerationError(f"metrics[{index}] score_range must be {expected_range}")
            evaluator = metric.get("evaluator")
            if not isinstance(evaluator, dict):
                raise PipelineGenerationError(f"metrics[{index}] requires executable evaluator")
            if not isinstance(evaluator.get("kind"), str) or not evaluator["kind"].strip():
                raise PipelineGenerationError(f"metrics[{index}].evaluator requires kind")
            if evaluator.get("source") not in {"runtime_rule", "external_llm"}:
                raise PipelineGenerationError(
                    f"metrics[{index}].evaluator.source must be runtime_rule or external_llm"
                )
            score_mapping = evaluator.get("score_mapping")
            if not isinstance(score_mapping, dict) or not score_mapping:
                raise PipelineGenerationError(f"metrics[{index}].evaluator requires score_mapping")
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < expected_range[0]
                or float(value) > expected_range[1]
                for value in score_mapping.values()
            ):
                raise PipelineGenerationError(
                    f"metrics[{index}].evaluator.score_mapping values must be within {expected_range}"
                )
            if category == "process":
                if (
                    evaluator.get("kind") != "hybrid_tool_call"
                    or evaluator.get("source") != "external_llm"
                    or evaluator.get("comparison") != "exact_tool_name_and_canonical_arguments"
                ):
                    raise PipelineGenerationError(
                        f"metrics[{index}] process evaluator must use external exact tool-call comparison"
                    )
                if set(evaluator.get("score_mapping", {})) != {"match", "mismatch"}:
                    raise PipelineGenerationError(
                        f"metrics[{index}] process evaluator score_mapping must contain match and mismatch"
                    )
            elif metric_type == "rule-based":
                if evaluator.get("kind") not in {
                    "business_state_rule", "document_rule", "trajectory_rule"
                } or evaluator.get("source") != "runtime_rule":
                    raise PipelineGenerationError(
                        f"metrics[{index}] rule-based evaluator must be an executable runtime rule"
                    )
                if not isinstance(evaluator.get("assertion"), str) or not evaluator["assertion"].strip():
                    raise PipelineGenerationError(f"metrics[{index}] rule-based evaluator requires assertion")
            elif metric_type == "model-based":
                if evaluator.get("kind") != "external_llm_judge" or evaluator.get("source") != "external_llm":
                    raise PipelineGenerationError(
                        f"metrics[{index}] model-based evaluator must use external_llm_judge"
                    )
            elif metric_type == "hybrid":
                if evaluator.get("kind") != "hybrid_outcome" or evaluator.get("source") != "external_llm":
                    raise PipelineGenerationError(
                        f"metrics[{index}] hybrid non-process evaluator must use hybrid_outcome"
                    )
                if not isinstance(evaluator.get("rule"), dict) or not isinstance(evaluator.get("external_llm"), dict):
                    raise PipelineGenerationError(
                        f"metrics[{index}] hybrid evaluator requires rule and external_llm sections"
                    )
            evaluation_inputs = metric.get("evaluation_inputs")
            if isinstance(evaluation_inputs, list):
                simulated_inputs = {
                    "user_profiles",
                    "user_scripts",
                    "dialogue_sessions",
                    "simulated_dialogue",
                    "user_simulation",
                }
                forbidden = sorted(
                    str(item) for item in evaluation_inputs if str(item) in simulated_inputs
                )
                if forbidden:
                    raise PipelineGenerationError(
                        f"metrics[{index}] cannot depend on generated simulation inputs: {forbidden}"
                    )
            if category == "process":
                if metric_type != "hybrid":
                    raise PipelineGenerationError("process metrics must be hybrid")
                if not isinstance(metric.get("target_action"), str) or not metric["target_action"].strip():
                    raise PipelineGenerationError(f"metrics[{index}] process metric requires target_action")
                if key_steps is not None and metric["target_action"] not in key_action_names:
                    raise PipelineGenerationError(
                        f"metrics[{index}] process target_action must reference a key step"
                    )
                if metric.get("condition") != "llm_expected_tool_call_exact_match":
                    raise PipelineGenerationError(
                        f"metrics[{index}] process condition must be llm_expected_tool_call_exact_match"
                    )
            if metric_type in {"model-based", "hybrid"}:
                if not isinstance(metric.get("evaluation_inputs"), list) or not metric["evaluation_inputs"]:
                    raise PipelineGenerationError(f"metrics[{index}] requires evaluation_inputs")
                if not isinstance(metric.get("criteria"), list) or not metric["criteria"]:
                    raise PipelineGenerationError(f"metrics[{index}] requires criteria")
            if metric_type in {"rule-based", "hybrid"}:
                if not isinstance(metric.get("condition"), str) or not metric["condition"].strip():
                    raise PipelineGenerationError(f"metrics[{index}] requires condition")
            category_weights[category] += float(weight)
            ids.add(metric_id)
        if not category_weights["outcome"]:
            raise PipelineGenerationError("metrics must include an outcome metric")
        if not category_weights["penalty"]:
            raise PipelineGenerationError("metrics must include a penalty metric")
        positive_weight_sum = category_weights["process"] + category_weights["outcome"]
        if not math.isclose(positive_weight_sum, 1.0, abs_tol=1e-6):
            raise PipelineGenerationError(f"positive metric weights must sum to 1, got {positive_weight_sum}")
        if not math.isclose(category_weights["penalty"], 1.0, abs_tol=1e-6):
            raise PipelineGenerationError(f"penalty metric weights must sum to 1, got {category_weights['penalty']}")
        if category_weights["outcome"] <= category_weights["process"]:
            raise PipelineGenerationError("outcome metric weight must exceed process metric weight")

    @staticmethod
    def _validate_metrics_for_environment(
        metrics: list[Any], *, environment_mode: str
    ) -> None:
        if environment_mode != "stateless":
            return
        for index, metric in enumerate(metrics):
            if not isinstance(metric, dict):
                continue
            if metric.get("category") == "process":
                raise PipelineGenerationError(
                    f"metrics[{index}] stateless tasks must not reward tool processes"
                )
            evaluator = metric.get("evaluator", {})
            if isinstance(evaluator, dict) and evaluator.get("kind") == "business_state_rule":
                raise PipelineGenerationError(
                    f"metrics[{index}] stateless tasks cannot depend on business state"
                )
            inputs = metric.get("evaluation_inputs", [])
            if isinstance(inputs, list) and "business_data" in inputs:
                raise PipelineGenerationError(
                    f"metrics[{index}] stateless tasks cannot evaluate business_data"
                )

    @staticmethod
    def _validate_metric_implementations(specs: list[Any], metrics: list[Any]) -> None:
        by_id = {
            metric.get("id"): metric for metric in metrics
            if isinstance(metric, dict) and isinstance(metric.get("id"), str)
        }
        seen: set[str] = set()
        allowed_sources = {"business_state", "trajectory", "final_agent_response", "observation"}
        allowed_operators = {"eq", "ne", "gte", "lte", "contains", "exists", "count_gte", "count_eq", "none_tool_calls"}
        for index, spec in enumerate(specs):
            if not isinstance(spec, dict):
                raise PipelineGenerationError(f"metric_implementations[{index}] must be an object")
            metric_id = spec.get("metric_id")
            metric = by_id.get(metric_id)
            if metric is None or metric_id in seen or metric.get("type") != "rule-based":
                raise PipelineGenerationError(f"metric_implementations[{index}] references a non-rule metric")
            if spec.get("source") not in allowed_sources or spec.get("operator") not in allowed_operators:
                raise PipelineGenerationError(f"metric_implementations[{index}] source/operator is invalid")
            if not isinstance(spec.get("path"), str) or not spec["path"]:
                raise PipelineGenerationError(f"metric_implementations[{index}] path is invalid")
            if spec.get("source") == "final_agent_response" and spec.get("path") not in {"$", ""}:
                raise PipelineGenerationError(
                    f"metric_implementations[{index}] cannot address fields on raw final_agent_response"
                )
            mapping = spec.get("score_mapping")
            low, high = metric.get("score_range", [0, 1])
            if (
                not isinstance(mapping, dict) or set(mapping) != {"pass", "fail"}
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high for value in mapping.values())
            ):
                raise PipelineGenerationError(f"metric_implementations[{index}] score_mapping is invalid")
            seen.add(metric_id)
        required = {
            metric_id for metric_id, metric in by_id.items()
            if metric.get("type") == "rule-based"
        }
        if seen != required:
            raise PipelineGenerationError(
                f"rule-based metrics require executable implementations; missing={sorted(required - seen)}"
            )

    @staticmethod
    def _promote_unimplemented_rule_metrics(
        metrics: list[Any], specs: list[Any], *, environment_mode: str | None = None
    ) -> None:
        """Never leave a declared rule metric without executable semantics.

        A rule that could not be compiled to the constrained DSL becomes an
        external judge.  This is slower, but preserves evaluability and is
        safer than asking the sandbox builder to interpret free-form text.
        """
        implemented = {
            spec.get("metric_id") for spec in specs if isinstance(spec, dict)
        }
        for metric in metrics:
            if not isinstance(metric, dict) or metric.get("type") != "rule-based":
                continue
            metric_id = metric.get("id")
            if metric_id in implemented:
                continue
            evaluator = metric.get("evaluator")
            assertion = evaluator.get("assertion") if isinstance(evaluator, dict) else None
            criterion = assertion if isinstance(assertion, str) and assertion.strip() else metric.get("rubric")
            scope = metric.get("scope")
            if scope == "trajectory":
                inputs = ["conversation", "tool_call", "tool_results", "final_agent_response"]
            elif scope == "terminal":
                inputs = ["final_agent_response", "terminal_observation"]
                if environment_mode != "stateless":
                    inputs.insert(1, "business_data")
            else:
                inputs = ["public_observation", "tool_results"]
                if environment_mode != "stateless":
                    inputs.insert(1, "business_data")
            metric["type"] = "model-based"
            metric["evaluation_inputs"] = inputs
            metric["criteria"] = [str(criterion or "判断该指标是否满足。")]
            metric["evaluator"] = {
                "kind": "external_llm_judge",
                "source": "external_llm",
                "score_mapping": dict(evaluator.get("score_mapping", {}))
                if isinstance(evaluator, dict) else {"pass": 1, "fail": 0},
            }
            logger.info("promoted uncompiled rule metric to external judge: metric=%s", metric_id)

    @staticmethod
    def _validate_reward_formula(formula: Any, metrics: list[Any]) -> None:
        if not isinstance(formula, dict):
            raise PipelineGenerationError("reward_formula must be an object")
        if formula.get("type") != "separate_sign_weighted_sum":
            raise PipelineGenerationError("reward_formula.type must be separate_sign_weighted_sum")
        if not isinstance(formula.get("formula"), str) or not formula["formula"].strip():
            raise PipelineGenerationError("reward_formula.formula must be non-empty")
        if formula.get("score_range") != [-1, 1]:
            raise PipelineGenerationError("reward_formula.score_range must be [-1, 1]")
        if formula.get("positive_weight_sum") != 1 or formula.get("negative_weight_sum") != 1:
            raise PipelineGenerationError("reward_formula positive and negative weights must each sum to 1")

    @staticmethod
    def _canonical_reward_formula(metrics: list[Any]) -> dict[str, Any]:
        """Return a bounded formula derived from the validated metric weights.

        Process and outcome are one positive feedback pool. Penalty is the
        independent negative feedback pool. Dividing process and outcome into
        separate averages would make the positive contribution reach 2, so
        the canonical formula uses the already-normalized metric weights once.
        """
        positive_terms: list[str] = []
        negative_terms: list[str] = []
        for metric in metrics:
            metric_id = str(metric["id"])
            term = f"{float(metric['weight']):.12g}*score({metric_id})"
            if metric["category"] in {"process", "outcome"}:
                positive_terms.append(term)
            else:
                negative_terms.append(term)
        positive_expression = " + ".join(positive_terms) or "0"
        negative_expression = " + ".join(negative_terms) or "0"
        return {
            "type": "separate_sign_weighted_sum",
            "formula": f"R = clip(({positive_expression}) + ({negative_expression}), -1, 1)",
            "positive_weight_sum": 1,
            "negative_weight_sum": 1,
            "score_range": [-1, 1],
            "positive_categories": ["process", "outcome"],
            "negative_categories": ["penalty"],
            "normalization": "metric weights are normalized once across all positive categories and independently across penalty metrics",
        }

    @staticmethod
    def _materialize_tools(tools: list[dict[str, Any]], artifact_dir: Path) -> dict[str, Any]:
        """Persist only standard OpenAI tool definitions.

        Bindings, Trainer actions, and implementation details deliberately stay
        outside this file. Code Agent consumes this schema and implements the
        corresponding sandbox behavior separately.
        """
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / "tools.json"
        path.write_text(
            json.dumps(tools, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return {
            "version": "1.0",
            "file": str(path.relative_to(artifact_dir)),
            "format": "openai_function_tools",
            "tool_count": len(tools),
        }

    @staticmethod
    def _validate_data_tables(tables: Any) -> None:
        if not isinstance(tables, list) or not tables:
            raise PipelineGenerationError("business data tables must be a non-empty list")
        TaskGenerationPipeline._validate_table_definitions(tables)
        names = {str(table["table_name"]) for table in tables}
        for index, table in enumerate(tables):
            columns = table["columns"]
            rows = table.get("rows")
            column_names = {str(column["name"]) for column in columns}
            if not isinstance(rows, list) or not rows:
                raise PipelineGenerationError(f"data_tables[{index}] requires complete non-empty rows")
            for row_index, row in enumerate(rows):
                if not isinstance(row, dict) or any(column not in row for column in column_names):
                    raise PipelineGenerationError(f"data_tables[{index}].rows[{row_index}] is not a complete row")

    @staticmethod
    def _validate_entities(entities: Any) -> None:
        if not isinstance(entities, list) or not entities:
            raise PipelineGenerationError("environment_entities.entities must be a non-empty list")
        ids: set[str] = set()
        for index, entity in enumerate(entities):
            if not isinstance(entity, dict):
                raise PipelineGenerationError(f"entities[{index}] must be an object")
            entity_id = entity.get("entity_id") or entity.get("name")
            if not isinstance(entity_id, str) or not entity_id.strip() or entity_id in ids:
                raise PipelineGenerationError(f"entities[{index}] has invalid or duplicate entity_id/name")
            if not isinstance(entity.get("description"), str) or not entity["description"].strip():
                raise PipelineGenerationError(f"entities[{index}] requires a description")
            if not isinstance(entity.get("required_facts", []), list):
                raise PipelineGenerationError(f"entities[{index}].required_facts must be a list")
            ids.add(entity_id)

    @staticmethod
    def _validate_table_definitions(tables: Any) -> None:
        if not isinstance(tables, list) or not tables:
            raise PipelineGenerationError("environment_table_design.tables must be a non-empty list")
        names: set[str] = set()
        for index, table in enumerate(tables):
            if not isinstance(table, dict):
                raise PipelineGenerationError(f"table definitions[{index}] must be an object")
            name = table.get("table_name")
            columns = table.get("columns")
            primary_key = table.get("primary_key")
            if not isinstance(name, str) or not name.strip() or name in names:
                raise PipelineGenerationError(f"table definitions[{index}] has invalid or duplicate table_name")
            if not isinstance(columns, list) or not columns:
                raise PipelineGenerationError(f"table definitions[{index}] requires columns")
            if not isinstance(primary_key, list) or not primary_key:
                raise PipelineGenerationError(f"table definitions[{index}] requires primary_key")
            column_names: set[str] = set()
            for column in columns:
                if not isinstance(column, dict) or not isinstance(column.get("name"), str) or not column["name"].strip():
                    raise PipelineGenerationError(f"table definitions[{index}] contains an invalid column")
                if column["name"] in column_names:
                    raise PipelineGenerationError(f"table definitions[{index}] contains duplicate columns")
                column_names.add(column["name"])
            if any(key not in column_names for key in primary_key):
                raise PipelineGenerationError(f"table definitions[{index}] primary_key references an unknown column")
            for field in ("foreign_keys", "indexes", "constraints"):
                if field in table and not isinstance(table[field], list):
                    raise PipelineGenerationError(f"table definitions[{index}].{field} must be a list")
            names.add(name)

    @staticmethod
    def _materialize_business_data(
        tables: list[dict[str, Any]],
        data_document: str,
        artifact_dir: Path,
        *,
        environment_mode: str = "business_data",
    ) -> dict[str, Any]:
        """Write table schemas/rows to files and return the compact task manifest."""
        artifact_dir.mkdir(parents=True, exist_ok=True)
        schema_dir = artifact_dir / "schemas"
        rows_dir = artifact_dir / "rows"
        schema_dir.mkdir(exist_ok=True)
        rows_dir.mkdir(exist_ok=True)
        used_names: set[str] = set()
        manifest_tables: list[dict[str, Any]] = []

        for index, table in enumerate(tables):
            raw_name = str(table["table_name"])
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_name).strip("._") or f"table_{index + 1}"
            if safe_name in used_names:
                safe_name = f"{safe_name}_{index + 1}"
            used_names.add(safe_name)
            schema = {key: value for key, value in table.items() if key != "rows"}
            schema_path = schema_dir / f"{safe_name}.json"
            rows_path = rows_dir / f"{safe_name}.jsonl"
            schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            with rows_path.open("w", encoding="utf-8") as handle:
                for row in table["rows"]:
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            manifest_tables.append({
                "table_name": raw_name,
                "schema_file": str(schema_path.relative_to(artifact_dir)),
                "rows_file": str(rows_path.relative_to(artifact_dir)),
                "row_count": len(table["rows"]),
                "primary_key": table["primary_key"],
            })

        document_path = artifact_dir / "data_document.md"
        document_path.write_text(data_document.rstrip() + "\n", encoding="utf-8")
        return {
            "version": "1.0",
            "environment_mode": environment_mode,
            "root": str(artifact_dir),
            "document_file": str(document_path.relative_to(artifact_dir)),
            "tables": manifest_tables,
        }

    @staticmethod
    def _materialize_user_simulation(
        profiles: list[Any],
        scripts: list[Any],
        sessions: list[dict[str, Any]],
        artifact_dir: Path,
    ) -> dict[str, Any]:
        """Persist user simulation inputs and sessions outside task.json."""
        sessions_dir = artifact_dir / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        profiles_path = artifact_dir / "user_profiles.json"
        scripts_path = artifact_dir / "user_scripts.json"
        profiles_path.write_text(json.dumps(profiles, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        scripts_path.write_text(json.dumps(scripts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        session_files: list[dict[str, str]] = []
        for index, session in enumerate(sessions):
            session_id = str(session.get("session_id") or f"session-{index + 1}")
            safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", session_id).strip("._") or f"session-{index + 1}"
            path = sessions_dir / f"{safe_id}.json"
            path.write_text(json.dumps(session, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            session_files.append({"session_id": session_id, "file": str(path.relative_to(artifact_dir))})
        return {
            "version": "1.0",
            "root": str(artifact_dir),
            "profiles_file": str(profiles_path.relative_to(artifact_dir)),
            "scripts_file": str(scripts_path.relative_to(artifact_dir)),
            "sessions": session_files,
        }

    @staticmethod
    def _environment_records(environment: dict[str, Any], actions: list[Any]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for item in environment.get("records", []):
            if isinstance(item, dict) and {"type", "field", "description", "value"} <= set(item):
                records.append({k: item[k] for k in ("type", "field", "description", "value")})
        for table in environment.get("data_tables", []):
            if isinstance(table, dict) and isinstance(table.get("table_name"), str):
                records.append({
                    "type": "business_table",
                    "field": table["table_name"],
                    "description": str(table.get("description") or "任务业务数据表"),
                    "value": table,
                })
        for action in actions:
            if isinstance(action, dict):
                name = action.get("name") or action.get("action")
                if isinstance(name, str) and name.strip():
                    records.append({"type": "action", "name": name, "description": action.get("description", name)})
        return records

"""Eight-stage external-LLM task generation pipeline.

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
    """Generate a complete task through eight independently prompted stages."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        script_count: int = 3,
        sessions_per_script: int = 2,
        minimum_dialogue_turns: int = 4,
        maximum_dialogue_turns: int = 12,
        retries: int = 3,
    ) -> None:
        if (
            script_count <= 0
            or sessions_per_script < 2
            or minimum_dialogue_turns < 2
            or maximum_dialogue_turns < minimum_dialogue_turns
            or retries <= 0
        ):
            raise ValueError("invalid pipeline counts or retries")
        self.llm = llm
        self.script_count = script_count
        self.sessions_per_script = sessions_per_script
        self.minimum_dialogue_turns = minimum_dialogue_turns
        self.maximum_dialogue_turns = maximum_dialogue_turns
        self.retries = retries

    def _call(self, stage: str, system: str, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        logger.info("pipeline stage started: stage=%s", stage)
        prompt = json.dumps(payload, ensure_ascii=False, indent=2)
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                response = self.llm.complete(
                    prompt,
                    system_prompt=system + "\n只输出合法 JSON 对象，不要解释。",
                    thinking=False,
                    temperature=0.2 if attempt > 1 else 0.5,
                    max_tokens=16_000,
                    response_format="json_object",
                )
                value = _json(response.content)
                if not isinstance(value, dict):
                    raise TypeError("stage result must be a JSON object")
                # Some OpenAI-compatible providers wrap structured output in
                # a single `output` object. Normalize that envelope once so
                # every pipeline stage can consume the same contract.
                if set(value) == {"output"} and isinstance(value.get("output"), dict):
                    value = value["output"]
                logger.info(
                    "pipeline stage completed: stage=%s attempt=%d duration_ms=%.1f result_keys=%s",
                    stage,
                    attempt,
                    (time.perf_counter() - started) * 1000,
                    sorted(value.keys()),
                )
                return value
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
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
        terminal_signal = "none"
        termination_reason: str | None = None
        user_signals = {"user_satisfied", "user_withdrew", "user_declined", "no_progress"}
        agent_signals = {"task_completed", "agent_completed", "no_progress"}

        while len(turns) < self.maximum_dialogue_turns:
            user_result = self._call(
                f"dialogue_sessions.{session_id}.user.{len(turns) + 1}",
                "你是 User LLM，只能按照用户画像和用户剧本扮演用户。用户剧本是嵌套决策树：根据当前对话和用户画像，"
                "选择一个 condition 与当前情况匹配的分支，沿 next 进入下一个节点；每轮只选择一个分支，不要把多个分支"
                "拼接成一条不符合树结构的行为。围绕当前任务进行口语化提问、补充、澄清、接受、拒绝、纠正、犹豫、沉默"
                "或表达不确定。不要替 Agent 完成任务，不要查看或猜测业务数据中的隐藏真值。只有在达到最少对话轮数后才输出终止信号。",
                {"task_description": description, "user_profile": profile, "user_script": script,
                 "conversation": turns, "minimum_turns": self.minimum_dialogue_turns,
                 "maximum_turns": self.maximum_dialogue_turns,
                 "output": {"message": "string", "terminal_signal": "none|user_satisfied|user_withdrew|user_declined|no_progress", "continue": True}},
            )
            # OpenAI-compatible chat providers commonly call the generated
            # text `content`; the pipeline contract uses `message`. Accept
            # both and normalize to the stored conversation format.
            message = user_result.get("message") or user_result.get("content")
            if not isinstance(message, str) or not message.strip():
                raise PipelineGenerationError(f"{session_id} user response must contain message/content")
            turns.append({"role": "user", "content": message.strip()})
            terminal_signal = str(user_result.get("terminal_signal") or "none")
            if len(turns) >= self.minimum_dialogue_turns and terminal_signal in user_signals:
                termination_reason = terminal_signal
                break
            if len(turns) >= self.maximum_dialogue_turns:
                termination_reason = "max_turns_reached"
                break

            agent_result = self._call(
                f"dialogue_sessions.{session_id}.agent.{len(turns) + 1}",
                "你是 Agent LLM。根据任务描述、公开业务环境数据、用户画像和当前对话回答用户。用户画像只用于调整表达方式和交互策略，不是业务事实来源；不要编造环境中不存在的数据，也不要把用户的话当作业务真值。回答应推进任务、提出必要澄清或给出基于环境数据的结果。只有在任务完成或确实无法推进时输出终止信号。",
                {"task_description": description, "environment": environment, "user_profile": profile, "conversation": turns,
                 "minimum_turns": self.minimum_dialogue_turns,
                 "maximum_turns": self.maximum_dialogue_turns,
                 "output": {"message": "string", "terminal_signal": "none|task_completed|agent_completed|no_progress", "continue": True}},
            )
            message = agent_result.get("message") or agent_result.get("content")
            if not isinstance(message, str) or not message.strip():
                raise PipelineGenerationError(f"{session_id} agent response must contain message/content")
            turns.append({"role": "agent", "content": message.strip()})
            terminal_signal = str(agent_result.get("terminal_signal") or "none")
            if len(turns) >= self.minimum_dialogue_turns and terminal_signal in agent_signals:
                termination_reason = terminal_signal
                break
            if len(turns) >= self.maximum_dialogue_turns:
                termination_reason = "max_turns_reached"
                break

        result = {
            "script_id": script_id,
            "session_id": session_id,
            "profile_id": profile.get("profile_id") if isinstance(profile, dict) else None,
            "turns": turns,
            "turn_count": len(turns),
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
        profile_terms: list[str],
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

        # 2a. Analyze business entities. This stage does not generate state or perform web search.
        entity_plan = self._call(
            "environment_entities",
            "根据任务描述分析完成任务所需的最小必要业务实体。只保留完成任务、支持 Agent 查询或修改、以及奖励评测真正需要持久化的业务事实；不需要独立查询、复用或更新的静态说明、标签和建议作为其他实体的字段或 JSON 保存。只有存在独立生命周期、独立查询/更新需求或明确业务关系时才拆分实体。输出实体、用途、必须保存的业务事实和实体关系，实体必须足够覆盖完整任务但遵循最小必要原则。",
            {"task_description": description, "task_type": task_type, "keywords": keywords,
             "profile_terms": profile_terms, "output": {"entities": [{"entity_id": "string", "name": "string", "description": "string", "required_facts": [], "relationships": []}]}},
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
                "根据已确认的单张表结构生成完整、真实、可直接插入数据库的业务数据。本阶段只返回 {\"rows\": [...]}；rows 必须非空，每行覆盖全部字段，主键唯一。",
                {"task_description": description, "entities": entity_plan["entities"], "table": table,
                 "output": {"rows": []}},
            )
            generated = None
            # The table design is authoritative; only rows cross this stage
            # boundary. Keep compatibility with older providers that returned
            # a table envelope while preferring the minimal rows-only contract.
            if isinstance(result.get("rows"), list):
                generated = table | {"rows": result["rows"]}
            if not isinstance(generated, dict):
                nested = result.get("table")
                if isinstance(nested, dict) and isinstance(nested.get("rows"), list):
                    generated = table | {"rows": nested["rows"]}
            if not isinstance(generated, dict):
                tables = result.get("tables")
                if isinstance(tables, list) and tables and isinstance(tables[0], dict) and isinstance(tables[0].get("rows"), list):
                    generated = table | {"rows": tables[0]["rows"]}
            if not isinstance(generated, dict) and isinstance(result.get("table_name"), str):
                generated = result
            if not isinstance(generated, dict):
                logger.error(
                    "environment table data invalid: table=%s result_keys=%s expected=table.rows",
                    table["table_name"],
                    sorted(result),
                )
                raise PipelineGenerationError(
                    f"environment_table_data.{table['table_name']} must return table.rows; "
                    f"received keys={sorted(result)}"
                )
            return generated

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
        materialized_dir = Path(artifact_dir) if artifact_dir is not None else Path("output/task_artifacts") / f"task-{uuid.uuid4().hex}"
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
            "分支必须围绕任务描述、实际业务实体、字段、公开记录和 Agent 可执行目标。可引用公开业务事实，"
            "但不得泄露隐藏真值、评测规则或实现细节。每个剧本提供非空 script_id 字符串。"
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
                    "next": {"node_id": "node-1", "user_behavior": "...", "branches": []},
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
            if session.get("termination_reason") not in {
                "user_satisfied", "task_completed", "user_withdrew", "user_declined",
                "agent_completed", "no_progress", "max_turns_reached",
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
        actions = self._call(
            "agent_actions",
            "根据任务描述、完整业务环境数据和模拟对话反推 Agent 必须做的原子动作。先分析任务目标依赖业务环境中的哪些实体、表、字段和业务事实，再分析模拟对话中 Agent 为什么能够作答：对话只作为行为证据，用于反推出 Agent 隐含执行的查询、读取、筛选、计算、比较、提交和用户交互，不要照搬 Agent 的自然语言回答或只把整段回答命名成一个动作。凡是回答依赖持久化业务数据的，必须拆出对应的查询/读取或业务处理动作；凡是仅根据用户输入和通用语言能力生成的最终表达，属于运行时回答生成，不作为 Trainer action。不要为未被任务目标或对话使用的表机械生成查询动作。持续细分每个动作，直到不存在更细粒度且仍需 Agent 独立选择的操作。每个输出动作必须是不可再拆的原子业务动作，并说明为什么不能继续拆分。inputs、outputs、preconditions、effects 使用当前任务的业务语义：inputs 只描述 Agent 执行动作时确实需要从公开 observation 或前序结果取得的输入，outputs 只描述动作产生的公开结果，不引入隐藏真值、内部主键、用户确认 ID 或预测答案输入。输出动作定义。",
            {"task_description": description, "environment": environment,
             "user_scripts": user_scripts, "dialogue_sessions": sessions,
             "output": {"actions": [{"name": "string", "description": "string", "atomicity_rationale": "string", "inputs": [{"name": "string", "description": "string"}], "outputs": [{"name": "string", "description": "string"}], "preconditions": ["string"], "effects": ["string"]}]}},
        )
        action_list = self._list(actions.get("actions"), "agent_actions")
        self._validate_actions(action_list)

        # 6. Generate strict OpenAI Function Tools from the atomic actions.
        tool_prompt = "根据任务、完整业务环境、模拟对话和已经确认的原子 Agent 动作，生成标准 OpenAI Function Tool 定义。工具设计要依据对话中的实际操作边界、信息来源和前序 observation，确定自然的工具名、描述、参数名、类型、枚举值、必填性和参数说明；参数必须与动作 inputs、业务环境和实际信息流一致。所有 Agent-用户交互动作统一为一个 ask_user 工具，其 parameters 是描述交互请求的 JSON Object schema；其他原子业务动作按业务语义分别生成工具。原子动作无需 Agent 输入时使用 properties={}、required=[]。如果全部动作都属于运行时直接回答，不需要工具调用，tools 可以是空数组；此时仍必须返回 tools 字段。输出标准工具 schema，并为顶层参数、嵌套对象属性和 array items schema 提供 description；数组对象同时定义 items.properties 和 items.required。工具 schema 不暴露隐藏真值、内部主键、数据库 ID 或用户确认 ID，只使用 observation 返回的公开引用或业务输入。"
        tool_payload = {"task_description": description, "environment": environment,
                        "user_scripts": user_scripts, "dialogue_sessions": sessions,
                        "agent_actions": actions,
                        "output": {"tools": []}}
        tool_list: list[dict[str, Any]] | None = None
        tool_bindings: list[dict[str, Any]] = []
        tool_error: PipelineGenerationError | None = None
        for tool_attempt in range(1, self.retries + 1):
            repair_hint = ""
            if tool_error is not None:
                repair_hint = f"\n上一版工具定义未通过校验，必须修复以下错误后重新输出完整结果：{tool_error}"
            tools = self._call(
                "openai_tools" if tool_attempt == 1 else "openai_tools.repair",
                tool_prompt + repair_hint,
                tool_payload,
            )
            try:
                candidate_tools = self._list(tools.get("tools"), "tools", minimum=0)
                self._validate_tools(candidate_tools)
                candidate_bindings = tools.get("tool_bindings", [])
                if not isinstance(candidate_bindings, list):
                    raise PipelineGenerationError("tool_bindings must be a list")
                for binding in candidate_bindings:
                    if not isinstance(binding, dict) or not all(isinstance(binding.get(key), str) and binding[key].strip() for key in ("tool_name", "trainer_action", "action_name")):
                        raise PipelineGenerationError("each tool binding needs tool_name, trainer_action and action_name")
                    if binding["tool_name"] not in {item["function"]["name"] for item in candidate_tools}:
                        raise PipelineGenerationError("tool binding references an unknown tool")
                    if binding["action_name"] not in {str(action.get("name") or action.get("action")) for action in action_list}:
                        raise PipelineGenerationError("tool binding references an unknown atomic action")
                tool_list = candidate_tools
                tool_bindings = candidate_bindings
                break
            except PipelineGenerationError as exc:
                tool_error = exc
                logger.warning("openai tool schema validation failed: attempt=%d/%d error=%s", tool_attempt, self.retries, exc)
        if tool_list is None:
            raise tool_error or PipelineGenerationError("openai tool generation failed")
        tools_manifest = self._materialize_tools(tool_list, materialized_dir)

        # 7. Design executable observations, rewards and terminal evaluation.
        reward_prompt = "根据任务、业务环境、原子动作和工具定义生成可执行的 observation 与 reward 设计。只保留与任务目标完成强相关的关键过程指标和目标结果指标，不为普通动作机械创建指标；如果任务无需关键工具动作或可直接生成答案，process 指标可以为空；如果存在关键工具动作，每个关键动作或关键动作链都可以有对应过程指标，不限制过程指标数量。观测指标不得依赖任务生成阶段的 user_profiles、user_scripts 或 dialogue_sessions 等模拟产物；evaluation_inputs 只能引用沙箱运行时实际产生的 conversation、public_observation、available_tools、tool_call、tool_results、business_data、final_document、terminal_observation 等输入。关键过程指标必须是 hybrid，并使用精简结构：声明 target_action、evaluation_inputs、criteria，以及固定 condition=llm_expected_tool_call_exact_match。沙箱 Code Agent 根据这个指标在实现评估器时调用外部 LLM，结合当前运行时 Context、可用工具和 criteria 生成期望的工具名与参数真值；然后由规则引擎对 Agent 实际 tool_call 的工具名和规范化参数进行确定性精确比对。任务 JSON 不要嵌入 LLM prompt、output_schema 或嵌套 expected-call 配置；不要用 LLM 直接给最终过程分数，也不要把工具选择错误或参数错误设计成 penalty。结果指标只判断任务目标是否完成或关键业务数据是否达到目标，优先使用可量化的业务数据变化；可使用 rule-based、model-based 或 hybrid。惩罚指标只有在直接影响任务目标时才保留，用于偏离用户诉求、无效循环或业务数据偏离预期，不评价工具选择或参数错误。每个 metric 包含 id、category、type、scope、rubric、weight 和 score_range；rule-based 或 hybrid 提供 condition，model-based 或 hybrid 提供 evaluation_inputs 和 criteria。process/outcome 分数范围为 [0,1]，penalty 分数范围为 [-1,0]；所有 process 与 outcome 指标的权重合计为 1，所有 penalty 指标的权重合计为 1，且 outcome 权重合计大于 process 权重合计。reward_formula 必须把所有 process/outcome 项放在同一个正反馈加权和中，把 penalty 项放在独立的负反馈加权和中，不得分别归一化 process 和 outcome；标准公式为 R = clip(sum(w_i*score_i, category in {process,outcome}) + sum(w_j*score_j, category == penalty), -1, 1)，该公式在分数范围和权重归一化成立时落在 [-1,1]。category 取 process、outcome、penalty，scope 取 step、state、terminal、trajectory。"
        reward_payload = {"task_description": description, "environment": environment,
                          "agent_actions": actions,
                          "tools": tool_list, "output": {"observation_schema": {}, "metrics": [{
                              "id": "process_key_tool_action", "category": "process", "type": "hybrid", "scope": "step",
                              "target_action": "key_tool_action",
                              "evaluation_inputs": ["recent_conversation", "public_observation", "tool_call", "tool_arguments"],
                              "criteria": ["当前上下文是否需要执行该关键动作"],
                              "condition": "llm_expected_tool_call_exact_match",
                              "score_range": [0, 1],
                              "weight": 0.2, "rubric": "过程动作选择正确"}],
                              "reward_formula": {"type": "separate_sign_weighted_sum", "formula": "R = clip(sum(w_i * score_i for category in {process,outcome}) + sum(w_j * score_j for category == penalty), -1, 1)", "positive_weight_sum": 1, "negative_weight_sum": 1, "score_range": [-1, 1]},
                              "termination": []}}
        rewards: dict[str, Any] | None = None
        reward_error: PipelineGenerationError | None = None
        for reward_attempt in range(1, self.retries + 1):
            repair_hint = ""
            if reward_error is not None:
                repair_hint = (
                    f"\n上一版 observation/reward 设计未通过校验，必须修复以下错误后重新输出完整结果：{reward_error}。"
                    "不要回显输入 payload，也不要返回 task_description、environment、agent_actions、tools 或 output 字段。"
                    "顶层只能返回 observation_schema、metrics、reward_formula、termination；"
                    "metrics 必须是非空数组，scope 只能是 step、state、terminal、trajectory，"
                    "category 只能是 process、outcome、penalty；process 指标只能对应关键动作，且 type 必须为 hybrid；"
                    "每个关键 process 指标必须包含非空 target_action、evaluation_inputs、criteria，且 condition 必须严格为 llm_expected_tool_call_exact_match；"
                    "每个 rule-based 或 hybrid 指标必须补充非空 condition；每个 model-based 或 hybrid 指标必须补充非空 evaluation_inputs 和 criteria 数组，"
                    "不能返回 criteria 缺失、空数组或 null；如果 rubric 已说明评价标准，也必须将其展开为 criteria 数组。"
                )
            candidate_rewards = self._call(
                "observations_rewards" if reward_attempt == 1 else "observations_rewards.repair",
                reward_prompt + repair_hint,
                reward_payload,
            )
            try:
                candidate_metrics = self._list(candidate_rewards.get("metrics"), "metrics")
                candidate_metrics = self._normalize_metric_weights(candidate_metrics)
                candidate_metrics = self._normalize_metric_evaluation_fields(candidate_metrics)
                candidate_rewards["metrics"] = candidate_metrics
                self._validate_metrics(candidate_metrics)
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

        requirements = description.get("requirements")
        if not isinstance(requirements, dict):
            requirements = {"input_modalities": ["text"]}
        requirements = dict(requirements)
        requirements.setdefault("media_truth_mode", "programmatic")
        complexity = description.get("complexity")
        if complexity not in {"simple", "standard", "complex"}:
            raise PipelineGenerationError("task_description.complexity must be simple, standard or complex")
        result = {
            "task": task_desc.strip(), "task_type": task_type, "task_intent": task_intent,
            "complexity": complexity,
            "requirements": requirements,
            "environment": self._environment_records(environment_summary, action_list),
            "media_generation": media_generation,
            "data_manifest": data_manifest,
            "user_simulation_manifest": user_simulation_manifest,
            "tools_manifest": tools_manifest,
            "actions": action_list, "tools": tool_list,
            "tool_bindings": tool_bindings,
            "observation_schema": rewards.get("observation_schema", {}), "metrics": rewards["metrics"],
            "reward_formula": rewards.get("reward_formula", {}), "termination": rewards.get("termination", []),
            "generation_pipeline": {"version": "1.0", "stages": [
                "task_description", "environment_entities", "environment_table_design",
                "environment_table_data", "environment_data_consistency", "environment_data_document",
                "environment_media_generation", "user_profiles", "user_scripts", "dialogue_sessions",
                "agent_actions", "openai_tools", "observations_rewards",
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
    def _validate_tools(tools: list[Any]) -> None:
        names: set[str] = set()
        for index, tool in enumerate(tools):
            function = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(tool, dict) or tool.get("type") != "function" or not isinstance(function, dict):
                raise PipelineGenerationError(f"tools[{index}] is not an OpenAI function tool")
            name = function.get("name")
            if not isinstance(name, str) or not name or name in names:
                raise PipelineGenerationError(f"tools[{index}].function.name is invalid or duplicated")
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
    def _validate_user_script_tree(script: dict[str, Any], index: int) -> None:
        """Validate the branching behavior tree consumed by User LLM."""
        tree = script.get("tree")
        if not isinstance(tree, dict):
            raise PipelineGenerationError(f"user_scripts[{index}] requires a tree object")
        seen_nodes: set[str] = set()
        seen_branches: set[str] = set()
        branch_count = 0

        def visit(node: Any, path: str, depth: int = 0) -> None:
            nonlocal branch_count
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
                next_node = branch.get("next")
                if not isinstance(branch_id, str) or not branch_id.strip() or branch_id in seen_branches:
                    raise PipelineGenerationError(f"user_scripts[{index}] branch {path}.{branch_index} requires a unique branch_id")
                if not isinstance(condition, str) or not condition.strip():
                    raise PipelineGenerationError(f"user_scripts[{index}] branch {branch_id} requires condition")
                if not isinstance(next_node, dict):
                    raise PipelineGenerationError(f"user_scripts[{index}] branch {branch_id} requires next node")
                seen_branches.add(branch_id)
                branch_count += 1
                visit(next_node, f"{path}.{branch_index}", depth + 1)

        visit(tree, "root")
        if branch_count < 2:
            raise PipelineGenerationError(f"user_scripts[{index}] tree must contain at least two branches")

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
    def _normalize_metric_evaluation_fields(metrics: list[Any]) -> list[Any]:
        """Repair a common LLM omission without changing evaluation semantics.

        Hybrid/model-based metrics need semantic criteria. When a provider
        returns a meaningful rubric but omits criteria, the rubric is the only
        safe deterministic fallback; the strict validator still rejects the
        metric when neither field contains usable text.
        """
        for index, metric in enumerate(metrics):
            if not isinstance(metric, dict):
                continue
            metric_type = metric.get("type")
            criteria = metric.get("criteria")
            if metric_type in {"model-based", "hybrid"} and (
                not isinstance(criteria, list) or not any(isinstance(item, str) and item.strip() for item in criteria)
            ):
                rubric = metric.get("rubric")
                if isinstance(rubric, str) and rubric.strip():
                    metric["criteria"] = [rubric.strip()]
                    logger.warning(
                        "filled missing metric criteria from rubric: metric_index=%d metric_id=%s",
                        index,
                        metric.get("id"),
                    )
        return metrics

    @staticmethod
    def _validate_metrics(metrics: list[Any]) -> None:
        ids: set[str] = set()
        category_weights = {"process": 0.0, "outcome": 0.0, "penalty": 0.0}
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

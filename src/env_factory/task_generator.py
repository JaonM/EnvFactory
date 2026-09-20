"""Generate long-horizon task descriptions from Scene graph paths."""

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .graph_builder import Neo4jGraphStore
from .knowledge_graph import SceneNode, TaskType
from .llm import LLMClient
from .task import Task
from .task_pipeline import TaskGenerationPipeline

logger = logging.getLogger(__name__)


class TaskGenerationError(ValueError):
    """Raised when a graph path or LLM task response is invalid."""



class TaskGenerator:
    """Sample event-element paths and turn their keywords into tasks."""

    INTENTS = (
        "query",
        "explain",
        "compare",
        "recommend",
        "diagnose",
        "modify",
        "execute",
        "plan",
        "summarize",
        "create",
        "extract",
        "classify",
        "validate",
        "audit",
        "calculate",
        "estimate",
        "schedule",
        "monitor",
        "troubleshoot",
        "transform",
        "decide",
        "simulate",
    )

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
    def __init__(
        self,
        store: Neo4jGraphStore,
        llm: LLMClient,
        *,
        hierarchy_child_limit: int = 10,
        hierarchy_workers: int = 4,
        user_script_count: int = 3,
        sessions_per_script: int = 2,
        minimum_dialogue_turns: int = 4,
        maximum_dialogue_turns: int = 12,
    ) -> None:
        if hierarchy_child_limit <= 0 or hierarchy_workers <= 0:
            raise ValueError("hierarchy_child_limit and hierarchy_workers must be greater than zero")
        self.store = store
        self.llm = llm
        self.hierarchy_child_limit = hierarchy_child_limit
        self.hierarchy_workers = hierarchy_workers
        self.pipeline = TaskGenerationPipeline(
            llm,
            script_count=user_script_count,
            sessions_per_script=sessions_per_script,
            minimum_dialogue_turns=minimum_dialogue_turns,
            maximum_dialogue_turns=maximum_dialogue_turns,
        )

    def generate(
        self,
        hops: int = 3,
        task_type: TaskType | str | None = None,
        task_style: str | None = None,
        artifact_dir: str | Path | None = None,
        task_intent: str | None = None,
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
        selected_intent = task_intent or random.choice(self.INTENTS)
        if selected_intent not in self.INTENTS:
            raise ValueError(f"unsupported task_intent: {selected_intent}")
        profile_terms = self._profile_terms(keywords)
        artifacts = self.pipeline.generate(
            keywords=list(keywords),
            task_type=selected_type.value,
            style=selected_style,
            task_intent=selected_intent,
            profile_terms=list(profile_terms),
            graph_context={"hops": selected_hops, "nodes": [node.name for node in path]},
            artifact_dir=artifact_dir,
        )
        if artifacts.get("complexity") not in {"simple", "standard", "complex"}:
            raise TaskGenerationError("task description returned an invalid complexity")
        return Task(
            desc=artifacts["task"], env=artifacts["environment"], metrics=artifacts["metrics"],
            task_type=selected_type, task_intent=artifacts.get("task_intent", selected_intent), complexity=artifacts["complexity"],
            artifacts=artifacts,
        )

    @staticmethod
    def _normalize_requirements(
        requirements: Any,
    ) -> dict[str, Any]:
        """Normalize explicit task requirements without guessing from prose."""
        raw = requirements if isinstance(requirements, dict) else {}
        allowed_modalities = {"text", "image", "audio", "video", "file", "structured_data"}
        modalities = raw.get("input_modalities", ["text"])
        if not isinstance(modalities, list):
            modalities = ["text"]
        modalities = list(dict.fromkeys(
            item for item in modalities if isinstance(item, str) and item in allowed_modalities
        )) or ["text"]

        def flag(name: str, fallback: bool = False) -> bool:
            value = raw.get(name, fallback)
            return value is True

        raw_interaction = raw.get("interaction")
        if not isinstance(raw_interaction, dict):
            # Accept older task JSON while normalizing it into the single
            # interaction contract. This is input compatibility, not a second
            # generation path.
            prior_multi_turn = flag("requires_multi_turn")
            prior_required = flag(
                "requires_user_interaction",
                flag("requires_clarification") or prior_multi_turn,
            )
            raw_interaction = {
                "required": prior_required,
                "mode": "multi_turn" if prior_multi_turn else ("single_turn" if prior_required else "none"),
                "triggers": ["clarification"] if flag("requires_clarification") else [],
            }
        mode = raw_interaction.get("mode")
        if mode not in {"none", "single_turn", "multi_turn"}:
            mode = "single_turn" if raw_interaction.get("required") else "none"
        required = raw_interaction.get("required") is True or mode != "none"
        if not required:
            mode = "none"
        interaction = {
            "required": required,
            "mode": mode,
            "triggers": list(dict.fromkeys(
                item for item in raw_interaction.get("triggers", [])
                if isinstance(item, str) and item.strip()
            )),
        }

        return {
            "input_modalities": modalities,
            "media_truth_mode": "programmatic" if any(
                item in {"image", "audio", "video", "file"} for item in modalities
            ) else "none",
            "interaction": interaction,
            "requires_external_tool": flag(
                "requires_external_tool",
                flag("requires_tool"),
            ),
            "requires_scheduling": flag("requires_scheduling"),
            "requires_clarification": flag("requires_clarification"),
        }

    @staticmethod
    def _build_constraints(
        *,
        task_desc: str,
        environment: list[dict[str, Any]],
        metrics: list[dict[str, Any]],
        intent: dict[str, Any],
        requirements: Any,
    ) -> dict[str, Any]:
        """Build machine-readable obligations for downstream sandbox agents.

        This is deliberately generated outside the sandbox Code Agent.  The
        natural-language spec remains useful for design, but these invariants
        are the stable contract used by the outer workflow for validation.
        """
        action_records = [item for item in environment if item.get("type") == "action"]
        action_names = [
            str(item.get("name") or item.get("field"))
            for item in action_records
            if isinstance(item.get("name") or item.get("field"), str)
        ]
        normalized_requirements = TaskGenerator._normalize_requirements(requirements)
        media_required = any(
            modality in {"image", "audio", "video", "file"}
            for modality in normalized_requirements["input_modalities"]
        )
        media_truth_mode = normalized_requirements["media_truth_mode"]
        requires_clarification = normalized_requirements["requires_clarification"]
        requires_multi_turn = normalized_requirements["interaction"]["mode"] == "multi_turn"
        requires_scheduling = normalized_requirements["requires_scheduling"]
        requires_external_tool = normalized_requirements["requires_external_tool"]
        intent_constraints = intent.get("constraints", [])
        if isinstance(intent_constraints, dict):
            intent_constraints = [intent_constraints]
        if not isinstance(intent_constraints, list):
            intent_constraints = []

        obligations = [
            {"id": "platform.persistence", "kind": "platform", "required": True},
            {"id": "platform.tool_schema", "kind": "platform", "required": True},
            {"id": "platform.hidden_truth_isolation", "kind": "platform", "required": True},
            {"id": "platform.user_simulator", "kind": "platform", "required": True},
            {"id": "platform.reward_interface", "kind": "platform", "required": True},
            *[
                {"id": f"task_action.{name}", "kind": "task_action", "required": True}
                for name in action_names
            ],
            *[
                {"id": f"capability.{capability_id}", "kind": "capability", "required": True}
                for capability_id, enabled in (
                    ("external_model_boundary", True),
                    ("user_clarification", requires_clarification),
                    ("multi_turn_interaction", requires_multi_turn),
                    ("scheduling", requires_scheduling),
                    ("external_business_tool", requires_external_tool),
                )
                if enabled
            ],
            {"id": "evaluation.reward_and_termination", "kind": "evaluation", "required": True},
        ]
        return {
            "schema_version": "1.0",
            "authority": "env_factory_outer_workflow",
            "mutable_by_code_agent": False,
            "obligations": obligations,
            "source": {
                "task_environment": True,
                "task_metrics": True,
                "intent_constraints": intent_constraints,
                "requirements": normalized_requirements,
            },
            "requirements": normalized_requirements,
            "platform": {
                "persistence": {"required": True, "episode_isolation": True},
                "tool_schema": "openai_function",
                "tool_trainer_action_mapping": "one_to_one",
                "hidden_truth_isolation": True,
                "runtime_credentials": "external_only",
                "user_simulator": {"required": True, "only_trigger": "ask_user"},
                "reward_interface": {"required": True},
                "media": {
                    "truth_mode": media_truth_mode,
                    "evaluation": "process_and_task_goal_only",
                    "multimodal_model_allowed": False,
                    "generation_source": "artifacts.media_generation",
                    "generation_owner": "code_agent_build_phase",
                },
            },
            "task_actions": [
                {
                    "name": str(item.get("name") or item.get("field")),
                    "description": str(item.get("description", "")),
                    "must_have": [
                        "preconditions", "observable_inputs", "observable_outputs",
                        "state_effects", "persistence_effects", "failure_behavior",
                        "acceptance_test",
                    ],
                }
                for item in action_records
            ],
            "capabilities": [
                {
                    "id": "external_model_boundary",
                    "required": True,
                    "detected": True,
                    "reason": "runtime user/data simulation uses externally supplied model credentials",
                    "credentials": "SANDBOX_LLM_API_KEY",
                    "owner": "sandbox_runtime_not_code_agent",
                },
                {
                    "id": "user_clarification",
                    "required": requires_clarification,
                    "detected": requires_clarification,
                    "reason": "task complexity explicitly requires clarification" if requires_clarification else "task does not require clarification",
                },
                {
                    "id": "multi_turn_interaction",
                    "required": requires_multi_turn,
                    "detected": requires_multi_turn,
                    "reason": "task requires multiple user turns" if requires_multi_turn else "task is not marked multi-turn",
                },
                {
                    "id": "scheduling",
                    "required": requires_scheduling,
                    "detected": requires_scheduling,
                    "reason": "task requires scheduling" if requires_scheduling else "task does not require scheduling",
                },
                {
                    "id": "external_business_tool",
                    "required": requires_external_tool,
                    "detected": requires_external_tool,
                    "reason": "task declares external tool use" if requires_external_tool else "task does not declare external tool use",
                },
            ],
            "evaluation": {
                "metric_ids": [str(metric.get("id")) for metric in metrics],
                "must_not_reward_from_agent_input": True,
                "must_not_expose_hidden_state": True,
                "media": {
                    "truth_mode": media_truth_mode,
                    "evaluation": "process_and_task_goal_only",
                    "media_content_is_not_evaluated": True,
                },
            },
        }

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

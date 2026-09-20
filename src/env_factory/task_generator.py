"""Generate long-horizon task descriptions from Scene graph paths."""

import logging
import random
import time
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
        user_script_count: int = 3,
        sessions_per_script: int = 2,
        minimum_dialogue_turns: int = 4,
        maximum_dialogue_turns: int = 12,
    ) -> None:
        self.store = store
        self.llm = llm
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
        artifacts = self.pipeline.generate(
            keywords=list(keywords),
            task_type=selected_type.value,
            style=selected_style,
            task_intent=selected_intent,
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

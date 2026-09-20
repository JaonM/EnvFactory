"""Generate one long-horizon task from the Scene graph."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

from env_factory import (
    LLMClient,
    Neo4jGraphStore,
    TaskGenerationError,
    TaskGenerator,
    TaskType,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="从知识图谱生成长程任务")
    parser.add_argument("--hops", type=int, default=3, help="随机路径最大跳数，实际范围为 0 到该值，默认 3")
    parser.add_argument("--count", type=int, default=1, help="生成任务数量，默认 1")
    parser.add_argument("--max-workers", type=int, default=4, help="任务生成并发数，默认 4")
    parser.add_argument(
        "--hierarchy-child-limit",
        type=int,
        default=10,
        help="每个路径关键词最多抽取的直接下位词数量，默认 10",
    )
    parser.add_argument(
        "--path-query-timeout",
        type=float,
        default=10.0,
        help="Neo4j 随机路径查询超时时间（秒），默认 10.0",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output"),
        help="任务输出根路径；最终文件写入 output/task_artifacts/task-N/task.json，默认 output",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("output/task_generation.log"),
        help="任务生成日志文件，默认 output/task_generation.log",
    )
    parser.add_argument(
        "--task-type",
        help="任务类型；支持逗号分隔多选，例如 QA,Event；默认从全部类型随机选择",
    )
    parser.add_argument(
        "--task-style",
        choices=TaskGenerator.STYLES,
        help="任务表达风格；默认随机选择",
    )
    parser.add_argument(
        "--task-intent",
        choices=TaskGenerator.INTENTS,
        help="任务意图；默认随机选择。可选：query、explain、compare、recommend、diagnose、modify、execute、plan、summarize、create、extract、classify、validate、audit、calculate、estimate、schedule、monitor、troubleshoot、transform、decide、simulate",
    )
    parser.add_argument(
        "--user-script-count",
        type=int,
        default=3,
        help="八阶段流程生成的用户剧本数量，默认 3",
    )
    parser.add_argument(
        "--sessions-per-script",
        type=int,
        default=2,
        help="每个用户剧本生成的多轮对话 session 数，至少 2，默认 2",
    )
    parser.add_argument(
        "--max-dialogue-turns",
        type=int,
        default=12,
        help="每个对话 session 的最大消息数，默认 12；达到后以 max_turns_reached 结束",
    )
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count 必须大于 0")
    if args.max_workers <= 0:
        parser.error("--max-workers 必须大于 0")
    if args.hierarchy_child_limit <= 0:
        parser.error("--hierarchy-child-limit 必须大于 0")
    if args.path_query_timeout <= 0:
        parser.error("--path-query-timeout 必须大于 0")
    if args.max_dialogue_turns < 4:
        parser.error("--max-dialogue-turns 必须至少为 4")
    load_dotenv()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(threadName)s %(name)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.log_file, encoding="utf-8")],
    )
    llm = LLMClient.from_env("LLM", timeout=float(os.getenv("LLM_TIMEOUT", "60")))
    logging.getLogger(__name__).info(
        "task generation run started: count=%d max_workers=%d output=%s log_file=%s",
        args.count, args.max_workers, args.output, args.log_file,
    )
    task_root = args.output if args.output.suffix == "" else args.output.parent
    task_root.mkdir(parents=True, exist_ok=True)
    with Neo4jGraphStore(
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
        path_query_timeout=args.path_query_timeout,
    ) as store:
        generator = TaskGenerator(
            store,
            llm,
            hierarchy_child_limit=args.hierarchy_child_limit,
            hierarchy_workers=args.max_workers,
            user_script_count=args.user_script_count,
            sessions_per_script=args.sessions_per_script,
            maximum_dialogue_turns=args.max_dialogue_turns,
        )
        with ThreadPoolExecutor(max_workers=min(args.max_workers, args.count)) as executor:
            def generate_one(index: int):
                logging.getLogger(__name__).info("task generation started: task=%d/%d", index, args.count)
                task_dir = task_root / "task_artifacts" / f"task-{index}"
                # A rerun must not mix artifacts from previous attempts (for
                # example script_1 and script-1 sessions).
                if task_dir.exists():
                    shutil.rmtree(task_dir)
                return generator.generate(
                    args.hops,
                    args.task_type,
                    args.task_style,
                    artifact_dir=task_root / "task_artifacts" / f"task-{index}",
                    task_intent=args.task_intent,
                )

            futures = {
                executor.submit(generate_one, index): index
                for index in range(1, args.count + 1)
            }
            failed = 0
            for completed, future in enumerate(as_completed(futures), start=1):
                index = futures[future]
                try:
                    task = future.result()
                except TaskGenerationError as exc:
                    failed += 1
                    task_dir = task_root / "task_artifacts" / f"task-{index}"
                    if task_dir.exists():
                        shutil.rmtree(task_dir)
                    logging.getLogger(__name__).warning(
                        "task generation skipped: task=%d/%d reason=%s",
                        index, args.count, exc,
                    )
                    continue
                except Exception:
                    failed += 1
                    task_dir = task_root / "task_artifacts" / f"task-{index}"
                    if task_dir.exists():
                        shutil.rmtree(task_dir)
                    logging.getLogger(__name__).exception(
                        "task generation failed: task=%d/%d", index, args.count
                    )
                    continue
                task_dir = task_root / "task_artifacts" / f"task-{index}"
                task_path = task_dir / "task.json"
                pipeline_artifacts = task.artifacts or {}
                artifact_manifest = {
                    key: pipeline_artifacts[key]
                    for key in (
                        "data_manifest",
                        "user_simulation_manifest",
                        "tools_manifest",
                        "media_generation",
                        "generation_pipeline",
                    )
                    if key in pipeline_artifacts
                }
                task_path.write_text(
                    json.dumps(
                        {
                            "task": task.desc,
                            "task_type": task.task_type.value,
                            "task_intent": task.task_intent,
                            "complexity": task.complexity,
                            "requirements": pipeline_artifacts.get("requirements", {}),
                            "environment": task.env,
                            "actions": pipeline_artifacts.get("actions", []),
                            "tools": pipeline_artifacts.get("tools", []),
                            "observation_schema": pipeline_artifacts.get("observation_schema", {}),
                            "metrics": task.metrics,
                            "reward_formula": pipeline_artifacts.get("reward_formula", {}),
                            "termination": pipeline_artifacts.get("termination", []),
                            "artifacts": artifact_manifest,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                logging.getLogger(__name__).info(
                    "task artifact written: task=%d/%d complexity=%s task_file=%s",
                    index, args.count, task.complexity, task_path,
                )
                logging.getLogger(__name__).info(
                    "task generation progress: completed=%d/%d success=%d failed=%d",
                    completed, args.count, completed - failed, failed,
                )
    logging.getLogger(__name__).info(
        "task generation run completed: success=%d failed=%d output=%s",
        args.count - failed, failed, task_root,
    )
    print(f"任务生成结束：成功={args.count - failed}，失败={failed}，目录={task_root / 'task_artifacts'}，日志={args.log_file}")


if __name__ == "__main__":
    main()

"""Generate one long-horizon task from the Scene graph."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from env_factory import LLMClient, Neo4jGraphStore, TaskEnvironmentMode, TaskGenerator, TaskType


def main() -> None:
    parser = argparse.ArgumentParser(description="从知识图谱生成长程任务")
    parser.add_argument("--hops", type=int, default=3, help="随机路径最大跳数，实际范围为 1 到该值")
    parser.add_argument("--count", type=int, default=1, help="生成任务数量")
    parser.add_argument("--max-workers", type=int, default=4, help="任务生成并发数")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/tasks.jsonl"),
        help="任务输出文件，默认 output/tasks.jsonl",
    )
    parser.add_argument(
        "--task-type",
        choices=[task_type.value for task_type in TaskType],
        help="任务类型；不指定时随机选择",
    )
    parser.add_argument(
        "--environment-mode",
        choices=[mode.value for mode in TaskEnvironmentMode],
        default=TaskEnvironmentMode.RANDOM.value,
        help="任务环境完整度",
    )
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count 必须大于 0")
    if args.max_workers <= 0:
        parser.error("--max-workers 必须大于 0")
    load_dotenv()
    llm = LLMClient.from_env("LLM", timeout=float(os.getenv("LLM_TIMEOUT", "60")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with Neo4jGraphStore(database=os.getenv("NEO4J_DATABASE", "neo4j")) as store:
        generator = TaskGenerator(store, llm)
        with args.output.open("a", encoding="utf-8") as output_file:
            with ThreadPoolExecutor(max_workers=min(args.max_workers, args.count)) as executor:
                futures = [
                    executor.submit(
                        generator.generate,
                        args.hops,
                        args.task_type,
                        args.environment_mode,
                    )
                    for _ in range(args.count)
                ]
                for future in as_completed(futures):
                    task = future.result()
                    output_file.write(
                        json.dumps(
                            {"task": task.desc, "environment": task.env, "metrics": task.metrics},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    output_file.flush()
    print(f"已生成 {args.count} 个任务：{args.output}")


if __name__ == "__main__":
    main()

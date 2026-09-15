"""Generate one long-horizon task from the Scene graph."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
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
        default=Path("output/tasks.jsonl"),
        help="任务输出文件，默认 output/tasks.jsonl",
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
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count 必须大于 0")
    if args.max_workers <= 0:
        parser.error("--max-workers 必须大于 0")
    if args.hierarchy_child_limit <= 0:
        parser.error("--hierarchy-child-limit 必须大于 0")
    if args.path_query_timeout <= 0:
        parser.error("--path-query-timeout 必须大于 0")
    load_dotenv()
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(threadName)s %(name)s - %(message)s",
    )
    llm = LLMClient.from_env("LLM", timeout=float(os.getenv("LLM_TIMEOUT", "60")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with Neo4jGraphStore(
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
        path_query_timeout=args.path_query_timeout,
    ) as store:
        generator = TaskGenerator(
            store,
            llm,
            hierarchy_child_limit=args.hierarchy_child_limit,
            hierarchy_workers=args.max_workers,
        )
        with args.output.open("a", encoding="utf-8") as output_file:
            with ThreadPoolExecutor(max_workers=min(args.max_workers, args.count)) as executor:
                def generate_one(index: int):
                    logging.getLogger(__name__).info("任务生成开始：任务=%d/%d", index, args.count)
                    return generator.generate(
                        args.hops,
                        args.task_type,
                        args.task_style,
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
                        logging.getLogger(__name__).warning(
                            "任务生成跳过：任务=%d/%d，原因=%s",
                            index, args.count, exc,
                        )
                        continue
                    except Exception:
                        failed += 1
                        logging.getLogger(__name__).exception(
                            "任务生成失败：任务=%d/%d", index, args.count
                        )
                        continue
                    output_file.write(
                        json.dumps(
                            {
                                "task": task.desc,
                                "task_type": task.task_type.value,
                                "complexity": task.complexity,
                                "complexity_features": task.complexity_features,
                                "environment": task.env,
                                "metrics": task.metrics,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    output_file.flush()
                    logging.getLogger(__name__).info(
                        "任务文件写入进度：完成=%d/%d，成功=%d，失败=%d",
                        completed, args.count, completed - failed, failed,
                    )
    print(f"任务生成结束：成功={args.count - failed}，失败={failed}，文件={args.output}")


if __name__ == "__main__":
    main()

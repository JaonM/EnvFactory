"""Build the knowledge graph from seed words and persist it to Neo4j."""

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from env_factory import (
    GraphExpansionConfig,
    LLMClient,
    Neo4jGraphStore,
    SearXNGClient,
    SeedGraphExpander,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建并持久化任务知识图谱")
    parser.add_argument("--rounds", type=int, default=None, help="最大扩展轮次")
    parser.add_argument("--max-scene-nodes", type=int, default=1000)
    parser.add_argument("--max-search-requests", type=int, default=500)
    parser.add_argument("--term-batch-size", type=int, default=16)
    parser.add_argument("--relation-candidate-limit", type=int, default=20)
    parser.add_argument("--max-workers", type=int, default=2, help="搜索 API 并发数")
    return parser.parse_args()


def load_seed_words() -> tuple[str, ...]:
    """Read one seed word per line from GRAPH_SEEDS_FILE."""

    path = seed_file_path()
    seeds = tuple(
        dict.fromkeys(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    )
    if not seeds:
        raise ValueError(f"seed file is empty: {path}")
    return seeds


def seed_file_path() -> Path:
    seed_file = os.getenv("GRAPH_SEEDS_FILE")
    if not seed_file:
        raise ValueError("GRAPH_SEEDS_FILE is not configured in .env")
    path = Path(seed_file)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"seed file does not exist: {path}")
    return path


def append_seed_words(words: tuple[str, ...]) -> int:
    path = seed_file_path()
    existing = {line.strip().casefold() for line in path.read_text(encoding="utf-8").splitlines()}
    new_words = tuple(dict.fromkeys(word.strip() for word in words if word.strip()))
    pending = tuple(word for word in new_words if word.casefold() not in existing)
    if pending:
        with path.open("a", encoding="utf-8") as file:
            file.write("\n" + "\n".join(pending) + "\n")
    return len(pending)


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args = parse_args()
    seeds = load_seed_words()
    config = GraphExpansionConfig(
        max_scene_nodes=args.max_scene_nodes,
        max_search_requests=args.max_search_requests,
        max_rounds=args.rounds or 3,
        term_batch_size=args.term_batch_size,
        relation_candidate_limit=args.relation_candidate_limit,
    )

    llm = LLMClient.from_env("LLM", timeout=float(os.getenv("LLM_TIMEOUT", "60")))
    search = SearXNGClient(timeout=float(os.getenv("SEARXNG_TIMEOUT", "10")))
    expander = SeedGraphExpander(
        search,
        llm,
        max_workers=args.max_workers,
        config=config,
    )

    with Neo4jGraphStore(database=os.getenv("NEO4J_DATABASE", "neo4j")) as store:
        store.verify_connectivity()
        logging.getLogger(__name__).info("开始写入 Neo4j 图谱")
        builder, groups = expander.expand_and_build(store, seeds, rounds=args.rounds)

    added_seeds = append_seed_words(
        tuple(word for group in groups for word in (group.name, *group.words))
    )

    print(
        f"图谱构建完成：{len(groups)} 个 scene 节点，{len(builder.edges())} 条关系，"
        f"新增种子词 {added_seeds} 个"
    )


if __name__ == "__main__":
    main()

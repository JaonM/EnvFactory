"""Create a local SQLite index from a Wikimedia dump."""

import argparse

from env_factory import WikipediaDumpIndexer


def main() -> None:
    parser = argparse.ArgumentParser(description="建立 Wikipedia 本地全文索引")
    parser.add_argument(
        "dump_file",
        help="Wikipedia pages-articles dump 文件路径（支持 .bz2），无默认值",
    )
    parser.add_argument(
        "database_file",
        help="输出的 SQLite FTS5 数据库文件路径，无默认值；脚本命令默认使用 data/wikipedia.sqlite3",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="建索引前删除已有的 pages_fts 表，默认追加到现有数据库",
    )
    args = parser.parse_args()
    count = WikipediaDumpIndexer(args.database_file).build(args.dump_file, replace=args.replace)
    print(f"索引完成：{count} 个页面")


if __name__ == "__main__":
    main()

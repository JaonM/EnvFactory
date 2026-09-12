"""Create a local SQLite index from a Wikimedia dump."""

import argparse

from env_factory import WikipediaDumpIndexer


def main() -> None:
    parser = argparse.ArgumentParser(description="建立 Wikipedia 本地全文索引")
    parser.add_argument("dump_file")
    parser.add_argument("database_file")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    count = WikipediaDumpIndexer(args.database_file).build(args.dump_file, replace=args.replace)
    print(f"索引完成：{count} 个页面")


if __name__ == "__main__":
    main()

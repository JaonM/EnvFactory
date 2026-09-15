#!/usr/bin/env bash

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
用法：scripts/index_wikipedia_dump.sh <dump-file> [database-file]

将 Wikipedia dump 建立为本地 SQLite FTS5 索引。

参数：
  dump-file      Wikipedia dump 文件路径，无默认值
  database-file  SQLite 数据库路径，默认：项目根目录/data/wikipedia.sqlite3

环境变量：
  WIKIPEDIA_DUMP_DB  覆盖 database-file 默认路径
EOF
  exit 0
fi

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dump_file="${1:?用法：$0 <dump-file> [database-file]}"
database_file="${2:-${WIKIPEDIA_DUMP_DB:-$project_dir/data/wikipedia.sqlite3}}"
cd "$project_dir"

exec uv run python examples/index_wikipedia_dump.py "$dump_file" "$database_file" --replace

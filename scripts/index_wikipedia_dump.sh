#!/usr/bin/env bash

set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dump_file="${1:?用法：$0 <dump-file> [database-file]}"
database_file="${2:-${WIKIPEDIA_DUMP_DB:-$project_dir/data/wikipedia.sqlite3}}"
cd "$project_dir"

exec uv run python examples/index_wikipedia_dump.py "$dump_file" "$database_file" --replace

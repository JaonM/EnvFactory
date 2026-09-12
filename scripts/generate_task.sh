#!/usr/bin/env bash

set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

if [[ ! -f .env ]]; then
  echo "未找到 .env，请先配置 Neo4j 和 LLM 参数。" >&2
  exit 1
fi

exec uv run python examples/generate_task.py "$@"

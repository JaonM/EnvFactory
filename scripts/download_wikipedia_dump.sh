#!/usr/bin/env bash

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
用法：scripts/download_wikipedia_dump.sh

下载中文 Wikipedia pages-articles multistream dump。

环境变量：
  WIKIPEDIA_DUMP_DIR  下载目录，默认：项目根目录/data
  WIKIPEDIA_DUMP_URL  下载地址，默认：https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles-multistream.xml.bz2
EOF
  exit 0
fi

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dump_dir="${WIKIPEDIA_DUMP_DIR:-$project_dir/data}"
dump_url="${WIKIPEDIA_DUMP_URL:-https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles-multistream.xml.bz2}"
dump_file="$dump_dir/$(basename "$dump_url")"

mkdir -p "$dump_dir"
echo "下载 Wikipedia dump：$dump_url"
curl --fail --location --retry 3 --output "$dump_file" "$dump_url"
echo "下载完成：$dump_file"

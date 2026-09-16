#!/usr/bin/env bash

set -euo pipefail

context=""
tag="env-factory-agent-sandbox"
start="false"
dockerfile=""

usage() {
  cat <<'EOF'
用法：scripts/build_docker_sandbox_image.sh --context DIR [选项]

构建 Docker 沙箱镜像。构建前会验证 Dockerfile 第一条 FROM 镜像的 manifest；
原镜像不可达时，按顺序尝试平替镜像站点。

选项：
  --context DIR     Docker 构建上下文目录，必填
  --dockerfile FILE Dockerfile 路径，默认：DIR/Dockerfile
  --tag NAME        镜像名称，默认：env-factory-agent-sandbox
  --start           构建后启动容器，默认：不启动
  -h, --help        显示帮助

环境变量：
  SANDBOX_BASE_IMAGE          覆盖 Dockerfile 第一条 FROM 镜像；默认模板使用 docker.m.daocloud.io/library/python:3.14-slim
  SANDBOX_BASE_IMAGE_MIRRORS  空格分隔的平替镜像列表；默认自动尝试常用镜像站
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --context) context="$2"; shift 2 ;;
    --dockerfile) dockerfile="$2"; shift 2 ;;
    --tag) tag="$2"; shift 2 ;;
    --start) start="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数：$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$context" ]]; then
  echo "缺少 --context" >&2
  usage >&2
  exit 2
fi
if [[ "$context" != /* ]]; then context="$(pwd)/$context"; fi
if [[ -z "$dockerfile" ]]; then dockerfile="$context/Dockerfile"; fi
if [[ "$dockerfile" != /* ]]; then dockerfile="$(pwd)/$dockerfile"; fi
[[ -d "$context" ]] || { echo "构建上下文不存在：$context" >&2; exit 3; }
[[ -f "$dockerfile" ]] || { echo "Dockerfile 不存在：$dockerfile" >&2; exit 3; }
command -v docker >/dev/null 2>&1 || { echo "未找到 docker CLI" >&2; exit 4; }
rm -f "$context/OK"

from_image="$(awk 'toupper($1) == "FROM" {print $2; exit}' "$dockerfile")"
if [[ -z "$from_image" ]]; then
  echo "Dockerfile 中没有找到 FROM 指令：$dockerfile" >&2
  exit 5
fi
if [[ "$from_image" == *'${'* ]]; then
  from_image="python:3.14-slim"
fi
base_image="${SANDBOX_BASE_IMAGE:-$from_image}"

normalised="$base_image"
if [[ "$normalised" != */* || "$normalised" == docker.io/* ]]; then
  normalised="${normalised#docker.io/}"
  if [[ "$normalised" != */* ]]; then normalised="library/$normalised"; fi
fi
tag_part=""
if [[ "$normalised" == *@* ]]; then
  tag_part="@${normalised#*@}"
  normalised="${normalised%@*}"
elif [[ "$normalised" == *:* ]]; then
  tag_part=":${normalised##*:}"
  normalised="${normalised%:*}"
fi
default_mirrors=("mirror.gcr.io/${normalised}${tag_part}" "docker.m.daocloud.io/${normalised}${tag_part}")
if [[ -n "${SANDBOX_BASE_IMAGE_MIRRORS:-}" ]]; then
  read -r -a mirrors <<< "$SANDBOX_BASE_IMAGE_MIRRORS"
else
  mirrors=("${default_mirrors[@]}")
fi
candidates=("$base_image" "${mirrors[@]}")

selected=""
for candidate in "${candidates[@]}"; do
  [[ -n "$candidate" ]] || continue
  echo "验证基础镜像可达性：$candidate" >&2
  if docker manifest inspect "$candidate" >/dev/null 2>&1; then
    selected="$candidate"
    break
  fi
done
if [[ -z "$selected" ]]; then
  echo "原始基础镜像及平替镜像均不可达：$base_image" >&2
  exit 6
fi
echo "使用基础镜像：$selected" >&2

resolved_dockerfile="$(mktemp "${TMPDIR:-/tmp}/env-factory-dockerfile.XXXXXX")"
cleanup() { rm -f "$resolved_dockerfile"; }
trap cleanup EXIT
awk -v image="$selected" '
  BEGIN { replaced = 0 }
  !replaced && toupper($1) == "FROM" {
    line = $0
    sub(/^[[:space:]]*FROM[[:space:]]+[^[:space:]]+/, "FROM " image, line)
    print line
    replaced = 1
    next
  }
  { print }
' "$dockerfile" > "$resolved_dockerfile"

docker build --file "$resolved_dockerfile" --tag "$tag" "$context"
# The marker is written only after the image build succeeds.
touch "$context/OK"
if [[ "$start" == "true" ]]; then
  exec docker run --rm --publish 8080:8080 \
    --mount "type=bind,source=$(cd "$context/data" && pwd),target=/workspace/data" \
    "$tag"
fi

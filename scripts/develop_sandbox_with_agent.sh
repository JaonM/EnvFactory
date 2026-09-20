#!/usr/bin/env bash

set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
input="examples/clothing_materials_task.json"
output="output/sandbox/agent_clothing_materials_sandbox"
agent="codex"
runtime="none"
tag="env-factory-agent-sandbox"
start="false"
background="true"
max_concurrency="2"

usage() {
  cat <<'EOF'
用法：scripts/develop_sandbox_with_agent.sh [选项]

调用 Code Agent 根据任务输入自主开发一个 RL 沙箱工程，然后可选构建并启动容器。
默认后台运行，Agent 日志写入目标目录的 agent.log。

选项：
  --input PATH       任务 JSON、JSON task list 或包含 task-N/task.json 的 artifacts 目录，默认：examples/clothing_materials_task.json
  --output DIR       单任务工作目录；输入为 list 时作为输出根目录，默认：output/sandbox/agent_clothing_materials_sandbox
  --agent NAME       codex、claude 或 opencode，默认：codex
  --runtime NAME     none 或 docker，默认：none
  --tag NAME         镜像名称，默认：env-factory-agent-sandbox
  --max-concurrency N 并发任务数，默认：2
  --start            构建后立即启动容器，默认：不启动
  --foreground       前台等待 Agent 完成，默认：后台运行
  -h, --help         显示帮助
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input|--output|--agent|--runtime|--tag|--max-concurrency)
      if (( $# < 2 )); then
        echo "$1 需要提供参数值" >&2
        exit 2
      fi
      case "$1" in
        --input) input="$2" ;;
        --output) output="$2" ;;
        --agent) agent="$2" ;;
        --runtime) runtime="$2" ;;
        --tag) tag="$2" ;;
        --max-concurrency) max_concurrency="$2" ;;
      esac
      shift 2
      ;;
    --start) start="true"; shift ;;
    --foreground) background="false"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数：$1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! [[ "$max_concurrency" =~ ^[1-9][0-9]*$ ]]; then
  echo "--max-concurrency 必须是正整数：$max_concurrency" >&2
  exit 2
fi

if [[ "$start" == "true" && "$runtime" != "docker" ]]; then
  echo "--start 仅在 --runtime docker 时可用" >&2
  exit 2
fi

input_path="$input"
if [[ "$input_path" != /* ]]; then input_path="$project_dir/$input_path"; fi
output_path="$output"
if [[ "$output_path" != /* ]]; then output_path="$project_dir/$output_path"; fi
# 规范化用户传入的目录，避免 --output xxx/ 与后续 /agent.log 等路径拼接产生双斜杠。
if [[ "$output_path" != "/" ]]; then output_path="${output_path%/}"; fi
root_output_path="$output_path"
temporary_input=""
cleanup_temporary_input() {
  if [[ -n "$temporary_input" ]]; then
    rm -f "$temporary_input"
  fi
}
trap cleanup_temporary_input EXIT

if [[ -d "$input_path" ]]; then
  if [[ -f "$input_path/task.json" ]]; then
    input_path="$input_path/task.json"
  else
    task_json_candidates=()
    while IFS= read -r candidate; do task_json_candidates+=("$candidate"); done < <(find "$input_path" -mindepth 2 -maxdepth 2 -type f -name task.json -print)
    if (( ${#task_json_candidates[@]} == 0 )); then
      echo "--input 目录必须直接包含 task.json，或包含一个或多个 task-N/task.json：$input_path" >&2
      exit 3
    elif (( ${#task_json_candidates[@]} == 1 )); then
      input_path="${task_json_candidates[0]}"
    else
      temporary_input="$(mktemp "${TMPDIR:-/tmp}/envfactory-task-list.XXXXXX.json")"
      python3 - "$temporary_input" "${task_json_candidates[@]}" <<'PY'
import json
import sys
from pathlib import Path

destination = Path(sys.argv[1])
tasks = [json.loads(Path(item).read_text(encoding="utf-8")) for item in sys.argv[2:]]
if not all(isinstance(task, dict) for task in tasks):
    raise SystemExit("任务目录中的 task.json 必须都是 JSON object")
destination.write_text(json.dumps(tasks, ensure_ascii=False) + "\n", encoding="utf-8")
PY
      input_path="$temporary_input"
    fi
  fi
fi
[[ -f "$input_path" ]] || { echo "任务输入不存在：$input_path" >&2; exit 3; }
task_count="$(python3 - "$input_path" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if isinstance(value, list):
    if not value:
        raise SystemExit("任务 JSON list 不能为空")
    if not all(isinstance(task, dict) for task in value):
        raise SystemExit("任务 JSON list 的每一项必须是 object")
    print(len(value))
elif isinstance(value, dict):
    print(1)
else:
    raise SystemExit("任务输入必须是 JSON object 或 JSON list")
PY
)" || exit 3

prepare_task() {
  local task_index="$1"
  local task_output="$2"
  rm -f \
    "$task_output/OK" "$task_output/status.json" "$task_output/status.json.tmp" \
    "$task_output/agent.log" \
    "$task_output/TASK_PROMPT.md" "$task_output/SPEC_TASK.md" "$task_output/AGENT_TASK.md" \
    "$task_output/BUILD_CONTRACT.json" \
    "$task_output/spec.md" "$task_output/action_plan.json" "$task_output/development_plan.json" "$task_output/tools.json" "$task_output/Dockerfile" \
    "$task_output/docker_build.sh" "$task_output/docker_run.sh" \
    "$task_output/acceptance.sh" "$task_output/IMPLEMENTATION_REPORT.md" \
    "$task_output/app.py"
  rm -rf "$task_output/data" "$task_output/tests"
  mkdir -p "$task_output/data"
  python3 - "$input_path" "$task_output/task.json" "$task_index" "$project_dir" "$task_output" <<'PY'
import json
import shutil
import sys
from pathlib import Path

source = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
index = int(sys.argv[3])
task = source[index] if isinstance(source, list) else source
project_dir = Path(sys.argv[4])
task_output = Path(sys.argv[5])
artifacts = task.get("artifacts") if isinstance(task, dict) else None
if isinstance(artifacts, dict):
    def copy_manifest(key, destination_name):
        manifest = artifacts.get(key)
        if not isinstance(manifest, dict) or not manifest.get("root"):
            return
        source_root = Path(str(manifest["root"]))
        if not source_root.is_absolute():
            source_root = project_dir / source_root
        if not source_root.is_dir():
            raise SystemExit(f"{key} 目录不存在：{source_root}")
        destination = task_output / "data" / destination_name
        shutil.copytree(source_root, destination, dirs_exist_ok=True)
        rewritten = dict(manifest)
        rewritten["root"] = f"data/{destination_name}"
        artifacts[key] = rewritten
        return rewritten

    data_manifest = copy_manifest("data_manifest", "business_data")
    user_manifest = copy_manifest("user_simulation_manifest", "user_simulation")
    task["artifacts"] = artifacts
    for item in task.get("environment", []):
        if isinstance(item, dict) and item.get("type") == "business_data_manifest" and data_manifest:
            item["value"] = data_manifest
Path(sys.argv[2]).write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
  python3 - "$task_output/task.json" "$task_output/BUILD_CONTRACT.json" <<'PY'
import json
import sys
from pathlib import Path

task = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
requirements = task.get("requirements", {})
actions = task.get("actions", [])
metrics = task.get("metrics", [])
if not isinstance(requirements, dict) or not isinstance(actions, list) or not isinstance(metrics, list):
    raise SystemExit("任务缺少 requirements、actions 或 metrics")
action_names = [item.get("name") for item in actions if isinstance(item, dict) and isinstance(item.get("name"), str) and item["name"]]
obligations = [
    {"id": "platform.persistence", "kind": "platform", "required": True},
    {"id": "platform.tool_schema", "kind": "platform", "required": True},
    {"id": "platform.hidden_truth_isolation", "kind": "platform", "required": True},
    {"id": "platform.user_simulator", "kind": "platform", "required": True},
    {"id": "platform.reward_interface", "kind": "platform", "required": True},
    *[{"id": f"task_action.{name}", "kind": "task_action", "required": True} for name in action_names],
    {"id": "evaluation.reward_and_termination", "kind": "evaluation", "required": True},
]
contract = {
    "schema_version": "1.0",
    "authority": "env_factory_outer_workflow",
    "mutable_by_code_agent": False,
    "obligations": obligations,
    "requirements": requirements,
    "platform": {
        "persistence": {"required": True, "episode_isolation": True},
        "tool_schema": "openai_function",
        "tool_trainer_action_mapping": "one_to_one",
        "hidden_truth_isolation": True,
        "runtime_credentials": "external_only",
        "user_simulator": {"required": True, "only_trigger": "ask_user"},
        "reward_interface": {"required": True},
    },
    "task_actions": actions,
    "capabilities": [{"id": "external_model_boundary", "required": True, "detected": True}],
    "evaluation": {"metric_ids": [item.get("id") for item in metrics if isinstance(item, dict) and item.get("id")]},
}
Path(sys.argv[2]).write_text(
    json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY
  chmod 444 "$task_output/BUILD_CONTRACT.json"
  write_task_prompt "phase1" "$task_output"
}

write_task_prompt() {
  local phase="$1"
  local task_output="$2"
  cp "$project_dir/docs/sandbox_spec_prompt.md" "$task_output/TASK_PROMPT.md"
  cat >> "$task_output/TASK_PROMPT.md" <<EOF

The complete task input is in ./task.json. The immutable outer-workflow contract is in ./BUILD_CONTRACT.json; it is generated during task generation and must be treated as authoritative. Phase 1 produces ./spec.md. Phase 2 implements the complete sandbox directly from ./spec.md.
The project output directory is: $task_output

## Active phase

$(if [[ "$phase" == "phase1" ]]; then
  echo "Phase 1 is active. Read ./task.json and ./BUILD_CONTRACT.json and write only ./spec.md. The spec must explicitly cover every BUILD_CONTRACT obligation and include each obligation id literally in the relevant section; do not weaken or replace the contract. Do not create action_plan.json, development_plan.json, code, tests, tools, HTTP handlers, Docker files, or development-plan files."
elif [[ "$phase" == "phase2" ]]; then
  echo "Phase 2 is active. Read ./spec.md, ./task.json, and ./BUILD_CONTRACT.json, then implement the complete sandbox in one pass. Treat spec.md as the authoritative implementation design and preserve every BUILD_CONTRACT obligation. Generate the complete runnable project, including persistence, business data loading, user simulator, Trainer actions, observations, rewards, standard tools.json, tests, acceptance.sh, IMPLEMENTATION_REPORT.md, Dockerfile, docker_build.sh, and docker_run.sh. Do not create action_plan.json or development_plan.json. Do not stop at a demo or plan: write the implementation and run its checks before finishing."
fi)
EOF
}

run_code_agent() {
  local phase_prompt="$1"
  case "$agent" in
    codex)
      (cd "$output_path" && codex exec --approve-for-me "$phase_prompt") </dev/null
      ;;
    claude)
      (cd "$output_path" && claude --dangerously-skip-permissions --print "$phase_prompt") </dev/null
      ;;
    opencode)
      (cd "$output_path" && opencode run "$phase_prompt") </dev/null
      ;;
    *)
      echo "不支持的 agent：${agent}；可选值为 codex、claude、opencode"
      return 2
      ;;
  esac
}

write_status() {
  local status="$1"
  local exit_code="$2"
  local message="${3:-}"
  python3 - "$output_path/status.json" "$status" "$exit_code" "$message" "$agent" "$runtime" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "status": sys.argv[2],
    "success": sys.argv[2] == "success",
    "exit_code": int(sys.argv[3]),
    "message": sys.argv[4],
    "agent": sys.argv[5],
    "runtime": sys.argv[6],
    "finished_at": datetime.now(timezone.utc).isoformat(),
}

tmp = path.with_name(path.name + ".tmp")
tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
tmp.replace(path)
PY
}

validate_spec_contract() {
  python3 - "$output_path/spec.md" "$output_path/BUILD_CONTRACT.json" <<'PY'
import json
import sys
from pathlib import Path

spec = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
contract = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
obligations = contract.get("obligations", [])
missing = [item["id"] for item in obligations if item.get("required", True) and item["id"] not in spec]
if missing:
    raise SystemExit("spec 未覆盖 BUILD_CONTRACT obligations：" + ", ".join(missing))
for action in contract.get("task_actions", []):
    name = action.get("name")
    if not isinstance(name, str) or name not in spec:
        raise SystemExit(f"spec 未覆盖结构化 task action：{name}")
print("spec contract validation: ok")
PY
}

validate_spec_implementation() {
  python3 - "$output_path/spec.md" "$output_path/task.json" "$output_path/tools.json" "$output_path/BUILD_CONTRACT.json" <<'PY'
import json
import sys
from pathlib import Path

spec_path, task_path, tools_path, contract_path = map(Path, sys.argv[1:])
spec = spec_path.read_text(encoding="utf-8", errors="replace")
task = json.loads(task_path.read_text(encoding="utf-8"))
tools = json.loads(tools_path.read_text(encoding="utf-8"))
contract = json.loads(contract_path.read_text(encoding="utf-8"))

if len(spec.strip()) < 1000:
    raise SystemExit("spec.md 过短，未形成可执行的详细实现规格")
required_topics = {
    "data": ("数据模型", "持久化", "业务数据"),
    "simulator": ("user simulator", "用户模拟", "用户剧本"),
    "tools": ("工具", "Trainer action", "Trainer Action"),
    "reward": ("奖励", "reward", "观测"),
    "acceptance": ("验收", "acceptance", "测试"),
    "runtime": ("Docker", "运行时", "API"),
}
for topic, markers in required_topics.items():
    if not any(marker in spec for marker in markers):
        raise SystemExit(f"spec.md 缺少实现主题：{topic}")

actions = []
action_sources = []
if isinstance(task, dict):
    action_sources.extend(task.get("actions", []))
    action_sources.extend(task.get("environment", []))
for item in action_sources:
    if isinstance(item, dict) and item.get("type") == "action":
        name = item.get("name") or item.get("field") or item.get("value")
        if isinstance(name, str) and name and name not in actions:
            actions.append(name)
missing_actions = [name for name in actions if name not in spec]
if missing_actions:
    raise SystemExit("spec.md 未覆盖任务动作：" + ", ".join(missing_actions))

if not isinstance(tools, list) or not tools:
    raise SystemExit("tools.json 必须是非空的顶层 Function Tool 数组")
names = set()
def check_schema(schema, location):
    if not isinstance(schema, dict):
        raise SystemExit(f"工具参数 {location} schema 必须是 object")
    if not isinstance(schema.get("description"), str) or not schema["description"].strip():
        raise SystemExit(f"工具参数 {location} 缺少 description")
    if schema.get("type") == "object":
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise SystemExit(f"工具参数 {location}.properties 必须是 object")
        required = schema.get("required", [])
        if not isinstance(required, list) or any(key not in properties for key in required):
            raise SystemExit(f"工具参数 {location}.required 无效")
        for key, value in properties.items():
            check_schema(value, f"{location}.{key}")
    elif schema.get("type") == "array" and "items" in schema:
        check_schema(schema["items"], f"{location}[]")

for index, tool in enumerate(tools, 1):
    if not isinstance(tool, dict) or tool.get("type") != "function":
        raise SystemExit(f"tools.json 第 {index} 项不是标准 function tool")
    function = tool.get("function")
    if not isinstance(function, dict):
        raise SystemExit(f"tools.json 第 {index} 项缺少 function")
    name = function.get("name")
    if not isinstance(name, str) or not name or name in names:
        raise SystemExit(f"tools.json 工具名无效或重复：{name!r}")
    if not isinstance(function.get("description"), str) or not function["description"].strip():
        raise SystemExit(f"工具 {name} 缺少 function.description")
    parameters = function.get("parameters")
    if not isinstance(parameters, dict) or parameters.get("type") != "object":
        raise SystemExit(f"工具 {name} 的 parameters 必须是 object schema")
    if not isinstance(parameters.get("properties"), dict):
        raise SystemExit(f"工具 {name} 缺少 parameters.properties")
    for key, value in parameters["properties"].items():
        check_schema(value, f"{name}.{key}")
    names.add(name)
    if name not in spec:
        raise SystemExit(f"spec.md 未定义实现工具：{name}")

required_obligations = {
    item.get("id") for item in contract.get("obligations", [])
    if item.get("required", True) and item.get("id")
}
missing_obligations = sorted(item for item in required_obligations if item not in spec)
if missing_obligations:
    raise SystemExit("spec.md 未覆盖 required obligations：" + ", ".join(missing_obligations))
print("spec implementation validation: ok")
PY
}

required_sandbox_files=(
  spec.md tools.json Dockerfile docker_build.sh docker_run.sh
  acceptance.sh IMPLEMENTATION_REPORT.md
)

validate_delivery() {
  local missing=()
  local file
  for file in "${required_sandbox_files[@]}"; do
    [[ -s "$output_path/$file" ]] || missing+=("$file")
  done
  if (( ${#missing[@]} > 0 )); then
    echo "Code Agent 未生成必需文件：${missing[*]}"
    return 1
  fi

  local legacy=()
  for file in action_plan.json development_plan.json; do
    [[ -e "$output_path/$file" ]] && legacy+=("$file")
  done
  if (( ${#legacy[@]} > 0 )); then
    echo "生成了已废弃的拓扑产物：${legacy[*]}"
    return 1
  fi

  local acceptance_log acceptance_status
  acceptance_log="$(mktemp "${TMPDIR:-/tmp}/sandbox-acceptance.XXXXXX")"
  set +e
  (cd "$output_path" && bash ./acceptance.sh) >"$acceptance_log" 2>&1
  acceptance_status=$?
  set -e
  if [[ "$acceptance_status" -ne 0 ]]; then
    echo "沙箱验收失败，退出码：$acceptance_status"
    cat "$acceptance_log"
    rm -f "$acceptance_log"
    return "$acceptance_status"
  fi
  rm -f "$acceptance_log"
  validate_spec_implementation
}

run_agent_and_finalize_impl() {
  echo "Code Agent 规格阶段开始：agent=$agent output=$output_path"
  spec_prompt="$(<"$output_path/TASK_PROMPT.md")"
  for spec_attempt in 1 2 3; do
    set +e
    run_code_agent "$spec_prompt"
    agent_status=$?
    set -e
    if [[ "$agent_status" -eq 0 && -s "$output_path/spec.md" ]]; then
      set +e
      spec_contract_error="$(validate_spec_contract 2>&1)"
      spec_contract_status=$?
      set -e
      if [[ "$spec_contract_status" -eq 0 ]]; then
        break
      fi
    else
      spec_contract_status=1
      spec_contract_error="Code Agent 未生成 spec.md，退出码：$agent_status"
    fi
    if [[ "$spec_attempt" -eq 3 ]]; then
      echo "规格阶段连续 3 次未通过：$spec_contract_error" >&2
      return 5
    fi
    echo "规格未通过外层契约校验，将错误发送给同一 Code Agent 重试：$spec_contract_error" >&2
    spec_prompt="$(cat <<EOF
$spec_prompt

上一轮生成的 spec.md 未通过 BUILD_CONTRACT 校验：
$spec_contract_error

请只修复 spec.md，逐项覆盖 ./BUILD_CONTRACT.json 中所有 required obligations，并在相关章节中原样写出 obligation id。不要修改 BUILD_CONTRACT.json，也不要生成其他文件。
EOF
)"
  done
  echo "Code Agent 规格阶段完成：$output_path/spec.md"

  write_task_prompt "phase2" "$output_path"
  implementation_prompt="$(<"$output_path/TASK_PROMPT.md")"
  implementation_error=""
  implementation_succeeded="false"
  for implementation_attempt in 1 2 3; do
    attempt_prompt="$implementation_prompt"
    if [[ "$implementation_attempt" -gt 1 ]]; then
      attempt_prompt="$(cat <<EOF
$implementation_prompt

上一轮一次性开发未通过外层检查，必须根据以下实际错误继续修复当前沙箱，不能只解释问题：
$implementation_error

请重新运行相关测试和 acceptance.sh，确保所有必需文件真实存在且非空。
EOF
)"
    fi
    set +e
    run_code_agent "$attempt_prompt"
    agent_status=$?
    set -e
    if [[ "$agent_status" -ne 0 ]]; then
      implementation_error="Code Agent 一次性开发退出码：$agent_status"
    else
      set +e
      delivery_error="$(validate_delivery 2>&1)"
      delivery_status=$?
      set -e
      if [[ "$delivery_status" -eq 0 ]]; then
        implementation_succeeded="true"
        break
      fi
      implementation_error="$delivery_error"
    fi
    echo "一次性开发第 ${implementation_attempt} 次尝试失败：$implementation_error" >&2
  done
  if [[ "$implementation_succeeded" != "true" ]]; then
    echo "Code Agent 一次性开发连续 3 次未通过：$implementation_error" >&2
    return 5
  fi
  echo "Code Agent 一次性开发完成：$output_path"
  echo "执行 spec 实现合规复核：$output_path/spec.md"
  if ! validate_spec_implementation; then
    echo "spec 实现合规复核失败" >&2
    return 5
  fi
  echo "执行运行时契约 trace 校验：$output_path/runtime_trace.jsonl"
  if ! validate_runtime_trace; then
    echo "运行时契约 trace 校验失败" >&2
    return 5
  fi

  case "$runtime" in
    none)
      touch "$output_path/OK"
      ;;
    docker)
      build_args=(--context "$output_path" --tag "$current_tag")
      if [[ "$start" == "true" ]]; then build_args+=(--start); fi
      "$project_dir/scripts/build_docker_sandbox_image.sh" "${build_args[@]}"
      ;;
    *)
      echo "不支持的 runtime：${runtime}；可选值为 none、docker"
      return 2
      ;;
  esac
  echo "沙箱构建完成标记：$output_path/OK"
  echo "Code Agent 已完成沙箱开发：$output_path"
}

run_agent_and_finalize() {
  set +e
  run_agent_and_finalize_impl
  local exit_code=$?
  set -e
  if [[ "$exit_code" -eq 0 ]]; then
    write_status "success" "$exit_code" "沙箱环境构建成功"
  else
    write_status "failed" "$exit_code" "沙箱环境构建失败，请查看 agent.log"
  fi
  return "$exit_code"
}

start_task() {
  local task_index="$1"
  local task_output="$2"
  output_path="$task_output"
  current_tag="$tag"
  if (( task_count > 1 )); then
    current_tag="${tag}-task-$(printf '%03d' "$((task_index + 1))")"
  fi
  prepare_task "$task_index" "$output_path"
  prompt="$(<"$output_path/TASK_PROMPT.md")"
  export project_dir agent runtime start background input_path output_path prompt current_tag
  log_file="$output_path/agent.log"
  if [[ "$background" == "true" ]]; then
    (run_agent_and_finalize) >"$log_file" 2>&1 </dev/null &
    pid=$!
    task_pids+=("$pid")
    echo "Code Agent 已后台启动：task=$((task_index + 1)) PID=$pid"
    echo "日志文件：$log_file"
  else
    run_agent_and_finalize
  fi
}

task_pids=()
background_failures=0
wait_for_task_slot() {
  while (( ${#task_pids[@]} >= max_concurrency )); do
    pid="${task_pids[0]}"
    set +e
    wait "$pid"
    task_status=$?
    set -e
    task_pids=("${task_pids[@]:1}")
    if [[ "$task_status" -ne 0 ]]; then
      echo "后台任务结束但未成功：PID=${pid}，退出码=${task_status}" >&2
      background_failures=1
    fi
  done
}

wait_for_all_tasks() {
  local pid task_status
  for pid in "${task_pids[@]}"; do
    set +e
    wait "$pid"
    task_status=$?
    set -e
    if [[ "$task_status" -ne 0 ]]; then
      echo "后台任务结束但未成功：PID=${pid}，退出码=${task_status}" >&2
      background_failures=1
    fi
  done
  return "$background_failures"
}

if (( task_count == 1 )); then
  start_task 0 "$output_path"
else
  mkdir -p "$root_output_path"
  for (( index = 0; index < task_count; index++ )); do
    if [[ "$background" == "true" ]]; then
      wait_for_task_slot
    fi
    start_task "$index" "$root_output_path/task_$(printf '%03d' "$((index + 1))")"
  done
  if [[ "$background" == "true" ]]; then
    wait_for_all_tasks
  fi
fi

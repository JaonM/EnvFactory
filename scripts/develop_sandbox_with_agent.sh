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
  --input FILE       任务 JSON 或 JSON task list，默认：examples/clothing_materials_task.json
  --output DIR       单任务工作目录；输入为 list 时作为输出根目录，默认：output/sandbox/agent_clothing_materials_sandbox
  --agent NAME       codex、claude 或 opencode，默认：codex
  --runtime NAME     none 或 docker，默认：none
  --tag NAME         镜像名称，默认：env-factory-agent-sandbox
  --max-concurrency N 并发开发任务数，默认：2
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
root_output_path="$output_path"

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
    "$task_output/agent.log" "$task_output/TASK_PROMPT.md" "$task_output/SPEC_TASK.md" "$task_output/AGENT_TASK.md" \
    "$task_output/spec.md" "$task_output/action_plan.json" "$task_output/tools.json" "$task_output/Dockerfile" \
    "$task_output/docker_build.sh" "$task_output/docker_run.sh" \
    "$task_output/acceptance.sh" "$task_output/IMPLEMENTATION_REPORT.md" \
    "$task_output/app.py"
  rm -rf "$task_output/data" "$task_output/tests"
  mkdir -p "$task_output/data"
  python3 - "$input_path" "$task_output/task.json" "$task_index" <<'PY'
import json
import sys
from pathlib import Path

source = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
index = int(sys.argv[3])
task = source[index] if isinstance(source, list) else source
Path(sys.argv[2]).write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
  write_task_prompt "phase1" "$task_output"
}

write_task_prompt() {
  local phase="$1"
  local task_output="$2"
  cp "$project_dir/docs/sandbox_spec_prompt.md" "$task_output/TASK_PROMPT.md"
  cat >> "$task_output/TASK_PROMPT.md" <<EOF

The complete task input is in ./task.json. The specification output must be ./spec.md and ./action_plan.json.
The project output directory is: $task_output

## Active phase

$(if [[ "$phase" == "phase1" ]]; then echo "Phase 1 is active. Write only ./spec.md and ./action_plan.json; do not implement code, tests, tools, HTTP handlers, or Docker files."; else echo "Phase 2 is active. Read ./spec.md and ./action_plan.json and implement the complete sandbox now. Do not regenerate the design unless a concrete implementation constraint requires a documented correction."; fi)
EOF
}

run_code_agent() {
  local phase_prompt="$1"
  case "$agent" in
    codex)
      (cd "$output_path" && codex exec --approve-for-me "$phase_prompt")
      ;;
    claude)
      (cd "$output_path" && claude --dangerously-skip-permissions --print "$phase_prompt")
      ;;
    opencode)
      (cd "$output_path" && opencode run "$phase_prompt")
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

validate_action_plan() {
  python3 - "$output_path/action_plan.json" "$output_path/task.json" <<'PY'
import json
import sys
from pathlib import Path

plan = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
task = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if not isinstance(plan, dict) or not isinstance(plan.get("actions"), list) or not plan["actions"]:
    raise SystemExit("action_plan.json 必须包含非空 actions 数组")
task_actions = {
    item.get("field") for item in task.get("environment", [])
    if item.get("type") == "action" and isinstance(item.get("field"), str)
}
seen_actions = set()
public_tools = set()
for index, action in enumerate(plan["actions"], 1):
    if not isinstance(action, dict):
        raise SystemExit(f"action_plan.actions 第 {index} 项必须是 object")
    name = action.get("task_action")
    if not isinstance(name, str) or not name or name in seen_actions:
        raise SystemExit(f"action_plan.actions 第 {index} 项 task_action 无效或重复")
    if name not in task_actions:
        raise SystemExit(f"action_plan 引用了不存在的 task action：{name}")
    if action.get("classification") not in {"atomic", "composite"}:
        raise SystemExit(f"action_plan action {name} 必须声明 atomic 或 composite")
    steps = action.get("steps")
    if not isinstance(steps, list) or not steps:
        raise SystemExit(f"action_plan action {name} 必须包含非空 steps")
    for step_index, step in enumerate(steps, 1):
        if not isinstance(step, dict) or step.get("kind") not in {"public_llm_tool", "internal", "llm_generate"}:
            raise SystemExit(f"action_plan action {name} 的第 {step_index} 步 kind 无效")
        if step["kind"] == "public_llm_tool":
            tool_name = step.get("tool_name")
            if not isinstance(tool_name, str) or not tool_name:
                raise SystemExit(f"action_plan action {name} 的公开工具步骤缺少 tool_name")
            if tool_name in public_tools:
                raise SystemExit(f"action_plan 中公开工具重复：{tool_name}")
            public_tools.add(tool_name)
    seen_actions.add(name)
if seen_actions != task_actions:
    raise SystemExit(f"action_plan 未覆盖全部 task action：缺少={sorted(task_actions-seen_actions)}")
print("action_plan validation: ok")
PY
}

run_agent_and_finalize_impl() {
  echo "Code Agent 规格阶段开始：agent=$agent output=$output_path"
  spec_prompt="$(<"$output_path/TASK_PROMPT.md")"
  for phase1_attempt in 1 2 3; do
    if [[ "$phase1_attempt" -eq 1 ]]; then
      phase1_prompt="$spec_prompt"
    else
      phase1_prompt="$(cat <<EOF
上一轮生成的 action_plan.json 未通过外部校验：
$plan_error

请只修正 spec.md 和 action_plan.json，严格按任务 action 重新分析原子/复合动作、公开 LLM Tool、串行/并行依赖，并重新保存文件。不要报告完成，先修复后结束。
EOF
)"
    fi
    set +e
    run_code_agent "$phase1_prompt"
    agent_status=$?
    set -e
    if [[ "$agent_status" -eq 0 && -s "$output_path/spec.md" && -s "$output_path/action_plan.json" ]]; then
      set +e
      plan_error="$(validate_action_plan 2>&1)"
      plan_status=$?
      set -e
      if [[ "$plan_status" -eq 0 ]]; then
        break
      fi
    else
      plan_error="Code Agent 规格阶段退出码为 $agent_status，或未生成 spec.md/action_plan.json"
      plan_status=1
    fi
    if [[ "$phase1_attempt" -eq 3 ]]; then
      echo "规格阶段连续 3 次未通过：$plan_error"
      return 5
    fi
    echo "规格方案校验失败，准备让 Code Agent 第 $((phase1_attempt + 1)) 次修复：$plan_error"
  done
  echo "Code Agent 规格阶段完成：$output_path/spec.md"

  echo "Code Agent 实现阶段开始：agent=$agent output=$output_path"
  write_task_prompt "phase2" "$output_path"
  prompt="$(<"$output_path/TASK_PROMPT.md")"
  set +e
  run_code_agent "$prompt"
  agent_status=$?
  set -e
  if [[ "$agent_status" -ne 0 ]]; then
    echo "Code Agent 实现阶段失败，退出码：$agent_status"
    return "$agent_status"
  fi

  if [[ -f "$output_path/Containerfile" ]]; then
    echo "移除不再使用的 Containerfile：$output_path/Containerfile"
    rm -f "$output_path/Containerfile"
  fi
  required=(spec.md tools.json Dockerfile docker_build.sh docker_run.sh acceptance.sh IMPLEMENTATION_REPORT.md)
  for file in "${required[@]}"; do
    if [[ ! -f "$output_path/$file" ]]; then
      echo "Code Agent 未生成必需文件：$output_path/$file"
      return 5
    fi
  done
  for file in acceptance.sh IMPLEMENTATION_REPORT.md; do
    if [[ ! -s "$output_path/$file" ]]; then
      echo "Code Agent 生成的文件为空：$output_path/$file"
      return 5
    fi
  done

  for tools_attempt in 1 2 3; do
    validation_error_file="$(mktemp "${TMPDIR:-/tmp}/sandbox-tools-validation.XXXXXX")"
    set +e
    python3 - "$output_path/tools.json" "$output_path/action_plan.json" 2>"$validation_error_file" <<'PY'
import json
import sys
from pathlib import Path

tools_path = Path(sys.argv[1])
plan_path = Path(sys.argv[2])
payload = json.loads(tools_path.read_text(encoding="utf-8"))
llm_tools = payload.get("llm_tools") if isinstance(payload, dict) else None
if not isinstance(llm_tools, list) or not llm_tools:
    raise SystemExit("tools.json 必须包含非空 llm_tools 数组")

llm_names = set()
def has_internal_annotation(value):
    if isinstance(value, str):
        return "role=" in value or "hidden_state" in value
    if isinstance(value, dict):
        return any(has_internal_annotation(item) for item in value.values())
    if isinstance(value, list):
        return any(has_internal_annotation(item) for item in value)
    return False

for index, tool in enumerate(llm_tools):
    if not isinstance(tool, dict) or tool.get("type") != "function":
        raise SystemExit(f"LLM 工具 {index + 1} 必须使用 type=function")
    function = tool.get("function")
    if not isinstance(function, dict) or not {"name", "description", "parameters"}.issubset(function):
        raise SystemExit(f"LLM 工具 {index + 1} 缺少标准 function 字段")
    name = function["name"]
    schema = function["parameters"]
    if not isinstance(name, str) or not name or name in llm_names:
        raise SystemExit(f"tools.json 工具名称无效或重复：{name!r}")
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise SystemExit(f"工具 {name} 的 parameters 必须是 object schema")
    if not isinstance(schema.get("properties"), dict):
        raise SystemExit(f"工具 {name} 缺少 parameters.properties")
    if not isinstance(schema.get("required", []), list):
        raise SystemExit(f"工具 {name} 的 parameters.required 必须是数组")
    if has_internal_annotation(function):
        raise SystemExit(f"LLM 工具 {name} 不得在 description 中包含内部 role/hidden_state 标记")
    llm_names.add(name)

plan = json.loads(plan_path.read_text(encoding="utf-8"))
public_tools = {
    step.get("tool_name")
    for action in plan.get("actions", [])
    for step in action.get("steps", [])
    if step.get("kind") == "public_llm_tool"
}
if public_tools != llm_names:
    raise SystemExit(
        f"llm_tools 与 action_plan.json 的公开工具不一致："
        f"缺少={sorted(public_tools - llm_names)}，多余={sorted(llm_names - public_tools)}"
    )

PY
    validation_status=$?
    set -e
    if [[ "$validation_status" -eq 0 ]]; then
      rm -f "$validation_error_file"
      break
    fi
    validation_detail="$(<"$validation_error_file")"
    rm -f "$validation_error_file"
    if [[ "$tools_attempt" -eq 3 ]]; then
      echo "tools.json 连续 3 次校验失败：$validation_detail" >&2
      return 5
    fi
    echo "tools.json 校验失败，准备让 Code Agent 第 $((tools_attempt + 1)) 次修复：$validation_detail" >&2
    repair_prompt="$(cat <<EOF
上一轮实现生成的 tools.json 未通过外部校验：
$validation_detail

请读取 ./spec.md 和 ./action_plan.json，只修复实现产物。确保 llm_tools 的公开工具集合严格等于 action_plan.json 中 kind=public_llm_tool 的 tool_name 集合，且每个 LLM tool 使用标准 function schema。不要把 task_action、trainer_actions、mappings 或 task_action_plans 的格式当作本次校验目标；修复后重新运行 acceptance.sh。
EOF
)"
    set +e
    run_code_agent "$repair_prompt"
    repair_status=$?
    set -e
    if [[ "$repair_status" -ne 0 ]]; then
      echo "Code Agent tools.json 修复阶段失败，退出码：$repair_status" >&2
      return "$repair_status"
    fi
  done

  echo "执行沙箱验收脚本：$output_path/acceptance.sh"
  set +e
  (cd "$output_path" && bash ./acceptance.sh)
  acceptance_status=$?
  set -e
  if [[ "$acceptance_status" -ne 0 ]]; then
    echo "沙箱验收失败，退出码：$acceptance_status" >&2
    return "$acceptance_status"
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
      echo "不支持的 runtime：$runtime；可选值为 none、docker"
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

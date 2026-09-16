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

usage() {
  cat <<'EOF'
用法：scripts/develop_sandbox_with_agent.sh [选项]

调用 Code Agent 根据任务输入自主开发一个 RL 沙箱工程，然后可选构建并启动容器。
默认后台运行，Agent 日志写入目标目录的 agent.log。

选项：
  --input FILE       任务 JSON，默认：examples/clothing_materials_task.json
  --output DIR       Agent 工作目录，默认：output/sandbox/agent_clothing_materials_sandbox
  --agent NAME       codex、claude 或 opencode，默认：codex
  --runtime NAME     none、docker、container 或 auto，默认：none
  --tag NAME         镜像名称，默认：env-factory-agent-sandbox
  --start            构建后立即启动容器，默认：不启动
  --foreground       前台等待 Agent 完成，默认：后台运行
  -h, --help         显示帮助
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) input="$2"; shift 2 ;;
    --output) output="$2"; shift 2 ;;
    --agent) agent="$2"; shift 2 ;;
    --runtime) runtime="$2"; shift 2 ;;
    --tag) tag="$2"; shift 2 ;;
    --start) start="true"; shift ;;
    --foreground) background="false"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数：$1" >&2; usage >&2; exit 2 ;;
  esac
done

input_path="$input"
if [[ "$input_path" != /* ]]; then input_path="$project_dir/$input_path"; fi
output_path="$output"
if [[ "$output_path" != /* ]]; then output_path="$project_dir/$output_path"; fi
mkdir -p "$output_path/data"
rm -f "$output_path/OK"
cp "$input_path" "$output_path/task.json"
cp "$project_dir/docs/sandbox_agent_prompt.md" "$output_path/AGENT_TASK.md"

cat >> "$output_path/AGENT_TASK.md" <<EOF

## Task-specific input

The complete task input is in ./task.json. Do not modify its semantics.
The project output directory is: $output_path
EOF

prompt="$(<"$output_path/AGENT_TASK.md")"
export project_dir input output agent runtime tag start background input_path output_path prompt

run_agent_and_finalize() {
  echo "Code Agent 开发开始：agent=$agent output=$output_path"
  set +e
  case "$agent" in
    codex)
      (cd "$output_path" && codex exec --approve-for-me "$prompt")
      ;;
    claude)
      (cd "$output_path" && claude --dangerously-skip-permissions --print "$prompt")
      ;;
    opencode)
      (cd "$output_path" && opencode run "$prompt")
      ;;
    *)
      echo "不支持的 agent：${agent}；可选值为 codex、claude、opencode"
      return 2
      ;;
  esac
  agent_status=$?
  set -e
  if [[ "$agent_status" -ne 0 ]]; then
    echo "Code Agent 失败，退出码：$agent_status"
    return "$agent_status"
  fi

  if [[ -f "$output_path/Containerfile" ]]; then
    echo "移除不再使用的 Containerfile：$output_path/Containerfile"
    rm -f "$output_path/Containerfile"
  fi
  required=(spec.md tools.json Dockerfile docker_build.sh docker_run.sh container_build.sh container_run.sh)
  for file in "${required[@]}"; do
    if [[ ! -f "$output_path/$file" ]]; then
      echo "Code Agent 未生成必需文件：$output_path/$file"
      return 5
    fi
  done

  if ! python3 - "$output_path/tools.json" "$output_path/task.json" <<'PY'
import json
import sys
from pathlib import Path

tools_path = Path(sys.argv[1])
task_path = Path(sys.argv[2])
payload = json.loads(tools_path.read_text(encoding="utf-8"))
llm_tools = payload.get("llm_tools") if isinstance(payload, dict) else None
trainer_actions = payload.get("trainer_actions") if isinstance(payload, dict) else None
if not isinstance(llm_tools, list) or not llm_tools:
    raise SystemExit("tools.json 必须包含非空 llm_tools 数组")
if not isinstance(trainer_actions, list) or not trainer_actions:
    raise SystemExit("tools.json 必须包含非空 trainer_actions 数组")

llm_mappings = payload.get("mappings") if isinstance(payload, dict) else None
if isinstance(llm_mappings, dict):
    llm_mappings = [
        {"llm_tool": llm_tool, "trainer_action": trainer_action}
        for llm_tool, trainer_action in llm_mappings.items()
    ]
elif not isinstance(llm_mappings, list):
    raise SystemExit("tools.json 必须包含 mappings 对象或数组")

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

trainer_names = set()
for index, action in enumerate(trainer_actions):
    if not isinstance(action, dict) or not {"name", "action_type", "description", "parameters"}.issubset(action):
        raise SystemExit(f"trainer_actions 第 {index + 1} 项缺少标准字段")
    name = action["name"]
    schema = action["parameters"]
    if not isinstance(name, str) or not name or name in trainer_names:
        raise SystemExit(f"Trainer 动作名称无效或重复：{name!r}")
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise SystemExit(f"Trainer 动作 {name} 的 parameters 必须是 object schema")
    if not isinstance(schema.get("properties"), dict) or not isinstance(schema.get("required", []), list):
        raise SystemExit(f"Trainer 动作 {name} 的 parameters 不完整")
    if action.get("visibility") not in {"trainer", "visible", "trainer-visible"}:
        raise SystemExit(f"Trainer 动作 {name} 的 visibility 必须为 trainer/visible")
    if action.get("action_type") == "llm_generate":
        raise SystemExit("llm_generate 只是执行计划标记，不能出现在 trainer_actions")
    trainer_names.add(name)

task = json.loads(task_path.read_text(encoding="utf-8"))
action_names = {
    item.get("field")
    for item in task.get("environment", [])
    if item.get("type") == "action"
}
plans = payload.get("task_action_plans") if isinstance(payload, dict) else None
if isinstance(plans, dict):
    plans = [
        {"task_action": task_action, **plan}
        for task_action, plan in plans.items()
        if isinstance(plan, dict)
    ]
elif not isinstance(plans, list):
    raise SystemExit("tools.json 必须包含 task_action_plans 对象或数组")
planned_actions = set()
for index, plan in enumerate(plans):
    if not isinstance(plan, dict) or not {"task_action", "steps"}.issubset(plan):
        raise SystemExit(f"task_action_plans 第 {index + 1} 项缺少 task_action 或 steps")
    task_action = plan["task_action"]
    steps = plan["steps"]
    if task_action not in action_names or task_action in planned_actions:
        raise SystemExit(f"任务 action 执行计划无效或重复：{task_action!r}")
    if not isinstance(steps, list) or not steps:
        raise SystemExit(f"任务 action {task_action} 必须至少包含一个执行步骤")
    for step in steps:
        if not isinstance(step, dict) or step.get("type") not in {"tool", "llm_generate"}:
            raise SystemExit(f"任务 action {task_action} 存在无效步骤类型")
        if step["type"] == "tool":
            if not {"llm_tool", "trainer_action"}.issubset(step):
                raise SystemExit(f"任务 action {task_action} 的 tool 步骤必须包含 llm_tool 和 trainer_action")
            if step["llm_tool"] not in llm_names or step["trainer_action"] not in trainer_names:
                raise SystemExit(f"任务 action {task_action} 引用了不存在的工具或 Trainer 动作")
        elif set(step) - {"type", "depends_on", "parallel_group"}:
            raise SystemExit(f"任务 action {task_action} 的 llm_generate 步骤不能包含工具参数")
    planned_actions.add(task_action)
if planned_actions != action_names:
    missing = sorted(action_names - planned_actions)
    extra = sorted(planned_actions - action_names)
    raise SystemExit(f"task_action_plans 覆盖不完整，缺少={missing}，多余={extra}")

mapped_llm = set()
for mapping in llm_mappings:
    if not isinstance(mapping, dict) or not {"llm_tool", "trainer_action"}.issubset(mapping):
        raise SystemExit("每个 mappings 项必须包含 llm_tool 和 trainer_action")
    if mapping["llm_tool"] in mapped_llm:
        raise SystemExit(f"LLM 工具存在重复映射：{mapping['llm_tool']}")
    if mapping["llm_tool"] not in llm_names or mapping["trainer_action"] not in trainer_names:
        raise SystemExit(f"无效的 LLM 工具映射：{mapping}")
    mapped_llm.add(mapping["llm_tool"])
if mapped_llm != llm_names:
    raise SystemExit("每个 LLM 工具都必须有且只有一个 Trainer action 映射")
mapping_by_llm = {mapping["llm_tool"]: mapping["trainer_action"] for mapping in llm_mappings}
for plan in plans:
    for step in plan["steps"]:
        if step.get("type") == "tool" and mapping_by_llm[step["llm_tool"]] != step["trainer_action"]:
            raise SystemExit(f"任务 action {plan['task_action']} 的工具映射与 mappings 不一致")
for required_name in ("ask_user",):
    if required_name in action_names and required_name not in trainer_names:
        raise SystemExit(f"tools.json 缺少 Trainer 动作：{required_name}")
PY
  then
    echo "Code Agent 生成的 tools.json 不符合标准工具 schema" >&2
    return 5
  fi

  if [[ "$runtime" == "auto" ]]; then
    if command -v container >/dev/null 2>&1; then runtime="container"
    elif command -v docker >/dev/null 2>&1; then runtime="docker"
    else echo "未找到 container 或 docker CLI"; return 6
    fi
  fi

  case "$runtime" in
    none)
      touch "$output_path/OK"
      ;;
    docker)
      build_args=(--context "$output_path" --tag "$tag")
      if [[ "$start" == "true" ]]; then build_args+=(--start); fi
      "$project_dir/scripts/build_docker_sandbox_image.sh" "${build_args[@]}"
      ;;
    container)
      command -v container >/dev/null 2>&1 || { echo "未找到 container CLI"; return 6; }
      container build --tag "$tag" "$output_path"
      touch "$output_path/OK"
      if [[ "$start" == "true" ]]; then
        exec container run --rm --publish 8080:8080 --mount "type=bind,source=$(cd "$output_path/data" && pwd),target=/workspace/data" "$tag"
      fi
      ;;
    *)
      echo "不支持的 runtime：$runtime；可选值为 none、docker、container、auto"
      return 2
      ;;
  esac
  echo "沙箱构建完成标记：$output_path/OK"
  echo "Code Agent 已完成沙箱开发：$output_path"
}

log_file="$output_path/agent.log"
if [[ "$background" == "true" ]]; then
  (run_agent_and_finalize) >"$log_file" 2>&1 </dev/null &
  pid=$!
  echo "Code Agent 已后台启动：PID=$pid"
  echo "日志文件：$log_file"
  echo "查看进程：ps -p $pid"
  echo "查看日志：tail -f $log_file"
else
  run_agent_and_finalize
fi

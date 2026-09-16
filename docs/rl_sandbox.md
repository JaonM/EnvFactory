# RL 沙箱构建与 Agent 工具协议

## Code Agent 开发沙箱

Code Agent 的职责是根据任务输入在独立目录中实际开发沙箱，而不是运行预先实现好的通用沙箱。使用：

```bash
./scripts/develop_sandbox_with_agent.sh \
  --agent codex \
  --input examples/clothing_materials_task.json \
  --output output/sandbox/agent_clothing_materials_sandbox \
  --runtime docker
```

该命令默认后台运行，返回 Agent PID；开发输出和后续文件校验/镜像构建日志写入目标目录的 `agent.log`。需要当前终端等待完成时，增加 `--foreground`。

支持 `codex`、`claude`、`opencode`。脚本会把任务复制到 Agent 工作目录，附加开发规范，然后在该目录内调用对应 CLI。Agent 必须生成 `spec.md`、业务状态机、工具接口、奖励函数、测试和容器文件；脚本会在 Agent 返回后检查必需文件。

对应的 Agent 命令是非交互模式：

- Codex：`codex exec --approve-for-me`
- Claude Code：`claude --dangerously-skip-permissions --print`
- OpenCode：`opencode run`

脚本不请求审批、不读取 stdin、不调用 sudo。Agent CLI 是否支持这些参数取决于本机安装版本；脚本不替 Agent 执行任务逻辑。

## 构建命令

任务输入为 JSON 文件，Code Agent 会在独立工程目录中开发状态机、业务数据、动作接口、奖励函数和容器文件：

```bash
./scripts/develop_sandbox_with_agent.sh \
  --agent codex \
  --input examples/clothing_materials_task.json \
  --output output/sandbox/agent_clothing_materials_sandbox \
  --runtime docker
```

脚本默认后台运行，使用 `--foreground` 等待完成。Agent 完成后，脚本校验 `spec.md`、包含 `llm_tools` 和 `trainer_actions` 的标准 `tools.json`、Dockerfile、构建脚本和运行脚本，并按参数构建镜像。流程成功完成后会在沙箱工程根目录生成 `OK` 文件；该文件表示 Agent 开发、文件校验和镜像构建（如启用）均已完成。Agent 必须自行分析任务 action 的参数需求，任务输入不会提供参数 schema；识别类动作必须区分 Agent 的预测输入和环境返回的识别结果。

Code Agent 的非交互调用方式：

```bash
codex exec --approve-for-me -- "./scripts/develop_sandbox_with_agent.sh --agent codex --input examples/clothing_materials_task.json --runtime docker"
```

也可以使用：

```bash
claude -p "运行 ./scripts/develop_sandbox_with_agent.sh --agent claude --runtime docker"
opencode run "运行 ./scripts/develop_sandbox_with_agent.sh --agent opencode --runtime docker"
```

Code Agent 是否允许执行命令由其自身运行参数决定；本项目脚本自身保持完全非交互。

## 容器服务

每个 Code Agent 生成的沙箱应在容器中提供工具服务。默认规范要求服务监听 `8080` 端口：

```text
GET  /health
POST /v1/reset
GET  /v1/observation
GET  /v1/actions
POST /v1/step
POST /v1/ask_user
GET  /v1/reward
```

`POST /v1/step` 请求示例：

```json
{
  "action_type": "identify_material",
  "params": {"item": "白色衬衫", "material": "棉"}
}
```

响应包含：

```json
{
  "observation": {},
  "result": {},
  "reward": 0.2,
  "reward_details": {},
  "done": false
}
```

每个动作都会推进离散时间，动作导致的状态修改会写入 SQLite。Agent 只能通过观测和工具结果获得允许暴露的信息，隐藏状态不会直接返回。

LLM 工具、Trainer 原子动作、调用映射、任务 action 执行计划、参数、返回结构和 HTTP 映射位于生成工程的 `tools.json`。LLM 只使用 `llm_tools`，RL Trainer 使用 `trainer_actions`、`task_action_plans` 和 `trainer_control` 与沙箱交互。一个任务 action 可以拆成串行、并行或混合步骤，也可以由 LLM 直接生成最终文本而不调用工具。LLM 工具描述只保留自然语言，参数角色等内部元数据不得写入 description。

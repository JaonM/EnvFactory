# 沙箱环境规范

## 目标

每个任务生成一个独立的沙箱工程。沙箱是离散时间状态机，Agent 每执行一个动作都会消耗至少一个时间单位；状态、业务数据、事件和奖励计算结果都可以持久化并复现。

## 输入

沙箱输入为一条任务记录：

- `task`：用户任务描述
- `environment`：用户画像、任务信息、状态、隐藏状态、动作、状态转移规则和终止条件
- `metrics`：step/state/terminal/trajectory 级奖励指标

## Agent 接口

- `reset()`：重置任务状态并返回初始观测
- `observation()`：获取当前 Agent 可见观测
- `available_actions()`：获取可执行动作定义
- `step(action_type, params)`：执行一个动作，推进离散时间并返回观测、奖励和终止状态
- `ask_user(prompt)`：向模拟用户提问，推进时间并返回用户消息
- `reward()`：计算当前状态下的奖励明细

## 工具定义

工程必须生成 `tools.json` 供 RL Trainer 使用。每个工具至少包含标准函数工具字段：

```json
{
  "name": "record_clothing_items",
  "description": "记录用户提供的衣物信息",
  "input_schema": {
    "type": "object",
    "properties": {
      "items": {
        "type": "array",
        "items": {"type": "string"},
        "description": "衣物名称或材质标签列表"
      }
    },
    "required": ["items"],
    "additionalProperties": false
  },
  "transport": "POST /v1/step",
  "returns": {"observation": "object", "reward": "number", "done": "boolean"},
  "visibility": "public"
}
```

任务输入中的 `action` 只声明动作名称，不声明参数。Agent 必须结合动作描述、状态转移规则和终止条件推导参数，并在业务代码、测试和 `tools.json` 中保持一致。

隐藏状态、转移规则、终止条件和内部业务数据不会出现在初始观测中。

## 持久化

默认使用 SQLite：

- `state`：当前状态和隐藏状态
- `events`：动作、参数、时间、结果和奖励
- `messages`：Agent 与模拟用户之间的消息

SQLite 适合单任务沙箱的事务性状态更新、回放和容器内持久化；后续可替换为 PostgreSQL 适配器。

## Apple Container

Apple Container 作为 OCI 容器运行时使用。每个生成工程只提供一个标准 `Dockerfile`，同时提供 `container_build.sh` 和 `container_run.sh`；沙箱数据通过 `/workspace/data` 挂载到宿主机。

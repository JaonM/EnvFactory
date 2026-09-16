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

## 工具与动作接口

工程必须生成 `tools.json`，明确区分 LLM 工具和 RL Trainer 动作接口：

- `llm_tools`：暴露给 LLM，供 LLM 生成结构化工具调用。
- `trainer_actions`：暴露给 RL Trainer，负责把 LLM 工具调用解析后真正执行到沙箱。
- `trainer_control`：RL Trainer 控制沙箱生命周期和读取结果的接口，例如 `reset`、`get_observation`、`get_reward`。

`llm_tools` 必须使用标准函数工具字段（与 LLM API 的 tools 参数一致）：

```json
{
  "type": "function",
  "function": {
    "name": "record_clothing_items",
    "description": "记录用户提供的衣物信息",
    "parameters": {
      "type": "object",
      "properties": {
        "items": {
          "type": "array",
          "items": {"type": "string"},
          "description": "衣物名称或材质标签列表"
        }
      },
      "required": ["items"]
    }
  }
}
```

对应的 `trainer_actions` 示例：

```json
{
  "name": "record_clothing_items",
  "action_type": "record_clothing_items",
  "description": "执行记录衣物动作",
  "parameters": {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {"type": "string"}}},
    "required": ["items"],
    "additionalProperties": false
  },
  "transport": "POST /v1/step",
  "request": {"action_type": "record_clothing_items", "params_from": "input"},
  "returns": {"observation": "object", "reward": "number", "done": "boolean"},
  "visibility": "trainer"
}
```

LLM 工具必须通过根级 `mappings` 映射到可执行动作，例如 `{"llm_tool":"record_clothing_items","trainer_action":"record_clothing_items"}`。任务输入中的 `action` 只声明动作名称，不声明参数。Agent 必须结合动作描述、状态转移规则和终止条件推导参数，并在业务代码、测试和两套接口定义中保持一致。

### 参数设计原则

参数不是动作描述中所有名词的简单罗列。Agent 应先判断字段角色：

- `selector/input`：Agent 已经知道、用于选择对象或提供证据的输入，例如 `clothing_id`、`basis`。
- `claim/guess`：Agent 为获得奖励而提交的预测，例如 `predicted_material`；环境用隐藏真值校验它。
- `result`：环境执行后返回的识别结果，不应作为识别动作的必填输入。
- `hidden_ground_truth`：只存在于环境内部，不能进入 LLM 工具或 Trainer 动作的输入 schema。

以“识别衣物材质”为例，推荐将动作设计为预测提交：

```json
{
  "name": "identify_material",
  "description": "根据衣物标签或描述提交材质判断",
  "parameters": {
    "type": "object",
    "properties": {
      "clothing_id": {"type": "integer", "description": "待识别衣物的 ID"},
      "predicted_material": {"type": "string", "description": "Agent 判断的材质"},
      "basis": {"type": "string", "description": "作出判断所依据的标签或描述"}
    },
    "required": ["clothing_id", "predicted_material"],
    "additionalProperties": false
  }
}
```

这里 `predicted_material` 是 Agent 的提交内容，不是环境真值。环境应从隐藏业务数据读取真实材质，比较两者后返回验证结果和奖励。若动作语义是“查看/读取材质”而非“提交判断”，则输入只需要 `clothing_id`，材质应作为环境返回结果；不能为了填充 schema 再要求 Agent 输入材质。

隐藏状态、转移规则、终止条件和内部业务数据不会出现在初始观测中。

## 持久化

默认使用 SQLite：

- `state`：当前状态和隐藏状态
- `events`：动作、参数、时间、结果和奖励
- `messages`：Agent 与模拟用户之间的消息

SQLite 适合单任务沙箱的事务性状态更新、回放和容器内持久化；后续可替换为 PostgreSQL 适配器。

## Apple Container

Apple Container 作为 OCI 容器运行时使用。每个生成工程只提供一个标准 `Dockerfile`，同时提供 `container_build.sh` 和 `container_run.sh`；沙箱数据通过 `/workspace/data` 挂载到宿主机。

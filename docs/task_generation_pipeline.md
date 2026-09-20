# 任务生成流程

`examples/generate_task.py` 统一使用分阶段外部 LLM 流程。每个任务先随机选择一个 `task_intent`，再由独立的外部 OpenAI 兼容 LLM 按该意图生成任务；进入下一阶段前进行 JSON 结构校验。

当前任务意图包括：`query`（查询）、`explain`（解释）、`compare`（比较）、`recommend`（推荐）、`diagnose`（诊断）、`modify`（修改）、`execute`（执行）、`plan`（规划）、`summarize`（总结）、`create`（创建）、`extract`（提取）、`classify`（分类）、`validate`（验证）、`audit`（审查）、`calculate`（计算）、`estimate`（估算）、`schedule`（排程）、`monitor`（监控）、`troubleshoot`（排障）、`transform`（转换）、`decide`（决策）和 `simulate`（模拟）。只有 `plan` 意图允许生成策划或执行方案，其他意图必须保持对应的目标和输出形式。可通过 `--task-intent` 指定意图，不指定时随机选择。

```text
主题/关键词
  1. task_description
  2. environment_data（内部拆分为多个独立提示词）
  3. user_profiles 与 user_scripts（画像独立，剧本参考任务和业务环境）
  4. dialogue_sessions（独立 User/Agent LLM 角色）
  5. agent_actions
  6. openai_tools
  7. observations_rewards
       ↓
     task.json
```

最终 `task.json` 的顶层字段是下游沙箱使用的唯一任务定义；任务文件不包含 `constraints` 字段；`artifacts` 只保存业务数据、用户模拟、工具文件、媒体生成和 Pipeline 版本等文件引用，不重复保存任务、动作、工具或指标。沙箱构建脚本根据任务顶层字段派生只读的 `BUILD_CONTRACT.json`。

第二阶段不是一个大提示词，而是按依赖顺序拆成以下独立提示词，每一步都校验输出后才进入下一步：

1. `environment_entities`：识别完成任务所需的最小必要业务实体，只保留需要查询、修改或评测的业务事实。
2. `environment_table_design`：根据最小必要实体设计原子数据库表，只生成表结构，不生成 rows。关系数据遵循 3NF，消除重复组、部分依赖和传递依赖；但 3NF 不意味着机械拆表。优先使用最少数量的表；只有存在独立生命周期、独立查询/更新需求或明确关系时才拆表或建立关联表，静态说明优先作为字段、枚举、JSON 或文本保存。
3. `environment_table_data.<table_name>`：按表并发生成完整、非空、可直接持久化的 rows，单张表失败可以单独重试。
4. `environment_data_consistency`：检查并修正主键、外键、字段完整性、业务关系和任务覆盖度，同时生成业务记录汇总。
5. `environment_data_document`：根据最终表结构和数据生成供 Code Agent 使用的持久化说明文档。
6. `environment_media_generation`：仅当第一阶段声明需要媒体时生成媒体数据程序、依赖、入口和输出目录。

这些提示词之间传递结构化结果，业务数据只根据任务描述和任务要求进行模拟。每张表的 schema 和 rows 会分别写入 `schemas/<table>.json` 与 `rows/<table>.jsonl`，`task.json` 通过 `artifacts.data_manifest` 保存文件清单；Code Agent 根据清单读取文件并初始化数据库。实体、表、数据、文档、媒体、动作、工具和奖励阶段各自只负责自己的输出。媒体不是每个任务的必选项，第一阶段根据 `requirements.input_modalities` 判断是否执行媒体生成；媒体生成代码由 Code Agent 在沙箱构建阶段执行，媒体识别和媒体评测不属于任务生成阶段。

生成结果的 `artifacts` 只包含 `data_manifest`、可选的 `media_generation`、用户模拟 manifest、`tools_manifest` 和 Pipeline 版本信息。`tools_manifest.file` 指向独立的 `tools.json`，该文件是顶层 OpenAI Function Tool 数组。动作、工具、指标和奖励定义直接位于 `task.json` 顶层；工具生成同时参考原子动作、业务环境和模拟对话，根据实际操作边界、信息来源和交互语义确定工具名、描述和参数；所有 Agent-用户交互动作统一映射为 `ask_user`。Code Agent 读取数据清单和工具 schema 后实现沙箱持久化与工具逻辑。

用户画像和用户剧本由两个独立提示词生成。用户画像不强制与任务主题或业务环境相关，但要求结构化描述 `profile_id`、身份背景、年龄阶段、职业或人生阶段、地域语境、教育背景、知识水平、目标动机、沟通风格、语言习惯、决策方式、风险容忍度、耐心、信任度、信息披露方式、提问方式、反馈方式、资源和时间敏感度、无障碍需求、挫败触发点、误解或偏见、已知事实、未知事实及行为倾向。数组字段应提供多个特征，画像之间要有明显差异；画像只影响用户的表达、节奏和决策，不提供任务业务真值。用户剧本必须同时参考任务描述和完整业务环境数据，采用嵌套决策树：节点包含 `node_id`、`user_behavior`、`branches`，分支包含 `branch_id`、`condition`、`next`。每个剧本应尽可能提供丰富且不重复的分支，通常覆盖 4--8 个分支和 2--4 层路径，包含信息补充、追问、接受、拒绝、纠正、犹豫、沉默、转移话题和结束等行为。User LLM 每轮根据当前对话选择一个匹配分支，不把分支树压扁为单一线性脚本。剧本生成不依赖用户画像，也不直接参与动作或奖励设计。生成每个 session 时，从用户画像集合中随机选择一个 profile 与当前剧本组合，形成不同的用户表达和决策风格，并在 session 中记录 `profile_id`；重新生成同一个 `task-N` 前会清理旧目录，避免残留中间产物混入本次结果。失败任务的部分产物也会清理，成功任务只保留 `task.json` 及其清单引用的运行依赖文件。

对话生成使用两个独立的外部 LLM 角色：User LLM 按画像和剧本发起或继续用户消息，Agent LLM 读取任务描述、业务环境、用户画像和当前对话进行回答。每个 session 都维护 `turn_count` 和 `termination_reason`，终止条件包括用户满意、用户拒绝、用户退出、Agent 完成、无进展以及达到最大消息数 `max_dialogue_turns`（默认 12）；每个 session 至少生成 4 条交替消息。终止信号由对应角色 LLM 提议，控制器校验并执行实际停止，最大轮数始终由控制器强制执行。

用户画像、用户剧本和对话 session 不内嵌到 `task.json`，而是写入 `user_simulation/` 目录，并由 `user_simulation_manifest` 引用：画像保存为 `user_profiles.json`，剧本保存为 `user_scripts.json`，每个 session 保存为独立 JSON 文件。用户画像和剧本用于对话模拟；模拟对话会作为 `agent_actions` 的输入，用于识别真实 Agent 交互边界和工具链，但不作为奖励设计或状态转移的输入；状态转移由后续 Code Agent 在沙箱中实现。

`agent_actions` 阶段先分析任务目标依赖的业务实体、表、字段和业务事实，再结合模拟对话反推 Agent 的隐含操作。模拟对话只作为行为证据：需要分析 Agent 为什么能作答，识别其隐含的查询、读取、筛选、计算、比较、提交和用户交互，而不是照搬 Agent 的自然语言回答。凡是回答依赖持久化业务数据，必须拆出对应的查询或业务处理动作；仅根据用户输入和通用语言能力组织最终表达的部分属于运行时回答生成，不作为 Trainer action；未被任务使用的数据不机械生成查询动作。每个动作必须是不可再由 Agent 独立细分的原子业务动作，并提供 `atomicity_rationale`、`inputs`、`outputs`、`preconditions`、`effects`；输入和输出是带有 `name` 与 `description` 的对象数组，前置条件和效果是字符串数组。后续 `openai_tools` 阶段结合任务、业务环境、模拟对话和动作输入设计参数；无 Agent 输入的动作使用空 object schema。

示例：

```bash
python examples/generate_task.py \
  --user-script-count 3 \
  --sessions-per-script 2 \
  --output output
```

每个任务最终写入 `output/task_artifacts/task-N/task.json`；业务数据、用户模拟文件和 `tools.json` 也保存在同一个 `task-N` 目录中。

任务生成已统一使用当前 pipeline，不再提供旧版生成模式或兼容分支。画像下位词扩展、constraints 构造和历史格式兼容解析均已删除。

观测与奖励设计规则：只保留与任务目标完成强相关的关键过程指标和目标结果指标，不能为每个普通动作机械创建指标；任务无需关键工具动作时 process 指标可以为空，存在多个关键动作时不限制过程指标数量。关键过程指标必须是 `hybrid`，使用精简字段 `target_action`、`evaluation_inputs`、`criteria` 和固定 `condition=llm_expected_tool_call_exact_match`。沙箱 Code Agent 根据该指标在实现评估器时调用外部 LLM，结合当前 Context、可用工具和 `criteria` 生成期望工具名及参数真值；随后由规则引擎对 Agent 实际工具名和规范化参数进行确定性精确比对，LLM 不直接输出最终过程分数，任务 JSON 也不嵌入完整 prompt 或 expected-call schema。结果指标判断任务目标是否完成或关键业务数据是否达到目标，可使用 `rule-based`、`model-based` 或 `hybrid`。惩罚指标只有在直接影响任务目标时才保留，用于偏离用户诉求、无效循环或业务数据偏离预期；工具选择错误和工具参数错误不作为惩罚项。

过程指标和结果指标是正反馈，`score_range` 固定为 `[0,1]`；惩罚指标是负反馈，`score_range` 固定为 `[-1,0]`。所有正反馈指标的权重之和独立归一化为 1，所有负反馈指标的权重之和独立归一化为 1；结果指标权重总和必须大于过程指标权重总和。最终奖励使用正负反馈分别归一化后相加的公式：

```text
R = clip(
  sum(w_pos_i * score_pos_i) / sum(w_pos_i)
  + sum(w_neg_j * score_neg_j) / sum(w_neg_j),
  -1, 1
)
```

其中正反馈得分位于 `[0,1]`，负反馈得分位于 `[-1,0]`；两组权重互不干扰，因此最终奖励始终落在 `[-1,1]`。`hybrid` 指标同时包含确定性 `condition` 和外部 LLM 语义 `criteria`。

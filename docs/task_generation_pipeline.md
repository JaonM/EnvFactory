# 任务生成流程

`examples/generate_task.py` 统一使用分阶段外部 LLM 流程。每个任务先随机选择一个 `task_intent`，再由独立的外部 OpenAI 兼容 LLM 按该意图生成任务；进入下一阶段前进行 JSON 结构校验。

当前任务意图包括：`query`（查询）、`explain`（解释）、`compare`（比较）、`recommend`（推荐）、`diagnose`（诊断）、`modify`（修改）、`execute`（执行）、`plan`（规划）、`summarize`（总结）、`create`（创建）、`extract`（提取）、`classify`（分类）、`validate`（验证）、`audit`（审查）、`calculate`（计算）、`estimate`（估算）、`schedule`（排程）、`monitor`（监控）、`troubleshoot`（排障）、`transform`（转换）、`decide`（决策）和 `simulate`（模拟）。只有 `plan` 意图允许生成策划或执行方案，其他意图必须保持对应的目标和输出形式。可通过 `--task-intent` 指定意图，不指定时随机选择。

```text
主题/关键词
  1. task_description
  1b. environment_plan（stateless/reference_data/stateful/external_capability）
  2. environment_data（仅 reference_data/stateful 生成业务实体和表）
  3. user_profiles 与 user_scripts（画像独立，剧本参考任务和业务环境）
  4. dialogue_sessions（独立 User/Agent LLM 角色）
  5. agent_actions
  5b. capability_plan（区分环境操作、Agent 推理和最终回答）
  6. openai_tools（只接收环境操作）
  6b. tool_implementation_specs（为可声明化的单表只读查询生成编译规格）
  7. reward_key_steps
  8. observations_rewards（基于关键步骤生成）
  8b. metric_implementation_specs（为规则指标生成可由共享运行时执行的 DSL）
  9. acceptance_contract（EnvFactory 独立业务验收基线）
  9b. acceptance_executable_scenarios（生成成功、失败、噪声和反事实业务轨迹）
  10. task_readiness（生成阶段硬门禁与复杂度重算）
       ↓
     task.json
```

`environment_plan` 只依据任务描述中的明确目标选择运行模式。`stateless` 任务直接处理用户提供的文本或结构化输入，跳过实体、表结构、rows、一致性和持久化文档生成；其 data manifest 明确标记 `environment_mode=stateless` 且允许空 `tables`。`extract`、`summarize`、`classify`、`transform`、`explain` 等意图在没有明确保存、更新或外部实时能力要求时会被确定性收敛为 `stateless`，避免用户模拟中的扩展请求污染正式任务范围。`reference_data` 生成只读资料，`stateful` 生成可持久化业务状态，`external_capability` 描述外部能力边界。

`acceptance_contract` 同时包含机器可执行的 `executable_scenarios` 和
`argument_probes`。新任务的 mutation 与黑盒验收直接消费这些结构化字段，
不再从自然语言步骤中用正则恢复工具调用；旧任务仍保留兼容解析。

最终 `task.json` 的顶层字段是下游沙箱使用的唯一任务定义；任务文件不包含 `constraints` 字段；`artifacts` 只保存业务数据、用户模拟、工具文件、媒体生成和 Pipeline 版本等文件引用，不重复保存任务、动作、工具或指标。任务生成在工具和奖励校验完成后，确定性生成 `requirements.runtime_interface`：它声明 HTTP 协议、系统接口、每一个 LLM tool 的请求 schema，以及 reward function 接口。沙箱构建脚本将 `task.json` 去除顶层 `actions` 字段后生成只读的 `BUILD_CONTRACT.json`，因此该 HTTP 约定会原样进入构建契约；不补充 task 中不存在的平台约束、能力、接口、评测或动作字段。沙箱只实现 `tools` 中声明的工具，不建立独立的 Trainer Action 注册表。

第二阶段不是一个大提示词，而是按依赖顺序拆成以下独立提示词，每一步都校验输出后才进入下一步：

1. `environment_entities`：识别完成任务所需的最小必要业务实体，只保留需要查询、修改或评测的业务事实。
2. `environment_table_design`：根据最小必要实体设计原子数据库表，只生成表结构，不生成 rows。关系数据遵循 3NF，消除重复组、部分依赖和传递依赖；但 3NF 不意味着机械拆表。优先使用最少数量的表；只有存在独立生命周期、独立查询/更新需求或明确关系时才拆表或建立关联表，静态说明优先作为字段、枚举、JSON 或文本保存。
3. `environment_table_data.<table_name>`：按表并发生成完整、非空、可直接持久化的 rows，单张表失败可以单独重试。
4. `environment_data_consistency`：检查并修正主键、外键、字段完整性、业务关系和任务覆盖度，同时生成业务记录汇总。
5. `environment_data_document`：根据最终表结构和数据生成供 Code Agent 使用的持久化说明文档。
6. `environment_media_generation`：仅当第一阶段声明需要媒体时生成媒体数据程序、依赖、入口和输出目录。

这些提示词之间传递结构化结果，业务数据只根据任务描述和任务要求进行模拟。每张表的 schema 和 rows 会分别写入 `schemas/<table>.json` 与 `rows/<table>.jsonl`，`task.json` 通过 `artifacts.data_manifest` 保存文件清单；Code Agent 根据清单读取文件并初始化数据库。实体、表、数据、文档、媒体、动作、工具和奖励阶段各自只负责自己的输出。媒体不是每个任务的必选项，第一阶段根据 `requirements.input_modalities` 判断是否执行媒体生成；媒体生成代码由 Code Agent 在沙箱构建阶段执行，媒体识别和媒体评测不属于任务生成阶段。

生成结果的 `artifacts` 只包含 `data_manifest`、可选的 `media_generation`、用户模拟 manifest、`tools_manifest` 和 Pipeline 版本信息。`tools_manifest.file` 指向独立的 `tools.json`。用户交互通过 Trainer-only 的 `POST /v1/user_simulator` 进入 User Simulator；待训练 Agent 的最终自然语言输出由 Trainer 通过 `POST /v1/agent_response` 提交并持久化为 `final_agent_response`，它不是 LLM Tool。业务工具使用 `POST /v1/tools/{tool_name}`，奖励使用 `GET /v1/reward`。这样最终回答无需伪装成工具，同时成功、失败和噪声验收轨迹可以真实覆盖最终结果奖励。

运行时接口同时声明 Trainer Bearer 鉴权、Agent/Trainer 访问边界、episode 隔离、seed 重置、幂等键、`/v1/replay` 回放、外部 LLM 适配器和 evaluator mock 配置。沙箱构建完成后，EnvFactory 会重新生成独立的外层 conformance；构建流程不会自动连接外部已运行服务，也不会默认启动沙箱。

`acceptance_contract` 由 EnvFactory 根据业务数据、原子动作、工具、关键奖励步骤和指标确定性生成，包含业务场景、数据不变量、工具非法输入、成功/失败奖励样例、数据变异策略和 mutation test 清单。Code Agent 不能修改该契约；外层验收使用它执行黑盒轨迹、前后数据快照、反事实奖励和实现缺陷注入测试。

用户画像和用户剧本由两个独立提示词生成。用户画像不强制与任务主题或业务环境相关，但要求结构化描述 `profile_id`、身份背景、年龄阶段、职业或人生阶段、地域语境、教育背景、知识水平、目标动机、沟通风格、语言习惯、决策方式、风险容忍度、耐心、信任度、信息披露方式、提问方式、反馈方式、资源和时间敏感度、无障碍需求、挫败触发点、误解或偏见、已知事实、未知事实及行为倾向。数组字段应提供多个特征，画像之间要有明显差异；画像只影响用户的表达、节奏和决策，不提供任务业务真值。用户剧本必须同时参考任务描述和完整业务环境数据，采用嵌套决策树：节点包含 `node_id`、`user_behavior`、`branches`，分支包含 `branch_id`、`condition`、`next`。每个剧本应尽可能提供丰富且不重复的分支，通常覆盖 4--8 个分支和 2--4 层路径，包含信息补充、追问、接受、拒绝、纠正、犹豫、沉默、转移话题和结束等行为。User LLM 每轮根据当前对话选择一个匹配分支，不把分支树压扁为单一线性脚本。剧本生成不依赖用户画像，也不直接参与动作或奖励设计。生成每个 session 时，从用户画像集合中随机选择一个 profile 与当前剧本组合，形成不同的用户表达和决策风格，并在 session 中记录 `profile_id`；重新生成同一个 `task-N` 前会清理旧目录，避免残留中间产物混入本次结果。失败任务的部分产物也会清理，成功任务只保留 `task.json` 及其清单引用的运行依赖文件。

启用噪声工具时，工具生成阶段至少生成一个噪声工具；默认上限为 3 时至少同时覆盖 `related_irrelevant` 和 `unrelated` 两类。候选工具还要经过独立的反事实有用性审查：凡是能提供原因分析证据、关键事实、比较依据、验证手段或排查资料的工具都不能作为噪声，会从候选集合中自动剔除；若候选集合被全部剔除，则注入一个不读取或修改业务状态的通用无关工具，避免正确的审计结论导致整项任务生成失败。噪声调用惩罚由共享运行时使用 `trajectory.events` 和 `none_tool_calls` 运算符确定性执行，不交给外部 LLM 判断。

对话生成使用两个独立的外部 LLM 角色：User LLM 按画像和剧本发起或继续用户消息；Agent LLM 只读取任务描述、公开业务环境和当前对话，不接收用户画像，也不能从隐藏画像推断用户属性。Agent 只能根据用户在对话中明确表达的内容进行回答。每个用户剧本分支包含 `should_end` 布尔字段；User LLM 消费该字段并在输出中返回 `message` 与 `should_end`，`true` 表示用户停止提问，`false` 表示继续对话。Agent LLM 不生成终止信号。每个 session 都维护 `turn_count` 和 `termination_reason`，控制器仅在 User LLM 输出 `should_end=true` 且达到最少对话轮数时结束；达到最大消息数 `max_dialogue_turns`（默认 12）只是安全兜底。每个 session 至少生成 4 条交替消息。

用户画像、用户剧本和对话 session 不内嵌到 `task.json`，而是写入 `user_simulation/` 目录，并由 `user_simulation_manifest` 引用：画像保存为 `user_profiles.json`，剧本保存为 `user_scripts.json`，每个 session 保存为独立 JSON 文件。用户画像和剧本用于对话模拟；模拟对话会作为 `agent_actions` 的输入，用于识别真实 Agent 交互边界和工具链，但不作为奖励设计或状态转移的输入；状态转移由后续 Code Agent 在沙箱中实现。

`agent_actions` 阶段先分析任务目标依赖的业务实体、表、字段和业务事实，再结合模拟对话反推 Agent 的隐含操作。其后新增 `capability_plan`，逐项分类为 `environment_operation`、`agent_reasoning` 或 `agent_response`。只有必须读取或修改沙箱私有状态、调用外部系统或使用确定性专用能力的 `environment_operation` 可以生成工具；比较、分析、选择和最终回答保留给待训练 Agent。每个动作必须提供 `atomicity_rationale`、`inputs`、`outputs`、`preconditions`、`effects`。后续 `openai_tools` 只接收通过资格判断的环境动作。

动作拆解阶段本身只接收业务模型，不接收业务实例数据。业务模型仅包含实体、表、字段、关系和约束定义；业务数据行、数据文档、隐藏真值、内部记录、具体字段值和数据库 ID 不会传入该阶段。这样可以让动作拆解判断“需要查询什么类型的数据”，但不能把某条真实模拟记录泄露到动作或工具定义中。

奖励生成分为两个阶段：`reward_key_steps` 先根据任务目标、业务模型、原子动作和工具梳理完成任务真正必需的关键步骤；`observations_rewards` 再读取该列表生成过程、结果和惩罚指标。过程指标的 `target_action` 必须属于 `reward_key_steps`，不会因为工具存在就自动获得过程奖励；如果没有关键工具步骤，process 指标可以为空。

每个 `rule-based` 指标必须具有对应的受限 DSL `metric_implementations`。无法编译的自然语言规则会被提升为 `model-based/external_llm_judge`，不会以空实现进入沙箱。任务输出前还会执行 task-level readiness 门禁，要求工具只绑定环境动作、规则指标实现完整、成功/失败轨迹齐全，并在启用噪声工具时要求 `noise_selection` 轨迹。成功场景使用独立生成的完整回答 fixture 并断言 `reward >= 0.5`；失败场景断言 `reward <= 0.2`；噪声场景断言 reward 非正。实际复杂度根据业务工具、关键步骤和指标数量重新计算。

共享运行时将 `final_agent_response` 保存为原始字符串，因此生成器禁止使用 `$.field` 访问虚构的文档字段；需要判断文档条目数量、结构或语义完整性的指标会提升为外部语义评估。噪声工具调用只保留一个确定性的 `penalty_noise_tool_usage`，其通过分数固定为 `0`、违规分数为 `-1`。readiness 同时检查指标 source/path 类型、正负奖励方向和噪声惩罚唯一性。没有业务工具而只有噪声工具的任务标记为 `task_readiness.training_profile=tool_abstention`，不会被误归类为多步工具调用训练样本。

示例：

```bash
python examples/generate_task.py \
  --user-script-count 3 \
  --sessions-per-script 2 \
  --output output
```

每个任务最终写入 `output/task_artifacts/task-N/task.json`；业务数据、用户模拟文件和 `tools.json` 也保存在同一个 `task-N` 目录中。

任务生成已统一使用当前 pipeline，不再提供旧版生成模式或兼容分支。画像下位词扩展、constraints 构造和历史格式兼容解析均已删除。

用户剧本的每个分支都包含布尔字段 `should_end`。该字段是用户终止信号：`true` 表示用户停止提问，`false` 表示继续对话。User Simulator 每轮消费当前剧本分支并输出 `message` 与 `should_end`；Agent Simulator 不生成终止信号。控制器仅在 User Simulator 输出 `should_end=true` 且达到最少对话轮数时结束 session，达到最大轮数只是安全兜底。

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

用户交互不生成 `ask_user` LLM Tool。工具生成阶段只生成业务工具；`POST /v1/user_simulator` 是仅供 RL Trainer 调用的内部接口，接收完整的 `messages` 对话数组，由 User Simulator 返回 `user_query` 和 `should_end`，不暴露给待训练 Agent。`GET /v1/reward` 同样标记为 `access=rl_trainer_only`，只有 Trainer 显式调用时才计算奖励，不暴露给待训练 Agent。

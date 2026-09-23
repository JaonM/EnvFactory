# env-factory

## 构建知识图谱

先确认 Neo4j 可访问，并在项目根目录配置 `.env`：

```bash
cp .env.example .env
```

至少配置以下变量：

```dotenv
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=your-model
LLM_TIMEOUT=60
WIKIPEDIA_API_URL=https://zh.wikipedia.org/w/api.php
WIKIPEDIA_TIMEOUT=10
# WIKIPEDIA_DUMP_DB=data/wikipedia.sqlite3
NEO4J_DATABASE=neo4j
GRAPH_SEEDS_FILE=data/scene_seeds.txt
LOG_LEVEL=INFO
```

`GRAPH_SEEDS_FILE` 指向种子词文本文件，每行一个词语；空行和以 `#` 开头的行会被忽略。
构建成功后，本轮发现的 scene 词语会自动追加到该文件；Neo4j 中已标记为扩展完成的词语下次会跳过。
`LOG_LEVEL` 支持 `DEBUG`、`INFO`、`WARNING` 等级别，默认使用 `INFO`。

执行默认图谱构建：

```bash
./scripts/build_graph.sh
```

指定扩展参数：

```bash
./scripts/build_graph.sh \
  --rounds 2 \
  --max-scene-nodes 500 \
  --max-search-requests 100 \
  --max-workers 2
```

脚本会并发调用 Wikipedia Action API，批量调用 LLM 抽取词语，增量合并 scene 节点和关系，最后写入 Neo4j。`task_type` 节点默认写入全部枚举值。

### 大规模构建

大规模任务建议先下载并建立本地索引：

```bash
./scripts/download_wikipedia_dump.sh
./scripts/index_wikipedia_dump.sh data/zhwiki-latest-pages-articles-multistream.xml.bz2
```

然后在 `.env` 中配置：

```dotenv
WIKIPEDIA_DUMP_DB=data/wikipedia.sqlite3
```

配置本地索引后，构建流程不再请求在线 Wikipedia API，直接使用 SQLite FTS5 检索页面正文。数据 dump 体积较大，下载和索引耗时取决于网络与磁盘性能。

## 生成长程任务

从 `SAME_EVENT_ELEMENT` 关系中随机抽取指定跳数的路径，并使用 LLM 生成任务描述：

```bash
./scripts/generate_task.sh --hops 3 --task-type Event
```

`--task-type` 支持 `QA`、`Event`、`Coding`、`Chat`、`Research`，可用英文逗号多选，例如 `--task-type QA,Event,Research`；每个任务会从指定类型中随机选择一个，省略时从全部类型随机选择。
`--hops 3` 表示每个任务随机选择 0、1、2 或 3 跳路径；0 跳表示随机选择一个 Scene 节点。
环境模式由 Pipeline 自动规划，不提供人工 `environment-mode` 参数。纯文本提取、总结、分类、转换和解释任务可使用 `stateless` 空业务数据环境；只有任务明确依赖只读资料或持久化状态时才生成业务实体、表和 rows。
任务描述会根据图谱路径和任务意图计算复杂度（`simple`、`standard`、`complex`），并据此控制描述和环境规模；可通过 `--task-style` 固定表达风格。
任务生成完成后，会继续根据任务描述和环境生成 `rule-based/model-based` 观测指标，写入 `Task.metrics`。为提高工具选择训练的辨别能力，默认生成 2–3 个噪声工具并覆盖相关无关与完全无关两类；噪声工具由共享运行时提供无任务关键副作用的通用实现，不占用业务 handler 实现成本，也不产生任务进度奖励。工具生成前会把动作分类为环境操作、Agent 推理和 Agent 回答，只有环境操作可以暴露为工具。任务规模不再绑定具体构建模型，结构有效性由 schema、契约、任务级 readiness、外层验收和训练可用性门禁统一判断。

沙箱通过普通 acceptance、outer conformance 和 mutation testing 后，还必须通过 `scripts/validate_training_readiness.py` 的 RL 环境硬门禁。该门禁执行结构化成功、失败、噪声及反事实轨迹，检查奖励可分离性、确定性和公开 observation 泄漏，并输出 `training_readiness.json`。
默认在 `output/task_artifacts/task-N/task.json` 写入每个任务的最终文件；可通过 `--output` 指定输出根目录。每次运行会扫描已有 `task-N`，从当前最大编号的下一号开始追加，绝不覆盖已有任务；并发进程通过原子目录预留避免编号冲突。Pipeline 运行日志默认追加写入 `output/task_generation.log`，也会输出到终端，可通过 `--log-file` 指定其他文件。日志记录任务级和阶段级开始、重试、成功、失败、耗时、产物路径和进度，不记录 Prompt 或凭据。任务默认并发生成 4 个，可通过 `--max-workers` 调整并发数。Neo4j 路径查询默认 10 秒超时，可通过 `--path-query-timeout` 调整。
使用 `--count N` 可批量新增 N 个独立的 `task-N` 目录；失败任务会清理本次预留目录，但不会修改任何历史任务。

任务环境由完整业务数据、数据说明文档、用户模拟素材、原子 Agent 动作和观测奖励设计组成，不再由任务生成阶段生成 `state`、`hidden_state` 或 `transition_rule` 记录。运行时状态、状态转移和终止执行逻辑由后续 Code Agent 在沙箱中根据业务数据和任务契约实现。
任务生成结果不再包含 `constraints` 字段。沙箱构建脚本根据 `task.json` 中的 requirements、actions 和 metrics 派生只读的 `BUILD_CONTRACT.json`，供 Code Agent 进行阶段验收；`spec.md` 是设计说明，`BUILD_CONTRACT.json` 是构建阶段派生的外层约束。
其中 `user_profile` 不再直接从任务描述臆造，而是从任务关键词随机选择 1 到全部关键词，并发查询其直接 `HIERARCHY` 下位节点，再由 LLM 润色生成；任务描述仅用于生成 `task_info`、状态和执行规则。
观测指标只保留与任务目标强相关的少量关键过程指标和目标结果指标。关键过程指标使用 `hybrid`：外部 LLM 生成当前上下文下的期望工具名和参数，规则引擎再对实际工具调用进行规范化比对；工具或参数错误不作为 penalty。结果指标用于判断目标是否完成，惩罚指标仅保留直接影响任务目标的偏离或无效循环。指标包含 `id`、`category`、`type`、`scope`、`condition/criteria`、`weight` 和 `score_range`，分别用于 step/state/terminal/trajectory 级别的奖励计算。

输出示例：

```json
{"task":"帮我挑选一套合适尺码的衣服并完成购买","task_type":"Event","complexity":"standard" ,"environment":[{"type":"user_profile","field":"interest","description":"用户兴趣","value":"服装","visibility":"observable"},{"type":"task_info","field":"goal","description":"任务目标","value":"提交订单"},{"type":"state","field":"order_status","description":"当前订单状态","value":"pending","visibility":"hidden"},{"type":"action","field":"submit_order","description":"提交订单","value":"submit_order","visibility":"hidden"},{"type":"transition_rule","field":"submit_order_rule","description":"提交订单后的状态变化","value":{"when":"submit_order is called","effect":"order_status becomes submitted"},"visibility":"hidden"},{"type":"termination","field":"success","description":"任务成功条件","value":["order_status == submitted"],"visibility":"hidden"}],"metrics":[{"id":"task_success","type":"rule-based","scope":"terminal","condition":"order_status == submitted","reward":1.0,"penalty":0.0,"once":true,"rubric":"订单提交成功","weight":1.0}]}
```

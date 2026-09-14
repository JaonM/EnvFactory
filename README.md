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
GRAPH_SEEDS_FILE=examples/scene_seeds.txt
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

`--task-type` 支持 `QA`、`Event`、`Coding`、`Chat`、`Research`；省略时随机选择。
`--hops 3` 表示每个任务随机选择 0、1、2 或 3 跳路径；0 跳表示随机选择一个 Scene 节点。
`--environment-mode` 支持 `complete`、`incomplete`、`random`，默认使用 `random` 随机生成任务环境。
任务描述会随机选择表达风格和复杂度，避免机械罗列关键词；可通过 `--task-style` 固定表达风格。
任务生成完成后，会继续根据任务描述和环境生成 `rule-based/model-based` 观测指标，写入 `Task.metrics`。
默认以追加方式输出到 `output/tasks.jsonl`，每生成一个任务立即写入一行 JSON；可通过 `--output` 指定其他文件。任务默认并发生成 4 个，可通过 `--max-workers` 调整并发数。Neo4j 路径查询默认 10 秒超时，可通过 `--path-query-timeout` 调整。
使用 `--count N` 可批量生成 N 个任务；所有任务均以 JSONL 方式逐行追加输出。

任务环境不是普通背景文本，而是由 `user_profile`、`task_info`、`state`、`hidden_state`、`action`、`transition_rule` 和 `termination` 记录组成。每条记录包含 `description` 字段说明业务含义，并带有 `visibility`：`observable` 会放入 Agent 初始观测，`hidden` 只保留在沙箱内部，需要通过用户交互、工具调用或环境事件获取。至少保留状态、动作、状态转移和终止条件，供 Agent 执行和评估。
观测指标包含 `id`、`type`、`scope`、`condition/criteria`、`reward`、`penalty`、`once` 和 `weight`，分别用于 step/state/terminal/trajectory 级别的奖励计算。

输出示例：

```json
{"task":"帮我挑选一套合适尺码的衣服并完成购买","task_type":"Event","environment-mode":"complete","environment":[{"type":"state","field":"order_status","description":"当前订单状态","value":"pending","visibility":"observable"},{"type":"action","field":"submit_order","description":"提交订单","value":{"params":{}},"visibility":"observable"},{"type":"transition_rule","field":"submit_order_rule","description":"提交订单后的状态变化","value":{"when":"submit_order is called","effect":"order_status becomes submitted"},"visibility":"hidden"},{"type":"termination","field":"success","description":"任务成功条件","value":["order_status == submitted"],"visibility":"hidden"}],"metrics":[{"id":"task_success","type":"rule-based","scope":"terminal","condition":"order_status == submitted","reward":1.0,"penalty":0.0,"once":true,"rubric":"订单提交成功","weight":1.0}]}
```

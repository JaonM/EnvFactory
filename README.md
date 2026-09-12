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
`--hops 3` 表示每个任务随机选择 1、2 或 3 跳路径。
`--environment-mode` 支持 `complete`、`incomplete`、`random`，默认使用 `random` 随机生成任务环境。
任务生成完成后，会继续根据任务描述和环境生成 `rule-based/model-based` 观测指标，写入 `Task.metrics`。
默认输出到 `output/tasks.jsonl`，每行一个 JSON 任务；可通过 `--output` 指定其他文件。
使用 `--count N` 可批量生成 N 个任务；数量大于 1 时输出 JSON 数组。

输出示例：

```json
{"task":"帮我挑选一套合适尺码的衣服并完成购买"}
```

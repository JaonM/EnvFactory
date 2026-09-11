# env-factory

## 构建知识图谱

先确认 Neo4j 和本地 SearXNG 已启动，并在项目根目录配置 `.env`：

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
SEARXNG_URL=http://localhost:8080
SEARXNG_TIMEOUT=10
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

脚本会并发搜索，批量调用 LLM 抽取词语，增量合并 scene 节点和关系，最后写入 Neo4j。`task_type` 节点默认写入全部枚举值。

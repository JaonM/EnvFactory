# 循环工程实验

`scripts/run_sandbox_build_loop.py` 使用可恢复实验调度器。默认不复用历史高分，构建模型仍固定为 `gpt-5.6-luna`。

正式良品率实验建议每轮生成 10 个新任务，固定使用
`direct_response=20%`、`simple_agentic=30%`、`multi_step_agentic=50%`。
实验分为两个阶段。开发阶段连续两轮达到任务良品率、条件构建良品率、端到端良品率、
分路由下限和合格样本均分后，自动进入生产准备留出集：

```bash
uv run python scripts/run_sandbox_build_loop.py \
  --generate-count 10 --max-rounds 0 --consecutive-rounds 2 \
  --validation live --build-mode clean \
  --sandbox-runtime docker \
  --experiment-seed 20260925 \
  --target-task-yield 0.85 --target-build-yield 0.80 \
  --target-end-to-end-rate 0.70 --target-qualified-mean 8.5 \
  --target-category-rate 0.60 \
  --output output/loop_experiment_v1 --hypothesis baseline
```

`--max-rounds 0` 是默认值，表示开发阶段没有轮数上限；仍受累计活跃时间预算和人工暂停控制。
默认 `--certification-profile production` 连续执行 3 个独立批次，每批使用 400 个留出请求并要求
至少形成 300 个全新任务；每个合格沙箱执行至少 10 次 rollout。各批次采用独立 seed、从零构建，
内容和 seed 不得与开发轮或其他留出批次
重复；数字、标点或轻微措辞改写形成的近重复任务家族同样不得跨越开发集和任何留出批次。
完成后由 `scripts/certify_training_materials.py` 根据置信区间、类别覆盖、重复率、跨分区家族隔离、运行时隔离、
奖励反事实和轨迹完整性生成 `production_readiness.json`。只有停止原因
`production_prepared_for_agentic_rl` 表示生产级训练素材准备认证通过。

需要快速验证候选实现时，可显式使用
`--certification-profile pilot --holdout-count 30 --holdout-rollout-episodes 3 --holdout-end-to-end-rate 0.70`。
Pilot 通过只表示可以进入更大规模认证，
不能表示生产准备完成。Pilot 只执行一个批次；生产认证的三个批次均为一次性留出集，失败后保留证据
并退出，不能回流调参后继续冒充留出集。

固定回归集可使用 `--task-ids 45,78,92,175`，不能与 `--generate-count` 混用。
`--build-mode repair` 允许固定任务继承上轮扩展实现，不复制 SQLite 状态及验收报告；其结果表示修复能力，不表示从零构建能力。新任务始终不继承旧任务实现。

## 配置和恢复

- `--threshold` 默认 8，统一采用大于等于；任务静态评分也是前置筛选。失败任务仍计入总请求数。
- `--validation offline` 只执行离线门禁，不能证明真实训练质量；默认 `live` 会追加真实模型 rollout。
- `--certification-profile production` 强制 `--sandbox-runtime docker`，每个候选环境必须实际构建内容寻址
  镜像并通过安全容器 smoke test；不能用 `none` 生成生产认证。pilot 可显式设置
  `--sandbox-runtime none`，但其结果仍只表示候选流程验证。并发 worker 使用互不相同的本地镜像 tag；
  smoke evidence 写入后立即移除临时镜像，避免大规模留出集发生 tag 串样或耗尽 Docker 存储。
- `--rollout-episodes 3 --rollout-steps 20` 控制开发阶段每个沙箱的轨迹次数和步数。开发阶段默认只要求至少
  一条成功证据；`--rollout-min-success-rate` 可提高该门槛。生产留出集默认使用 10 次 episode 和
  `--holdout-rollout-success-rate 0.6666666666666666`。
- live 模式的最终 10 分由离线可执行证据占 9 分、真实 rollout 占 1 分组成；任何 live 硬失败仍直接取消训练资格，不能依靠离线高分抵消。
- 离线评分通过后、任何外部模型调用前，流水线先运行数据治理审计并生成 `data_governance.json`。
  缺少合成数据来源声明、发现凭证或无法确定 Agent/User/Judge provider 时停止该样本；疑似 PII 会被记录，
  只有明确声明为合成 fixture 时才允许继续。该技术审计不替代实际外部端点所需的组织授权。
- live rollout 通过后会以 `SANDBOX_EVALUATOR_MOCK=0` 执行奖励反事实校准，写入
  `agentic_training_value_live.json`。真实 evaluator 的成功轨迹、失败轨迹、无工具、错参数、跳步、乱序和
  噪声轨迹不满足奖励分离时，样本仍不合格；离线 mock 报告不能替代该证据。
- production profile 保留完成冒烟验证的本地镜像直到 live rollout 结束；Agent 驱动器在宿主侧运行，
  但所有工具、状态持久化、User Simulator 和奖励调用都通过随机 loopback 端口进入实际容器。容器停止后
  才删除镜像。轨迹中的 image ID 与构建 provenance 不一致，或退回进程内 app，都会取消生产资格。
- live rollout 与奖励校准之间执行 policy-visible 隐私审计，写入 `trajectory_privacy.json`。User
  Simulator 内部 outcome/FSM 标签只进入 trainer-only evidence，Agent 可见 `result` 只保留实际传给
  Agent 的 `user_query`；发现凭证或隐藏控制字段时停止样本，不能进入便携训练素材包。
- `--build-timeout`、`--generation-timeout`、`--score-timeout`、`--rollout-timeout` 分阶段限时，超时清理进程组。
- `--max-total-seconds` 默认 259200，只累计实验进程的活跃执行时间；正常暂停不消耗预算。`runtime_state.json` 保存累计活跃时间和运行状态。质量循环不再因“停滞”或固定 20 轮提前结束；显式设置正数 `--max-rounds` 才启用轮数预算。基础设施超时属于异常中止而非质量收敛。
- 同一个实验目录只允许一个运行进程；每个任务完成即原子保存。相同命令重启可恢复，不重复执行已经完成的任务；未完成构建使用新的 attempt 目录。中断的生成不会自动重新抽样，缺失任务计为失败。
- 代码、配置或固定输入变化时必须换实验目录，以免把不同版本的结果混为一谈。实验还会冻结 Python、
  OS/架构、项目直接依赖的已安装版本以及 `pyproject.toml`、`uv.lock` 摘要；这些执行环境信息发生漂移时
  同样拒绝恢复。快照不采集主机名、路径、环境变量或密钥。旧版 history 不能直接作为新实验续跑。

`experiment.json` 保存配置及代码/输入指纹；每个生成请求在模型调用前写入
`sample_manifest.json`，记录 batch index、route、run seed、sample seed、可用环境能力和生成 provider。
每次 route attempt 还记录实际响应模型计数、finish reason、token usage 与 provider response ID 哈希；
不保存 prompt、响应正文、凭证或原始 response ID。生产认证会独立验证 provider、seed 和 attempt 链。
生成失败时保留目录并写入 `failure.json`，不会再删除失败样本的身份和归因证据。
`round-*/round_report.json` 保存任务状态；`history.json` 同时保存生成完成率、任务良品率、
条件构建良品率、端到端良品率、合格样本均分、分类型统计和停止原因。

## 模型默认值

Rollout Agent 使用任务生成的 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL`、`LLM_TIMEOUT`。
User Simulator 和奖励 LLM 按字段优先使用 `SANDBOX_LLM_*`；缺失或空值回退到对应 `LLM_*`（超时回退到 `LLM_TIMEOUT`）。显式配置某字段不会覆盖其他字段。没有任何模型/密钥配置时明确报错，不静默使用 mock。

CLI 从项目 `.env` 加载环境变量；独立沙箱/容器应由启动器注入这些变量，不复制 `.env` 或密钥进入沙箱文件。离线模式仍强制 mock。

## Rollout 证据

Agent 只获得题面、公开工具及观测，使用 JSON 动作协议自主调用工具或回复用户；不读取成功答案、奖励规则和隐藏业务状态。User Simulator 使用真实 LLM，协议失败的 fallback 会单独标记并阻止 live 验证通过。

保存 transition schema v2：每一步包含完整公开模型输入、原始输出、解析动作、工具/User Simulator 结果、
前后观察、奖励、terminated/truncated 和调用用量；同时保留 HTTP trace 与 replay。轨迹绑定任务哈希、
沙箱可执行输入摘要、模型及 provider 摘要。检查无操作高奖励、重复奖励不稳定，以及 stateful 目标不满足
却得到高奖励。模型没有完成任务记录为缺少成功证据，不直接断言环境有错。

这里的“User Simulator 结果”对策略侧仅指公开 `user_query`；outcome category、transition ID、match
status 和 termination reasoning 属于 trainer-only metadata。便携 JSONL 采用字段白名单重新投影，不会
因为未来在内部 rollout 结构中增加调试字段而自动把它们泄漏给训练策略。

开发阶段的默认 live 门要求至少一条成功轨迹、全部轨迹无已检测环境问题且无 LLM fallback；
发布留出集将成功率门提高到至少 2/3，并要求轨迹池同时包含成功和失败样本。
报告将失败责任区分为 `agent`、`environment` 和 `infrastructure`；Agent 未完成任务不再自动归咎于沙箱实现。
它仍是有限成功证据，不是训练质量的完备证明；同模型担任 Agent/User/Judge 有相关性偏差。

循环工程的版本单位是“冻结源码的一次实验”，不是在同一源码上反复抽样。每个候选版本先使用相同
`--experiment-seed` 做配对回归，再用新 seed 测量分布外良品率。一次候选版本只修复一个可复用根因；
不允许在运行中的实验目录对应源码上继续修改。最终候选冻结后由调度器自动运行生产留出集，
每个新构建还必须通过 `task_lineage.json` 证明生成任务与沙箱内运行任务字节一致，且没有发生旧式绝对
manifest 路径迁移；否则以 `TASK_LINEAGE` 失败，在评分和 live rollout 前终止。
`history.json` 的最终停止原因只有 `production_prepared_for_agentic_rl` 才表示生产准备认证通过；
`holdout_target_met` 仅属于 pilot 门禁。

生产认证通过统计、真实性和不可变性门禁后，还必须成功导出并复验
`training_materials_bundle/`。该包使用相对路径和内容摘要，可直接迁移到后续 RL 数据转换/训练系统；
仅存在指向本机输出目录的清单不再足以触发生产准备停止原因。包验证器会从原始 rollout 重建
transition 投影并验证 episode/step/终止关系；它证明训练素材可摄取，不宣称已经执行 RL 训练。
Bundle v10 内置 `certification.json`、`dataset_card.json` 和机器可读 `consumer_contract.json`；后者固定
transition JSON Schema、记录顺序、任务生成来源、环境重建入口以及 policy/trainer 可见性边界。数据集卡的构成统计、模型偏差、用途限制与
内部使用边界会同实际 transition 交叉验证，不能通过重新计算 manifest 哈希伪造更宽泛的认证结论。
生产 profile 还必须传入 `--bundle-signing-private-key` 与 `--bundle-trusted-public-key`（或在 `.env` 中设置
`ENVFACTORY_BUNDLE_SIGNING_PRIVATE_KEY`、`ENVFACTORY_BUNDLE_TRUSTED_PUBLIC_KEY`）。私钥应由组织密钥管理
系统保管且不得提交到仓库；实验清单只冻结公钥身份。未签名包和只携带自声明公钥的包均不能通过生产门禁。
任务会在每个训练类别内按近重复任务家族确定性分层为 80% train、10% validation、10% test；同一家族及
一个任务的所有 episode/transition 共享同一 split。生产包要求每个类别覆盖三个 split，且任务家族不得
跨类别或跨 split，防止模板变体泄漏到下游评估集。

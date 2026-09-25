# Agentic RL 训练素材生产准备认证

EnvFactory 的当前认证边界是 `production_prepared_for_agentic_rl`：证明任务、沙箱、工具、业务状态、
奖励和 rollout 轨迹满足下游 RL 系统接入前的素材契约。EnvFactory 本身不执行 RL 训练，因此该状态
不是“可直接开训”或训练平台上线认证；训练框架适配、算法兼容性、资源容量、RL 算法收敛、训练后
策略提升和跨模型泛化都必须在下游系统另行验证。

## 认证硬门禁

`scripts/certify_training_materials.py` 读取冻结实验的 `history.json`，使用以下默认策略：

- 连续 3 个独立留出批次，每批至少 300 个实际生成的新任务；开发阶段和各留出批次之间的 seed、
  任务内容均不重叠。
- 任务良品率不低于 90%，Wilson 95% 置信区间下界不低于 85%。
- 条件沙箱构建良品率不低于 90%，Wilson 95% 置信区间下界不低于 85%。
- 端到端训练就绪率不低于 85%，Wilson 95% 置信区间下界不低于 80%。
- 每个出现的训练类别端到端合格率不低于 75%。
- 三个留出批次分别保持训练构成骨架：`direct_response` 不少于 10% 且不高于 30%，
  `simple_agentic` 不少于 20%，`multi_step_agentic` 不少于 35%；未知或失败路由仍进入分母。
- 精确重复率为 0，近似重复率不高于 5%。
- 每个合格沙箱至少 10 个真实 rollout，三个批次累计至少 7500 个 episode；轨迹池同时包含成功和失败。
- 每个 episode 必须使用 transition schema v2，逐步保存模型输入、原始输出、解析动作、公开观察、
  工具或 User Simulator 结果、下一观察、奖励以及 terminated/truncated 标记；仅有 HTTP trace 不合格。
- policy-visible transition 只能包含 Agent 实际收到的输入、公开观察和公开结果。User Simulator 的
  outcome、FSM transition、match status，以及 replay、初末业务状态等保留为 trainer-only evidence；
  不得混入供策略摄取的 `transitions.jsonl`。
- episode 环境错误率不高于 0.1%，LLM fallback 为 0。
- User Simulator 调用协议有效率不低于 99.5%，并且至少产生一条真实用户响应证据。
- 真实 User Simulator 轨迹至少覆盖 3 类结果，同时包含成功/接受类结果和需要继续交互或恢复的结果，
  防止只验证“用户永远接受”的退化模拟器。
- 每个合格沙箱的固定 seed reset、episode 隔离、replay 稳定性、隐藏字段扫描和确定性验证全部通过。
- 每个合格沙箱必须实际完成 Docker 构建，并在禁网、只读根文件系统、非 root、丢弃全部 capabilities、
  `no-new-privileges` 和资源上限下通过容器内 pytest smoke test。基础镜像必须解析并固化为
  `sha256` 内容地址，测试依赖必须精确锁版本；镜像 ID、平台、Dockerfile 和依赖摘要由认证器独立复核。
  安全 smoke 容器还会使用 `importlib.metadata` 导出排序后的实际 Python 分发包名称与版本到
  `python_packages.json`；认证器复核其摘要、结构、唯一性，并确认每个无条件直接 pin 的安装版本一致。
  该 inventory 用于依赖可追溯，不等同于漏洞扫描、许可证判断或 SBOM 法务审批。
- 工具契约、真实业务结果、mutation resistance 和奖励反事实全部通过。
- 奖励假阳性率不高于 0.5%，假阴性率不高于 2%。
- 奖励反事实必须使用生产配置的真实 evaluator 重跑；mock 报告只用于离线构建测试，不能进入生产认证。
- 每份 rollout 保存任务 SHA-256、沙箱可执行输入摘要、Agent/User/Judge 模型及 provider 摘要；认证时重新
  计算并拒绝把其他任务或沙箱的轨迹挂接到当前样本。
- 每个成功任务保存任务生成 provider、配置模型、实际响应模型计数、逐次 route attempt seed、完成原因、
  token usage 和 provider response ID 的 SHA-256；不保存 prompt、响应正文、凭证或原始 response ID。
  认证器从冻结的 `sample_manifest.json` 独立重建该快照，拒绝模型身份、attempt 顺序、seed、任务类别或
  响应计数不一致的样本。该溯源证明素材由声明的生成流程产生，不证明模型输出本身正确。
- 每个任务必须声明业务数据为模型生成的合成数据且不包含真实用户数据。真实 rollout 前执行出站载荷
  审计，记录 Agent、User Simulator 和 Reward Judge 的 provider 身份与可见字段，扫描凭证及疑似 PII；
  发现凭证、缺少合成来源声明或缺少 provider 身份时禁止外发并取消样本资格。
  最终认证不会信任流水线写出的审计结论，而是对冻结的任务与业务 fixture 重新扫描，并要求命中路径、
  合成来源声明和报告完全一致。

数据治理审计只证明素材满足项目内的技术门禁，不等同于组织层面的联网、供应商或数据出境授权。
实际调用某个外部端点前，运行方仍须取得适用于该端点和这些载荷的明确授权。

认证采用比率和置信区间双门禁，避免小样本的高点估计被误认为稳定良品率。任务生成失败仍计入
总请求分母；沙箱构建率以通过任务门禁的任务为条件分母；端到端率使用所有请求作为分母。

## 单沙箱证据

`validate_training_readiness.py` 通过公开运行时接口验证：

- 相同 seed reset 恢复完全一致的业务基线；
- 两个 episode 的状态和 replay 不串扰；
- 重复读取 replay 内容及 trace hash 稳定；
- observation 不包含 ground truth、凭证或隐藏状态；
- 相同成功轨迹可确定性复现；
- 成功和失败轨迹的奖励可分离。

`validate_agentic_training_value.py` 进一步执行无工具回答、参数破坏、跳步、乱序和噪声工具等反事实，
验证任务工具确实必要、参数会影响结果、依赖顺序真实存在，并防止只输出正确措辞骗取高奖励。

`audit_data_governance.py` 在任何 live rollout 前验证任务的合成数据声明，扫描会进入公开任务、工具定义和
业务 fixture 的凭证与疑似 PII，并将目标 provider、允许的出站面和禁止出站字段写入
`data_governance.json`。报告只保存命中类型与 JSON 路径，不回写疑似敏感值。

`audit_trajectory_privacy.py` 在 live rollout 后扫描 Agent 实际可见的 messages、observation、action、
tool/User 公开结果和 next observation。凭证或 acceptance contract、ground truth、future user turns 等
内部控制字段一旦进入可见轨迹，该样本立即失去导出资格。生产认证器和便携包导出器都会针对原始 rollout
独立重算，不能用人工修改的干净报告绕过。

`build_docker_sandbox_image.sh` 将可达镜像标签解析为当前 Docker 平台的 manifest digest，以该内容地址
构建并回写最终 Dockerfile；随后在生产安全参数下执行镜像内测试，成功后才写
  `docker_image_metadata.json`。每个并发 attempt 使用由其输出路径派生的唯一镜像 tag，认证时要求 tag、
Dockerfile 基础镜像和 provenance 三者一致；验证后删除本地临时镜像，只把内容身份和可重建源码纳入
  训练素材。没有这份可复核证据的沙箱不能通过生产准备认证。
- 实验同时冻结 Python 实现与版本、操作系统/架构、项目直接依赖的实际安装版本，以及
  `pyproject.toml`、`uv.lock` 摘要。运行中或恢复时发生环境漂移会立即终止并要求使用新输出目录；
  生产认证会独立重算该快照，便携包和数据集卡也保留同一内容身份。快照不包含主机名、路径、环境变量
  或密钥。它证明认证执行环境可追溯，不要求训练消费端使用相同平台。

## 结果解释

- `production_prepared_for_agentic_rl`：生产级训练素材准备认证通过。
- `production_readiness_failed`：至少一个生产门禁失败，应根据 `production_readiness.json.failed_gates`
  修复通用根因后使用全新的留出集重新认证。
- `holdout_target_met`：仅 pilot 门禁通过，不能宣称生产准备完成。
- `offline_target_met`：只证明离线契约可执行，不能证明真实模型轨迹可采集。

认证报告会显式保存 `does_not_certify`，防止将训练前环境质量误表述为训练效果。

通过或失败时都会生成 `training_materials_manifest.json`，列出当前合格候选的任务 SHA-256、沙箱
可移植文件树 SHA-256、原始证据指纹、rollout 摘要哈希、类别、分数和 episode 数量，并记录认证时的
评估器源码摘要，再为整批清单生成 `dataset_sha256`。
认证要求每个合格样本都有唯一证据指纹；后续导入训练系统时应重新计算这些摘要，拒绝认证后被修改、
替换或额外混入的任务、沙箱和轨迹。只有认证报告 `certified=true` 时，这份清单才能作为正式输入。

生产认证还会原子生成 `training_materials_bundle/`。该目录不保留本机绝对路径，按内容身份保存每个
环境的任务、运行时代码、业务数据、验收证据和 `live_rollout.json`，并生成 `transitions.jsonl` 与
`bundle_manifest.json`。Bundle v8 同时包含去除本机路径的 `certification.json`、机器可读
`dataset_card.json`。JSONL 每行是一条可重建的 schema v2 transition，并携带任务、类别、episode、
任务生成模型与 provider 身份、生成 seed、Agent 模型和 User/Judge 模型身份。整个目录通过文件清册与
`bundle_sha256` 再次校验；导出失败或包校验失败时，即使此前统计门禁通过，也不会产生
`production_prepared_for_agentic_rl`。

便携包验证器还会从每个环境内的原始 `live_rollout.json` 重新生成预期 transition 流，并与
`transitions.jsonl` 逐条精确比较。校验范围包含 item/episode/step 引用、模型身份、parsed action 与原始
输出的一致性、observation 链、reward、usage 以及 terminated/truncated 终止语义，防止只重算文件哈希
就让语义损坏的轨迹通过。JSONL 由固定字段白名单投影产生；完整 trainer-only 证据仍保存在对应环境目录，
并由 manifest 的 `transition_visibility` 显式区分，避免下游把评估标签作为策略观测。

Bundle v8 还包含 `consumer_contract.json`：以 JSON Schema 固定 transition 记录字段，以机器可读形式声明
记录身份与排序、环境目录和 Docker 重建入口、运行接口来源，以及 policy input/output、环境反馈和
trainer-only 证据边界。验证器使用内置规范与文件逐项比较；即使同时修改契约并重算所有外层哈希，也不能
把 trainer-only 字段伪装成策略输入。历史 v3/v4 包仍可验证完整性，但不能满足当前生产认证的受信发布门禁。

生产发布还要求使用组织持有的 Ed25519 私钥对最终 `bundle_manifest.json` 生成 detached signature，并由
显式指定的受信任公钥复验。报告和包中只保存公钥 SHA-256 身份，不保存私钥、私钥路径或公钥内容。
因此重写整个包并重算所有普通哈希仍不能伪造认证来源。开发检查可以导出未签名包，但
`trusted_attestation=false`，不能触发 `production_prepared_for_agentic_rl`。

任务按内容身份在每个任务类别内确定性分配到 `train`、`validation`、`test`，目标比例为 80/10/10。
split 的最小单位是近重复任务家族而不是 transition：认证器使用与近重复率门禁相同的标准化与
3-shingle Jaccard 规则聚类，同一数字变体、轻微措辞变体和传递相似链整体进入同一集合；同一沙箱的
所有 episode 和 transition 也永远属于同一集合。
生产门禁要求三个集合都非空且每个任务类别在三个集合中均有覆盖；分组同时写入 item、JSONL、数据集卡
和消费契约，并由验证器独立重算。跨类别家族或跨 split 家族重叠都会使生产认证失败，避免下游
训练/评估泄漏或靠重写清单改变分组。

数据集卡记录任务类别、模型组合、同模型评估数量、episode 成败和 transition 数量，并明确只适合重建
沙箱、验证数据适配器、收集新鲜 on-policy rollout 和准备 policy-visible 输入。它显式禁止把本认证解释为
RL 收敛、训练后提升、跨模型泛化或任意离线 RL 算法兼容性证明。EnvFactory 不替组织声明数据分发权利；
便携包默认为 `internal_only_until_legal_and_security_review`。验证器从实际文件重新计算数据集卡统计，单纯
重写卡片并更新哈希不能改变这些边界。

导入前执行：

```bash
uv run python scripts/verify_training_materials.py \
  output/loop_experiment_v1/training_materials_manifest.json
```

验证器会分别重新计算任务文件、沙箱可移植文件树、rollout JSON 和整批清单摘要；任何文件变化、
轨迹数量变化或重复沙箱身份都会返回非零退出码。评估器版本作为 provenance 固化，但后续评估器升级
不会被误判成已认证沙箱遭到篡改。v1/v2/v3 清单仍可按各自规则验证，但应重新认证并升级为同时包含
执行环境和任务生成 provenance 的 v4 后再导入训练系统。

迁移或导入训练平台后可独立验证便携包：

```bash
uv run python scripts/export_training_materials.py \
  output/loop_experiment_v1/production_readiness.json \
  --output output/loop_experiment_v1/training_materials_bundle \
  --verify-only
```

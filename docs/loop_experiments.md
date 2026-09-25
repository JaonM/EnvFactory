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
  --experiment-seed 20260925 \
  --target-task-yield 0.85 --target-build-yield 0.80 \
  --target-end-to-end-rate 0.70 --target-qualified-mean 8.5 \
  --target-category-rate 0.60 \
  --output output/loop_experiment_v1 --hypothesis baseline
```

`--max-rounds 0` 是默认值，表示开发阶段没有轮数上限；仍受累计活跃时间预算和人工暂停控制。
默认 `--certification-profile production` 使用 400 个留出请求，要求至少形成 300 个全新任务；
每个合格沙箱执行至少 10 次 rollout。留出集采用独立 seed、从零构建，内容和 seed 不得与开发轮
重复。完成后由 `scripts/certify_training_materials.py` 根据置信区间、类别覆盖、重复率、运行时隔离、
奖励反事实和轨迹完整性生成 `production_readiness.json`。只有停止原因
`production_prepared_for_agentic_rl` 表示生产级训练素材准备认证通过。

需要快速验证候选实现时，可显式使用
`--certification-profile pilot --holdout-count 30 --holdout-rollout-episodes 3 --holdout-end-to-end-rate 0.70`。
Pilot 通过只表示可以进入更大规模认证，
不能表示生产准备完成。两种留出集都只执行一次，失败后保留证据并退出，不能回流调参后继续冒充留出集。

固定回归集可使用 `--task-ids 45,78,92,175`，不能与 `--generate-count` 混用。
`--build-mode repair` 允许固定任务继承上轮扩展实现，不复制 SQLite 状态及验收报告；其结果表示修复能力，不表示从零构建能力。新任务始终不继承旧任务实现。

## 配置和恢复

- `--threshold` 默认 8，统一采用大于等于；任务静态评分也是前置筛选。失败任务仍计入总请求数。
- `--validation offline` 只执行离线门禁，不能证明真实训练质量；默认 `live` 会追加真实模型 rollout。
- `--rollout-episodes 3 --rollout-steps 20` 控制开发阶段每个沙箱的轨迹次数和步数。开发阶段默认只要求至少
  一条成功证据；`--rollout-min-success-rate` 可提高该门槛。生产留出集默认使用 10 次 episode 和
  `--holdout-rollout-success-rate 0.6666666666666666`。
- live 模式的最终 10 分由离线可执行证据占 9 分、真实 rollout 占 1 分组成；任何 live 硬失败仍直接取消训练资格，不能依靠离线高分抵消。
- `--build-timeout`、`--generation-timeout`、`--score-timeout`、`--rollout-timeout` 分阶段限时，超时清理进程组。
- `--max-total-seconds` 默认 172800，只累计实验进程的活跃执行时间；正常暂停不消耗预算。`runtime_state.json` 保存累计活跃时间和运行状态。质量循环不再因“停滞”或固定 20 轮提前结束；显式设置正数 `--max-rounds` 才启用轮数预算。基础设施超时属于异常中止而非质量收敛。
- 同一个实验目录只允许一个运行进程；每个任务完成即原子保存。相同命令重启可恢复，不重复执行已经完成的任务；未完成构建使用新的 attempt 目录。中断的生成不会自动重新抽样，缺失任务计为失败。
- 代码、配置或固定输入变化时必须换实验目录，以免把不同版本的结果混为一谈。旧版 history 不能直接作为新实验续跑。

`experiment.json` 保存配置及代码/输入指纹；每个生成请求在模型调用前写入
`sample_manifest.json`，记录 batch index、route、run seed、sample seed 和可用环境能力。
生成失败时保留目录并写入 `failure.json`，不会再删除失败样本的身份和归因证据。
`round-*/round_report.json` 保存任务状态；`history.json` 同时保存生成完成率、任务良品率、
条件构建良品率、端到端良品率、合格样本均分、分类型统计和停止原因。

## 模型默认值

Rollout Agent 使用任务生成的 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL`、`LLM_TIMEOUT`。
User Simulator 和奖励 LLM 按字段优先使用 `SANDBOX_LLM_*`；缺失或空值回退到对应 `LLM_*`（超时回退到 `LLM_TIMEOUT`）。显式配置某字段不会覆盖其他字段。没有任何模型/密钥配置时明确报错，不静默使用 mock。

CLI 从项目 `.env` 加载环境变量；独立沙箱/容器应由启动器注入这些变量，不复制 `.env` 或密钥进入沙箱文件。离线模式仍强制 mock。

## Rollout 证据

Agent 只获得题面、公开工具及观测，使用 JSON 动作协议自主调用工具或回复用户；不读取成功答案、奖励规则和隐藏业务状态。User Simulator 使用真实 LLM，协议失败的 fallback 会单独标记并阻止 live 验证通过。

保存工具结果、用户交互、奖励、状态变化及调用用量；检查无操作高奖励、重复奖励不稳定，以及 stateful 目标不满足却得到高奖励。模型没有完成任务记录为缺少成功证据，不直接断言环境有错。

开发阶段的默认 live 门要求至少一条成功轨迹、全部轨迹无已检测环境问题且无 LLM fallback；
发布留出集将成功率门提高到至少 2/3，并要求轨迹池同时包含成功和失败样本。
报告将失败责任区分为 `agent`、`environment` 和 `infrastructure`；Agent 未完成任务不再自动归咎于沙箱实现。
它仍是有限成功证据，不是训练质量的完备证明；同模型担任 Agent/User/Judge 有相关性偏差。

循环工程的版本单位是“冻结源码的一次实验”，不是在同一源码上反复抽样。每个候选版本先使用相同
`--experiment-seed` 做配对回归，再用新 seed 测量分布外良品率。一次候选版本只修复一个可复用根因；
不允许在运行中的实验目录对应源码上继续修改。最终候选冻结后由调度器自动运行生产留出集，
`history.json` 的最终停止原因只有 `production_prepared_for_agentic_rl` 才表示生产准备认证通过；
`holdout_target_met` 仅属于 pilot 门禁。

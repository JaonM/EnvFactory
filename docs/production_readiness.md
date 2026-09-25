# Agentic RL 训练素材生产准备认证

EnvFactory 的当前认证边界是 `production_prepared_for_agentic_rl`：证明任务、沙箱、工具、业务状态、
奖励和 rollout 轨迹可以作为后续 RL 系统的生产级输入。它不证明 RL 算法收敛、训练后策略提升，
也不证明跨模型泛化；这些结论必须在接入训练框架后另行验证。

## 认证硬门禁

`scripts/certify_training_materials.py` 读取冻结实验的 `history.json`，使用以下默认策略：

- 至少 300 个实际生成的全新留出任务；开发阶段和留出集 seed、任务内容不重叠。
- 任务良品率不低于 90%，Wilson 95% 置信区间下界不低于 85%。
- 条件沙箱构建良品率不低于 90%，Wilson 95% 置信区间下界不低于 85%。
- 端到端训练就绪率不低于 85%，Wilson 95% 置信区间下界不低于 80%。
- 每个出现的训练类别端到端合格率不低于 75%。
- 精确重复率为 0，近似重复率不高于 5%。
- 每个合格沙箱至少 10 个真实 rollout，累计至少 2500 个 episode；轨迹池同时包含成功和失败。
- episode 环境错误率不高于 0.1%，LLM fallback 为 0。
- User Simulator 调用协议有效率不低于 99.5%，并且至少产生一条真实用户响应证据。
- 每个合格沙箱的固定 seed reset、episode 隔离、replay 稳定性、隐藏字段扫描和确定性验证全部通过。
- 工具契约、真实业务结果、mutation resistance 和奖励反事实全部通过。
- 奖励假阳性率不高于 0.5%，假阴性率不高于 2%。

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

## 结果解释

- `production_prepared_for_agentic_rl`：生产级训练素材准备认证通过。
- `production_readiness_failed`：至少一个生产门禁失败，应根据 `production_readiness.json.failed_gates`
  修复通用根因后使用全新的留出集重新认证。
- `holdout_target_met`：仅 pilot 门禁通过，不能宣称生产准备完成。
- `offline_target_met`：只证明离线契约可执行，不能证明真实模型轨迹可采集。

认证报告会显式保存 `does_not_certify`，防止将训练前环境质量误表述为训练效果。

通过或失败时都会生成 `training_materials_manifest.json`，列出当前合格候选的任务 SHA-256、沙箱
证据指纹、rollout 摘要哈希、类别、分数和 episode 数量，并为整批清单生成 `dataset_sha256`。
认证要求每个合格样本都有唯一证据指纹；后续导入训练系统时应重新计算这些摘要，拒绝认证后被修改、
替换或额外混入的任务、沙箱和轨迹。只有认证报告 `certified=true` 时，这份清单才能作为正式输入。

导入前执行：

```bash
uv run python scripts/verify_training_materials.py \
  output/loop_experiment_v1/training_materials_manifest.json
```

验证器会重新计算任务文件、沙箱实现与评测代码联合指纹、rollout JSON 和整批清单摘要；任何文件变化、
轨迹数量变化或重复沙箱身份都会返回非零退出码。

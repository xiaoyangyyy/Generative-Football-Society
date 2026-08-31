# 世界模型动作采用链路 V1

## 目标

本链路解决“世界模型可以预测，但几乎不改变实际动作”的结构性问题。它只允许通过独立验证质量门的动作证据进入正常采样器，不允许模型绕过物理规则、角色可行性或发布门。

```text
比赛状态
  -> checkpoint 独立验证质量门
  -> pass vs hold 反事实价值
  -> 置信度/单次预测确定性取较小值
  -> 有界概率策略混合（最大权重 0.35）
  -> 与教练干预组合
  -> 一个共享随机数
       |-> 无世界模型基线动作
       +-> 世界模型动作
  -> 执行动作
  -> 采用、概率位移、反事实改变和归因审计
```

## 当前质量授权

- 封存 checkpoint：`latent_wm_rollout_calibrated_candidate.pt`
- `pass_planner_quality = 0.176654`，超过 `min_planner_quality = 0.15`，允许以低权重进入动作策略。
- `shot_planner_quality = 0.005400`，未过门，射门策略保持关闭。
- 横传和持球没有独立动作头验证，不借用传球质量冒充已验证能力。

验证质量是硬授权门；当前观测覆盖率和单次预测不确定性只会继续降低融合权重，不会把已经通过独立验证的动作头错误地再次归零。

## 归因语义

每次动作机会保存：

- 物理/现有策略的基线概率；
- 世界模型混合后的概率；
- 共享随机数；
- 无世界模型时的反事实基线动作；
- 实际执行动作；
- 世界模型是否改变了动作；
- 总变差距离，即该机会的预期反事实动作改变概率；
- 是否存在外部计划覆盖或教练共干预；
- 是否允许把改变归因给直接世界模型策略。

外部计划覆盖和教练共干预不会进入独立归因分母。概率变化不等于实际动作变化；短样本可以出现非零总变差但零次随机边界命中，因此两者必须同时报告。

## 当前证据状态

新候选已在封存代码身份下完成预注册机制实验。旧 confirmatory/academic replication 结果仍属于旧代码身份，是历史记录而不是当前候选的结果证据；新的 V1 结果只确认动作采纳机制，不替代全长度赛果研究。

新的固定协议：

- 协议：`data/evaluation/action_adoption_protocol_v1.json`
- 运行器：`scripts/action_adoption_study.py`
- 两个臂：同 checkpoint 的 `M1_predict_only` 与 `M1_action_policy`
- 6个固定对阵 × 2个固定样本，共12个配对、24次15分钟运行
- 无中期分析、无可选停止、无训练、无供应商调用
- 结果只能确认或否定动作采用机制，禁止直接触发产品或学术晋级

查看状态：

```bash
python scripts/action_adoption_study.py --status
```

2026-08-31 的固定预算已完整执行：两个臂各12次，共24次，无材料偏差、无训练、无供应商调用。1954次动作机会中1951次受到非零策略影响，预期反事实动作变化25.246次、实际变化25次，12/12配对均出现可测行为变化；负对照臂保持零影响。全部冻结机制门通过，正式状态为 `mechanism_confirmed`。

正式产物：

- `data/evaluation/action_adoption_v1/decision.json`
- `data/evaluation/action_adoption_v1/rows.csv`
- `data/evaluation/action_adoption_v1/summary.md`

独立重放验证：

```bash
python scripts/verify_action_adoption_result.py
```

该结果确认“世界模型进入采样器并稳定改变微动作”。后续全长度研究也已完成：30 个 M0 与 30 个 M1 全场运行组成 30 个冻结配对，30/30 配对均出现可测行为变化。M1 相对 M0 的外部校准损失差为 `-0.0318925`，95% 分层配对 bootstrap 区间为 `[-26.0619546, 5.1374296]`。区间跨零且未通过最小有意义改善门，候选也未通过全部外部有效性门，因此正式决策为 `inconclusive_keep_research_only`，不支持产品或论文晋级。

全长度正式产物与独立重放：

- `data/evaluation/action_outcome_protocol_v1.json`
- `data/evaluation/action_outcome_v1/progress.json`
- `data/evaluation/action_outcome_v1/decision.json`

```bash
python scripts/verify_action_outcome_result.py
```

当前可以严格声明“世界模型改变微动作并传导为完整比赛行为分布变化”；仍不能声明“世界模型可靠改善校准、赛果或真实足球决策”。

## 后续门序

1. 已完成固定24次机制研究，且全部冻结门通过；当前只声明“动作采用链路已确认”。
2. 已完成60次全长度结果研究；结果为不确定且保持研究态。下一次方差、目标或策略改造必须使用新协议和新代码身份，不得修改本次冻结结果。
3. 完成经理顾问12/24/48固定窗口的真实交互与执行覆盖验证；不得用模拟点击冒充用户证据。
4. 只有新代码身份下的结果门、外部有效性门和发布审查全部通过，才能恢复研究候选或稳定产品授权。

## 两种动作采纳证据不可混用

上述 V1 机制研究只回答比赛内部 `pass vs hold` 策略是否在共享随机数下改变微动作。经理赛前顾问属于另一层级：它比较七种可玩战术，记录玩家明确采用或查看后改选，并把最终战术连接到直接执行证据。微动作研究通过不能证明经理建议有用，经理点击采用也不能证明比赛内部动作头有效。

经理级证据使用独立冻结协议 `data/evaluation/manager_advisor_protocol_v1.json` 和只读分析器 `scripts/manager_advisor_study.py`。其固定信息窗口为 12、24、48 条已建议决策；窗口中的决策全部执行后，才检查明确交互覆盖、直接执行覆盖、未绑定建议比例和推荐战术多样性。分析器首先重放 entry、派生汇总和 ledger 三层身份，拒绝修改汇总后重新计算外层哈希的伪造证据。

```bash
python scripts/manager_advisor_study.py --status
python scripts/manager_advisor_study.py --analyze --base-dir <studio-directory>
```

该协议只能确认或否定采纳取证链是否完整。它不计算赛果效应，不授权因果结论、现实足球推广、产品晋级或论文晋级。当前只完成代码和预注册，尚未收集 12 条经理建议决策。

# Generative Football Society 总验收报告

验收日期：2026-07-26

## 当前结论

项目当前生产版本为 v7.0.0，v6.0.0 保留为回滚版本。v8.0-v8.9 均为研究候选，没有替换生产模型。最新 v8.9 已完成第二强标签 provider 接入和双向严格 leave-one-provider-out 验收，状态为 `evaluated`，未 sealed、未 deployed。

## 发布边界

| 层级 | 版本 | 状态 | 用途 |
| --- | --- | --- | --- |
| 生产 | v7.0.0 | deployed | 当前比赛模拟与公开接入 |
| 回滚 | v6.0.0 | frozen | 生产故障回退 |
| 帧级研究 | v8.0-v8.8 | evaluated candidates | tracking、动作条件与事件驱动实验 |
| 跨 provider 时间模型 | v8.9 | evaluated candidate | 严格 LODO 研究证据 |

`data/releases/current.json` 是唯一部署指针。研究候选不得通过文件存在、模型注册或验收通过自动进入生产。

## V8.9 数据合同

StatsBomb Pass 通过 `related_events` UUID 与显式 `Ball Receipt*` 事件关联，并校验 recipient 身份。只保留成功接球；后续动作使用自身事件时间戳；五秒内无合格动作按右删失处理。

- SkillCorner：10 场，5,451 条成功接球链。
- StatsBomb 360：417 场，306,584 条成功接球链。
- 两个 provider 统一使用 22 维静态几何。
- 两端都遮蔽速度与加速度，避免缺失模式泄露 provider。
- 场区上下文使用镜像不变的球门距离、边线距离和实体间距。

## 严格 LODO 结果

| Held provider | Train provider | 时间 MAE | 同网格常数基线 | Continue BA | Pass/Shot BA | 2 秒 Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SkillCorner | StatsBomb 360 | 0.478 s | 0.506 s | 0.572 | 0.557 | 0.139 |
| StatsBomb 360 | SkillCorner | 0.936 s | 0.965 s | 0.617 | 0.544 | 0.225 |

训练、checkpoint 选择和阈值校准只使用训练 provider。时间基线被限制在与模型相同的 0.5 秒输出网格。连续中位数基线仍然更强，因此 v8.9 不具备部署条件；该失败边界保留在机器报告中。

## 架构结论

- 帧级世界模型通过 evidence-gated router 接入，unsupported provider 或无效身份回退到物理路径。
- 事件驱动 rollout 在预测 mark 到达后终止上一动作效应。
- 稀有 shot 分支使用可解释、镜像不变的球门邻近单调分数，避免共享隐状态的 provider 捷径。
- LLM 继续负责稀疏高层决策、持久叙事状态和有审计的元学习提案，不介入连续物理。
- v8.9 没有增加新的 LLM agent 或客户端。

## 自动验收

CI 执行：

```powershell
python -m pytest -q
python scripts/verify_release_manifest.py data/releases/v6.0.0.json
python scripts/verify_v7_release.py --artifacts-only
python scripts/verify_release_manifest.py data/releases/v8.9.0-candidate.json
python scripts/verify_temporal_lopo_v89.py
```

v8.9 快速验收会完整校验 427 个正式 NPZ、双向 provider 隔离、两折 gate、模型注册状态、所有冻结哈希以及 v7/v6 部署边界。

## 下一研究目标

连续时间目标已由 v9.0 达成。本轮继续禁止扩展动作类别、router 分支和 LLM 层；下一门槛收束为跨 provider 分布校准与不确定性覆盖。

规范证据：

- `data/releases/v8.9.0-candidate.json`
- `data/frame_world/v89_temporal_providers/manifest.json`
- `reports/acceptance/temporal_mark_v89_lopo.json`
- `docs/V8_9_STRICT_PROVIDER_LODO_ACCEPTANCE.md`

## V9.0 连续时间补充验收

v9.0 使用三分量 log-normal mixture 和右删失似然替代 0.5 秒离散时间格点。严格 LODO 下，SkillCorner held-out MAE 为 `0.412s`，优于连续基线 `0.463s`；StatsBomb held-out MAE 为 `0.912s`，优于连续基线 `0.928s`。路由器保留连续秒数，仅在固定步执行边界取整。

v9.0 状态为 `evaluated` research candidate。StatsBomb 折仅提升 `0.015s`，下一门槛是跨 provider 分布校准和不确定性覆盖，仍不允许部署。

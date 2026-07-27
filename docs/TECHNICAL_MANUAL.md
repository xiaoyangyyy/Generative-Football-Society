# GFS World Cup 2026 Simulator — 项目技术手册

> **用途**：开发、运维、校准与扩展的一站式参考。  
> **数学细节**：见 [`PROJECT_FULL_SPEC.md`](PROJECT_FULL_SPEC.md)（公式与默认参数全表）。  
> **校准命令**：见 [`CALIBRATION.md`](CALIBRATION.md)。  
> **版本**：与仓库当前实现同步（微观物理 + 情感耦合 + 可选 LLM 认知 + 潜变量世界模型 + StatsBomb 连续标定）。

---

## 目录

1. [项目定位](#1-项目定位)
2. [系统架构](#2-系统架构)
3. [仓库结构](#3-仓库结构)
4. [核心模块手册](#4-核心模块手册)
5. [单场数据流](#5-单场数据流)
6. [数据资产](#6-数据资产)
7. [安装与运行](#7-安装与运行)
8. [环境变量](#8-环境变量)
9. [校准体系](#9-校准体系)
10. [测试与 CI](#10-测试与-ci)
11. [持久化与输出](#11-持久化与输出)
12. [扩展开发指南](#12-扩展开发指南)
13. [专题文档索引](#13-专题文档索引)
14. [附录：设计不变量](#附录-设计不变量)

---

## 1. 项目定位

**Generative Football Society (GFS) V13** 是一个世界杯规模的**多智能体社会—足球仿真框架**。

| 维度 | 说明 |
|------|------|
| 赛事 | 2026 世界杯：48 队、12 组、32 强淘汰赛、点球、伤停补时式微观 |
| 宏观 | 球队心理、记忆、信念、媒体、融合决策 → 有效实力与宏观 xG |
| 微观 | 6 s/tick 物理引擎：传球、射门、犯规、空间压迫、情感调制 |
| 可选层 | LLM 认知（System 2）、潜变量世界模型（WM 规划） |
| 标定 | StatsBomb WC2022 开放数据；连续 z-score 评分，无 p10/p90 硬裁剪 |

**核心流水线**（宏观叙事）：

```
Event → Appraisal → Emotion → Coping → Memory → Belief → State/Tactics → Match Outcome
```

**比分来源**（默认）：微观射门物理（`MATCH_MICRO_SCORE=1`），宏观 λ ODE 提供 xG 先验与叙事，不直接“掷骰子改分”。

**坐标系**：球场归一化 \([0,1]^2\)；\(x\) 纵向（主队攻向 \(x\to 1\)），\(y\) 横向。

---

## 2. 系统架构

### 2.1 三层时间尺度

| 尺度 | 时间步 | 主要模块 | 输出 |
|------|--------|----------|------|
| **宏观** | 赛前/赛后 | `macro_goal_dynamics`, `FusionController`, `SocietyAgent` | 有效实力、宏观 xG、心理漂移 |
| **中观** | 15 min 窗 | `MesoAggregator` | `MicroAppraisalPacket` → GFS 叙事 |
| **微观** | 6 s/tick（默认） | `match_micro_runner` | 传球/射门/犯规统计、物理进球、micro_xg |

### 2.2 逻辑分层

```
┌─────────────────────────────────────────────────────────────┐
│  锦标赛层  tournament_2026 / knockout_bracket / checkpoint   │
├─────────────────────────────────────────────────────────────┤
│  社会智能体  SocietyAgent (z_state, 记忆, 信念, 战术)         │
│  融合层      FusionController (五专家 softmax 融合)           │
│  宏观进球    macro_goal_dynamics → Poisson / xG 先验          │
├─────────────────────────────────────────────────────────────┤
│  微观引擎    match_micro_runner                               │
│    ├─ 情感     affective_coupling → PlayerModulators         │
│    ├─ 空间     spatial_field + spatial_intelligence          │
│    ├─ 传球     passing_engine + pass_calibration + ball_phys │
│    ├─ 射门     shot_engine + goal_generator                  │
│    ├─ 纪律     discipline_schedule + event_schedule          │
│    └─ 动作     action_engine (pass/shot/cross/hold softmax)  │
├─────────────────────────────────────────────────────────────┤
│  可选层                                                       │
│    ├─ 认知 LLM   cognitive/* (MATCH_COGNITIVE=1)              │
│    └─ 世界模型   world_model/* (MATCH_WORLD_MODEL=1)         │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 System 1 / System 2

| 系统 | 实现 | 职责 |
|------|------|------|
| **System 1** | 连续 ODE / softmax / sigmoid 数学引擎 | 状态演化、物理、比分、标定 |
| **System 2** | `llm_engine` + `cognitive/*` | 有界、带置信度的战术/心理建议；**不直接决定比分** |

校准模式（`CALIBRATION_MODE=1`）下，L0 统计与 LLM/WM 隔离；M1/C1 作为 additive ablation 单独评估。

---

## 3. 仓库结构

```
sim/
├── run_world_cup_2026_full.py    # 全赛主入口
├── run_match_micro.py            # 单场微观测试
├── run_affective_phase1b.py      # 情感层单测入口
├── main_monte_carlo.py           # 蒙特卡洛实验
│
├── src/
│   ├── simulation/               # 锦标赛、Agent、LLM、融合、赛程
│   ├── match_engine/             # 微观物理 + 校准 + 认知 + WM
│   ├── memory_engine/            # 宏观 λ、Poisson、status_score
│   ├── data_engine/              # 名单、教练、FM、实体动力学
│   ├── agents/                   # 社会角色 Agent（Prophet 等）
│   └── visualization/            # 图表
│
├── data/
│   ├── rosters/                  # 48 队 JSON 名单
│   ├── coaches/                  # 教练 preset / 心理
│   ├── calibration/              # StatsBomb 基准、contract、param_registry
│   ├── persistence/              # checkpoint、squad_carryover、cognitive_log
│   ├── world_model/              # latent_wm.pt、traces/
│   └── schemas/                  # player/coach JSON schema
│
├── scripts/                      # 构建、训练、基准、校准、分析
├── tests/                        # 单元 + slow 校准测试
├── docs/                         # 设计与本手册
├── reports/                      # gate、ablation、baseline 报告
└── outputs/                      # full_run log、ball_log
```

---

## 4. 核心模块手册

### 4.1 锦标赛与社会层 (`src/simulation/`)

| 模块 | 文件 | 职责 |
|------|------|------|
| 全赛编排 | `world_cup_runner.py`, `tournament_2026.py` | 构建 48 队、小组赛、淘汰赛 |
| 单场管线 | `match_pipeline.py` | carryover → 微观 → 比分 → 赛后反馈 |
| 智能体 | `agent.py` | `z_state`、记忆、信念、赛后递归、战术 |
| 融合 | `fusion_controller.py` | phys/affect/social/tactic/governance 五专家 |
| LLM | `llm_engine.py` | 教练决策、叙事、认知 API |
| 赛程 | `knockout_bracket.py`, `group_context.py` |  bracket、小组形势 |
| 断点 | `tournament_checkpoint.py` | 可恢复全赛 |
| 跨场状态 | `cross_match_state.py` | 伤病、停赛、体能 carryover |
| 主场 | `venue_policy.py` | 2026 联合主办、crowd ψ 加成 |

### 4.2 宏观进球 (`src/memory_engine/`)

| 模块 | 文件 | 职责 |
|------|------|------|
| λ ODE | `macro_goal_dynamics.py` | 进球强度连续动力学、宏观 xG 积分 |
| Poisson | `poisson_simulator.py` | 宏观进球采样（非物理优先时） |
| 实力 | `status_score.py` | 历史赛果 → status_score |
| 宏观—微观融合 | `macro_micro_fusion.py` | `resolve_unified_score()` |
| 中观聚合 | `macro_micro_fusion.py` 等 | 15 min 窗叙事包 |

### 4.3 微观引擎 (`src/match_engine/`)

| 子系统 | 关键文件 | 说明 |
|--------|----------|------|
| **入口** | `match_micro_runner.py` | `run_match_micro_simulation()` |
| **配置** | `micro_config.py`, `config.py` | `MicroMatchConfig` 全参数 |
| **情感** | `affective_coupling.py`, `match_affective_runner.py` | ψ、z_emo、PlayerModulators |
| **空间** | `spatial_field.py`, `spatial_intelligence.py`, `kinematic_position.py` | ρ/Press/Φ 网格、通道、越位 |
| **传球** | `passing_engine.py`, `pass_calibration.py`, `ball_physics.py`, `pass_intercept.py` | logit 标定 + 物理轨迹 |
| **射门** | `shot_engine.py`, `goal_generator.py` | xG 几何、GK、射正、进球 |
| **动作** | `action_engine.py` | pass/shot/cross/hold softmax |
| **纪律** | `discipline_schedule.py`, `event_schedule.py` | tick 犯规、黄红牌 |
| **战术** | `tactical_engine.py`, `tactical_catalog.py`, `tactical_profile.py` | 21 维战术向量 |
| **名单** | `squad_factory.py`, `formation.py` | 首发、阵型、替补 |
| **输出** | `meso_aggregator.py`, `player_match_stats.py`, `ball_path_logger.py` | 统计、ball_log |
| **适配** | `adapter.py`, `macro_bridge.py` | 宏观—微观接口 |

**Tick 循环顺序**（摘要）：

1. 日历事件 `event_schedule`
2. 战术压力 `tactical_engine`
3. 空间场 `spatial` + `spatial_intelligence`
4. 情感 `affective` → modulators
5. 纪律 / 抢断 / 传中 emit
6. 运动学 `kinematic`
7. `ActionEngine` → Pass / Shot / Aerial
8. 换人、认知层、中观记录

### 4.4 认知层 (`src/match_engine/cognitive/`)

| 文件 | 职责 |
|------|------|
| `routing.py`, `salience.py` | 事件路由、显著性门控 |
| `executor.py`, `apply.py` | LLM 计划执行、写回 modulator |
| `entity_registry.py`, `schemas.py` | Tier A/B/C 实体、有界 delta |
| `bus.py`, `events.py` | 认知事件总线 |
| `config.py` | `MATCH_COGNITIVE_*` 解析 |

缓存：`data/cache/cognitive/*.json`  
日志：`data/persistence/cognitive_log/*.json`

### 4.5 世界模型 (`src/match_engine/world_model/`)

| 文件 | 职责 |
|------|------|
| `model.py` | GRU 潜变量转移 |
| `observation.py`, `action_codec.py` | obs∈R³⁰⁷, action∈R¹⁸ |
| `planner.py`, `inference.py` | 规划 bonus 注入传球/射门 softmax |
| `recorder.py`, `ball_log_dataset.py` | trace 采集与训练集 |

权重：`data/world_model/latent_wm.pt`  
详见 [`WORLD_MODEL.md`](WORLD_MODEL.md)

### 4.6 校准子系统 (`src/match_engine/calibration/`)

| 文件 | 职责 |
|------|------|
| `contract.py` | 加载 observable contract；`evaluate_rows()` 连续评分 |
| `benchmark_core.py` | 6 .fixture 基准跑、team_level_rows |
| `profile.py`, `apply_params.py` | `param_registry.json` → `MicroMatchConfig` |
| `ablation.py` | M0/M1/T0/C1 预设、subtractive ablation、env 隔离 |
| `objective.py` | `calibration_loss`, `calibration_score` |

### 4.7 数据工程 (`src/data_engine/`)

| 文件 | 职责 |
|------|------|
| `roster_loader.py`, `roster_builder.py` | JSON 名单 → 微观球员状态 |
| `coach_loader.py` | 教练 preset、心理 |
| `fm_import.py`, `fm_roster_merge.py` | FM 导出合并 CA/属性 |
| `entity_dynamics.py` | 球队向量 x_T、教练心理均衡 |
| `player_loader.py` | 球员能力映射 |
| `team_name_resolver.py` | 队名归一化 |

---

## 5. 单场数据流

```
TournamentManager.play_match(home, away)
  │
  ├─ 1. prepare_match_agents()          # squad_carryover 伤病/停赛
  ├─ 2. LLM coach_decide_tactics()      # 阵型 + 四旋钮
  ├─ 3. apply_beliefs_to_tactics()      # 信念 → 战术微调
  ├─ 4. simulate_internal_game()        # 教练—球员博弈
  ├─ 5. FusionController.fuse()         # → effective_status
  ├─ 6. expected_match_xg()             # λ ODE → 宏观 xG 先验
  ├─ 7. run_match_micro_simulation()    # MATCH_MICRO=1 时
  ├─ 8. resolve_unified_score()         # 物理进球 或 融合
  ├─ 9. recursive_update()              # 赛后心理/记忆
  └─ 10. save checkpoint + carryover
```

**三条叙事链**（便于理解耦合）：

1. **System 1 情感**（M0 默认开）：crowd ψ → 球员情绪 → modulators → 传球/射门/犯规
2. **LLM 认知**（`MATCH_COGNITIVE=1`）：事件触发 → 有界 writeback
3. **宏观媒体**（`SocietyAgent` + 总统/媒体逻辑）：场间影响 `eff_status`，经 Fusion 进入微观先验

---

## 6. 数据资产

### 6.1 名单与教练

| 路径 | 格式 | 说明 |
|------|------|------|
| `data/rosters/{Team}.json` | player schema | CA、位置、能力、carryover |
| `data/coaches/wc2026_coaches.json` | coach schema | preset_affinities、mental |
| `data/schemas/*.schema.json` | JSON Schema | 校验用 |

构建流水线：

```powershell
python scripts/build_rosters_from_transfermarkt.py
python scripts/import_fm_export.py
python scripts/verify_rosters.py
```

### 6.2 校准基准

| 路径 | 内容 |
|------|------|
| `statsbomb_match_baselines.json` | WC2022 128 队×场，15 硬指标 mean/p10/p90 |
| `statsbomb_pass_rates.json` | 短/直塞/长传成功率锚点 |
| `joint_baselines.json` | soft 约束分位数 + 相关参考 |
| `observable_contract.json` | 硬/软指标定义、gate 阈值 |
| `param_registry.json` | 可标定参数注册表 + profile overrides |

### 6.3 持久化

| 路径 | 内容 |
|------|------|
| `persistence/tournament_checkpoint.json` | 赛程进度 |
| `persistence/squad_carryover.json` | 跨场伤病/停赛/统计 |
| `persistence/cognitive_log/` | 认知计划日志 |
| `persistence/roster_verification.json` | 名单校验结果 |

---

## 7. 安装与运行

### 7.1 环境

```bash
pip install -r requirements.txt
```

| 依赖 | 用途 |
|------|------|
| numpy, pandas | 数值与数据 |
| torch | 世界模型 |
| openai, python-dotenv | LLM API |
| statsbombpy | 开放数据拉取 |
| matplotlib | 可视化 |

推荐 **Python 3.10+**。Windows 下设置：

```powershell
$env:PYTHONPATH="."
```

### 7.2 配置 LLM（全赛/认知需要）

`.env` 或环境变量：

```env
API_KEY=your_key
BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4-turbo-preview
GFS_SEED=42
```

### 7.3 常用入口

| 命令 | 用途 |
|------|------|
| `python run_world_cup_2026_full.py` | 全赛（写 `outputs/full_run_latest.log`） |
| `python run_world_cup_2026_full.py --resume` | 从 checkpoint 续跑 |
| `python run_match_micro.py` | 单场微观 smoke |
| `python scripts/run_single_match_cognitive.py` | 单场认知测试 |

**推荐全赛 flags**（物理优先）：

```powershell
$env:MATCH_MICRO="1"
$env:MATCH_MICRO_SCORE="1"
$env:MATCH_COGNITIVE="0"
$env:MATCH_WORLD_MODEL="0"
python run_world_cup_2026_full.py
```

**断点续跑**：

```powershell
.\scripts\resume_full_run.ps1
# 或
python run_world_cup_2026_full.py --resume
```

---

## 8. 环境变量

### 8.1 微观 / 全赛

| 变量 | 默认 | 说明 |
|------|------|------|
| `MATCH_MICRO` | 0 | 启用微观引擎 |
| `MATCH_MICRO_SCORE` | 1 | 物理进球为官方比分 |
| `MATCH_COGNITIVE` | 0 | LLM 认知层 |
| `MATCH_SCHEDULED_SHOTS` | 0 | 日历射门（物理优先时关） |
| `MATCH_BALL_LOG` | 0 | 写 ball_log jsonl |
| `MATCH_BALL_LOG_TXT` | 0 | 人类可读 log |
| `GFS_SEED` | 42 | 全局 RNG |
| `XG_SUPPLEMENT` | 0 | 物理进球过少时用 μxG 补球 |

### 8.2 世界模型

| 变量 | 默认 | 说明 |
|------|------|------|
| `MATCH_WORLD_MODEL` | 0 | 启用 WM 规划 |
| `MATCH_WM_PLAN` | 1 | 规划 bonus |
| `MATCH_WM_PLANNER_BLEND` | 0.30 | 质量门控后的传球 blend 上限 |
| `MATCH_WM_SHOT_BLEND` | 0.40 | 射门 blend |
| `MATCH_WM_RECORD` | 0 | 录制 trace |

### 8.3 校准

| 变量 | 说明 |
|------|------|
| `CALIBRATION_MODE` | 1 = 隔离叙事层，固定 eff_status |
| `CALIBRATION_NO_BLEND` | 1 = 纯物理传球（无 StatsBomb logit blend） |
| `CALIBRATION_LOGIT_BLEND` | logit 混合权重 override |
| `CALIBRATION_INTERCEPT_RISK_SCALE` | 拦截标定 scale |
| `CALIBRATION_ROSTER_STATUS` | T0：用名单 status 而非固定值 |

### 8.4 认知

| 变量 | 说明 |
|------|------|
| `MATCH_COGNITIVE_MAX_PER_TIER` | 如 `6,4,4,2,3` |
| `MATCH_COGNITIVE_S0` | 显著性阈值基线 |
| `MATCH_COGNITIVE_SYNC` | 同步模式 |

---

## 9. 校准体系

### 9.1 阶段概览

| 阶段 | 产物 | 目的 |
|------|------|------|
| **Phase 0** | `observable_contract.json`, gate runner | 可观测指标契约 + 连续评分 |
| **Phase 1** | `param_registry.json` | 参数注册 + profile `v5_physics_first` |
| **Phase 2** | `ablation.py`, `run_ablation_matrix.py` | 分层 ablation，Δz vs M0 |
| **Phase 3** | `joint_calibrate.py` | 导出 profile、全 gate、ablation 验证 |
| **Phase 4** | `score_path.py` | 单一官方比分路径；宏观 λ 叙事-only |
| **Phase 5** | `narrative_isolation.py`, `layer_gate.py` | L0 隔离；M1/C1 additive gate |

See [`PHASE4_IMPLEMENTATION.md`](PHASE4_IMPLEMENTATION.md), [`PHASE5_IMPLEMENTATION.md`](PHASE5_IMPLEMENTATION.md).

### 9.2 评分方法（当前）

- **无 p10/p90 硬 band**、无 sim 内 `if x > t` 裁剪
- 每指标：`z = |sim - target| / scale`，`score = exp(-0.5·z²)`
- scale：硬指标用 StatsBomb IQR；soft range 用 joint_baselines IQR
- 相关约束：`shots_goals_correlation` 等 — target 可锚定 **M0 sim 联合分布**（StatsBomb 作 reference）
- Gate：`min_metric_score=0.30`，`min_combined_score=0.52`，全部 pass 才 exit 0

### 9.3 校准 Pipeline

| Pipeline | 环境 | 用途 |
|----------|------|------|
| **M0** | WM=0, Cognitive=0, CALIBRATION_MODE=1 | 纯微观物理 baseline |
| **M1** | M0 + MATCH_WORLD_MODEL=1 | WM additive |
| **C1** | M0 + MATCH_COGNITIVE=1 | 认知 additive |
| **T0** | M0 + CALIBRATION_ROSTER_STATUS=1 | 名单驱动 status |

### 9.4 Ablation 矩阵

**Subtractive**（相对 M0 关闭）：

- `no_affective` — 情感调制冻结
- `no_spatial` — 空间场/智能关闭
- `no_blend` — StatsBomb logit blend=0
- `no_discipline_tick` — tick 犯规关闭
- `no_tactical_bias` — 战术偏置关闭

**Additive**（相对 M0 开启）：`M1`, `C1`

### 9.5 常用命令

```powershell
$env:PYTHONPATH="."

# 拉 StatsBomb 基准
python scripts/fetch_match_baselines_statsbomb.py
python scripts/fetch_pass_calibration_statsbomb.py

# 快速 smoke（20 min，非正式 pass）
python scripts/run_calibration_baselines.py --quick

# 全 gate（6×3×90min，~2h）
python scripts/run_calibration_gate.py --pipeline M0 --samples 3 --match-seconds 5400

# Phase 3：gate + ablation（~8–16h）
python scripts/joint_calibrate.py --validate-only

# 单独 ablation
python scripts/run_ablation_matrix.py --samples 3

# 开放数据对比
python scripts/benchmark_sim_vs_open_data.py --samples 3 --pipeline M0
```

### 9.6 报告位置

| 报告 | 路径 |
|------|------|
| Gate | `reports/calibration_gate.json` |
| Baselines | `reports/calibration_baselines/{M0,M1,T0}.json` |
| Ablation | `reports/ablation/matrix.json`, `matrix.md` |
| Profile | `reports/joint_calibration/v5_physics_first.json` |

### 9.7 关键可调参数

| 区域 | 位置 |
|------|------|
| 传射动作频率 | `MicroMatchConfig` `action_*` |
| 传球 completion | `pass_calibration.py`, `statsbomb_pass_rates.json` |
| xG / GK | `shot.xg_geom_base`, `shot.gk_b_reflex` |
| 犯规/牌 | `discipline_schedule.py`, `discipline_*` |
| 注册表导出 | `data/calibration/param_registry.json` |

---

## 10. 测试与 CI

```powershell
# 快速单元测试
pytest tests/ -m "not slow"

# 慢测试（含校准）
$env:RUN_SLOW_TESTS="1"
pytest tests/test_calibration_suite.py tests/test_pass_calibration.py -m slow
```

| 测试文件 | 覆盖 |
|----------|------|
| `test_match_micro_phase2.py` | 空间/传球 |
| `test_match_phase3.py`, `test_match_phase3b.py` | 射门/弧线 |
| `test_affective_phase1b.py` | 情感耦合 |
| `test_calibration_suite.py` | 校准基础设施 |
| `test_pass_calibration.py` | 传球 logit |
| `test_world_model.py` | WM  smoke |
| `test_cognitive_*.py` | 认知路由/应用 |
| `test_macro_goal_dynamics.py` | 宏观 λ |
| `test_tactical_system.py` | 21 维战术 |

---

## 11. 持久化与输出

| 路径 | 内容 |
|------|------|
| `outputs/full_run_latest.log` | 全赛文本日志（Tee） |
| `outputs/ball_log/*.jsonl` | 传球/射门物理 trace |
| `reports/` | 校准 gate、ablation、benchmark |
| `data/world_model/traces/*.jsonl` | WM 训练 trace |

---

## 12. 扩展开发指南

### 12.1 新增可观测指标

1. 在 `benchmark_core` 聚合字段  
2. 写入 `observable_contract.json`（hard 或 soft）  
3. 若需 StatsBomb 基准，更新 `fetch_match_baselines_statsbomb.py`  
4. 在 `param_registry.json` 标注 `affects` 关联参数  

### 12.2 新增 Ablation

1. 在 `ablation.py` 增加 `AblationSpec`（env + config_overrides）  
2. 加入 `SUBTRACTIVE_ABLATIONS` 或 `ADDITIVE_LAYERS`  
3. 跑 `run_ablation_matrix.py` 验证 Δz  

### 12.3 新增微观行为

1. 在 tick 循环合适位置 emit（参考 `micro_events.py`）  
2. 用连续概率（sigmoid/softmax），避免硬阈值  
3. 补充 slow test + 检查 gate 是否回归  

### 12.4 修改比分路径

- 物理优先：确保 `MATCH_MICRO_SCORE=1`，进球仅来自 `shot_engine`  
- 宏观叙事：λ ODE 与 `resolve_unified_score()` 分工见 `macro_micro_fusion.py`  
- **不要**在 LLM 层直接写比分  

---

## 13. 专题文档索引

| 文档 | 内容 |
|------|------|
| [`PROJECT_FULL_SPEC.md`](PROJECT_FULL_SPEC.md) | **完整数学规格**（公式、默认值、附录） |
| [`TECHNICAL_MANUAL.md`](TECHNICAL_MANUAL.md) | **本手册**（架构、运行、校准） |
| [`CALIBRATION.md`](CALIBRATION.md) | 校准命令速查 |
| [`SINGLE_MATCH_ENGINE_DESIGN.md`](SINGLE_MATCH_ENGINE_DESIGN.md) | 微观引擎顶层设计 |
| [`SPATIAL_INTELLIGENCE_FULL_SPEC.md`](SPATIAL_INTELLIGENCE_FULL_SPEC.md) | SAC 空间智能 |
| [`COGNITIVE_DUAL_SYSTEM.md`](COGNITIVE_DUAL_SYSTEM.md) | 认知双系统 |
| [`WORLD_MODEL.md`](WORLD_MODEL.md) | 潜变量 WM |
| [`ENTITY_DYNAMICS.md`](ENTITY_DYNAMICS.md) | 实体动力学 |
| [`TACTICAL_SYSTEM.md`](TACTICAL_SYSTEM.md) | 21 维战术 |
| [`COACH_AND_FM_DATA.md`](COACH_AND_FM_DATA.md) | 数据管线 |
| [`PHASE1B_IMPLEMENTATION.md`](PHASE1B_IMPLEMENTATION.md) | 情感—空间耦合 |
| [`PHASE2A_2B_IMPLEMENTATION.md`](PHASE2A_2B_IMPLEMENTATION.md) | 空间场与传球 |
| [`PHASE3_IMPLEMENTATION.md`](PHASE3_IMPLEMENTATION.md) | 射门与动作 |
| [`PHASE3B_IMPLEMENTATION.md`](PHASE3B_IMPLEMENTATION.md) | 弧线传球 |

---

## 附录：设计不变量

1. **无硬阈值门控**：传球、拦截、射正、犯规均为连续概率模型。  
2. **校准锚定开放数据**：传球 logit 锚定 StatsBomb WC2022；gate 用连续 score，不用 band 裁剪 sim。  
3. **物理优先比分**：全赛默认 `MATCH_MICRO_SCORE=1`。  
4. **风格可辨**：战术向量 + foul_impulse + 压迫改变传射与犯规结构。  
5. **可断点续赛**：checkpoint + squad_carryover 保证 48 队长周期一致。  
6. **校准隔离**：`CALIBRATION_MODE` 下 L0 与 LLM/WM 分离；M1/C1 单独 ablation 评估。

---

## 脚本全索引

| 脚本 | 用途 |
|------|------|
| `run_world_cup_2026_full.py` | 全赛 |
| `run_match_micro.py` | 单场微观 |
| `scripts/resume_full_run.ps1` | 断点续跑 |
| `scripts/restart_full_run.ps1` | 清 checkpoint 重开 |
| `scripts/build_rosters_from_transfermarkt.py` | 生成名单 |
| `scripts/import_fm_export.py` | FM 合并 |
| `scripts/verify_rosters.py` | 名单校验 |
| `scripts/fetch_match_baselines_statsbomb.py` | StatsBomb 比赛基准 |
| `scripts/fetch_pass_calibration_statsbomb.py` | 传球率基准 |
| `scripts/run_calibration_gate.py` | 全 gate |
| `scripts/run_calibration_baselines.py` | M0/M1/T0 baseline |
| `scripts/run_ablation_matrix.py` | Ablation 矩阵 |
| `scripts/joint_calibrate.py` | Phase 3 联合校准 |
| `scripts/benchmark_sim_vs_open_data.py` | 开放数据对比 |
| `scripts/benchmark_style_contrast.py` | 风格对比 |
| `scripts/benchmark_tournament_smoke.py` | 锦标赛 smoke |
| `scripts/benchmark_roster_calibration.py` | 名单校准 |
| `scripts/ensure_world_model.py` | WM 一键训练 |
| `scripts/train_world_model.py` | WM 训练 |
| `scripts/validate_world_model.py` | WM holdout |
| `scripts/collect_world_model_traces.py` | 采集 trace |
| `scripts/run_single_match_cognitive.py` | 单场认知 |
| `scripts/analyze_group_stage.py` | 小组赛 log 分析 |

---

*实现变更以 `src/` 为准；公式与参数默认值以 `PROJECT_FULL_SPEC.md` 为准。*

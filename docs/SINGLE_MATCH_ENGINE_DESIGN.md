# GFS 单场微观仿真引擎 — 完整设计文档

**版本**: 1.2  
**状态**: 架构设计（待实现）  
**变更**: v1.1 §7 空间智能；v1.2 链至 [**空间智能全规格 v2.0**](SPATIAL_INTELLIGENCE_FULL_SPEC.md)（情绪耦合 + 球路连续体 + 位置体态）  
**目标**: 在 **不破坏** 现有 Generative Football Society (GFS) V13 框架的前提下，新增「真实球员 + 连续数学」的单场仿真层，并向上游锦标赛/社会层提供 **可聚合、可校准** 的接口。

---

## 目录

1. [总体判断与建议](#1-总体判断与建议)
2. [设计原则](#2-设计原则)
3. [与现有 GFS 的关系](#3-与现有-gfs-的关系)
4. [系统架构总览](#4-系统架构总览)
5. [数学基础（无阈值范式）](#5-数学基础无阈值范式)
6. [空间与控球场](#6-空间与控球场)
7. [空间智能引擎](#7-空间智能引擎) — **扩展阅读**: [SPATIAL_INTELLIGENCE_FULL_SPEC.md](SPATIAL_INTELLIGENCE_FULL_SPEC.md)
8. [球员实体与阵容](#8-球员实体与阵容)
9. [传球网络与射门决策](#9-传球网络与射门决策)
10. [体能、士气与认知状态](#10-体能士气与认知状态)
11. [教练战术与换人](#11-教练战术与换人)
12. [观众—裁判—场上双向耦合](#12-观众裁判场上双向耦合)
13. [跨场记忆与赛程负荷](#13-跨场记忆与赛程负荷)
14. [进球/xG 涌现与宏观接口](#14-进球xg-涌现与宏观接口)
15. [真实球员与教练数据层](#15-真实球员与教练数据层)
16. [与 LLM / Fusion 的边界](#16-与-llm--fusion-的边界)
17. [代码结构与集成点](#17-代码结构与集成点)
18. [实现路线图](#18-实现路线图)
19. [验证与校准](#19-验证与校准)
20. [附录：符号表与默认方程](#20-附录符号表与默认方程)

---

## 1. 总体判断与建议

### 1.1 可行性结论

**值得做，且与 GFS 定位高度一致。** 当前仓库的比赛核心是 `simulate_match_score`（泊松 λ 由球队 `effective_status` 驱动），球队智能体 `SocietyAgent` 已具备连续潜变量 `z_state`、情感—记忆—信念管线、融合控制器与 LLM 战术层。你要的单场模块（传球、空间、个人士气/体能、换人、观众—裁判耦合、球员网络、上场遗留）本质上是 **在现有「宏观社会—心理」层之下插入「微观运动—决策」层**，而不是重写上层。

### 1.2 推荐策略（三层尺度）

| 尺度 | 时间步 | 职责 | 与现有框架 |
|------|--------|------|------------|
| **L3 宏观** | 赛后 | `SocietyAgent.recursive_update`、媒体、信念 | **保持不变** |
| **L2  meso** | 90 分钟 → 相位/15 分钟块 | 聚合 xG、势头、犯规负荷、换人效果 | **新增适配器** |
| **L1 微观** | Δt ≈ 2–5 秒（可配置） | 空间场、**空间智能**、传球图、个人 ODE | **新增 `match_engine`** |

**关键约束**: L1 不直接改比分；L2 通过 **连续强度场 + 空间智能指标** 产生 xG 过程；最终比分仍走 **随机过程**（见 §14），但 λ 来自仿真积分而非 `status/35` 启发式。

### 1.3 不建议的做法

- ❌ 用 `if stamina < 60: pass_accuracy -= 0.2` 这类硬阈值（与你要求冲突，也与 GFS 的 sigmoid/tanh/softmax 风格不一致）
- ❌ 让 LLM 逐秒决定传球（成本、不可复现、破坏数学主引擎原则）
- ❌ 在 `SocietyAgent` 内塞 23×2 球员对象（类爆炸、破坏现有球队抽象）
- ❌ 一次性替换 `play_match` 全流程（风险高）

### 1.4 建议采纳的核心技术路线

1. **新包** `src/match_engine/`（微观），通过 **适配器** 挂到 `poisson_simulator.simulate_match_score` 或 `TournamentManager.play_match` 第 6 步。
2. **连续场 + 空间智能层 + 图消息传递** 描述空间感知、无球跑动与传球，不用离散「格子 IF」。
3. **ODE + 指数核** 描述体能、士气、观众情绪。
4. **真实 roster** 用独立数据 schema（JSON/SQLite），与 `data/tactics_final_en.json` 的叙事描述解耦。
5. **可切换后端**: `MATCH_ENGINE=legacy|micro`，默认 `legacy`，保证现有 CI/脚本零破坏。

---

## 2. 设计原则

### P1 — 非侵入扩展 (Open-Closed)

- 现有 `SocietyAgent`、`FusionController`、`SimulationLLM` 接口 **不改签名**。
- 微观层输出 **MatchMicroSummary** DTO，由适配器转为 `(goals_a, goals_b, xg_a, xg_b)` 及 `play_match` 所需的赛后事件字典。

### P2 — 连续数学优先 (No Hard Thresholds)

所有「开关」表达为：

- **软门控**: \(\sigma(x) = 1/(1+e^{-x})\)，\(\tanh\)
- **分布选择**: \(\mathrm{softmax}(u/\tau)\)
- **生存/强度**: Hawkes 过程、非齐次 Poisson、Gamma-Exponential 混合（非「大于 3 次就…」）
- **约束**: 投影到单纯形、盒约束 \([0,1]\)，不用分段常数规则表

> 注: `np.clip` 用于数值稳定（避免 `log(0)`），**语义上**不表示「低于 X 就不发生」；发生概率由平滑函数控制。

### P3 — 可识别、可校准

每个模块输出 **可观测充分统计量**（传球成功率、推进距离、压迫强度、xG 增量），便于与 Opta/StatsBomb 类数据或自建 replay 对齐。

### P4 — 真实身份，参数可缺省

有真实球员时用 `player_id`；缺失时用 **位置原型**（Role Archetype）+ 球队 tier 先验，保证 48 队均可跑通。

### P5 — 复现性

`GFS_SEED` 控制微观 RNG；与宏观种子分层：`seed_micro = hash(GFS_SEED, match_id)`。

---

## 3. 与现有 GFS 的关系

### 3.1 当前 `play_match` 流程（节选）

```
recover → LLM coach tactics → internal_game + referee sample
→ fusion → simulate_match_score(eff_status_1, eff_status_2)  ← 替换/包装点
→ media / dialogue → recursive_update → apply_match_wear
```

**挂钩点（推荐）**:

```python
# tournament_2026.py 第 6 步 — 保持返回值契约
from src.match_engine.adapter import simulate_match_score_micro

s1, s2, xg1, xg2, micro = simulate_match_score_micro(
    home_agent=a1, away_agent=a2,
    eff_status_home=eff_status_1, eff_status_away=eff_status_2,
    fusion_home=fused_1, fusion_away=fused_2,
    referee=referee, stage_pressure=pressure,
    is_knockout=is_knockout,
)
```

### 3.2 职责划分

| 层级 | 现有 GFS | 新增单场引擎 |
|------|----------|--------------|
| 赛前心态/信念 | `SocietyAgent`, beliefs | 接收 `z_state` 投影为球员先验 |
| 战术意图 | LLM `coach_decide_tactics` + `tactical_controls` | 映射为 **连续控制向量** \(\mathbf{u}_{coach}\) |
| 比赛过程 | Poisson λ | 空间场 + 传球图 + 事件流 |
| 比分 | `simulate_match_score` | 由 **xG 强度积分** + 终结层产生 |
| 赛后心理 | `recursive_update` | 接收 `MicroMatchEvent` 列表（红牌、点球争议、球星失误） |

### 3.3 向上聚合（球队状态反馈）

微观赛后导出：

- `team_fatigue_delta`, `injury_load_delta`（已有 `apply_match_wear` 可改为读 micro）
- `star_morale_shock`（影响 Icon 角色）
- `referee_grievance_delta`（来自争议事件积分）
- `decision_memory` 特征：控球率、被压迫、换人时机误差

---

## 4. 系统架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                    TournamentManager.play_match                  │
│  (现有: tactics, fusion, media, recursive_update — 保留)         │
└────────────────────────────┬────────────────────────────────────┘
                             │ eff_status, fusion, referee, agents
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              match_engine.adapter.simulate_match_score_micro     │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐ │
│  │ MatchState   │──▶│ TickIntegrator│──▶│ EventRecorder       │ │
│  │ Builder      │   │ (Δt loop)     │   │ (pass, shot, foul…)  │ │
│  └──────────────┘   └───────┬──────┘   └──────────┬───────────┘ │
│                             │                      │             │
│         ┌───────────────────┼──────────────────────┘             │
│         ▼                   ▼                                    │
│  SpatialFieldEngine ──▶ SpatialIntelligenceEngine (核心)            │
│         │                    │                                    │
│         └────────┬───────────┘                                    │
│                  ▼                                                │
│  PassingGraphEngine   StaminaMoraleODE   CoachPolicy              │
│  CrowdRefereeCoupling  CarryoverKernel                            │
└────────────────────────────┬────────────────────────────────────┘
                             │ MatchMicroSummary
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  GoalGenerator (NH-Poisson on xG rate) → goals, xg_a, xg_b     │
└─────────────────────────────────────────────────────────────────┘
```

### 4.1 核心类型（建议）

```python
@dataclass
class PlayerState:
    player_id: str
    name: str
    role: str          # GK|CB|FB|DM|CM|AM|W|ST
    on_pitch: bool
    minutes: float
    position_xy: np.ndarray  # [0,1]^2 归一化球场坐标
    stamina: float           # 对数体能 logit
    morale: float            # 对数士气 logit
    cognitive_load: float
    spatial_cognition: float # 空间认知 S_i（logit），见 §7
    yellow_accum: float      # 连续化牌负荷，非整数阈值
    touch_heat: float        # Hawkes 活跃度
    off_ball_target: np.ndarray  # 无球目标点，由空间智能更新

@dataclass
class CoachControls:
    line_height: float       # [0,1]
    width: float
    press_intensity: float
    risk_budget: float
    build_up_tempo: float
    counter_weight: float
    sub_aggression: float    # 换人倾向

@dataclass
class MatchMicroSummary:
    xg_home: float
    xg_away: float
    goals_home: int
    goals_away: int
    events: list
    player_logs: dict
    phase_stats: dict
    carryover_state: dict
```

---

## 5. 数学基础（无阈值范式）

### 5.1 统一状态向量

每球员 \(i\):

\[
\mathbf{x}_i = [\; s_i,\; m_i,\; S_i,\; \mathbf{p}_i,\; h_i,\; q_i,\; c_i \;]^\top
\]

- \(s_i\): 体能（logit，\(\mathbb{R}\) 无界，显示用 \(\sigma(s_i)\)）
- \(m_i\): 士气（logit）
- \(S_i\): **空间认知**（logit，见 §7.2）
- \(\mathbf{p}_i \in [0,1]^2\): 位置（投影保持，由空间智能导航更新）
- \(h_i\): touch Hawkes 激发值
- \(q_i\): 认知负荷（连续）
- \(c_i\): 与教练/队友/观众的 **耦合标量**

### 5.2 软门控通用形式

任意「是否尝试射门/直塞」决策，用 **效用 + softmax**:

\[
\mathbb{P}(a=k) = \frac{\exp(u_k / \tau)}{\sum_j \exp(u_j / \tau)}
\]

\(u_k\) 由空间优势、网络中心性、体能、士气 **线性+交叉项** 组成，\(\tau\) 随认知负荷上升（决策更「钝」）。

### 5.3 时间推进

全场 \(T=90\)（加时 `\extend`），步长 \(\Delta t\):

\[
\mathbf{x}^{t+\Delta t} = \mathbf{x}^t + \Delta t \cdot f(\mathbf{x}^t, \mathbf{u}_{coach}, \mathbf{F}_{field}, \boldsymbol{\xi}^t)
\]

\(f\) 分项由下列模块叠加；\(\boldsymbol{\xi}\) 为高斯噪声（强度由 fusion `volatility` 调制）。

### 5.4 禁止清单 → 替代方案

| 传统阈值做法 | 本设计替代 |
|--------------|------------|
| 体能 < 60 不传球 | 传球成功率 \(p = \sigma(\alpha_0 + \alpha_1 s_i + \alpha_2 \|\mathbf{p}_i-\mathbf{p}_{target}\|)\) |
| 士气高就浪射 | 射门效用含 \(-\beta_{risk} \cdot \sigma(m_i)\cdot(1-\text{space})\) 连续惩罚 |
| 观众 > 80dB 裁判偏 | 偏置 \(\delta_{ref} = \kappa \tanh(\psi_{crowd}-\psi_0)\) 连续项 |
| 第 70 分钟必换人 | 换人强度 \(\lambda_{sub}(t) = \lambda_0 \sigma(a_{sub} t + b_{sub})\)  Hazard，非固定分钟 |

---

## 6. 空间与控球场

### 6.1 _pitch 连续场（推荐：势场 + 扩散）

将球场归一化为 \(\Omega=[0,1]^2\)。每队 \(k \in \{H,A\}\) 维护 **控球密度** \(\rho_k(\mathbf{r},t)\)，满足：

\[
\frac{\partial \rho_k}{\partial t} = D_k \nabla^2 \rho_k + \nabla \cdot (\rho_k \mathbf{v}_k) - \lambda_{press}\,\rho_k\,\rho_{opp} + S_k(\mathbf{r},t)
\]

- \(\mathbf{v}_k\): 由教练 `line_height`, `width` 决定的 **漂移场**（解析向量场，非格点 IF）
- \(S_k\): 事件源（抢断、解围 → 负源；推进传球 → 正源）
- 对抗项 \(\rho_k \rho_{opp}\): 压迫吞噬控球（双场耦合）

**空间优势**（用于传球/射门效用）:

\[
A_k(\mathbf{r}) = \frac{\rho_k(\mathbf{r})}{\rho_k(\mathbf{r})+\rho_{opp}(\mathbf{r})+\varepsilon}
\]

### 6.2 通道与宽度

宽度 \(w\in[0,1]\) 调制扩散各向异性：

\[
D_k = D_0 \cdot \mathrm{diag}(1+\alpha_w w,\; 1-\alpha_w w)
\]

### 6.3 定位球与禁区

禁区用 **平滑指示核** \(B_{box}(\mathbf{r})=\sigma(\gamma(\| \mathbf{r}-\mathbf{r}_{goal}\|-r_{box}))\)，不硬裁剪。

xG 局部强度:

\[
\chi(\mathbf{r}) = \chi_0 \cdot B_{box}(\mathbf{r}) \cdot A_k(\mathbf{r}) \cdot \sigma(\eta^\top \mathbf{f}_{shot})
\]

> §6 提供 **物理层** 的控球扩散；§7 在其上叠加 **认知—决策层** 的空间智能，回答「谁该往哪跑、线在哪、何时利用半空间」，全部用连续场与梯度，无区域硬切换。

---

## 7. 空间智能引擎

**Spatial Intelligence Engine (SIE)** 是本设计的核心差异化模块：不仅模拟「球在哪」，还模拟 **球员如何阅读、创造、压缩与利用空间**。

### 7.1 定义：四层空间智能

| 层级 | 名称 | 数学对象 | 足球语义 |
|------|------|----------|----------|
| L0 | **感知** | 局部采样 \(\mathcal{P}_i\) | 视野内队友/对手密度、传球线遮挡 |
| L1 | **解释** | 场 \(\Phi, \mathcal{N}, \mathcal{L}\) | 哪里空、哪里人多、走廊质量 |
| L2 | **决策** | 效用 \(u_{move}, u_{pass}\) | 跑位、传射选择 |
| L3 | **协调** | 团队形 \(\mathbf{S}_{team}\) | 阵型弹性、集体前压/回撤 |

四层均由 **连续函数** 构成；「进入禁区」等语义用平滑核 \(B_{box}\) 表达，不用 `if zone == "penalty"`。

### 7.2 个体空间认知状态 \(S_i\)

在 §5 状态向量中增加 **空间认知** \(S_i \in \mathbb{R}\)（logit 存储，显示 \(\sigma(S_i)\in(0,1)\)）：

\[
\dot{S}_i = -\rho_S S_i + \kappa_v \theta_i^{vision} \cdot \|\nabla \Phi(\mathbf{p}_i)\| - \kappa_f \tanh(q_i) + \kappa_m \tanh(m_i - m_{team})
\]

- \(\|\nabla \Phi\|\) 大 → 球员「看见」并利用空当，认知增益  
- 认知负荷 \(q_i\) 高 → 空间阅读变差（与 §10 决策温度 \(\tau_{dec}\) 联动）  
- 静态先验：\(\theta_i^{spatial} \in \boldsymbol{\theta}_i\)，与 `vision` 相关但不等同（中卫重占位、边卫重宽度）

**无球 vs 有球**：有球时 \(S_i\) 调制 **出球选项熵**；无球时调制 **跑位速度** \(\|\dot{\mathbf{p}}_i\|\)。

### 7.3 自由空间场 \(\Phi(\mathbf{r},t)\)

\[
\Phi(\mathbf{r}) = \sigma\Bigl( a_0 + a_1 \frac{\rho_{own}(\mathbf{r})}{\rho_{own}+\rho_{opp}+\varepsilon} - a_2 \sum_{j\in opp} K_\sigma(\mathbf{r},\mathbf{p}_j) - a_3 \sum_{j\in own\setminus i} K_\sigma(\mathbf{r},\mathbf{p}_j) \Bigr)
\]

\(K_\sigma(\mathbf{r},\mathbf{p})=\exp(-\|\mathbf{r}-\mathbf{p}\|^2/(2\sigma_p^2))\) 为软占有核（**非 Voronoi 硬边界**）。

**语义**：\(\Phi\approx 1\) 可推进；\(\Phi\approx 0\) 被压迫或拥堵。传球目标评分、无球跑动均沿 \(\nabla \Phi\) 与 \(-\nabla \text{Press}\) 组合。

### 7.4 软 Voronoi 占有与人数优势场

每球员对场地的 **软支配**：

\[
O_i(\mathbf{r}) = \frac{\exp(-d_i(\mathbf{r})/\tau_i)}{\sum_{j\in pitch} \exp(-d_j(\mathbf{r})/\tau_j)},\quad d_i(\mathbf{r})=\|\mathbf{r}-\mathbf{p}_i\|
\]

\(\tau_i\) 由角色与 \(\theta^{press}\) 调制（后腰覆盖半径大、边锋小）。

**人数优势**（用于判断是否可传身后）：

\[
\mathcal{N}(\mathbf{r}) = \tanh\Bigl( \sum_{i\in own} O_i(\mathbf{r}) - \sum_{j\in opp} O_j(\mathbf{r}) \Bigr) \in (-1,1)
\]

\(\mathcal{N}>0\) 连续表示局部过载，非「2v1 才传」阈值。

### 7.5 传球走廊质量 \(\mathcal{L}_{ij}\)

边 \(i\to j\) 的空间智能权重，用 **视线积分**（连续，非 LOS 布尔）：

\[
\mathcal{L}_{ij} = \exp\Bigl( -\int_{\mathbf{p}_i}^{\mathbf{p}_j} \bigl( \mu_1 \rho_{opp}(\mathbf{s}) + \mu_2 \sum_{k\in opp} K_\sigma(\mathbf{s},\mathbf{p}_k) \bigr)\, d\mathbf{s} \Bigr)
\]

实现：沿 \(\mathbf{p}_i\to\mathbf{p}_j\) 均匀采样 \(M\) 点近似积分。 \(\mathcal{L}_{ij}\to 0\) 表示线路被压缩，**自然降低** \(w_{ij}\)，无需「被挡就不传」规则。

### 7.6 战术软区域（半空间、肋部、Zone 14）

不用离散区域表，用 **高斯软核** 叠加：

\[
Z_{halfL}(\mathbf{r}) = \exp\bigl(-\| \mathbf{r}-\mathbf{r}_{halfL}\|^2 / 2\sigma_z^2\bigr),\quad
Z_{halfR}(\mathbf{r}) = \exp\bigl(-\| \mathbf{r}-\mathbf{r}_{halfR}\|^2 / 2\sigma_z^2\bigr)
\]

\[
Z_{14}(\mathbf{r}) = \exp\bigl(-\| \mathbf{r}-\mathbf{r}_{zone14}\|^2 / 2\sigma_{14}^2\bigr)
\]

教练 `width` 增大 → 边锋无球目标沿 \(Z_{half}\) 梯度加权；`risk_budget` 增大 → 前腰向 \(Z_{14}\) 漂移权重上升。

**进入次数**（赛后统计）：\(\int_0^T \mathbb{1}[\sigma(Z_{14}(\mathbf{p}_i))>0.5]\,dt\) 的软近似 \(\int \sigma(Z_{14})\,dt\)。

### 7.7 无球跑动：势能场导航

对无球球员 \(i\)：

\[
\dot{\mathbf{p}}_i = \alpha_i \nabla V_i(\mathbf{p}_i) - \beta_i \nabla \mathrm{Press}(\mathbf{p}_i) + \boldsymbol{\eta}_i
\]

综合势能：

\[
V_i = w_\Phi \Phi + w_{lane} \cdot \mathrm{dist\_to\_goal\_lane}(\mathbf{p}_i) + w_{supp} \cdot K_\sigma(\mathbf{p}_i,\mathbf{p}_{ball}) + w_{form} \cdot \|\mathbf{p}_i - \mathbf{M}_{form}[i]\|^{-1}_{soft}
\]

- \(\mathbf{M}_{form}[i]\)：阵型锚点（§8）  
- \(\mathrm{dist\_to\_goal\_lane}\)：到对方球门中轴垂距的 **平滑递减** 函数  
- \(\alpha_i = \alpha_0 \cdot \sigma(S_i) \cdot \sigma(s_i)\)：空间认知 × 体能  

**防守方** 对应势能 \(V_i^{def}\)：最大化压迫（沿 \(+\nabla \mathrm{Press}\)）+ 维持防线紧凑（见 §7.9）。

### 7.8 预判场 \(\hat{\mathbf{B}}(\mathbf{r},t)\)（球路预测）

球位置 \(\mathbf{b}(t)\)，速度 \(\mathbf{v}_b\)：

\[
\hat{\mathbf{B}}(\mathbf{r},t) = K_\sigma(\mathbf{r},\; \mathbf{b} + \mathbf{v}_b \Delta t_{pred}) \cdot \sigma(\|\mathbf{v}_b\|)
\]

防守球员跑向 \(-\nabla_{\mathbf{p}_i} \|\mathbf{p}_i - \int \mathbf{r}\hat{\mathbf{B}}(\mathbf{r})\,d\mathbf{r}\|\) 的软目标，实现 **提前封锁传球线路** 而非反应式跟随。

### 7.9 团队空间形态张量 \(\mathbf{S}_{team}\)

全队 10 外场位置协方差 \(\boldsymbol{\Sigma}_{team}\) 与质心 \(\bar{\mathbf{p}}\)：

- **紧凑度**（连续，非「防线≤4m」）：\(C_{def} = \mathrm{tr}(\boldsymbol{\Sigma}_{team})^{-1/2}\)  
- **纵向深度**：\(\sigma_{xx}\)  
- **宽度**：\(\sigma_{yy}\)  

教练 \(\mathbf{u}_{coach}\) 设定 **目标形态** \(\boldsymbol{\Sigma}^\*\)、\(\bar{\mathbf{p}}^\*\)：

\[
\mathcal{L}_{shape} = \|\boldsymbol{\Sigma}_{team}-\boldsymbol{\Sigma}^\*\|_F^2 + \|\bar{\mathbf{p}}-\bar{\mathbf{p}}^\*\|^2
\]

队内共享惩罚进入无球 \(V_i^{form}\)，实现 **集体保持阵型** 而非每人独立随机游走。

### 7.10 压迫陷阱与空间封锁（势阱）

当 `press_intensity` 高，在球附近构造 **势阱**：

\[
W_{trap}(\mathbf{r}) = -w_{trap} \exp\bigl(-\|\mathbf{r}-\mathbf{p}_{ball}\|^2/(2\sigma_{trap}^2)\bigr)
\]

防守者沿 \(-\nabla W_{trap}\) 收拢；进攻方沿 \(+\nabla \Phi - \nabla W_{trap}\) 寻找脱出通道。陷阱成功率连续体现在 **传球 \(q_{ij}\)** 下降，而非「触发陷阱事件」布尔。

### 7.11 空间智能 → 传球 / 射门（与 §9 接口）

传球边权扩展为：

\[
w_{ij} = \sigma(\cdots) \cdot \exp(\beta C_{ij}) \cdot \sigma(s_i) \cdot \mathcal{L}_{ij} \cdot \Phi(\mathbf{p}_j) \cdot \bigl(1+\lambda_N \mathcal{N}(\mathbf{p}_j)\bigr)
\]

射门效用扩展：

\[
u_{shot} \mathrel{+}= c_\Phi \Phi(\mathbf{p}_i) + c_N \mathcal{N}(\mathbf{p}_i) - c_{trap} W_{trap}(\mathbf{p}_i)
\]

**空间崩溃率**（连续）：\(\kappa_{collapse} = \|\nabla \Phi(\mathbf{p}_{ball})\|\)，用于 L2 相位统计与 Fusion `pitch` 专家。

### 7.12 空间智能指标（导出与校准）

写入 `MatchMicroSummary.phase_stats`：

| 指标 | 公式/含义 |
|------|-----------|
| `phi_integral` | \(\int_\Omega \Phi\,d\mathbf{r}\) 全队可推进空间体积 |
| `halfspace_exposure` | \(\sum_i \int \sigma(Z_{half}(\mathbf{p}_i))\,dt\) |
| `lane_quality_mean` | \(\mathbb{E}_{pass}[\mathcal{L}_{ij}]\) |
| `overload_created` | \(\int \max(0,\mathcal{N}(\mathbf{r}))\,d\mathbf{r}\,dt\) |
| `shape_compactness` | \(C_{def}\) 时间平均 |
| `spatial_iq_util` | \(\mathbb{E}_i[\sigma(S_i)\cdot \|\nabla\Phi(\mathbf{p}_i)\|]\) |

用于：赛后叙事、校准真实数据、向上层 `decision_memory` 特征。

### 7.13 与控球扩散场的关系（模块依赖）

```
SpatialFieldEngine (§6)     →  ρ_k, A_k, Press
        ↓
SpatialIntelligenceEngine   →  Φ, O_i, N, L_ij, V_i, B̂, Σ_team
        ↓
PassingGraphEngine (§9)     →  w_ij, q_ij, u_shot
```

每 tick 顺序：**先更新 \(\rho\) → 再算 SIE 场 → 再更新 \(\mathbf{p}_i\) → 再传球决策**。

### 7.14 计算实现要点

- 网格：\(32\times 22\) 或 \(48\times 33\) 均匀格；\(\Phi,\rho,\mathcal{N}\) 用 `numpy` 卷积/梯度  
- 快速模式 `MATCH_FAST_MODE=1`：降为 \(16\times 11\)，SIE 每 2 个 \(\Delta t\) 更新一次  
- 禁止：基于格点 `if grid[i,j]==EMPTY` 的离散路径搜索；允许：**势场梯度跟踪** + 小噪声（可复现 RNG）

### 7.15 全规格延伸（v2.0）

§7 为 **空间认知骨架**。情绪四主体耦合、位置/朝向/相位、香蕉球/落叶球/头球/直塞等球路，见独立文档：

**[docs/SPATIAL_INTELLIGENCE_FULL_SPEC.md](SPATIAL_INTELLIGENCE_FULL_SPEC.md)** — 空间—情绪—球路耦合 (SAC) 最细方案。

---

## 8. 球员实体与阵容

### 8.1 阵容与阵型

- 从 `data/rosters/{team_id}.json` 加载 23 人名单，赛前选 **18+5**。
- 阵型矩阵 \(\mathbf{M}_{form}\in\mathbb{R}^{N\times 2}\): 将 10 外场 + GK 锚定到默认坐标，教练 `formation` 字符串映射到 \(\mathbf{M}\)（查表 + 连续插值，如 4-3-3 → 4-2-3-1 用凸组合）。

### 8.2 能力向量（真实球员）

每名球员静态 \(\boldsymbol{\theta}_i\):

\[
[\text{tech},\text{phys},\text{mental},\text{pace},\text{vision},\text{spatial},\text{press},\text{aerial},\text{gk}]^\top \in [0,1]^9
\]

来源：手工标注 / CSV / 外部 API；缺失时用 **tier + role** 先验：

\[
\boldsymbol{\theta}_i \sim \mathcal{N}(\boldsymbol{\mu}_{role,tier}, \boldsymbol{\Sigma}_{role})
\]

### 8.3 角色耦合矩阵（球队化学）

\[
\mathbf{C}_{chem} \in \mathbb{R}^{N\times N},\quad C_{ij}=f(\text{club_overlap}, \text{history}, \text{position_compatibility})
\]

俱乐部重叠、国家队搭档历史可从 `goalscorers.csv` / 新建 `pair_chemistry.json` 估计。

---

## 9. 传球网络与射门决策

### 9.1 传球为图上的消息传递

每 \(\Delta t\)，构造 **可行传球边** \(i\to j\)（距离、角度、压迫约束）。边权 **必须** 接入 §7 空间智能因子 \(\mathcal{L}_{ij},\Phi,\mathcal{N}\)（完整式见 §7.11）:

\[
w_{ij} = \underbrace{\sigma(\alpha^\top [\theta_i,\theta_j, A(\mathbf{p}_j), -\|\mathbf{p}_i-\mathbf{p}_j\|])}_{技术/空间} \cdot \underbrace{\exp(\beta C_{ij})}_{化学} \cdot \underbrace{\sigma(s_i)}_{体能} \cdot \mathcal{L}_{ij} \cdot \Phi(\mathbf{p}_j)
\]

出球人选 \(i^\*\): \(\mathrm{softmax}_i(h_i + \text{role_bias})\)  
目标 \(j^\*\): \(\mathrm{softmax}_j(w_{ij})\)

### 9.2 传球结果（无布尔成功/失败表）

传球到达强度 \(q_{ij}\in[0,1]\):

\[
q_{ij} = \sigma\bigl( b_0 + b_1 w_{ij} - b_2 \cdot \text{Press}_{opp}(\mathbf{p}_j) \bigr)
\]

\(q_{ij}\) 解释为 **控球转移比例**；部分失败表现为 \(\rho\) 场回流对方而非离散丢球。

### 9.3 射门决策

射门效用:

\[
u_{shot} = c_0 + c_1 \chi(\mathbf{p}_i) + c_2 \sigma(m_i) - c_3 q_i + c_4 \tanh(h_i - h_{fatigue})
\]

以概率 \(\sigma(u_{shot}/\tau_{dec})\) 进入射门相位；否则继续传球循环。

### 9.4 射门结果（xG 核）

条件强度 \(\lambda_{shot} = \lambda_0 \cdot \chi(\mathbf{p}) \cdot \sigma(\text{GK\_block})\)，其中守门员阻挡:

\[
\text{GK\_block} = g_0 + g_1 \theta_{GK} - g_2 \|\mathbf{p}_{ball}-\mathbf{p}_{gk}\|
\]

实际进球：\(\mathrm{Bernoulli}(1-e^{-\lambda_{shot}\Delta t_{shot}})\) 或 Poisson(1) 近似。

---

## 10. 体能、士气与认知状态

### 10.1 体能 ODE（含战术负荷）

\[
\dot{s}_i = -\kappa_s^{base}(\mathbf{u}_{coach}) - \kappa_s^{press}\cdot \text{Press}_i + \kappa_s^{recover}\cdot \mathbb{1}_{off}(i) + \kappa_s^{sub}\cdot \Delta_{fresh}(i)
\]

- 高位逼抢 `press_intensity` 增大 \(\kappa_s^{press}\)
- 替补上场: \(\Delta_{fresh}(i)=\sigma(\text{minutes\_bench})\) 脉冲

**无「跑不动就不传」阈值** — 仅通过 §9 的 \(\sigma(s_i)\) 连续衰减。

### 10.2 士气 ODE（事件驱动 + 社交）

\[
\dot{m}_i = -\rho_m m_i + \sum_{e\in\mathcal{E}} \phi_m(e,i) + \gamma_m \sum_j C_{ij}\tanh(m_j-m_i)
\]

事件 \(\phi_m\): 助攻、失误、被换下、观众嘘声、教练鼓励（见 §12）。

与 GFS 情感层对齐：赛后把 \(\bar{m}_i\) 聚合为 `appraisal` 输入的 `impact` 分量。

### 10.3 认知负荷

\[
\dot{q}_i = a_q \cdot \text{Press}_i + b_q \cdot \mathbb{1}_{score\_behind} - c_q \cdot \text{coach\_clarity}
\]

\(\tau_{dec} = \tau_0(1 + d_q q_i)\) 连接决策模糊度；与 §7 空间认知 \(S_i\) 共同调制「阅读比赛」能力。

### 10.4 Hawkes 触球热度

\[
\dot{h}_i = -\beta_h h_i + \alpha_h \sum_{t_k<t} e^{-\beta_h(t-t_k)}
\]

用于「球权找状态热的人」而非硬编码「核心球员必触球」。

---

## 11. 教练战术与换人

### 11.1 连续战术控制向量

与现有 `tactical_controls` 对齐并扩展：

\[
\mathbf{u}_{coach} = [\text{line\_height}, \text{width}, \text{press}, \text{risk}, \text{tempo}, \text{counter}, \text{sub\_agg}]^\top
\]

**赛前**: LLM JSON → `set_tactical_controls`（已有）  
**赛中**: 教练策略 \(\pi_{coach}\) 每 \(M\) 分钟更新 \(\mathbf{u}\):

\[
\mathbf{u}^{t+M} = \mathbf{u}^t + \mathbf{K}_{PI}(\mathbf{x}_{score}-\mathbf{x}_{target}) + \mathbf{K}_{match}\mathbf{s}_{phase}
\]

\(\mathbf{s}_{phase}\): 相位统计（被 xG、控球、压迫）。这是 **连续 PID / 梯度步**，非「落后就 4231」规则表。

### 11.2 换人：非齐次 Poisson 过程

替补池 \(b\in\mathcal{B}\)，换人时刻 \( \tau \sim \mathrm{NHPP}(\lambda_{sub}(t)) \):

\[
\lambda_{sub}(t) = \lambda_0 \cdot \sigma(\text{sub\_agg}) \cdot \exp\bigl(\eta_1 \cdot \text{fatigue\_avg} + \eta_2 \cdot \text{card\_load} - \eta_3 \cdot \text{score\_lead}\bigr)
\]

选人：\(\mathrm{softmax}_b( w_{out\to b}^{tactical} + w_{fresh})\)。

### 11.3 真实教练

`CoachProfile`: `id`, `name`, `risk_appetite`, `adapt_speed`, `press_bias`, `youth_bias`  
LLM 仅生成 **自然语言 reasoning**；数值 \(\mathbf{K}_{PI}\) 来自数据或默认。

---

## 12. 观众—裁判—场上双向耦合

### 12.1 观众场 \(\psi(t)\)

\[
\dot{\psi} = a_\psi \cdot \mathrm{GoalPulse}(t) + b_\psi \cdot \mathrm{FoulControversy}(t) - c_\psi \psi + d_\psi \cdot \text{HomeAdvantage}
\]

- `HomeAdvantage`: 主场系数（世界杯中立场地可设 0）
- 输出 \([0,1]\): \(\sigma(\psi)\)

### 12.2 对球员/教练

\[
\dot{m}_i \mathrel{+}= \kappa_{crowd\to player} \cdot \tanh(\psi - \psi_{neutral}) \cdot \text{fan\_affinity}_i
\]

\[
\dot{c}_{coach} \mathrel{+}= \kappa_{crowd\to coach} \cdot \tanh(\psi)\cdot (1-\text{experience}_{coach})
\]

### 12.3 对裁判（延续现有 Beta 严格度）

在 `referee["strictness"]` 基础上：

\[
\text{strictness}' = \text{strictness} + \kappa_{r1}\tanh(\psi-\psi_0) + \kappa_{r2}\tanh(\text{grievance}_{home}-\text{grievance}_{away})
\]

犯规强度 \(\lambda_{foul}\):

\[
\lambda_{foul} = \lambda_{f0} \exp(\delta_{strict}\cdot \text{strictness}' + \delta_{bias}\cdot \text{bias}_k)
\]

### 12.4 球员/教练 → 观众（反作用）

进球、技巧传球、暴力犯规 → \(\psi\) 方程源项；教练抗议 → 短暂 \(\psi\) 尖峰。

### 12.5 与现有 `apply_referee_dynamics` 集成

微观产生的 `referee_controversy_integral` 写入赛后 `recursive_update` 的 `referee_controversy`，替代纯随机争议。

---

## 13. 跨场记忆与赛程负荷

### 13.1 球员级 carryover 核

上场结束保存 `CarryoverState`:

\[
\mathbf{k}_i^{(match)} = \bigl[ s_i^{end},\; m_i^{end},\; \text{minutes},\; \text{knock},\; \text{yellow\_load},\; \text{highlight\_valence} \bigr]
\]

下一场初值:

\[
s_i^{0} = s_i^{end} + \kappa_{rest}(rest\_days) + \epsilon,\quad
m_i^{0} = \rho_{carry} m_i^{end} + (1-\rho_{carry}) m_{baseline}
\]

**rest_days** 来自 `TournamentManager.play_match` 已有 `recover(rest_units)`。

### 13.2 球队级（对接 `SocietyAgent`）

| 微观输出 | 宏观字段 |
|----------|----------|
| 平均体能下降 | `fatigue` |
| 医疗事件积分 | `injury_load` |
| 关键球员红牌/伤病 | `injury_list` 条目 |
| 争议犯规 | `referee_grievance` |
| 球星表现 | Icon `ego` / `patience` 微调 |

### 13.3 决策记忆增强

`record_decision_event` 增加特征: `xg_phase_profile`, `sub_timing_error`, `press_collapse_integral` — 供 `retrieve_similar_decision_memories` 连续相似度（余弦核，非阈值过滤）。

---

## 14. 进球 / xG 涌现与宏观接口

### 14.1 微观 xG 累积

\[
XG_k = \int_0^T \int_\Omega \chi_k(\mathbf{r},t)\, d\mathbf{r}\, dt
\]

离散实现: 每步 \(\Delta XG \mathrel{+}= \chi \cdot \Delta t \cdot \mathrm{area\_cell}\)。

### 14.2 宏观比分生成（保持 Poisson 契约，改进 λ）

**推荐**: λ 由微观与宏观融合：

\[
\lambda_k = \underbrace{\lambda_{micro}(XG_k)}_{\text{主}} \cdot \Bigl(1 + \eta \cdot \tanh\bigl(\frac{eff\_status_k - \bar{s}}{scale}\bigr)\Bigr)
\]

\[
\lambda_{micro}(XG) = \lambda_0 \cdot XG^{\gamma} \quad (\gamma \approx 0.85\text{–}1.0)
\]

仍调用 `np.random.poisson`（或现有 knockout 双抽），**替换** `status/35` 启发式，但函数签名不变。

### 14.3 加时 / 点球

- 加时: 降低 \(\kappa_s^{recover}\)，提高 \(\kappa_s^{press}\)，\(T=30\) 分钟微观子仿真。
- 点球: 独立 `PenaltyDuelEngine`，球员 \(\theta_{mental}\)、\(m_i\)、`\penalty_anxiety`（球队已有）进入 \(\sigma(\cdot)\) 转化概率。

---

## 15. 真实球员与教练数据层

### 15.1 目录结构（建议）

```
data/
  rosters/
    Brazil.json
    Argentina.json
    ...
  coaches/
    Brazil.json
  schemas/
    player.schema.json
    coach.schema.json
  external/          # gitignore，用户自备
    fbref_2026.csv
```

### 15.2 `player.schema.json`（核心字段）

```json
{
  "player_id": "br_vinicius_7",
  "name": "Vinícius Júnior",
  "team_id": "Brazil",
  "role": "LW",
  "club": "Real Madrid",
  "age": 25,
  "abilities": {
    "tech": 0.92, "phys": 0.88, "mental": 0.80,
    "pace": 0.95, "vision": 0.85, "spatial": 0.90, "press": 0.70,
    "aerial": 0.45, "gk": 0.0
  },
  "traits": ["inverted_winger", "clutch"],
  "fan_affinity": 0.95
}
```

### 15.3 与现有 `tactics_final_en.json` 关系

- `style_desc` 继续服务 LLM 叙事与 `style_archetype`。
- **仿真数值** 只读 `rosters/*.json`；无 roster 时回退 **SyntheticRosterGenerator**（tier + formation）。

### 15.4 数据获取建议

| 来源 | 用途 | 许可注意 |
|------|------|----------|
| FIFA/公开名单 | 姓名、号码、位置 | 核对授权 |
| FBref / StatsBomb 开源 | 能力先验校准 | 非商业条款 |
| 手工 YAML | 世界杯 48 队 baseline | 首版可交付 |

实现工具: `scripts/build_rosters_from_csv.py`（不入核心仿真路径）。

---

## 16. 与 LLM / Fusion 的边界

### 16.1 LLM 仍负责（不变）

- 赛前战术 JSON、信念整合
- 赛后媒体矩阵、反思日记
- 社会对话

### 16.2 LLM 不负责（新增约束）

- 每秒传球选择、比分、犯规次数

### 16.3 Fusion 扩展（可选 Phase 2）

新增专家通道 `pitch`:

```python
pitch_signal = {
    "status": +1.8 * tanh(xg_for - xg_against),
    "volatility": 0.2 + 0.4 * press_chaos_integral,
    "chaos": 0.1 + 0.5 * turnover_rate,
    # 空间智能导出（§7.12）
    "spatial_status": +1.2 * tanh(phi_integral_for - phi_integral_against),
}
```

在 `FusionController.export_expert_signals` 以 **可选参数** 注入，默认权重 0（不改变 legacy 行为）。

---

## 17. 代码结构与集成点

### 17.1 新包布局

```
src/match_engine/
  __init__.py
  adapter.py              # simulate_match_score_micro
  state.py                # MatchState, PlayerState, CoachControls
  spatial_field.py        # SpatialFieldEngine (§6)
  spatial_intelligence.py # SpatialIntelligenceEngine (§7)
  passing_graph.py        # PassingGraphEngine
  stamina_morale.py       # StaminaMoraleODE
  coach_policy.py         # CoachPolicy, substitution NHPP
  crowd_referee.py        # CrowdRefereeCoupling
  carryover.py            # CarryoverKernel
  goal_generator.py       # λ from XG, Poisson finalizer
  roster_loader.py
  synthetic_roster.py
  integrator.py           # TickIntegrator 主循环
```

### 17.2 环境变量

| 变量 | 默认 | 含义 |
|------|------|------|
| `MATCH_ENGINE` | `legacy` | `legacy` \| `micro` |
| `MATCH_DT` | `3.0` | 仿真步长（秒） |
| `MATCH_FAST_MODE` | `0` | 1=粗空间网格，加速 |
| `ROSTER_PATH` | `data/rosters` | 名单目录 |

### 17.3 入口脚本（新增）

```bash
# 单场调试，不跑全杯赛
python run_single_match.py --home Brazil --away Argentina --seed 42 --engine micro
```

### 17.4 测试策略

- **单元**: 各 ODE 能量守恒/有界性；softmax 和为 1
- **回归**: `MATCH_ENGINE=legacy` 与当前 main 输出分布一致
- **统计**: micro 模式 xG–进球 相关系数合理（≈0.3–0.6 量级）

---

## 18. 实现路线图

### Phase 0 — 脚手架（1 周）

- [ ] `match_engine` 包 + `adapter` 透传 legacy
- [ ] `MatchMicroSummary` DTO + `run_single_match.py`
- [ ] `MATCH_ENGINE` 开关 + CI 双模式

### Phase 1 — 最小微观闭环（2–3 周）

- [ ] 空间场 \(\rho\) 1D 简化版（纵向推进）
- [ ] **空间智能 v1**：\(\Phi\)、\(\mathcal{L}_{ij}\)、无球 \(\nabla V_i\)
- [ ] 传球图 + xG 累积
- [ ] 体能/士气 ODE
- [ ] λ 来自 XG 的 `goal_generator`
- [ ] 接入 `play_match` 第 6 步

### Phase 2 — 真实 roster + 换人 + 教练 PID（2 周）

- [ ] `data/rosters/*.json` 首批 8 强球队
- [ ] NHPP 换人 + 教练 \(\mathbf{u}(t)\) 演化
- [ ] `apply_match_wear` 读 micro 导出

### Phase 3 — 观众裁判耦合 + 跨场（2 周）

- [ ] \(\psi(t)\) 与 `apply_referee_dynamics` 联动
- [ ] `CarryoverState` 序列化进 `SocietyAgent` 扩展字段

### Phase 4 — 空间智能增强（2 周）

- [ ] 软 Voronoi \(O_i\)、人数优势 \(\mathcal{N}\)
- [ ] 预判场 \(\hat{\mathbf{B}}\)、压迫势阱 \(W_{trap}\)
- [ ] 团队形态 \(\boldsymbol{\Sigma}_{team}\) 与半空间/Zone14 统计

### Phase 5 — 性能与校准（持续）

- [ ] `MATCH_FAST_MODE`、向量化场更新
- [ ] 与历史 CSV xG 代理指标校准
- [ ] Fusion `pitch` 专家（可选）

---

## 19. 验证与校准

### 19.1 内禀检验

- 长时间仿真: \(\sigma(s_i)\) 不漂移出 \([0,1]\) 显示域
- 平局倾向: 可通过对 \(\lambda_{micro}\) 的 \(\gamma\) 调节
- 主场优势: \(\psi\) 中性场应为 0 对称

### 19.2 外禀检验（若有数据）

- 传球成功率 ~ 75–85%（顶级队）
- 单场 xG 总和 ~ 2.0–3.5
- 替补后 15 分钟 xG 强度变化方向合理

### 19.3 与 GFS 上层一致性

- `eff_status` 高的一方 **边际上** \(\mathbb{E}[\lambda]\) 更高，但不强制每场获胜
- 赛后 `recursive_update` 输入的 `score_diff` 与微观 `key_events` 一致

---

## 20. 附录：符号表与默认方程

### 20.1 主要符号

| 符号 | 含义 |
|------|------|
| \(\rho_k\) | 球队 k 控球密度场 |
| \(\Phi\) | 自由空间场 |
| \(\mathcal{L}_{ij}\) | 传球走廊质量 |
| \(\mathcal{N}\) | 人数优势场 |
| \(S_i\) | 个体空间认知 |
| \(O_i\) | 软 Voronoi 占有 |
| \(A_k\) | 空间优势（控球比） |
| \(\mathbf{u}_{coach}\) | 连续战术控制 |
| \(w_{ij}\) | 传球边权重 |
| \(\chi\) | 射门/xG 局部强度 |
| \(\psi\) | 观众情绪状态 |
| \(s_i, m_i\) | 体能、士气（logit） |
| \(XG_k\) | 累积期望进球 |

### 20.2 默认超参（初值，待校准）

```yaml
spatial:
  D0: 0.08
  lambda_press: 0.35
  alpha_w: 0.25
spatial_intelligence:
  a0: 0.0
  a1: 1.2
  a2: 1.8
  a3: 0.6
  sigma_p: 0.06
  mu1_lane: 2.2
  mu2_lane: 1.4
  w_phi_move: 0.55
  w_lane_move: 0.25
  w_supp_move: 0.20
  w_trap: 0.45
  sigma_trap: 0.08
  rho_S: 0.04
  kappa_v: 0.08
passing:
  tau_dec: 0.45
  beta_chem: 0.60
stamina:
  kappa_s_base: 0.012
  kappa_s_press: 0.028
  kappa_s_recover: 0.008
morale:
  rho_m: 0.05
  gamma_m: 0.02
crowd:
  a_psi: 0.40
  c_psi: 0.12
  kappa_crowd_player: 0.015
goal_link:
  lambda0: 0.95
  gamma_xg: 0.90
  eta_status_blend: 0.12
```

### 20.3 伪代码：主循环

```python
def run_micro_match(state: MatchState, dt: float, T: float) -> MatchMicroSummary:
    t = 0.0
    while t < T:
        spatial.step(state, dt)              # §6: ρ, Press
        sie.step(state, dt)                # §7: Φ, L_ij, N, V_i, p_i drift
        coach.update_controls(state, t)
        maybe_substitution(state, t)
        crowd_ref.step(state, dt)
        for _ in range(n_ticks_per_dt):
            i_star = select_on_ball(state)
            action = softmax_sample([
                ("pass", u_pass(i_star)),
                ("shot", u_shot(i_star)),
                ("dribble", u_dribble(i_star)),
            ], tau=state.tau_dec(i_star))
            execute(action, state)
            stamina_morale.step(state, dt_inner)
        t += dt
    goals = goal_generator.finalize(state)
    return build_summary(state, goals)
```

---

## 结语

该方案在 **工程上** 把单场真实感拆成可实现的连续模块，在 **架构上** 尊重 GFS「数学主引擎 + LLM 约束认知 + 社会心理外层」的分层，在 **数据上** 为真实球员/教练预留一等公民 schema，同时用 `MATCH_ENGINE=legacy` 保证现有世界杯脚本零回归。

**我的建议**: 优先完成 **Phase 0 + Phase 1**（空间场 + **空间智能 v1** + 传球 + xG → λ → 比分 + adapter），用 8 支球队真实 roster 做 demo，再迭代观众—裁判、跨场 carryover 与 Phase 4 高阶空间智能。这样你可以在几周内看到「会跑位、会找空当」的单场，而不必等全模块一次到位。

---

*文档维护: 随 `src/match_engine/` 实现进度更新 Phase checklist。*

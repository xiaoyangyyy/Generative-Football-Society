# 空间智能全规格书（SIE+ 情绪耦合 + 球路连续体）

**版本**: 2.0  
**关联文档**: [`SINGLE_MATCH_ENGINE_DESIGN.md`](SINGLE_MATCH_ENGINE_DESIGN.md)  
**定位**: 在 v1.1 §7 基础上，回答「是否全面」「情绪要不要建」「位置怎么建」「香蕉球/头球/直塞等怎么建」的 **最细实现级方案**。

---

## 目录

1. [诚实评估：v1.1 全面吗？](#1-诚实评估v11-全面吗)
2. [总架构：空间—情绪—球路耦合 (SAC)](#2-总架构空间情绪球路耦合-sac)
3. [多主体情绪场（球员 / 教练 / 裁判 / 观众）](#3-多主体情绪场球员--教练--裁判--观众)
4. [位置与体态：从锚点到连续跑位](#4-位置与体态从锚点到连续跑位)
5. [球路连续体：不传「类型标签」，传「物理参数」](#5-球路连续体不传类型标签传物理参数)
6. [动作目录：短传 / 直塞 / 传中 / 各类射门 / 头球](#6-动作目录短传--直塞--传中--各类射门--头球)
7. [空中对抗与二点球](#7-空中对抗与二点球)
8. [单 tick 完整计算管线](#8-单-tick-完整计算管线)
9. [与 GFS 宏观层的双向接口](#9-与-gfs-宏观层的双向接口)
10. [数据 schema 与能力维度扩展](#10-数据-schema-与能力维度扩展)
11. [实现模块与文件结构](#11-实现模块与文件结构)
12. [分阶段交付（最细路线图）](#12-分阶段交付最细路线图)
13. [校准、验证与可观测指标](#13-校准验证与可观测指标)
14. [附录 A：符号总表](#附录-a符号总表)
15. [附录 B：默认超参 YAML](#附录-b默认超参-yaml)

---

## 1. 诚实评估：v1.1 全面吗？

### 1.1 已覆盖（v1.1 §7 做得对的部分）

| 维度 | 覆盖度 | 说明 |
|------|--------|------|
| 自由空间 / 压迫 | ★★★★☆ | \(\Phi\)、Press、走廊 \(\mathcal{L}_{ij}\) |
| 无球跑动 | ★★★☆☆ | 势能梯度，缺 **朝向 / 速度矢量 / 角色通道** |
| 团队形态 | ★★★☆☆ | \(\boldsymbol{\Sigma}_{team}\)，缺 **相位切换（攻/守/转换）** |
| 传球选择 | ★★★☆☆ | 图上的 softmax，缺 **球路物理与动作参数** |
| 射门 xG | ★★☆☆☆ | 位置核 \(\chi\)，缺 **轨迹、旋转、门将反应** |
| **情绪** | ★☆☆☆☆ | 仅 \(S_i\) 受 \(m_i,q_i\) 影响，**未**建裁判/观众/教练当场耦合 |
| **球路类型** | ☆☆☆☆☆ | 未建模 |
| **头球 / 空中** | ☆☆☆☆☆ | 未建模 |
| **裁判尺度** | ★★☆☆☆ | 宏观有，微观犯规未与 **空间侵犯** 连续耦合 |

**结论**: v1.1 是优秀的 **「空间认知骨架」**，但还不是你要的「非常真实」的完整方案。  
**v2.0（本文）** 在骨架上增加三块一等公民模块：

1. **AffectiveSpatialCoupling** — 球员/教练/裁判/观众情绪 ↔ 空间行为  
2. **KinematicPositionLayer** — 位置、朝向、速度、角色通道、相位  
3. **BallManifoldEngine** — 球路连续参数 + 轨迹 ODE（涵盖短传、直塞、香蕉球、落叶球、头球等）

---

## 2. 总架构：空间—情绪—球路耦合 (SAC)

### 2.1 设计信条（延续 GFS，禁止阈值）

- 不定义 `if action == "banana_shot"`；定义 **球状态** \(\mathbf{B}(t)=[\mathbf{r},\mathbf{v},\boldsymbol{\omega},h_{air}]\) 与 **动作参数** \(\mathbf{a}\in\mathbb{R}^d\)，由 softmax 选 \(\mathbf{a}\)。  
- 情绪不用「愤怒 > 0.7 就犯规」；用 **连续冲动** \(\iota_{foul}\)、**决策温度** \(\tau\)、**视野收缩** \(\sigma_{vision}\downarrow\)。  
- 「直塞 vs 短传」不是枚举，是 \((v,\theta_{elev},spin,\text{line curvature})\) 的 **不同区域**。

### 2.2 模块栈（每 \(\Delta t\)）

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Layer A  Macro inject (赛前)                                             │
│   SocietyAgent.z_state, emotion_profile, tactical_controls, referee priors│
└───────────────────────────────┬─────────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Layer B  SpatialFieldEngine (§6)  →  ρ, Press, A                         │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Layer C  AffectiveSpatialCoupling  →  e_i, e_coach, e_ref, ψ_crowd       │
│          modulates: S_i, τ_dec, σ_vision, Φ weights, foul intensity      │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Layer D  SpatialIntelligenceEngine (§7)  →  Φ, N, L_ij, V_i, Σ_team      │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Layer E  KinematicPositionLayer  →  p_i, v_i, θ_i, lane_i, role phase   │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Layer F  BallManifoldEngine  →  choose â, integrate B(t), contacts       │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Layer G  Event + xG + foul/referee + stamina/morale ODE                  │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Layer H  Meso aggregate (15') →  team appraisal micro-events → GFS       │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.3 核心状态（单场）

**球员 \(i\)**

\[
\mathbf{X}_i = \bigl[s_i,\; m_i,\; S_i,\; \mathbf{p}_i,\; \mathbf{v}_i,\; \theta_i,\; \mathbf{e}_i,\; h_i,\; q_i,\; \kappa_i\bigr]^\top
\]

- \(\mathbf{e}_i \in \Delta^4\) 当场微情绪（pride, anger, fear, determination）— 与 GFS 5 维可投影对齐  
- \(\theta_i\) 身体朝向（弧度）  
- \(\kappa_i\) 空中争顶能力状态（跳跃高度包络，连续）

**教练 \(c\)**

\[
\mathbf{X}_{coach} = [\mathbf{u}_{coach},\; e_{stress},\; e_{trust},\; e_{rage},\; \mathbf{u}_{target}]
\]

**裁判 \(r\)**

\[
\mathbf{X}_{ref} = [\text{strictness},\; e_{calm},\; e_{defensive},\; \text{bias}_H,\; \text{card\_load}]
\]

**观众（场级）**

\[
\psi \in \mathbb{R},\quad \dot\psi = f(\text{events},\; \text{home},\; \text{media})
\]

**球**

\[
\mathbf{B} = [\mathbf{r}_b,\; \mathbf{v}_b,\; \omega_b,\; \hat{\mathbf{n}}_b,\; h_{air}]
\]

\(\omega_b\) 旋转标量；\(\hat{\mathbf{n}}_b\) 旋转轴；\(h_{air}\) 离地高度（地面传时为 0）。

---

## 3. 多主体情绪场（球员 / 教练 / 裁判 / 观众）

### 3.1 要不要建模？——要，且必须与空间智能耦合

| 主体 | 当场是否建模 | 理由 |
|------|--------------|------|
| **球员** | **必须** | 士气影响跑位胆量、传球选择、射门冲动、犯规 |
| **教练** | **必须** | 临场变阵、换人、压迫调档受压力驱动 |
| **裁判** | **必须** | 尺度、偏哨、卡牌连续化，影响空间侵略 |
| **观众** | **必须** | 主场声浪、嘘声、进球浪潮反馈球员与裁判 |

**与 GFS 关系**:  
- **当场**: 微情绪 \(\mathbf{e}_i(t)\) 快速 ODE（秒—分钟级）  
- **赛后/节间**: 聚合为 `appraisal` 事件 → `SocietyAgent.recursive_update`（现有 6D appraisal + 5D emotion **不变**）  
- **不是两套系统**: 微情绪是宏观情感的 **高时间分辨率实现**，通过 §9 聚合核连接。

### 3.2 球员微情绪 \(\mathbf{e}_i\)（4D softmax，可扩 5D）

\[
\mathbf{e}_i = \mathrm{softmax}\bigl(\mathbf{W}_e \mathbf{z}_{i}^{emo}\bigr),\quad
\mathbf{z}_{i}^{emo} \in \mathbb{R}^4
\]

\[
\dot{\mathbf{z}}_i^{emo} = -\mathbf{R}_e \mathbf{z}_i^{emo} + \mathbf{G}_e(\text{micro-events}_i) + \mathbf{H}_e \boldsymbol{\Psi}_{crowd} + \sum_j \mathbf{K}_{ij}(\mathbf{z}_j^{emo}-\mathbf{z}_i^{emo})
\]

**微事件源 \(\mathbf{G}_e\)**（连续，非阈值）:

| 事件 | 对 \(\mathbf{z}^{emo}\) 的驱动 |
|------|-------------------------------|
| 完成关键传球 | \(+\delta_{pride}\) |
| 被抢断 | \(+\delta_{anger}, +\delta_{fear}\) |
| 射门未果 | \(+\delta_{shame}\) 投影到 fear/determination |
| 进球 | \(+\delta_{pride}, -\delta_{fear}\) |
| 被判犯规/黄 | \(+\delta_{anger}\)（若 \(\text{norm\_violation}\) 高） |
| 观众嘘声 | \(\propto -\tanh(\psi)\cdot \text{fan\_affinity}_i\) |

**空间智能耦合（关键）**:

\[
\tau_{dec,i} = \tau_0 \bigl(1 + d_q q_i + d_f \|\mathbf{e}_{fear}\| + d_a \|\mathbf{e}_{anger}\|\bigr)
\]

\[
\sigma_{vision,i} = \sigma_{v0} \cdot \bigl(1 - v_f \|\mathbf{e}_{fear}\| - v_a \|\mathbf{e}_{anger}\|\bigr)
\]

视野收缩 → \(\mathcal{L}_{ij}\) 积分路径采样变短 → **「压力下看不见直塞」** 自然出现。

\[
\alpha_{move,i} = \alpha_0 \cdot \sigma(s_i) \cdot \sigma(S_i) \cdot \bigl(1 + b_{det}\, e_{determination} - b_{fear}\, e_{fear}\bigr)
\]

\[
u_{shot,i} \mathrel{+}= \rho_{anger}\, e_{anger} - \rho_{fear}\, e_{fear}
\]

### 3.3 教练情绪 \(\mathbf{e}_{coach}\)

教练不是场上 23 人之一，但控制 \(\mathbf{u}_{coach}(t)\) 的 **PID/梯度步**:

\[
\dot{e}_{stress} = a_s \cdot |\text{score\_diff}| + a_x \cdot (\text{xGA} - \text{xGF}) - b_s e_{stress}
\]

\[
\dot{e}_{trust} = a_t \cdot \mathrm{coordination} - a_c \cdot \mathrm{conflict\_heat} - b_t e_{trust}
\]

**对空间智能的影响**:

\[
\mathbf{u}_{coach}^{t+\Delta} = \mathbf{u}_{coach}^t + \mathbf{K}_{PI}(\mathbf{s}^*-\mathbf{s}) + \mathbf{K}_{emo}\bigl[e_{stress},\, e_{trust},\, e_{rage}\bigr]
\]

- 压力大 → `press_intensity`↑、`risk_budget` 可能↓（由 \(\mathbf{K}_{emo}\) 符号决定，数据校准）  
- `sub_aggression` \(\propto \sigma(e_{stress} - e_{trust})\)

与现有 `simulate_internal_game`（教练 vs Icon）:  
Icon 抗议 → \(e_{coach}^{stress}\) 脉冲 → 战术抖动 \(\|\Delta \mathbf{u}_{coach}\|\) 增大。

### 3.4 裁判情绪 \(\mathbf{e}_{ref}\)

在现有 `strictness ~ Beta` 上叠加 **动态尺度**:

\[
\text{strictness}' = \text{strictness} + \kappa_{r\psi}\tanh(\psi-\psi_0) + \kappa_{rg}\tanh(e_{rage}^{crowd}) - \kappa_{rc} e_{calm}
\]

\[
\lambda_{foul}(\mathbf{r},t) = \lambda_{f0} \exp\bigl(\delta_s \cdot \text{strictness}' + \delta_p \cdot \mathrm{Press}(\mathbf{r}) - \delta_{trust}\cdot \text{referee\_trust}_{local}\bigr)
\]

**空间侵犯连续化**（无「战术犯规开关」）:

\[
\mathrm{Infringement}_{ij} = \int_{\text{duel}} \max(0, \|\mathbf{v}_i-\mathbf{v}_j\| - v_{allowed}) \cdot K_\sigma(\mathbf{s},\mathbf{p}_{duel})\, ds
\]

\(\mathrm{Infringement} > \text{连续} \Rightarrow\) 卡牌强度 \(\sim \text{Poisson}(\lambda_{card})\)，非 `if foul`.

**对球员**: 争议判罚 → 全队 \(\mathbf{z}^{emo}\) 的 `norm_violation` 通道上升 → 与 GFS `referee_grievance` 同步。

### 3.5 观众场 \(\psi\) 与双向耦合

沿用设计文档 §12，扩展 **空间反馈**:

\[
\dot\psi = a_\psi G(t) + b_\psi H_{home} - c_\psi \psi
\]

\(G(t)\): 进球、红牌、VAR 争议、精彩扑救（由事件强度积分）

**观众 → 空间**:

| 通道 | 效应 |
|------|------|
| 球员 | \(\dot m_i \mathrel{+}= k_{c\to p} \tanh(\psi)\cdot \text{affinity}_i\) |
| 教练 | \(\dot e_{stress} \mathrel{+}= k_{c\to c} \tanh(-\psi)\) 若落后主场 |
| 裁判 | \(\dot e_{calm} \mathrel{-}= k_{c\to r} |\psi|\) 尺度收紧或松动（校准） |
| 对手心理 | 客场球员 fear ↑ |

**空间 → 观众**:

\[
G(t) \mathrel{+}= \eta_1 \|\nabla \Phi(\mathbf{p}_{ball})\| + \eta_2 \cdot \mathbb{1}_{shot} \cdot \chi + \eta_3 \cdot \mathrm{SkillMove}_i
\]

### 3.6 情绪—空间耦合矩阵（一览）

```
         │  Φ    │  L_ij │  τ_dec │  λ_foul │  u_shot │  α_move │
─────────┼───────┼───────┼────────┼─────────┼─────────┼─────────┤
e_fear   │  ↓权  │  ↓    │  ↑     │  ↓      │  ↓      │  ↓      │
e_anger  │  乱   │  ↓    │  ↑     │  ↑      │  ↑      │  ↑      │
e_determ │  ↑权  │  ↑    │  ↓     │  —      │  ↑      │  ↑      │
e_stress │  —    │  —    │  —     │  —      │  —      │  coach↑press │
ψ_crowd  │  —    │  —    │  ↑/↓   │  ref↑   │  home↑  │  home↑  │
```

所有「↑↓」通过 **乘法调制** 与 **logit 偏移** 实现，不用 if-else。

---

## 4. 位置与体态：从锚点到连续跑位

### 4.1 三层位置表示

| 层 | 变量 | 含义 |
|----|------|------|
| **战术锚** | \(\mathbf{M}_{form}[i](t)\) | 阵型 4-3-3 理论站位，随 \(\mathbf{u}_{coach}\) 漂移 |
| **实际位** | \(\mathbf{p}_i \in [0,1]^2\) | 连续坐标（纵向 x：己方→对方，横向 y：左→右） |
| **体态** | \(\theta_i, \mathbf{v}_i\) | 朝向与速度，影响抢断面积、传球脚法 |

### 4.2 角色通道（Role Lane）

每球员角色 \(R_i \in \{\mathrm{GK,LB,CB,RB,DM,CM,AM,LW,RW,ST}\}\) 绑定 **通道中心线** \(y_{lane}(R_i)\):

\[
\mathcal{C}_i(\mathbf{p}) = \exp\bigl(-(y - y_{lane}(R_i))^2 / 2\sigma_{lane}^2\bigr)
\]

无球势能加项:

\[
V_i \mathrel{+}= w_{lane} \cdot \mathcal{C}_i(\mathbf{p})
\]

**防止边锋漂移到中路堆积** — 不用硬规则「W 不能进禁区中央」。

### 4.3 相位（Phase）连续变量

全队进攻相位 \(P \in [0,1]\)（0=深度防守，1=高位进攻）:

\[
\dot P = \kappa_P \bigl( \sigma(\text{possession}) + \sigma(\text{line\_height}) - \sigma(\text{press\_against}) - P \bigr)
\]

锚点随相位插值:

\[
\mathbf{M}_{form}[i] = (1-P)\mathbf{M}_{def}[i] + P\mathbf{M}_{att}[i]
\]

### 4.4 朝向与「身位」

\[
\dot\theta_i = \omega_i^{target} - \theta_i,\quad
\omega_i^{target} = \mathrm{atan2}\bigl([\mathbf{p}_{ball}-\mathbf{p}_i]_y, [\mathbf{p}_{ball}-\mathbf{p}_i]_x\bigr)
\]

**身位优势**（防守连续）:

\[
\mathrm{BodyAdv}_{ij} = \cos(\theta_i - \angle(\mathbf{p}_i\to\mathbf{p}_j)) \cdot \exp(-\|\mathbf{p}_i-\mathbf{p}_j\|/\ell)
\]

进入抢断成功率 \(\sigma(\cdots + \mathrm{BodyAdv})\).

### 4.5 位置相关空间智能

- **肋部** \(Z_{half}\): 边卫内收、内锋拉边由 \(w_{half}(R_i)\) 连续权重控制  
- **防线深度**: \(\bar x_{def} = \frac{1}{|D|}\sum_{i\in D} p_{i,x}\) 进入 offside 风险场（软）:

\[
\mathrm{OffsideRisk}(\mathbf{r}) = \sigma\bigl(\gamma_{off}(r_x - \bar x_{def} - \delta_{off})\bigr)
\]

直塞效用:

\[
u_{pass\_through} \mathrel{+}= \log\bigl(1 - \mathrm{OffsideRisk}(\mathbf{p}_{target})\bigr) \cdot \Phi(\mathbf{p}_{target})
\]

### 4.6 门将特殊坐标

\(\mathbf{p}_{GK}\) 受 **射门角度覆盖** 优化:

\[
\mathbf{p}_{GK}^* = \arg\min_{\mathbf{p}} \int_{\Omega_{goal}} \chi(\mathbf{r})\, d\mathbf{r}
\]

用梯度下降 1–2 步近似，非离散站位表。

---

## 5. 球路连续体：不传「类型标签」，传「物理参数」

### 5.1 核心思想

所有「香蕉球 / 落叶球 / 直塞 / 短传」都是 **同一 ODE** 在不同参数区域的特解：

\[
\dot{\mathbf{r}}_b = \mathbf{v}_b,\quad
\dot{\mathbf{v}}_b = \mathbf{g} + \mathbf{F}_{drag} + \mathbf{F}_{Magnus} + \mathbf{F}_{knuckle}
\]

| 力 | 公式 | 足球现象 |
|----|------|----------|
| 重力 | \(\mathbf{g}=(0,0,-g)\) | 弧线下落 |
| 阻力 | \(-c_d \|\mathbf{v}\|\mathbf{v}\) | 减速 |
| 马格努斯 | \(c_m (\boldsymbol{\omega}\times\mathbf{v})\) | **香蕉球**、弧线传球 |
| 抖动阻力 | \(-c_k \eta(t) \mathbf{v}\) | **落叶球**（低旋高漂） |

\(\boldsymbol{\omega} = \omega_b \hat{\mathbf{n}}_b\).

### 5.2 动作参数向量 \(\mathbf{a}\)（传球 / 射门统一）

\[
\mathbf{a} = [\,v_{0},\; \theta_{elev},\; \theta_{azim},\; \omega_b,\; n_x,\; n_y,\; n_z,\; t_{contact}\,]^\top
\]

- \(v_0\): 初速标量  
- \(\theta_{elev}\): 仰角（地面传 ≈ 0，传中 ≈ 0.4–0.8 rad）  
- \(\theta_{azim}\): 方位角  
- \(\omega_b\): 转速（香蕉高，落叶低）  
- \(\hat{\mathbf{n}}\): 旋转轴（侧旋 → 香蕉）  

**不存「直塞」标签**；直塞 = \((v_0 \text{大}, \theta_{elev} \text{小}, \text{目标 } \mathbf{p}_{ahead})\) 且 \(\Phi(\mathbf{p}_{target})\) 高。

### 5.3 动作选择：条件 softmax（给定场上状态）

候选集不是离散 200 种「招」，而是 **采样 \(K\) 组 \(\mathbf{a}_k\)** + 解析梯度修正:

\[
\mathbb{P}(\mathbf{a}_k) = \frac{\exp(u(\mathbf{a}_k)/\tau_{dec})}{\sum_j \exp(u(\mathbf{a}_j)/\tau_{dec})}
\]

效用 \(u(\mathbf{a})\) 含：

\[
u = w_\Phi \Phi(\mathbf{p}_{land}) + w_\chi \chi(\mathbf{p}_{land}) + w_{safe}\mathcal{L}_{ij} - w_{risk}\mathrm{OffsideRisk} - w_{spin}\cdot \text{exec\_error}(\omega)
\]

\(\text{exec\_error}(\omega) = (\omega - \omega_{skill})^2 / \sigma_{skill}^2\) 体现球员 **脚法能力** \(\theta^{tech}\).

### 5.4 「香蕉球」在方程里是什么？

**条件**: \(\omega_b\) 高，\(\hat{\mathbf{n}} \parallel\) 水平且 \(\perp \mathbf{v}\)（侧旋）  

曲率:

\[
\kappa_{curve} = c_m \omega_b / \|\mathbf{v}\|^2
\]

落地/门框概率通过 **轨迹积分** 与门将覆盖区相交面积 → xG 增量。

**球员能力**: \(\theta^{curve}\) 调制 \(c_m^{eff}\).

### 5.5 「落叶球」是什么？

**条件**: \(\omega_b \approx 0\)，\(c_k\) 大，\(\eta(t)\) 为 Ornstein-Uhlenbeck 噪声  

轨迹末端垂直速度突变 → 门将 \(\Delta t_{react}\) 不足:

\[
p_{goal} \mathrel{+}= \kappa_{knuckle} \cdot \exp(-\theta_{GK}^{aerial}) \cdot (1-\omega_b)
\]

### 5.6 「头球」是什么？

不是独立球类，是 **接触模式** \(contact = header\):

- 球到达 \(\mathbf{p}_{aerial}\) 时 \(h_{air} > h_{head,i}\)  
- 争顶胜率 \(\sigma(\theta^{aerial} + \kappa_i^{jump} - \theta_{opp}^{aerial})\)  
- 接触后 \(\mathbf{v}_b\) 重定向，\(\omega_b \approx 0\)  

\(\kappa_i^{jump} = \kappa_0 \sigma(s_i) \cdot \sigma(\text{timing})\).

---

## 6. 动作目录：短传 / 直塞 / 传中 / 各类射门 / 头球

下列「类型」均为 **\((\mathbf{a}, \text{目标点}, \text{接触模式})\) 的命名区域**，实现时用高斯先验在 \(\mathbf{a}\) 空间采样，**不是** `match action_type`.

### 6.1 地面传球谱系

| 名称 | 参数区域（示意） | 空间智能入口 |
|------|------------------|--------------|
| **短传** | \(v_0\in[3,8]\), \(\theta_{elev}<0.05\), 距离 \(<15m\) | \(\mathcal{L}_{ij}\), \(\Phi(\mathbf{p}_j)\) 高即可 |
| **直塞** | \(v_0\in[12,22]\), \(\theta_{elev}<0.08\), 目标在 \(\nabla\Phi\) 前方 | \(-\mathrm{OffsideRisk}\), \(\mathcal{N}(\mathbf{p}_j)>0\) |
| **长传转移** | \(v_0\in[18,28]\), \(\theta_{elev}\in[0.1,0.25]\), 跨半场 | 弱侧 \(\Phi\) 积分最大点 |
| **传中** | \(v_0\in[14,24]\), \(\theta_{elev}\in[0.3,0.6]\), 目标 \(Z_{box}\) | \(Z_{half}\) 起球点 |
| **外脚背/弧线传** | \(\omega_b\) 高，\(\hat n\) 水平 | 同香蕉，目标 \(\mathbf{p}_j\) |

**成功率**（连续）:

\[
q_{ij}(\mathbf{a}) = \sigma\bigl(b_0 + b_1 \mathcal{L}_{ij} + b_2 \Phi(\mathbf{p}_j) - b_3 \|\mathbf{p}_j-\mathbf{p}_{pred}(t+\Delta t)\| + b_4 \theta_i^{pass}\bigr)
\]

### 6.2 射门谱系

| 名称 | 参数区域 | 额外效应 |
|------|----------|----------|
| **推射** | \(v_0\) 中，\(\omega\approx 0\), \(\theta_{elev}\) 低 | 稳定，xG 由 \(\chi\) 主导 |
| **搓射/弧线（香蕉）** | \(\omega\) 高，侧旋 | 门将站位偏移误差 \(\epsilon_{GK}\) |
| **落叶球** | \(\omega\approx 0\), \(c_k\) 大 | 末端 \(\Delta v_z\) 随机 |
| **远射** | \(v_0\) 高，距离远 | \(\chi\) 基线低，仅明星 \(\theta^{shot}\) 足够时 \(u_{shot}\) 高 |
| **头球攻门** | contact=header | §7 空中对抗 |
| **凌空/半凌空** | 接触时 \(h_{air}>0.3\) | 执行误差 \(\uparrow\) |

### 6.3 带球（dribble）

\[
\dot{\mathbf{p}}_i = \mathbf{v}_i^{carry},\quad
\mathbf{v}_i^{carry} = v_{carry} \cdot \frac{\nabla \Phi}{\|\nabla \Phi\|+\varepsilon} - \beta \nabla \mathrm{Press}
\]

丢球概率 \(\lambda_{lose} = \exp(\text{Press}(\mathbf{p}_i) - \theta_i^{dribble})\) → Hawkes 抢断事件。

### 6.4 动作选择总效用（每 tick 持球者）

\[
u_k \in \{u_{pass}(\mathbf{a}_k),\; u_{shot}(\mathbf{a}_k),\; u_{dribble},\; u_{hold}\}
\]

\[
\mathbb{P}(k) = \mathrm{softmax}(u_k / \tau_{dec,i})
\]

---

## 7. 空中对抗与二点球

### 7.1 空中密度场

\[
\mathcal{A}(\mathbf{r}, h) = \sum_i \kappa_i^{jump} \cdot K_\sigma(\mathbf{r},\mathbf{p}_i) \cdot \sigma(h - h_{head,i})
\]

### 7.2 争顶结果

\[
P_{win,i} = \sigma\bigl(a_0 + a_1 \theta_i^{aerial} + a_2 \kappa_i^{jump} - a_3 \max_{j\in opp} \theta_j^{aerial}\bigr)
\]

赢家重定向 \(\mathbf{v}_b\)；输家 \(\dot m_i\) 负向脉冲。

### 7.3 二点球

球落地后 \(\rho\) 场在落点突变 → \(\Phi\) 尖峰 → 周围球员 \(\nabla V_i\) 指向落点（连续「二点争抢」）。

---

## 8. 单 tick 完整计算管线

**输入**: `MatchState` at \(t\)  
**步长**: \(\Delta t\)（建议 0.5–3.0 秒，可自适应）

| 步骤 | 模块 | 输出 |
|------|------|------|
| 1 | `SpatialFieldEngine.step` | \(\rho,\ Press,\ A\) |
| 2 | `AffectiveSpatialCoupling.step` | \(\mathbf{e}_i,\ e_{coach},\ e_{ref},\ \psi\) 及调制系数 |
| 3 | `SpatialIntelligenceEngine.step` | \(\Phi,\ \mathcal{N},\ \mathcal{L}_{ij},\ W_{trap}\) |
| 4 | `KinematicPositionLayer.step` | \(\mathbf{p}_i,\ \mathbf{v}_i,\ \theta_i,\ P\) |
| 5 | 持球者选择动作 \(\mathbf{a}\) | softmax over candidates |
| 6 | `BallManifoldEngine.integrate` | \(\mathbf{B}(t+\Delta t)\) |
| 7 | 接触检测 | pass complete / tackle / foul / shot / goal |
| 8 | `StaminaMoraleODE.step` | \(s_i, m_i, S_i\) |
| 9 | `EventRecorder` | 日志 + xG 增量 |
| 10 | 每 15' | `MesoAggregator` → micro-appraisal 包 |

**自适应 \(\Delta t\)**: 球速高或事件密集时 \(\Delta t \to 0.5s\)，死球时 \(\to 3s\).

---

## 9. 与 GFS 宏观层的双向接口

### 9.1 赛前：Macro → Micro

| GFS 字段 | 微观初始化 |
|----------|------------|
| `z_state.morale` | 全队 \(m_i\) 先验均值 |
| `emotion_profile` | \(\mathbf{z}_i^{emo}\) 初始 logits |
| `tactical_controls` | \(\mathbf{u}_{coach}\) |
| `referee_trust`, `referee_grievance` | \(\lambda_{foul}\) 偏置 |
| `fatigue`, `injury_load` | \(s_i\) 初值 |
| `fusion volatility` | 轨迹噪声 \(\boldsymbol{\xi}\) 幅度 |

### 9.2 赛中：Micro → Macro（每 15' 或半场）

生成 `MicroAppraisalPacket`:

```python
@dataclass
class MicroAppraisalPacket:
    team: str
    phase: str              # "0-15", "15-30", ...
    xg_for: float
    xg_against: float
    phi_integral: float
    foul_controversy: float # ∫ |∇referee_bias| dt
    icon_morale_shock: float # 球星 e_pride/e_shame 积分
    crowd_psi_mean: float
    tactical_drift: float   # ||u_coach(t) - u_coach(0)||
```

注入 `SocietyAgent._appraise_event({...})` —— **复用现有 6D appraisal**，不新造轮子。

### 9.3 赛后：Micro → `recursive_update`

| 微观积分 | 宏观事件键 |
|----------|------------|
| 红牌 | `norm_violation` ↑ |
| 点球争议 | `referee_controversy` ↑ |
| 球星致命失误 | `impact` ↓, Icon 通道 |
| 观众浪潮 | `social_chaos` hint |

### 9.4 与 LLM 边界（不变）

LLM **不选** \(\mathbf{a}\)；赛后可读 `MatchEventLog` 生成叙事：「第 67 分钟弧线直塞撕开肋部」——由 \(\omega_b,\ \mathbf{p}_{land}\) 反推话术，非幻觉比分。

---

## 10. 数据 schema 与能力维度扩展

### 10.1 球员能力扩展（建议 16 维）

```json
"abilities": {
  "tech": 0.0, "pass": 0.0, "curve": 0.0, "dribble": 0.0,
  "shot": 0.0, "power": 0.0, "knuckle": 0.0,
  "pace": 0.0, "stamina": 0.0, "aerial": 0.0, "heading": 0.0,
  "vision": 0.0, "spatial": 0.0, "press": 0.0, "mental": 0.0,
  "gk_reflex": 0.0, "gk_aerial": 0.0
}
```

| 能力 | 影响 |
|------|------|
| `curve` | \(c_m^{eff}\)，香蕉球/弧线传 |
| `knuckle` | \(c_k\)，落叶球稳定性 |
| `heading` | 头球争顶、攻门 |
| `pass` | \(q_{ij}\) |
| `spatial` | \(S_i\) 增益 |

### 10.2 教练 profile 扩展

```json
"coach": {
  "id": "coach_scaloni",
  "adapt_speed": 0.82,
  "press_bias": 0.65,
  "risk_appetite": 0.58,
  "emotional_volatility": 0.35,
  "youth_bias": 0.40
}
```

`emotional_volatility` → \(\|\mathbf{K}_{emo}\|\) 范数。

### 10.3 裁判 profile 扩展

```json
"referee": {
  "strictness_mean": 0.62,
  "card_propensity": 0.45,
  "foul_tolerance": 0.50,
  "home_bias_tendency": 0.08,
  "var_calming": 0.70
}
```

---

## 11. 实现模块与文件结构

```
src/match_engine/
  spatial_field.py
  spatial_intelligence.py
  affective_coupling.py      # NEW: §3 多主体情绪
  kinematic_position.py      # NEW: §4 位置/相位/朝向
  ball_manifold.py           # NEW: §5-6 球路 ODE
  aerial_duel.py             # NEW: §7 空中
  action_sampler.py          # 候选 â 生成 + softmax
  meso_aggregator.py         # 15' 聚合 → MicroAppraisalPacket
  integrator.py              # §8 主循环
  adapter.py
```

**依赖**: 仅 `numpy`；可选 `scipy.integrate` 做球路细分（无则 Euler）。

---

## 12. 分阶段交付（最细路线图）

### Phase 1a — 空间骨架（已有 v1.1）

- [ ] \(\Phi, \mathcal{L}_{ij}, \nabla V_i\)

### Phase 1b — 情绪耦合 v1 ✅

- [x] \(\mathbf{e}_i\) 4D ODE + \(\tau_{dec}, \sigma_{vision}\) 调制 — `affective_coupling.py`  
- [x] \(\psi \leftrightarrow m_i\) — `CrowdState` + 球员/教练通道  
- [x] \(e_{coach} \to \mathbf{u}_{coach}\) — `CoachAffectiveState.tactical_current`  
- [x] 实现说明 — [`PHASE1B_IMPLEMENTATION.md`](PHASE1B_IMPLEMENTATION.md)

### Phase 2a — 位置层 ✅

- [x] \(\mathbf{p}_i, \mathbf{v}_i, \theta_i, P\) — `kinematic_position.py`  
- [x] \(\rho\), Press, \(\Phi\), offside — `spatial_field.py`, `spatial_intelligence.py`  
- [x] 文档 — [`PHASE2A_2B_IMPLEMENTATION.md`](PHASE2A_2B_IMPLEMENTATION.md)

### Phase 2b — 球路 v1（地面）✅

- [x] 短传 / 直塞 / 长传 — `passing_engine.py`（连续效用 + 成功率）  
- [x] 与 1b 调制量 `tau_dec`, `vision_scale` 对接  
- [x] 统一入口 — `match_micro_runner.py`, `run_match_micro.py`

### Phase 3 — 球路 v2（旋转 + 射门）

- [x] Magnus → 香蕉球/弧线传 (`ball_physics.py`, `shot_engine` curved)  
- [x] Knuckle 噪声 → 落叶球 (`phys_knuckle`, `sample_shot_params`)  
- [x] 射门 xG 与门将覆盖 (`shot_engine._gk_save_prob`, `goal_generator`)

### Phase 3b — 弧线传球 + 空中

- [x] Magnus 弧线传球 — `ball_physics.integrate_pass_trajectory`, `passing_engine` 连续 ω  
- [x] 外脚背 — `build_pass_spin_axis` 倾斜 \(\hat{n}\)，`outside_foot_factor`  
- [x] 贴地低传 — `ground_pass_weight`, 草地滚动阻力  
- [x] 移动拦截 — `pass_intercept.evaluate_pass_intercept` vs \(p_{pred}(t+\Delta t)\)  
- [x] 防守 pursuit 朝落点 — `predict_player_xy` 分子步追击  
- [x] 二过一墙传 — `wall_pass.py` 双 ODE 连贯  
- [x] 头球争顶 + 传中 → header shot (`aerial_duel.py`, `action_engine` cross)

### Phase 4 — 裁判犯规连续化 + 完整 SAC 校准

- [ ] \(\lambda_{foul}\), \(\mathrm{Infringement}\)  
- [ ] 15' `MicroAppraisalPacket` → GFS

### Phase 5 — 性能 / 快速模式

- [ ] 粗网格 + 自适应 \(\Delta t\)

---

## 13. 校准、验证与可观测指标

### 13.1 空间智能

| 指标 | 合理范围（顶级赛） |
|------|-------------------|
| 场均 \(\int\Phi\,d\mathbf{r}\) | 校准到控球率 45–55% |
| `lane_quality_mean` | 0.55–0.75 |
| `halfspace_exposure` | 队间差异可辨 |

### 13.2 球路

| 指标 | 说明 |
|------|------|
| 传球成功率 vs 距离 | 应单调降 |
| 直塞 xG 链 | 直塞后 10s 内 xG↑ |
| 香蕉球进球率 | 低样本高方差，仅趋势 |

### 13.3 情绪

| 指标 | 说明 |
|------|------|
| 落后时 \(e_{fear}\) 均值 | > 领先时 |
| \(\psi\) 与主场犯规差 | 可检测主场哨 |

### 13.4 回归

- `MATCH_ENGINE=legacy` 比分分布不变  
- micro 模式 1000 场 λ 与 xG 相关性 \(\rho > 0.25\)

---

## 附录 A：符号总表

| 符号 | 含义 |
|------|------|
| \(\Phi\) | 自由空间场 |
| \(\mathcal{L}_{ij}\) | 传球走廊 |
| \(\mathbf{e}_i\) | 球员微情绪（softmax） |
| \(\psi\) | 观众场 |
| \(\mathbf{B}\) | 球状态 \((\mathbf{r},\mathbf{v},\omega,\hat n,h_{air})\) |
| \(\mathbf{a}\) | 动作参数向量 |
| \(P\) | 进攻相位 |
| \(\kappa_{curve}\) | 轨迹曲率（香蕉） |
| \(c_k,\eta\) | 落叶球抖动 |

---

## 附录 B：默认超参 YAML

```yaml
affective:
  R_e: 0.08
  d_q: 0.35
  d_f: 0.55
  d_a: 0.40
  v_f: 0.25
  k_crowd_player: 0.018
  k_crowd_ref: 0.012
  K_emotion_press: 0.15

kinematic:
  sigma_lane: 0.08
  kappa_P: 0.12
  gamma_off: 8.0
  delta_off: 0.02

ball:
  g: 9.81
  c_drag: 0.24
  c_magnus: 0.035
  c_knuckle: 0.18
  ou_eta: 0.4
  v0_short: [4, 9]
  v0_through: [14, 22]
  elev_cross: [0.35, 0.65]
  omega_banana: [15, 35]   # rad/s scale
  omega_knuckle: [0, 3]
  n_candidates: 12

aerial:
  a0_win: 0.0
  a1_aerial: 2.2
  jump_kappa0: 0.85

foul:
  lambda_f0: 0.06
  delta_s: 1.4
  delta_p: 0.9
```

---

## 结语：对你三个问题的直接回答

1. **全面吗？**  
   v1.1 **不全面**；v2.0 本文补齐情绪四主体、位置体态、球路连续体，才接近「非常真实」目标。

2. **空间智能要不要建球员/裁判/观众/教练情绪？**  
   **要。** 但必须作为 **AffectiveSpatialCoupling** 调制空间行为（视野、冲动、战术 PID、犯规强度），并与 GFS 宏观 `appraisal→emotion→coping` 通过 15' 聚合包衔接，避免两套心理系统。

3. **香蕉球 / 头球 / 落叶球 / 直塞 / 短传怎么考虑？**  
   **不建离散类型表**；建 **球状态 ODE + 动作参数 \(\mathbf{a}\)**，各类动作是参数空间中的区域 + 不同物理项（Magnus / knuckle / header contact）。语义名称仅用于日志与 LLM 叙事。

---

*维护: 实现时同步更新 [`SINGLE_MATCH_ENGINE_DESIGN.md`](SINGLE_MATCH_ENGINE_DESIGN.md) §7 交叉引用。*

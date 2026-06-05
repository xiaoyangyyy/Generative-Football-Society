# 实体动力学模型（球员 / 教练 / 球队）

原先在 `coach_loader`、`roster_builder` 里用 `np.clip`、分段 `if role == GK`、硬编码名帅加成都已改为 **连续动力学先验**，与微观层 `affective_coupling` 的 ODE 思路一致。

## 核心思想

| 层级 | 状态 | 生成方式 |
|------|------|----------|
| **教练** | \(z \in \mathbb{R}^6\) 心理 latent | 线性松弛平衡 \(z^\* = K^{-1} B u\) |
| **球员** | `condition` + `channel_affinities` + `abilities` | \(z^*=K_p^{-1}B_p u\)，\(\pi=\text{softmax}(P z)\)，\(a=\sum\pi_k\) 通道模板 |
| **球队** | \(x_T \in \mathbb{R}^5\) | 阵容能力均值 + 教练 mental 的 `tanh` 耦合 |
| **战术** | 21 维向量 | \(\sum_k \pi_k \cdot \text{preset}_k\)，\(\pi = \text{softmax}(P z)\) |

输入 \(u\) 均为可观测外生量（执教年限、FIFA 排名、市值对数、国脚场次、身高等），**不用阈值表**。

## 教练动力学

外生输入：

\[
u_{\text{coach}} = [\tau,\ \pi_{\text{press}},\ \rho_{\text{rep}},\ \sigma_{\text{style}}]^\top
\]

\[
\tau = \tanh(\text{tenure}/6),\quad \pi_{\text{press}} = \tanh\frac{25-\text{rank}}{8}
\]

平衡态（对应 \(\dot z = -Kz + Bu = 0\)）：

\[
z^\* = K^{-1} B u,\quad \text{mental}_i = \sigma(z^\*_i)
\]

实现：`entity_dynamics.coach_mental_equilibrium`

战术风格为 **softmax 亲和度** `preset_affinities`，再混合为连续 21 维向量（`blend_preset_from_affinities`），而非单一硬标签。

## 球员（与教练同规格）

| 字段 | 教练类比 | 说明 |
|------|----------|------|
| `condition` | `mental` | \(z^*=K_p^{-1}B_p u\)，6 维读出 |
| `channel_affinities` | `preset_affinities` | 8 通道 softmax |
| `primary_channel` | `preferred_preset` | 主通道名 |
| `dynamics_input` | `dynamics_input` | \(u_0..u_4\) |
| `role_embedding` | — | 8 维 Fourier |
| `abilities` | 战术混合向量 | \(\sum_k \pi_k \cdot\) 通道模板 |

实现：`build_player_dynamics_payload` / `player_loader.PlayerProfile` — Schema：`data/schemas/player.schema.json`

## 球队向量

\[
x_T = [\text{attack},\ \text{defense},\ \text{press},\ \text{morale\_field},\ \text{institutional\_pressure}]^\top
\]

写入 `data/rosters/{Team}.json` 的 `team_dynamics` 字段。

## 宏观进球率（λ 动力学）

世界杯默认路径已改为 **λ̇ = f(x_T, x_opp, λ, λ_opp)**，不再用 `status/35` 查表定 λ。

**状态** \(x_T \in \mathbb{R}^5\)：`attack, defense, press, morale_field, institutional_pressure`（来自 `team_dynamics` 或 status 平滑映射）。

**速率方程**（`macro_goal_dynamics.lambda_dot`）：

\[
\dot\lambda = \underbrace{\text{softplus}(x^{self}_{atk}-x^{opp}_{def}) + \cdots}_{\text{drive}(x_T,x_{opp})}
- \underbrace{\kappa\lambda + \cdots}_{\text{decay}}
- \text{KO/stage drag}
\]

**xG** = \(\int_0^{90} \lambda(t)\,dt\)（欧拉积分 90 步）；**进球** ~ Poisson(xG)（仅观测噪声，非 λ 查表）。

实现：`src/memory_engine/macro_goal_dynamics.py`  
世界杯：`tournament_2026.play_match` → `simulate_match_score_dynamics(a1, a2, eff_status, …)`  

回退旧 Poisson：`MATCH_LEGACY_POISSON=1`

## 与比赛内 ODE 的关系

- **JSON / 建队阶段**：上面为 **静态先验**（\(t=0\) 平衡态）。
- **比赛中**：`CoachAffectiveState.z_stress`、球员 `z_emo` 仍由 `affective_coupling.step` 积分演化。

先验 latent 通过 `latent_logit_from_unit_interval` 映射到 logit 初值，避免 `np.clip` 硬截断。

## 重新生成数据

```powershell
python scripts\fetch_football_baseline.py
```

教练 JSON 会多出 `preset_affinities`、`dynamics_input`；球员 JSON 会多出 `team_dynamics`、`dynamics_model`。

## 代码入口

`src/data_engine/entity_dynamics.py`

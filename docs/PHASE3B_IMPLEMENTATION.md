# Phase 3b — 弧线传球与传球物理增强

## 弧线传球（Magnus）

见 `integrate_pass_trajectory` + `desired_pass_omega`：连续 \(\omega_b\)，无「香蕉传」标签。

## 外脚背（旋转轴倾斜）

| 项 | 实现 |
|----|------|
| 连续强度 | `outside_foot_factor(start, target, curve, ω*)` → \([0,1]\) |
| 旋转轴 | `build_pass_spin_axis(azim, sign, outside)`：\( \hat{n} \) 由竖直向传球法向倾斜 `pass_outside_tilt_max` |
| 执行误差 | 成功率含 \((outside - outside^*)^2\) 惩罚 |
| 效用 | 大角度 + 高 `curve` 时外脚背传略增效用 |

## 贴地低传（\(\theta_{elev}\approx 0\)）

| 项 | 实现 |
|----|------|
| 权重 | `ground_pass_weight(elev) = exp(-elev/σ)` |
| 阻力 | \(c_d^{eff} = c_d(1 + \lambda_{ground}\, w_{ground})\) |
| 草皮滚动 | \(z < z_{contact}\) 时水平加速度 \(-\mu_{roll} w_{ground}\,\mathbf{v}_{xy}\) |
| 初速 | 贴地传略增 \(v_0\)（`pass_ground_v0_boost`） |

## 移动拦截（\(p_{pred}(t+\Delta t)\)）

| 项 | 实现 |
|----|------|
| 预测 | `predict_player_xy(p, Δt) = p + v·Δt·scale` |
| 落点误差 | `pred_miss = ‖land - p_pred_recv‖` 进入 \(q_{ij}\) |
| 抢断 | 若 \(\|land - p_{opp}^{pred}\| < r_{int}\) 且更近于接球人 → `TACKLE_WON` + 球权 |
| 风险 | `intercept_risk = σ(b_0 + b_{pace}\theta^{pace} - b_{dist} d_{opp} + …)` |

模块：`pass_intercept.py`

## 统计字段

`outside_foot_passes_*`, `ground_passes_*`, `pass_intercepts_*`

## 防守人主动跑位（pursuit）

- `predict_player_xy(..., pursuit_target, pursuit_weight)`：惯性 + 朝落点匀速逼近（分子步积分）
- `pursuit_weight_defender`：压迫 / 线路 / 距落点距离 → 连续权重
- 接球人亦朝落点轻度移动（`pass_recv_pursuit_weight`）

## 二过一 / 墙式传球

- `wall_pass.py`：`find_wall_partner` → 第一脚 ODE → `_execute_wall_return` 第二脚
- 候选 `kind="wall"`，高压 + 近距离队友效用更高
- 成功：KEY_PASS + ASSIST，微量 μxG；统计 `wall_passes_*` / `wall_combos_*`

## 测试

```bash
python -m unittest tests.test_match_phase3b -v
```

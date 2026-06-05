# Phase 2a + 2b 实现说明

**状态**: 已实现（与 Phase 1b 统一在 `match_micro_runner`）

---

## Phase 2a — 位置与空间场

| 模块 | 文件 | 内容 |
|------|------|------|
| 阵型锚点 | `formation.py` | 4-3-3 攻防锚点、通道、教练 line/width 调制 |
| 控球扩散 | `spatial_field.py` | \(\rho_{home},\rho_{away}\), Press 网格扩散 |
| 空间智能 | `spatial_intelligence.py` | \(\Phi\) 场、传球走廊 \(\mathcal{L}_{ij}\)、越位风险、人数优势 |
| 运动学 | `kinematic_position.py` | \(\mathbf{p}_i,\mathbf{v}_i,\theta_i\)、进攻相位 \(P\)、无球跑位 |

球员状态扩展：`position`, `velocity`, `orientation`, `abilities`（pass/vision/spatial/pace）。

---

## Phase 2b — 地面传球

| 模块 | `passing_engine.py` |
|------|---------------------|
| 短传 | 距离近、目标为队友当前位置 |
| 直塞 | 目标点前移 + vision/pace 加权 |
| 长传转移 | 远距离边路球员、距离惩罚 |

选择：`softmax(u/τ)`，\(\tau\) 来自 Phase 1b `tau_dec`。  
成功率：\(\sigma(b_0 + b_1 \mathcal{L} + skill - press - dist)\)，失败 → 抢断转换。

---

## 统一仿真栈（每 tick）

1. `SpatialFieldEngine.step`  
2. `SpatialIntelligenceEngine.step`  
3. `AffectiveSpatialCoupling.step`  
4. `KinematicPositionLayer.step`  
5. `PassingEngine.step`  
6. 同步球位置 + 事件表进球/射门  

---

## 运行

```bash
# 完整微观仿真（推荐）
python run_match_micro.py --home Brazil --away Argentina --seed 42
python run_match_micro.py --fast

# 世界杯单场接入（泊松比分不变，输出 [MICRO] 行）
set MATCH_MICRO=1
python run_world_cup_2026_tactical.py --quick --no-interactive
```

仅 Phase 1b（无空间/传球）：`MATCH_AFFECTIVE=1`（不含 `MATCH_MICRO`）。

---

## 输出指标 (`MicroMatchSummary`)

- `possession_home` — 控球 tick 比例  
- `passes_*` / `pass_completion_*` — 传球次数与成功率  
- `through_balls_*` — 直塞尝试次数  
- `phi_integral_*` — 平均自由空间场  
- `micro_xg_*` — 射门事件累积 xG 代理  

---

## 测试

```bash
python -m unittest tests.test_match_micro_phase2 tests.test_affective_phase1b -v
```

---

## 后续 Phase 3

- 旋转球（Magnus）、射门轨迹、门将覆盖  
- 头球与传中争顶  
- `simulate_match_score` 的 λ 接入 `micro_xg` 积分  

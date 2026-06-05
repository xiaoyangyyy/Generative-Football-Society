# Phase 1b 实现说明 — 情绪空间耦合 (AffectiveSpatialCoupling)

**状态**: 已实现  
**包路径**: `src/match_engine/`

---

## 已实现内容

| 模块 | 文件 | 功能 |
|------|------|------|
| 配置 | `config.py` | 连续 ODE / 脉冲超参（无硬阈值规则） |
| 状态 | `state.py` | 球员/教练/裁判/观众/全场状态 |
| 微事件 | `micro_events.py` | 16 类事件 → 情绪 logit 脉冲 |
| 引擎 | `affective_coupling.py` | `AffectiveSpatialCoupling.step()` |
| 阵容 | `squad_factory.py` | 从 `SocietyAgent` 生成 11 人 + 队友耦合矩阵 |
| 宏观桥 | `macro_bridge.py` | 赛前注入 / 赛后写回战术与情绪 |
| 事件表 | `event_schedule.py` | 按比分/xG/裁判/戏剧度生成 90 分钟事件流 |
| 全场跑 | `match_affective_runner.py` | 180–540 tick 仿真 + 汇总 |
| Meso | `meso_aggregator.py` | 15' 窗口 `MicroAppraisalPacket` |

### 空间调制输出（每名球员）

- `tau_dec` — 决策温度（恐惧/愤怒/认知负荷 ↑）
- `vision_scale` / `lane_sampling_scale` — 视野与传球线采样
- `move_alpha` — 无球跑动强度（体能×空间认知×决心）
- `shot_utility_bias` — 射门冲动
- `foul_impulse` — 犯规倾向（连续，供 Phase 4 犯规模块使用）

### 四主体情绪

1. **球员** — \(z^{emo}\) 4 维 logit → softmax(pride, anger, fear, determination) + 队友扩散 + 观众  
2. **教练** — stress / trust / rage → 实时改写 `tactical_controls`  
3. **裁判** — calm + 动态 `strictness_effective` + `lambda_foul`  
4. **观众** — \(\psi\) 双向：进球/红牌/VAR → 浪潮；\(\psi\) → 球员士气与教练压力  

---

## 如何运行

### 独立演示（无需 LLM）

```bash
python run_affective_phase1b.py --home Brazil --away Argentina --seed 42
python run_affective_phase1b.py --fast   # dt=30s，更快
```

### 接入世界杯 `play_match`

```bash
set MATCH_AFFECTIVE=1
set GFS_SEED=42
python run_world_cup_2026_tactical.py --quick --no-interactive
```

日志示例：`[AFFECTIVE] ψ=... | coach stress ... | ref strict ...`

### 单元测试

```bash
python -m unittest tests.test_affective_phase1b -v
```

---

## 环境变量

| 变量 | 默认 | 含义 |
|------|------|------|
| `MATCH_AFFECTIVE` | off | `1`/`true` 启用 play_match 后处理 |
| `GFS_SEED` | 42 | 事件表与仿真 RNG |

---

## 与后续 Phase 的关系

- **Phase 2a** 位置层将读取 `PlayerModulators.move_alpha`, `vision_scale`  
- **Phase 2b** 传球层将读取 `tau_dec`, `lane_sampling_scale`  
- **Phase 4** 犯规模块将读取 `foul_impulse`, `lambda_foul`  

当前 **不改变泊松比分**；仅写回 `_agents` 的 `tactical_controls` 与 `emotion_profile`（blend=0.35）。

---

## 设计文档

- [`SPATIAL_INTELLIGENCE_FULL_SPEC.md`](SPATIAL_INTELLIGENCE_FULL_SPEC.md) §3  
- [`SINGLE_MATCH_ENGINE_DESIGN.md`](SINGLE_MATCH_ENGINE_DESIGN.md) §7.15  

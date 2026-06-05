# 主流战术体系（连续向量）

GFS 微观层使用 **21 维连续战术向量**，不用 `if tactic == 'gegenpress'`。每个命名风格是 `R^21` 中的预设区域；教练四旋钮 + 阵型 + 球队风格描述与之混合。

## 命名预设（`tactical_catalog.py`）

| 类别 | 预设 id |
|------|---------|
| 压迫 | `gegenpress`, `high_press`, `bielsa_man_marking`, `pressing_trap` |
| 防守块 | `low_block_counter`, `catenaccio`, `park_the_bus`, `mid_block` |
| 控球 | `tiki_taka`, `positional_play`, `possession_control`, `double_pivot_control` |
| 直接 | `direct_vertical`, `long_ball`, `route_one`, `counter_attack`, `transitional_vertical` |
| 边路 | `wing_play`, `crossing_target`, `wide_overload` |
| 特殊 | `false_nine`, `total_football`, `offside_trap_high_line`, `narrow_compact`, `balanced` |

## 连续维度

`pressing_intensity`, `risk_budget`, `line_height`, `rotation_aggressiveness`（与 GFS 兼容）  
扩展：`possession_orientation`, `verticality`, `width_play`, `tempo`, `counter_attack`, `high_press`, `low_block`, `wing_focus`, `cross_frequency`, `through_ball_bias`, `long_ball_bias`, `build_up_short`, `offside_trap`, `counterpress`, `man_oriented_press`, `target_man`, `overlap_fullbacks`, `compactness`

## 阵型锚点（`formation.py`）

支持：`433`, `442`, `4231`, `352`, `532`, `541`, `4222`, `343`, `4141`, `451`, `3421`, `523`  
从 `tactics_master.json` 的 `formation` 字符串自动解析。

## 微观挂钩

| 模块 | 效应 |
|------|------|
| `spatial_field` | 压迫场 × counterpress / man_press |
| `kinematic_position` | 宽度、套边、防线高度 |
| `passing_engine` | 短/直塞/长传效用偏置 |
| `action_engine` | 射门效用 × verticality |
| `tactical_engine` | 落后加压 / 领先收缩（连续漂移） |

## 测试

```bash
python -m unittest tests.test_tactical_system tests.test_match_phase3b -v
```

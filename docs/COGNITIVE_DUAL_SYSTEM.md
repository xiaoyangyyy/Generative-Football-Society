# 单场双系统：系统1 情绪动力学 + 系统2 事件触发 LLM

## 概述

| 系统 | 职责 | 频率 |
|------|------|------|
| **系统1** | `AffectiveSpatialCoupling`、传球/射门 ODE、裁判/观众/边裁连续状态 | 每 tick |
| **系统2** | `SimulationLLM` 结构化 JSON 计划写回系统1 | 高显著性事件 + 冷却 |

**硬约束**：系统2 **不得**输出比分、xG、逐脚传球结果；仅调制战术向量、情绪脉冲、裁判倾向、观众 ψ。

## LLM 接入（Qwen / OpenAI 兼容）

在项目根目录创建 `.env`（已在 `.gitignore`，勿提交）：

```env
DEEPSEEK_API_KEY=replace_with_rotated_key
BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-v4-flash
```

`src/simulation/llm_gateway.py` 通过 `environment_snapshot()` 读取这些变量；`SimulationLLM` 使用 `openai` SDK 的 `base_url` + `model`。

## 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `MATCH_MICRO` | 0 | 开启微观仿真（认知层依赖微观循环） |
| `MATCH_COGNITIVE` | 0 | 开启赛中认知/LLM |
| `MATCH_COGNITIVE_SYNC` | 0 | 1=同步等待每次 LLM（慢） |
| `MATCH_COGNITIVE_CACHE` | `data/cache/cognitive` | LLM 响应缓存目录 |
| `MATCH_COGNITIVE_S0` | 0.55 | salience sigmoid 中心 |
| `MATCH_COGNITIVE_COOLDOWN` | 180 | 同实体触发冷却（秒） |
| `MATCH_COGNITIVE_MAX_PER_TIER` | `6,4,4,2,3` | 教练,主裁,球员/队,边裁,观众 每场 cap |
| `MATCH_CROWD_LLM_NUMERIC` | 1 | 观众 LLM 是否调制 ψ |

## 实体 Tier

- **A 教练**：进球、失球、红牌、半场、xG 劣势 → `coach_in_match_plan`
- **A 主裁**：红牌、VAR、争议 → `referee_in_match_judgment`
- **B 球员**：进球者、被罚、换人当事人；每队每场 ≤ cap/2
- **C 边裁**：VAR 等；调制越位 strictness → 主裁争议
- **C 观众**：进球、ψ 尖峰；`crowd_collective_reaction`

## 代码布局

```
src/match_engine/cognitive/
  config.py          # 环境配置
  events.py          # CognitiveTriggerEvent, CognitivePlanRecord
  salience.py        # 连续 salience + should_fire
  bus.py             # MatchCognitiveBus
  entity_registry.py # 事件→多实体触发 + FACTS_LEDGER
  routing.py         # TierBudget
  schemas.py         # JSON 校验/裁剪
  apply.py           # 写回系统1
  executor.py        # LLM + 规则回退 + 缓存
  assistant_dynamics.py
```

集成点：`match_micro_runner.run_match_micro_simulation` 每 tick 调用 bus + executor。

## 输出

- `MicroMatchSummary.cognitive_triggers` / `cognitive_plans` / `cognitive_tier_usage`
- 持久化：`data/persistence/cognitive_log/{Home}_vs_{Away}_{stage}.json`
- 赛后：`SocietyAgent.ingest_micro_cognitive_memory`

## 单场脚本

```powershell
cd D:\path\to\GFS-sparse
$env:PYTHONPATH="."
$env:MATCH_MICRO="1"
$env:MATCH_COGNITIVE="1"
python scripts/run_single_match_cognitive.py --home Brazil --away Argentina --seed 42 --out reports/match_cognitive.json
```

无 API Key 时自动使用 **规则回退计划**（`executor._rule_fallback_plan`），仍可验证写回与日志。

## 与锦标赛

- `play_match` 在 `[MICRO]` 块打印 `[COGNITIVE]` 统计
- `facts_ledger` 含 `cognitive_events`
- `finalize_match_feedback` 写入 carryover + cognitive_log

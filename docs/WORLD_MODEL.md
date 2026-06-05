# Latent World Model (Spatial WM) v2

Learned transition model over fixed match observations for imagination-based **pass and shot** planning.

## Architecture

```
obs_t (307-d) ──encoder──► z_t (96-d)
(z_t, action_t) ──GRU / Transformer──► z_{t+1}
z_{t+1} ──decoder──► obs_{t+1}
         ├── head_pass  → P(complete / intercept)
         ├── head_shot  → P(goal threat)
         └── head_xg    → ΔxG (attacking)
```

Actions include **pass, shot, hold, cross, intercept, tackle** (18-d).

- **Observation**: downsampled ρ/Press/Φ grids + ball + 22 player xy/v + tactics.
- **Action**: pass / shot / hold / cross one-hot + target xy + pass kind + receiver slot.
- **Planning**: `pass_imagination_bonuses` adds WM score to `PassingEngine` softmax utilities.

## Environment flags

| Variable | Default | Meaning |
|----------|---------|---------|
| `MATCH_WORLD_MODEL` | `0` | Load checkpoint + enable WM runtime |
| `MATCH_WM_PLAN` | `1` | Use imagination for pass selection (needs checkpoint) |
| `MATCH_WM_RECORD` | `0` | Write `data/world_model/traces/*.jsonl` each tick with action |
| `MATCH_WM_CHECKPOINT` | `data/world_model/latent_wm.pt` | Model weights |
| `MATCH_WM_TRANSITION` | `gru` | `gru` or `transformer` |
| `MATCH_WM_SHOT_BLEND` | `0.40` | Shot branch imagination weight |
| `MATCH_WM_SNAPSHOT_IN_BALL_LOG` | `1` with BALL_LOG | Store obs in ball_log jsonl |
| `MATCH_WM_PLANNER_BLEND` | `0.45` | How much WM shifts pass utilities |
| `MATCH_WM_IMAGINATION_STEPS` | `3` | Latent rollout depth per candidate |

## Workflow

```bash
# One-shot before full LLM tournament (also in restart_full_run.ps1)
python scripts/ensure_world_model.py --collect-pairs 32 --epochs 45

# Manual steps:
python scripts/collect_world_model_traces.py --pairs 48
python scripts/backfill_wm_from_ball_log.py   # after matches with BALL_LOG=1
python scripts/train_world_model.py --epochs 50 --use-ball-log --transition gru

# Full World Cup (restart script sets MATCH_WORLD_MODEL=1)
.\scripts\restart_full_run.ps1
```

## vs handwritten SIE

| SIE (rules) | Latent WM |
|-------------|-----------|
| Φ, lane, offside analytic | Learned from traces |
| Current tick only | Rollout K steps in latent space |
| Always on if spatial enabled | Needs trained `latent_wm.pt` |

Both can run together: SIE heuristic + WM bonus on pass utilities.

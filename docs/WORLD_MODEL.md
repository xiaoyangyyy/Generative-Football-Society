# World Model v6

The world model learns action-conditioned match transitions for counterfactual
pass and shot planning. It is an optional advisory layer: the deterministic
match engine remains authoritative.

## Core design

```text
full observation + candidate action
              |
 CNN grid + player attention + context encoder
              |
         GRU/Transformer
              |
        residual state delta
              |
 next observation + bootstrap outcome ensemble
              |
 quality/OOD gate -> bounded planner advantage
```

The 307-dimensional observation contains spatial fields, ball state, score and
clock, all 22 player positions and velocities, tactics, and attack direction.
Actions contain type, target, pass/shot kind, receiver slot, pre-action
estimates, and a normalized transition horizon.

Training-only outcome slots contain the actual pass result, goal result, and
on-target result. These labels are extracted and zeroed before the action enters
the model, preventing outcome leakage.

## Important design properties

- Predicts a bounded residual over the current observation instead of rebuilding
  the whole state from one latent vector.
- Uses real pass completion and goal outcomes, not pre-action probability or xG
  as fake labels.
- Weights ball, score, player, and tactical features by semantic importance so
  sparse grids cannot dominate the loss.
- Uses one-step counterfactual planning. Repeating one action for several latent
  steps is no longer the default.
- Stores holdout metrics and a `planner_quality` score in every checkpoint.
- Scales planner bonuses by checkpoint quality and observation coverage.
- Loads v2/v3 checkpoints for replay compatibility, but gives them zero planning
  authority under the default quality gate.
- Splits train and validation data by complete match rather than adjacent rows.
- Excludes aggregate backfill traces when raw ball logs are loaded, preventing
  duplicate training examples.
- Uses independent pass and shot quality gates; a weak branch contributes zero.
- Uses a legal hold action as the counterfactual baseline.
- Uses ensemble disagreement and observation coverage to reduce OOD influence.
- Territorial progress is exposed only as `progress_delta`; the misleading
  historical xG alias has been removed.

## Environment flags

| Variable | Default | Meaning |
|---|---:|---|
| `MATCH_WORLD_MODEL` | `0` | Load the world-model runtime |
| `MATCH_WM_PLAN` | `1` | Enable quality-gated planning |
| `MATCH_WM_RECORD` | `0` | Record full-state transitions |
| `MATCH_WM_CHECKPOINT` | `data/world_model/latent_wm.pt` | Checkpoint path |
| `MATCH_WM_TRANSITION` | `gru` | `gru` or `transformer` |
| `MATCH_WM_PLANNER_BLEND` | `0.30` | Maximum pass advantage blend |
| `MATCH_WM_SHOT_BLEND` | `0.25` | Maximum shot advantage blend |
| `MATCH_WM_IMAGINATION_STEPS` | `1` | Counterfactual rollout depth |
| `MATCH_WM_MIN_QUALITY` | `0.15` | Minimum branch confidence after coverage discount |

## Train and validate

```bash
python scripts/ensure_world_model.py --force --collect-pairs 48 --epochs 50
python scripts/validate_world_model.py
```

Manual workflow:

```bash
python scripts/collect_world_model_traces.py --pairs 48
python scripts/backfill_wm_from_ball_log.py
python scripts/train_world_model.py --epochs 50 --use-ball-log --transition gru
python scripts/validate_world_model.py
```

Only validated v6 checkpoints are allowed to influence planning. A missing,
legacy, low-quality, or sparse-input model contributes a zero bonus.

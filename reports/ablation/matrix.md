# Ablation matrix (continuous z, Δ vs M0)

Baseline M0: score=0.8936 loss=4.275 pass=True

## Subtractive (turn OFF vs M0)

| Ablation | score | Δloss | pass | top Δz |
|----------|-------|-------|------|--------|
| no_affective | 0.8336 | +2.64 | True | fouls_committed_per_team_match:+0.51, goals_per_team_match:+0.37, goals_to_micro_xg_ratio:+0.35 |
| no_spatial | 0.8413 | +27.29 | False | goals_to_micro_xg_ratio:-0.89, goals_per_team_match:-0.66, shots_per_team_match:+0.44 |
| no_blend | 0.9126 | -0.80 | True | goals_to_micro_xg_ratio:-0.50, pass_completion:+0.39, goals_per_team_match:-0.36 |
| no_discipline_tick | 0.8091 | +28.77 | False | goals_to_micro_xg_ratio:-0.54, goals_per_team_match:-0.45, fouls_committed_per_team_match:+0.27 |
| no_tactical_bias | 0.9092 | -0.66 | True | goals_to_micro_xg_ratio:-0.51, goals_per_team_match:-0.39, fouls_committed_per_team_match:+0.15 |

## Additive (turn ON vs M0)

| Layer | score | Δloss | pass | top Δz |
|-------|-------|-------|------|--------|
| M1 | 0.8947 | -0.05 | True | fouls_committed_per_team_match:-0.13, pass_completion:+0.04, goals_per_team_match:-0.04 |
| C1 | 0.9200 | -1.11 | True | goals_to_micro_xg_ratio:-0.39, goals_per_team_match:-0.30, shots_goals_correlation:+0.10 |

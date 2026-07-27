# V9.1 Cross-Provider Calibration Acceptance

## Corrected result

V9.1 point prediction and cross-provider interval calibration pass after correcting the interval support from the model's numerical floor (`0.025 s`) to the label support (`0 s`). Zero-delay events are valid same-frame actions and must be covered by intervals whose lower endpoint is zero.

| Held provider | Point MAE | Baseline MAE | 50% coverage | 90% coverage |
|---|---:|---:|---:|---:|
| SkillCorner | 0.9098 s | 0.9214 s | 46.56% | 91.57% |
| StatsBomb 360 | 0.8987 s | 0.9138 s | 57.11% | 92.13% |

Integrated Brier scores are 0.1613 and 0.1561. Both strict provider holdout folds pass all point and coverage gates.

## Boundary

V9.1 is evaluated evidence, not the active production release. Production remains v7 with v6 rollback. V9.2 supersedes this calibration experiment with a third explicit-clock provider and worst-source multi-provider calibration.
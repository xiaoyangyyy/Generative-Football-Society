# V8.9 Strict Provider LODO Acceptance

Date: 2026-07-26

## Decision

V8.9 adds StatsBomb 360 as a second provider with explicit reception identity and timing, and passes bidirectional strict leave-one-provider-out evaluation. It is accepted as an evaluated research candidate only. V7 remains deployed and v6 remains the rollback release.

## Data Contract

StatsBomb Pass events are joined to their explicit `Ball Receipt*` events through UUIDs in `related_events`. The recipient identity must agree on both events. Only successful receptions with a valid 360 freeze-frame are included. Following pass, shot, dispossession, miscontrol, failed dribble, foul-won, clearance, and offside marks use their own event timestamps. Chains without a qualifying mark within five seconds are right-censored.

The resulting dataset contains 5,451 SkillCorner receptions from 10 matches and 306,584 StatsBomb 360 receptions from 417 valid matches. One malformed upstream 360 file and matches without qualifying chains are excluded rather than repaired silently.

Both providers use the same 22-dimensional static geometry contract. Velocity and acceleration are unavailable in StatsBomb 360 and therefore excluded from both providers. Absolute context is represented by mirror-invariant goal/side-line proximity and entity distances, preventing coordinate-direction shortcuts.

## Strict LODO Results

Thresholds and checkpoints are selected only on matches from the training provider. No held-provider examples are used for training, model selection, or threshold calibration.

| Held provider | Train provider | Time MAE | Quantized baseline | Continue BA | Pass/shot BA | 2s Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SkillCorner | StatsBomb 360 | 0.478 s | 0.506 s | 0.572 | 0.557 | 0.139 |
| StatsBomb 360 | SkillCorner | 0.936 s | 0.965 s | 0.617 | 0.544 | 0.225 |

The timing comparator is selected on the training provider and restricted to the same 0.5-second output grid as the survival model. Continuous train-median baselines remain in the report as diagnostics and are not hidden.

## Structured Shot Head

A free shared-latent shot head failed cross-provider evaluation because rare shot labels inherited provider-specific timing representations. V8.9 uses a monotonic receiver-to-nearest-goal risk score and calibrates only its threshold on the source-provider development set. This deliberately lower-capacity head transfers better and remains interpretable.

Canonical evidence:

- `data/frame_world/v89_temporal_providers/manifest.json`
- `reports/acceptance/temporal_mark_v89_lopo.json`
- `data/frame_world/temporal_mark_v89_lopo_skillcorner.pt`
- `data/frame_world/temporal_mark_v89_lopo_statsbomb360.pt`

Source provenance follows the StatsBomb Open Data repository and specification: https://github.com/statsbomb/open-data

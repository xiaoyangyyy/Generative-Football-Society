# V9.0 Continuous-Time Survival Acceptance

Date: 2026-07-26

## Decision

V9.0 replaces the 0.5-second discrete hazard with a three-component log-normal mixture survival model. It passes the previously frozen requirement: both strict provider holdout folds beat the continuous training-provider median baseline. V9.0 is accepted as an evaluated research candidate, not deployed. V7 remains active and v6 remains rollback.

## Model Contract

The model predicts mixture weight, log-time location, and positive scale for three continuous components. Observed events use mixture log-density likelihood. Events without a qualifying mark inside five seconds use mixture survival probability as a right-censored likelihood. The scheduled time is the numerical median of the mixture CDF on `[0.025, 5.0]` seconds.

The continue head shares the geometry encoder. The rare pass/shot mark remains the mirror-invariant monotonic receiver-to-goal risk score established in v8.9. Training data, static geometry, successful-reception filtering, and provider split are unchanged from the frozen v8.9 contract.

## Strict LODO Results

| Held provider | Train provider | Continuous MAE | Continuous baseline | Gain | Continue BA | Pass/shot BA | 2s Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SkillCorner | StatsBomb 360 | 0.412 s | 0.463 s | 0.051 s | 0.579 | 0.681 | 0.144 |
| StatsBomb 360 | SkillCorner | 0.912 s | 0.928 s | 0.015 s | 0.582 | 0.757 | 0.233 |

Checkpoint selection and thresholds use only the training provider. The held provider is untouched until final evaluation.

## Router Acceptance

The router preserves the continuous `delay_seconds`. Quantization occurs only at the fixed-step execution boundary with `ceil(delay_seconds / step_seconds)`.

- Held-out feature rows: 5,451.
- Mean prediction change after deterministic feature permutation: 0.469 s.
- Predictions outside the old 0.5-second grid: 99.74%.
- Real SkillCorner closed-loop contexts: 64.
- Finite and bounded rollouts: 64/64.
- Execution-step consistency: 64/64.
- Receiver identity deletion blocked the learned route: 64/64.

## Boundary

The StatsBomb-held time gain is positive but small at 0.015 s. This is sufficient for the frozen v9.0 research gate but not for production promotion. Distributional calibration and uncertainty coverage across providers should be tested before considering sealing.

Canonical evidence:

- `reports/acceptance/continuous_time_v90_lopo.json`
- `reports/acceptance/continuous_router_v90.json`
- `data/frame_world/continuous_router_v90.json`
- `data/frame_world/continuous_time_v90_lopo_skillcorner.pt`
- `data/frame_world/continuous_time_v90_lopo_statsbomb360.pt`

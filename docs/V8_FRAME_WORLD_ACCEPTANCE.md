# V8 Frame World Model Acceptance

Date: 2026-07-25

## Decision

`frame_world:8.0.0-candidate` is accepted as an evaluated research candidate. It is not sealed or deployed, and the active v7 release and v6 rollback pointer remain unchanged.

## Scope

- Provider-neutral 10 Hz frame contract for 16 slots per team plus ball.
- 1,106,890 frames from 19 matches across Metrica, SkillCorner, and Sportec.
- Explicit visibility masks; hidden or extrapolated coordinates are never used as labels.
- Five-frame temporal encoder, entity graph transformer, residual constant-velocity prediction, possession auxiliary head, and heteroscedastic uncertainty.
- Horizons: 0.1, 0.5, 1.0, and 2.0 seconds.

## Held-Out Results

| Metric | Model | Constant velocity |
| --- | ---: | ---: |
| Overall ADE | 0.634 m | 0.876 m |
| Ball ADE | 2.838 m | 3.485 m |
| 0.1 s ADE | 0.062 m | 0.066 m |
| 0.5 s ADE | 0.218 m | 0.301 m |
| 1.0 s ADE | 0.581 m | 0.818 m |
| 2.0 s ADE | 1.705 m | 2.360 m |

Every held-out provider beat its own constant-velocity baseline: Metrica 0.680 vs 1.011 m, SkillCorner 0.695 vs 1.009 m, and Sportec 0.568 vs 0.698 m.

Uncertainty temperature was fitted on dev only. Held-out radial coverage is 57.2% at the nominal 50% region and 83.8% at the nominal 90% region. Both configured acceptance bands pass without test-set calibration.

## Correctness Controls

- Sportec is timestamp-resampled to the nearest 0.1-second grid, rather than frame-number decimation.
- Metrica and SkillCorner are normalized to the same pitch coordinates and sample rate.
- Velocity labels require consecutive visibility and all masked tensor storage remains finite.
- Train, dev, and test are match-disjoint. Test contains one held-out match from every provider.
- Data files are SHA-256 pinned in `data/frame_world/v8/manifest.json`.
- The model artifact is registered as `evaluated`, not deployed.

## Remaining Boundary

This is a direct multi-horizon predictor, not yet an autoregressive long rollout model. It is suitable for short-horizon simulation proposals and counterfactual scoring. Deployment should wait for closed-loop rollout stability, event-conditioned interventions, and strict leave-one-provider-out retraining rather than only per-provider match holdout.

Canonical evidence: `reports/acceptance/frame_world_v8.json`.

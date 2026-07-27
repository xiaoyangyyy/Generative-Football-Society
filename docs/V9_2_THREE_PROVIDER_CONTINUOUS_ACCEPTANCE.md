# V9.2 Three-Provider Continuous-Time Acceptance

## Data contract

V9.2 adds Metrica as the third explicit-clock provider. A reception interval begins at PASS `End Time`; receiver identity is PASS `To`; the next action is the first event within five seconds whose actor `From` is that receiver. Sportec was rejected for this role because the available PASS events do not contain end timestamps.

Provider quality gates pass for 10 SkillCorner matches, 417 StatsBomb 360 matches, and 2 Metrica matches. Observed event counts are 5,208, 294,804, and 1,707. Median delays are 1.10, 1.131, and 1.12 seconds.

## Strict LOPO point results

Training uses equal provider mass per epoch and provider-stratified development splits.

| Held provider | Model MAE | Provider-equal baseline | Result |
|---|---:|---:|---|
| Metrica | 0.9096 s | 0.9617 s | Pass |
| SkillCorner | 0.9171 s | 0.9223 s | Pass |
| StatsBomb 360 | 0.8855 s | 0.9180 s | Pass |

Continuation, pass/shot, Brier, finite-output, and provider-isolation gates pass in every fold.

## Multi-source calibration

Each held provider is scored only after calibration on the other two providers. The interval radius is the maximum source-provider split-conformal absolute-residual radius.

| Held provider | 50% coverage | 90% coverage | 90% width | Integrated Brier |
|---|---:|---:|---:|---:|
| Metrica | 59.40% | 90.92% | 3.219 s | 0.1476 |
| SkillCorner | 47.20% | 91.78% | 3.271 s | 0.1654 |
| StatsBomb 360 | 59.77% | 91.62% | 3.270 s | 0.1546 |

The statistical interval support is `[0, 5]` seconds. The model's `0.025 s` floor is only for stable log-likelihood evaluation.

## Routing and release

`ActionTransitionRouter.plan_next_mark_interval()` is opt-in and returns `None` unless an accepted provider radius is supplied. V9.2 is frozen and registered as evaluated only. Active production remains v7 and rollback remains v6.

## Verification

```bash
python scripts/audit_temporal_providers_v92.py
python scripts/evaluate_continuous_calibration_v92.py
python scripts/verify_continuous_v92.py
python -m pytest -q
```
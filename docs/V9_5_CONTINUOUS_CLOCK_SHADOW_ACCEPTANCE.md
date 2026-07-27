# V9.5 Continuous Clock Shadow Acceptance

## Decision

V9.5 passes the matched-seed, mirrored-fixture shadow gate. It is frozen as a shadow candidate, not deployed. Production remains v7.0.0 and rollback remains v6.0.0.

## Runtime integration

`ContinuousMicroEventClock` is a default-off adapter between micro-match state and the verified v9.3 calibrated temporal router. It maps on-pitch identities into the 33-slot frame contract, selects the nearest opposing defender, schedules every resolved reception including turnovers, applies the predicted pass/shot mark once, and treats terminal as a five-second no-follow-up horizon. Corrupt model hashes and unsupported providers fail closed.

## Shadow protocol

Eight 120-second matches use four home/away mirrored fixture pairs and fixed seeds. Each fixture is run once with the deployed tick clock and once with the v9.3 clock. Real-rate targets are scaled from the project's StatsBomb calibration anchors: 535.27 passes and 11.67 shots per team per 90 minutes; total xG uses 2.6 per match.

All nine gates pass. Mean absolute target error changes:

| Metric | v7 clock | v9.3 clock |
|---|---:|---:|
| Passes | 212.085 | 19.335 |
| Shots | 3.731 | 2.361 |
| Total xG | 0.895 | 0.615 |

Median pass-density reduction is 81.5%. Completion drift is 0.078. The model produced 345 plans and gated 1,568 ticks. Treatment is deterministic under repeat execution and mirrored possession bias remains within the 0.08 gate.

## Verification

```bash
python scripts/evaluate_continuous_clock_shadow_v95.py --seconds 120 --pairs 8
python scripts/freeze_v95_release.py
python scripts/verify_continuous_v95.py
python -m pytest -q
```

The next decision is explicit deployment promotion or a longer full-match shadow. No deployment pointer is mutated by this release.

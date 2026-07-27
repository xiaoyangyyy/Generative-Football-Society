# V9.7 Dual-Clock Reception Queue Acceptance

## Decision

V9.7 passes production-cadence full-match shadow as an evaluated, default-off research candidate. It is not deployed. Production remains v7.0.0 and rollback remains v6.0.0.

## Architecture

The 5.5-second macro clock remains responsible for spatial fields, tactics, affect, schedules, and at most one high-level action per tick. A deterministic sub-tick queue owns only completed single-pass reception responses.

After a pass, the queue:

1. Captures the pre-pass 33-slot frame and stable identities.
2. Uses the verified v9.3 temporal model for the due time.
3. Uses the v8.5 pass-triplet model for ball/receiver/defender transition evidence.
4. Subtracts the constant-velocity baseline and merges only the learned residual.
5. Applies a 0.5 residual blend atomically before ActionEngine returns.
6. Cancels stale identity chains and consumes no random numbers.

This corrects the v9.6 failure: temporal events no longer gate global actions and no longer duplicate baseline kinematics.

## Full-Match Shadow

Four complete 90-minute matches use two mirrored fixture pairs and fixed paired seeds at the production 5.5-second cadence.

- Scheduled events: 2,941
- Applied events: 2,941
- Cancelled events: 0
- RNG draws: 0
- Mean learned displacement applied: 0.0278 m
- Signed mirrored possession drift: 0.0227
- Deterministic repeat: passed

Mean absolute real-target error remains non-inferior:

| Metric | Baseline | V9.7 |
|---|---:|---:|
| Passes | 101.290 | 102.540 |
| Shots | 14.590 | 14.840 |
| Total xG | 0.887 | 0.906 |

All eleven gates pass without changing thresholds after the final residual-contract correction.

## Release Boundary

V9.7 source evidence is copied into immutable `data/releases/snapshots/v9.7` artifacts. Live source paths are not used as frozen artifacts. Deployment remains an explicit future decision.

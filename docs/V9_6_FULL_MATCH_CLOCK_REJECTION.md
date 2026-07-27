# V9.6 Full-Match Continuous Clock Rejection

## Decision

V9.6 rejects production promotion of the v9.5 continuous-clock integration. Production remains v7.0.0 and rollback remains v6.0.0. This is a successful negative acceptance result: the long-horizon protocol found a cadence confound that the 120-second shadow could not expose.

## Experiments

Four complete 90-minute matches use two home/away mirrored fixture pairs and fixed seeds. The first diagnostic used a 2-second tick to make 1-5 second temporal marks observable. It improved pass volume but reduced shots and xG. The production comparison then used the actual 5.5-second macro tick.

At production cadence, the temporal model generated 3,893 plans and gated zero ticks. Its 1-5 second reception horizon is shorter than one macro simulation step, so the current global action gate cannot represent the learned timing.

| Mean absolute real-target error | v7 clock | continuous clock |
|---|---:|---:|
| Passes | 101.290 | 97.290 |
| Shots | 14.590 | 16.840 |
| Total xG | 0.887 | 1.194 |

Pass completion drift remained 0.010 and signed mirrored possession drift remained 0.025. Determinism passed. The failure is therefore localized to clock resolution and action composition, not rollout instability.

## Supersession

V9.5 remains hash-verifiable historical evidence, but its short-window `promotion_ready` result is superseded for deployment decisions by this full-match production-cadence rejection. Short-window results compared against a 0.5-second baseline that was not the production action cadence.

## Required architecture

The next candidate must separate:

1. A 5.5-second macro clock for spatial, affective, tactical, and tournament state.
2. An independent sub-tick reception event queue for pass arrival, receiver/defender response, and the next marked action.
3. A hazard-based action policy calibrated per second rather than per tick.
4. Explicit reconciliation at macro boundaries, with no duplicate RNG draws or actions.

No threshold relaxation or delay multiplier is accepted as a substitute for this separation.

## Verification

```bash
python scripts/evaluate_continuous_clock_long_shadow_v96.py --seconds 5400 --dt 5.5 --pairs 4
python scripts/freeze_v96_release.py
python scripts/verify_continuous_v96.py
python -m pytest -q
```

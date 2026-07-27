# V8.4 Pressure-Onset Pair Transition Acceptance

Date: 2026-07-25

## Decision

V8.4 passes the SkillCorner match-held-out pressure-onset transition gates and is accepted as an evaluated research candidate. It is not deployed. V7 remains deployed and V6 remains the rollback release.

## Why V8.3 Failed

The 7,063 pressure events span 130,175 positive state frames, with a median duration of 1.7 seconds. Training on every positive frame primarily identifies an already-pressured state. Its held-out local transition error was 0.349 m versus a 0.331 m kinematic fallback, so the transition branch was rejected.

## Event Contract

V8.4 preserves all 7,063 event-level starts, including concurrent pressure events that would otherwise collapse into 4,975 merged rising edges. Of these, 6,375 events (90.3%) have visible, mapped actor and target geometry.

Each valid onset contains:

- pressure actor and possession target slots;
- actor-to-target distance;
- pair closing speed;
- metric relative direction;
- both players' current velocity and one-frame velocity change.

Unknown or invisible pairs are excluded from transition supervision. The model can only return bounded residuals for the two identified players. Disabled inference is exactly the constant-velocity fallback.

## Match-Held-Out Result

Eight matches train the model, one selects the epoch, and one is untouched test data.

| Test condition | Mean entity error |
| --- | ---: |
| Correct onset geometry | 0.319 m |
| Zero intervention | 0.341 m |
| Geometry shuffled within batch | 0.357 m |

Correct conditioning improves on the fallback by 6.7% and on shuffled geometry by 10.8%. The shuffled test keeps the same player pair, current coordinates, velocity, and baseline; only event geometry is reassigned. This is a materially stricter specificity test than shuffling absolute trajectories.

## Evidence Boundary

Pressure transition labels are strong only in SkillCorner, so provider LODO cannot identify cross-provider pressure generalization. The candidate is approved for SkillCorner-compatible inputs only. Sportec tackles remain weak proxies and Metrica pressure remains unavailable. Cross-provider deployment requires a second provider with strong, time-aligned pressure onset and actor/target identities.

Canonical evidence:

- `data/frame_world/v84_pressure/manifest.json`
- `reports/acceptance/frame_pressure_v84.json`
- `data/frame_world/frame_world_v84_pressure_pair.pt`

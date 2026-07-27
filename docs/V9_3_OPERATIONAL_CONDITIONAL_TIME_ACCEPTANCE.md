# V9.3 Operational Conditional-Time Acceptance

## What closed

V9.3 turns the v9.2 LOPO evidence into an operational candidate without weakening evidence isolation. A single model is trained with equal provider mass while one provider-stratified calibration partition remains untouched. The frozen calibration artifact is loaded with model SHA-256 verification and includes predicted-action Mondrian intervals with a global fallback for groups below 100 observations.

Operational calibration MAE passes its provider-equal anchor baseline on all three held calibration partitions: Metrica `0.8803 s`, SkillCorner `0.9545 s`, and StatsBomb 360 `0.9039 s`.

## Conditional LOPO

The calibration method is independently tested with strict three-way LOPO. Overall 90% coverage is `90.92%`, `91.55%`, and `91.19%` for held Metrica, SkillCorner, and StatsBomb 360. Every predicted pass, shot, and terminal group with at least 100 held observations remains within 10 percentage points of nominal coverage.

## Runtime

`load_calibrated_temporal_router()` verifies every model hash and atomically loads models, thresholds, and interval radii. `plan_next_mark_interval()` chooses the predicted-action radius and falls back to global calibration. `rollout_event_driven_interval()` supports lower, median, and upper timing interventions.

The event transition no longer restores stale pre-pass velocities. A non-terminal mark attaches the ball to the predicted actor and inherits actor velocity; a terminal mark stops the ball while preserving player motion.

## Real tracking rollout

Forty Metrica tracking contexts were evaluated across lower, median, and upper timing interventions. All trajectories are finite, pitch-bounded, deterministic, ordered, and nontrivially different. Maximum counterfactual spread is `76.67 m`, equal to `56.4%` of its per-sample physical reachable bound.

## Remaining external boundary

Metrica contributes only two public sample matches. This cannot be repaired honestly in code. Sportec remains excluded from explicit reception-time supervision because its available PASS end timestamps are absent. Broader promotion requires additional licensed explicit-clock matches.

## Release decision

V9.3 is frozen and registered as evaluated only. Active production remains v7 with v6 rollback.
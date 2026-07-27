# V8.6 Action Transition Router Acceptance

Date: 2026-07-25

## Decision

V8.6 passes unified routing, evidence fallback, physical safety, possession contract, and 1-5 second closed-loop counterfactual gates. It is an evaluated research candidate and is not deployed. V7 remains active and V6 remains the rollback release.

## Unified Contract

`ActionTransitionRouter` accepts a `FrameContext` and one or more `ActionRequest` values and returns a `TransitionResult` containing positions, velocities, route decisions, possessor identity, and possession team.

- Pass routes to the v8.5 ball/receiver/interceptor triplet model.
- Pressure routes to v8.4 only for a visible SkillCorner-compatible onset pair.
- Shot routes to v8.3 only with a validated direction and supported provider.
- Missing identities, invalid team roles, unsupported providers, and unavailable directions use exact kinematic fallback.

Simultaneous actions share one 0.5-second kinematic baseline. Local residuals are composed after routing, so two actions cannot advance simulation time twice. Ball ownership conflicts use the ball action route; compatible player residuals may compose.

## Physical And Identity Safety

The router enforces metric displacement limits after residual composition: 45 m/s for the ball and 12 m/s for players. Positions are clipped to the normalized pitch buffer. Pass actor/receiver roles must share a team and the interceptor must be an opponent; pressure actor and target must be opponents.

Possession is recomputed from visible players within 2 m of the predicted ball. Unknown possession is represented by `-1`, never by a fabricated player.

## Real-Checkpoint Rollout

The audit uses 112 real aligned event contexts: 96 passes and 16 pressure onsets, plus supported and unsupported shot route probes.

| Horizon | Mean action-deletion effect |
| --- | ---: |
| 1 s | 2.329 m |
| 2 s | 4.627 m |
| 3 s | 6.688 m |
| 5 s | 9.194 m |

- Every rollout is finite and bounded.
- Maximum observed speed is 45.000004 m/s, within float tolerance of the ball cap.
- Moving the action by 0.5 seconds changes the 5-second result by 1.933 m on average.
- Metrica shot routing is enabled; Sportec shot routing falls back exactly.
- Possessor and possession-team outputs satisfy the identity contract.

The increasing deletion effect is evidence of a persistent velocity intervention, not proof of calibrated 5-second trajectory accuracy. Future deployment still requires multi-action ground-truth rollout evaluation.

Canonical evidence:

- `data/frame_world/action_router_v86.json`
- `reports/acceptance/action_router_v86.json`

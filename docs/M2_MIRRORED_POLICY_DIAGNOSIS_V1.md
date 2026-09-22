# M2 mirrored-policy diagnosis V1

> This note diagnoses a completed, immutable study. It does not promote the
> candidate, change thresholds, retrain, or re-run the 360-match budget.

## Question and scope

The preregistered M2 study asked whether a sealed, outcome-aligned world-model
policy changes simulator-internal team-scoped behavior enough to clear every
frozen promotion gate. The design is six fixtures, 20 seeds, three arms
(M0 / M2_home / M2_away), 5400-second matches, and a 10,000-draw
fixture-stratified cluster bootstrap. The primary metric is controlled
micro-xG margin. This is not a real-football causal estimate.

Protocol next action after the complete result is
`retain_research_only_and_diagnose`.

## Frozen result

The complete budget finished on 2026-09-18 under the bound protocol, checkpoint
and code identities.

| Quantity | Result |
|---|---:|
| Runs | 360/360 |
| Frozen decision | `research_only_default_off` |
| Promotion supported | no |
| Controlled micro-xG | 0.005, 95% CI [-0.013, 0.023] |
| Attributable realized utility | -0.056, CI entirely negative |
| Expected counterfactual change rate | 0.000617, 95% CI [0.00049, 0.00077] |
| Influenced / opportunities | 117366 / 117366 |
| High-level counterfactual action changes | 65 |
| Pass-target changes | 201 |

Coverage and external-calibration noninferiority passed. The three mechanism
and primary-effect gates failed. All gates are conjunctive, so promotion is
impossible.

## What failed mechanically

The controller is authorization-rich and behavior-poor. An opportunity counts
as influenced whenever any authorized signal exists
(`src/match_engine/world_model/action_adoption.py`). Planner utility deltas are
audit-only; sampling happens once in `mix_direct_action_probabilities`, with
authority clipped at 0.35 and advantage scaled by 0.10
(`src/match_engine/action_engine.py`).

Under M2, hold is pinned to exact zero persistence, so most signals are
suppression rather than a positive alternative
(`src/match_engine/world_model/planner.py`). High-level pass is scored on a
synthetic shared encoding, not the executable receiver. Target ranking is a
second surface that fail-closes on tiny spread; its 201 changes do not feed
the high-level 0.02 CF-rate gate.

Home-arm mean recommended probability shift is about `1e-9`. Away-arm is about
`9e-5`. That is why 100% influence coexists with an expected CF rate 33 times
below the floor, and why attributable realized utility stays tightly negative.

## What to modify next (new protocol only)

Do not patch this bound study. A later candidate needs a new protocol and code
identity before training or sealed qualification.

1. Do not treat “signal present” as “behavior changed”; keep influenced and
   TV / expected CF separate for attribution.
2. Score high-level pass on executable candidates consistent with
   `pass_target_ranking`.
3. When suppressing pass versus zero persistence, boost a validated
   alternative instead of suppression-only.
4. Write an authority × blend contract that can reach the 0.02 CF floor.
   `planner_blend = 0.30` multiplied by sealed authority often yields an
   effective blend near 0.06.
5. Retrain and re-qualify only after that identity amendment.

## Authoritative artifacts

- `data/evaluation/m2_mirrored_policy_protocol_v1.json`
- `data/evaluation/m2_mirrored_policy_progress_v1.json`
- `data/evaluation/m2_candidate_eligibility_v1.json`
- `data/evaluation/m2_mirrored_policy_diagnosis_v1.json`
- `data/world_model/m2_outcome_aligned_candidate.pt`

The verified conclusion is:

> The sealed M2 policy attaches and opens its gates, but it barely changes
> high-level actions. This fixed study does not support promotion. Keep the
> layer research-only and default-off.

Any controller redesign belongs to a new preregistered study. The present
evidence must remain immutable.

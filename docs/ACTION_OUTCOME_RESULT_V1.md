# World-model action-policy full-match result V1

## Question and scope

This study asks whether the mechanism-confirmed M1 action policy improves the
simulator's external calibration loss over M0 while changing observable
full-match behavior. It is a simulator-internal, checkpoint- and
identity-bound comparison. It does not estimate a causal effect in real
football and cannot authorize deployment by itself.

The preregistered design used six fixed fixtures and five fixed sample indices,
giving 30 matched fixture-seed pairs and 60 full 90-minute simulations. M0 was
executed first and M1 second. The primary analysis was a 10,000-draw paired
bootstrap stratified by fixture with seed 260806. The minimum meaningful loss
improvement was frozen at 0.10. No interim effect analysis or optional stopping
was allowed.

## Sealed result

The complete budget ran on 2026-08-31 under the frozen protocol, checkpoint,
mechanism decision and critical-code identities.

| Quantity | Result |
|---|---:|
| M0 runs | 30/30 |
| M1 runs | 30/30 |
| Matched pairs | 30 |
| M1 minus M0 external calibration loss | -0.03189252164278855 |
| 95% paired bootstrap interval | [-26.061954645805613, 5.137429588662575] |
| Behavior-changing pairs | 30/30 (100%) |
| Frozen decision | `inconclusive_keep_research_only` |
| Promotion supported | no |

The action policy changed observable full-match behavior in every matched
pair, so the learned component is no longer merely a prediction-side
spectator. The primary interval is extremely wide, crosses zero and does not
place its upper bound below `-0.10`. The M1 candidate also failed the frozen
external-validity gate. Therefore the study does not show a reliable or
meaningful outcome improvement.

## Frozen promotion gates

| Gate | Result |
|---|---|
| Candidate passes all external-validity gates | fail |
| Upper 95% bound is below -0.10 | fail |
| At least 10% of pairs change behavior | pass |
| Execution identity verified | pass |

All gates were conjunctive. Two failures make promotion impossible regardless
of favorable-looking secondary metrics.

## Descriptive secondary differences

All values are M1 minus M0 paired means and are descriptive only.

| Metric | Difference |
|---|---:|
| Pass completion | +0.0015651204712201123 |
| Interceptions per pass | +0.00038060289207508526 |
| Passes per team-match | -4.683333333333334 |
| Long-pass share | -0.001770864278869562 |
| Shots per team-match | +0.7333333333333333 |
| Shots-on-target rate | -0.09611518111518112 |
| Goals per team-match | +0.13333333333333333 |
| Fouls per team-match | +0.016666666666666666 |
| Yellow cards per team-match | +0.2 |
| Red cards per team-match | +0.016666666666666666 |
| Possession share | +0.011518858307849138 |
| Crosses per team-match | +0.6166666666666667 |
| Headers per team-match | -0.1 |
| Tackles per team-match | +0.43333333333333335 |
| Micro-xG per team-match | +0.05955436101753413 |

These differences demonstrate distributional change, not tactical benefit.
They were not multiplicity-adjusted and cannot override the primary decision.

## Reproduction and claim boundary

The authoritative artifacts are:

- `data/evaluation/action_outcome_protocol_v1.json`
- `data/evaluation/action_outcome_v1/progress.json`
- `data/evaluation/action_outcome_v1/decision.json`

Replay the exact analysis and verify all identities with:

```bash
python scripts/verify_action_outcome_result.py
```

The verified conclusion is:

> The sealed world-model policy changes micro-actions and full-match behavior,
> but this fixed study does not establish a reliable improvement in external
> calibration or match outcomes. M1 remains research-only and default-off.

Any variance diagnosis, objective redesign or policy change belongs to a new
preregistered study with a new code identity. The present evidence must remain
immutable and reportable as an inconclusive result.

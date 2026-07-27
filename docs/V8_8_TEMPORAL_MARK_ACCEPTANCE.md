# V8.8 Temporal Mark Acceptance

Date: 2026-07-26

## Decision

V8.8 passes match-held-out timing, marked-action, counterfactual, identity, and closed-loop gates for SkillCorner. It is registered as an evaluated research candidate. The deployed release remains v7, with v6 retained for rollback.

## Contract

The module models the next post-reception event with a shared encoder and three explicit outputs: a ten-bin discrete hazard over 0.5-second intervals, continue versus terminal, and pass versus shot conditional on continuation. Events after five seconds and unknown next actions are right-censored rather than mislabeled as negatives. Thresholds are selected on the development match and stored in the checkpoint; the model and router use one shared decoder.

The event-driven router schedules the predicted mark, applies the existing pass-triplet transition only at the initial action, and terminates its learned velocity effect when the mark is reached. Invalid or invisible receiver/defender identities and unsupported providers disable the learned path.

## Held-Out Results

Eight SkillCorner matches train the model, one selects the checkpoint and thresholds, and match `2017461` is untouched test data.

| Measure | Result | Gate |
| --- | ---: | ---: |
| Observed timing MAE | 0.498 s | < 1.5 s |
| Survival NLL | 1.587 | finite |
| Two-second Brier score | 0.156 | < 0.25 |
| Continue/terminal balanced accuracy | 0.554 | > 0.55 |
| Pass/shot balanced accuracy | 0.656 | > 0.55 |

## Counterfactual And Rollout Checks

On all 547 held-out feature rows, deterministic feature permutation changes 72.4% of scheduled times with a mean absolute shift of 1.071 bins, and changes 49.4% of action marks. These are sensitivity checks, not causal effect estimates: they establish that both decoder branches consume event geometry instead of collapsing to constants.

Across 64 real held-out contexts, all ten-step rollouts are finite and bounded. Predicted schedules span steps 2 through 5. Removing receiver visibility blocks every temporal route. Terminating the prior action effect at the predicted mark changes the final ball/receiver/defender state by 7.958 m on average relative to an indefinitely persistent effect.

## Scope Boundary

Only SkillCorner supplies strong, explicitly timed reception chains. Metrica labels are inferred and Sportec lacks equivalent timing supervision, so v8.8 is not strict leave-one-provider-out evidence and is not deployable cross-provider. The next expansion should require a second provider with explicit reception and next-action timestamps rather than relaxing this boundary.

Canonical evidence:

- `reports/acceptance/temporal_mark_v88.json`
- `reports/acceptance/temporal_router_v88.json`
- `data/frame_world/temporal_router_v88.json`

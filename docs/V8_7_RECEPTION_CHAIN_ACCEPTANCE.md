# V8.7 Reception Chain Acceptance

Date: 2026-07-25

## Decision

V8.7 passes SkillCorner match-held-out gates for pass completion and the next state-machine action. It is accepted as an evaluated SkillCorner-scoped research candidate and is not deployed. Metrica is an inferred-label external audit only; Sportec remains unavailable.

## Event Chain Contract

The dataset contains 10,348 aligned pass chains:

- 8,585 SkillCorner events with strong `pass_outcome` supervision;
- 1,763 Metrica events inferred from the next event, excluded from training and promotion evidence;
- 8,029 SkillCorner events linked to a following possession action within five seconds.

Reception outcomes are `complete`, `turnover`, `offside`, or `unknown`. The predictive completion head trains only on complete versus turnover. The next-action state machine uses `pass`, `shot`, or `terminal`; loss and stoppage are merged because both terminate the current controlled continuation. Unknown and unmatched actions are not forced into a class.

## Held-Out Result

Eight SkillCorner matches train the model, one selects the epoch, and one remains untouched test data.

- Completion balanced accuracy: 0.769.
- Completion true-positive rate: 0.703.
- Completion true-negative rate: 0.835.
- Next-action macro F1: 0.306.
- Pass F1: 0.600.
- Shot F1: 0.126.
- Terminal F1: 0.193.

The completion gate uses balanced accuracy specifically to prevent a recurrence of a majority-class BA 0.50 head. Rare next actions remain substantially weaker than pass and are retained as an explicit boundary.

## State Transition

The router can resolve a supported pass into a `ReceptionState` containing outcome, possessor slot, possession team, and next action. Completion transfers possession to the receiver; turnover transfers it to the selected interceptor. Unsupported providers return no inferred chain state rather than fabricating possession.

## External Audit

Applying the SkillCorner model to Metrica inferred labels gives completion BA 0.603 and next-action macro F1 0.257. This is useful evidence of distribution shift, not provider-LODO acceptance. A second provider with explicit pass outcomes is required before cross-provider promotion.

Canonical evidence:

- `data/frame_world/v87_reception_chains/manifest.json`
- `reports/acceptance/reception_chain_v87.json`
- `reports/acceptance/reception_chain_v87_external.json`

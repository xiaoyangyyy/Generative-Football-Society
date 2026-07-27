# V8.1 Controlled Rollout Acceptance

Date: 2026-07-25

## Decision

`8.1.0-candidate` passes the research gates for closed-loop stability, explicit continuous-action intervention, and strict leave-one-provider-out evaluation. The three fold artifacts are evaluated candidates only. V7 remains the deployed release.

## Contract

The control tensor is per entity: normalized x acceleration, normalized y acceleration, and an explicit intervention mask. Acceleration estimated from observed history is context only. It cannot alter physical state unless the mask is set, preventing provider-specific tracking noise from being interpreted as an external command.

Each transition uses bounded physical integration plus a learned graph-temporal residual. Training includes a differentiable three-step closed-loop loss. Evaluation feeds predictions back for 20 steps without teacher forcing.

## Strict LODO

For each fold, every match from the held provider is excluded from training, early stopping, and development selection.

| Held provider | 1-step model / baseline | 2-second model / baseline | Maximum speed | Result |
| --- | ---: | ---: | ---: | --- |
| Metrica | 0.094446 / 0.094446 m | 2.524 / 2.525 m | 43.8 m/s | pass |
| SkillCorner | 0.031287 / 0.031257 m | 2.918 / 2.904 m | 29.4 m/s | pass |
| Sportec | 0.050360 / 0.050353 m | 1.931 / 1.926 m | 43.8 m/s | pass |

The acceptance rule is strict cross-provider non-inferiority within 5%, not a claim that every fold beats constant velocity. Metrica improves at two seconds; SkillCorner and Sportec remain within 0.5%.

## Intervention

A decaying positive-x ball acceleration produces 3.90-3.94 m mean forward displacement after two seconds in all folds. All counterfactual rollouts are finite and bounded. Other-entity response is measurable but tiny; the available observational data does not identify a strong causal reaction to synthetic controls, so no such claim is made.

## Stability

- 20-step rollouts contain no NaN or Inf.
- Positions remain inside the configured pitch buffer.
- Maximum rollout speed remains below the 45 m/s gate.
- Action effects require an explicit mask.
- Every fold checkpoint records its held provider.

Canonical evidence: `reports/acceptance/frame_world_v81.json`.

# V8.3 Actor-Aware Action Acceptance

Date: 2026-07-25

## Decision

V8.3 passes strict provider leave-one-out gates for direction-conditioned pass and shot transitions. Actor and target slots are now part of the action contract. Pressure recognition passes, but pressure-conditioned player displacement fails its causal transition gate and is disabled by default. V8.3 remains an evaluated research candidate; V7 remains deployed and V6 remains the rollback release.

## Actor And Target Alignment

- Metrica actor mapping: 100%; target mapping: 97.3%.
- SkillCorner actor mapping: 100%; target mapping: 99.7%.
- Sportec actor mapping: 97.1%; target mapping: 94.7%.
- Pressure actor and target mapping on SkillCorner: 100%.
- Missing, out-of-range, or unsupported identities use `-1` and cannot move an entity.

The provider identity mapping reconstructs the same stable entity slots as the frame builder. Action residuals are structurally masked: pass and shot affect only the ball; pressure can affect only its actor and target slots.

## Strict Provider LODO

| Held provider | Event-ball conditioned | Direction removed | Result |
| --- | ---: | ---: | --- |
| Metrica | 4.285 m | 6.115 m | pass |
| SkillCorner | 2.816 m | 3.029 m | pass |
| Sportec | exact fallback | exact fallback | pass |

The held provider is absent from training and development selection. Pressure conditioning is enabled only when strong SkillCorner pressure evidence exists in the training providers. Holding out SkillCorner therefore disables the pressure branch rather than treating unavailable labels as negatives.

## Pressure Falsification

The match-held-out SkillCorner pressure classifier reaches F1 0.655. The local transition experiment does not pass: actor/target error is 0.349 m with pressure conditioning versus 0.331 m with the kinematic fallback. Role-specific actor/target embeddings, a tenfold residual cap, and trajectory-based early stopping reduced but did not reverse the gap.

This is an important negative result. A pressure occurrence label identifies a state but does not identify the counterfactual displacement caused by pressure. Runtime pressure displacement therefore remains off. Enabling it requires richer labels such as pressure onset, approach vector, distance, intensity, and a matched no-pressure counterfactual or an accepted causal proxy.

## Verification

Canonical evidence:

- `data/frame_world/v82_actions/manifest.json`
- `reports/acceptance/frame_world_v83.json`
- `reports/acceptance/frame_pressure_v83.json`

The action manifest pins each aligned label file by SHA-256. Unit tests cover ball-only scope, actor/target-only pressure scope, unknown identity handling, and exact fallback without training evidence.

# GFS Research Direction and Freeze

## Decision

GFS is primarily a **football world-model research platform**. The deployed
v7 simulator remains a stable reference environment and public integration
surface; it is not automatically replaced by later research candidates.

The next milestone is external validity, not another world-model concept.
New model heads, LLM roles, routers, action classes, and mechanism protocols are
frozen until the external evidence gate and a full matched-seed ablation are
both frozen and passing.

## Frozen evidence boundary

- External marginal reference: StatsBomb Open Data, FIFA World Cup 2022,
  128 team-match rows and 15 declared observables.
- Pass evidence: strict provider-isolated completion evaluation; providers with
  fewer than 100 positive or 100 negative labels cannot authorize a learned
  cross-provider residual.
- Shot evidence: at least 1,000 held-out shots, 100 goals, multiple providers,
  and Brier improvement over the non-leaking baseline.
- Continuous-state evidence: at least two providers, five matches and 500
  observed temporal events per provider.
- Promotion evidence: matched seeds, frozen artifacts, complete tests, and an
  explicit release-manifest change. File presence never promotes a model.

Run the evidence audit with:

```bash
python scripts/audit_research_evidence.py --check
```

## Current module decisions

- Retain spatial intelligence and tick discipline in the core simulator.
- Retain the validated v7 shot model.
- Keep pass learning behind provider evidence and a physics anchor.
- Put affective coupling on watch: the diagnostic ablation improved when it was
  removed, but one fixture is not sufficient evidence for deletion.
- Keep advanced world-model, LLM cognition, narrative, memory, and meta-learning
  research-only. They remain default-off and cannot claim real-world effects.

The quick ablation is diagnostic only. It found that M0 itself fails the frozen
observable contract, so no additive module can be promoted from that run. A
full multi-fixture matched-seed report is the next promotion gate.

## Six-step completion contract

1. Concept expansion is frozen by `research_freeze_v1.json`.
2. The external football benchmark is named and machine checked.
3. Pass, shot, goal, and continuous-state coverage are audited by provider.
4. Matched-seed ablation exists; evidence strength is explicit.
5. Unsupported layers are isolated or provider-gated instead of receiving
   implicit authority.
6. The project identity is research-first; v7 remains the stable simulator.

This contract does not claim that GFS is externally calibrated. It makes the
remaining failure visible and prevents research complexity from hiding it.

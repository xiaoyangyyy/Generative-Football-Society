# GFS Research Direction and Freeze

## Decision

GFS is primarily a **football world-model research platform**. The deployed
v7 simulator remains a stable reference environment and public integration
surface; it is not automatically replaced by later research candidates.

The formal external-validity baseline and full matched-seed ablation are now
complete. New model heads, LLM roles, routers, action classes, and mechanism
protocols remain evidence-gated: completion of an experiment does not promote
a layer whose validation gate failed or whose benefit is unresolved.

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

- Retain spatial intelligence, affective coupling, tick discipline, and
  tactical bias in the core simulator; removing each broke a formal gate.
- Retain the validated v7 shot model.
- Keep pass learning behind provider evidence and a physics anchor. The current
  pass-logit blend is optional and default-off because its removal passed all
  formal gates without a resolved loss difference.
- Keep advanced world-model, LLM cognition, narrative, memory, and meta-learning
  research-only. M1 had no measurable formal effect; the retrained v9 candidate
  failed its two-step planning gate. C1 had no statistically resolved gain and
  has no live-provider evidence. These layers remain default-off.

The earlier quick ablation remains diagnostic only. The authoritative report is
the six-fixture, three-seed, full-90-minute matched protocol in
`data/evaluation/formal_ablation_summary_v1.json`. See
`docs/FORMAL_RESEARCH_VALIDATION_2026-08.md` for the completed plan.

## Six-step completion contract

1. Concept expansion is frozen by `research_freeze_v1.json`.
2. The external football benchmark is named and machine checked.
3. Pass, shot, goal, and continuous-state coverage are audited by provider.
4. Matched-seed ablation exists; evidence strength is explicit.
5. Unsupported layers are isolated or provider-gated instead of receiving
   implicit authority.
6. The project identity is research-first; v7 remains the stable simulator.

This contract now has machine-checked completion evidence in
`data/evaluation/staged_completion_v1.json`. It establishes bounded external
observable validity for the declared benchmark; it does not claim universal
football realism or live-provider LLM benefit.

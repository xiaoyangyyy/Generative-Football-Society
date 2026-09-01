# Project Architecture Audit

## Standard

This project treats advanced as measurable improvement in correctness,
calibration, uncertainty handling, causal separation, replayability, or
maintainability. Mathematical notation, LLM call count, and module count are
not quality signals by themselves.

## Layer Review

| Layer | Current design | Decision | Remaining work |
|---|---|---|---|
| Raw data | Explicit CSV boundary, schema checks, source/version manifests, sealed group splits | Reproducible baseline | Add temporal snapshots when upstream datasets change |
| Entity priors | Continuous player/coach/team fields with football-semantic role axes | Useful and interpretable | Fit coefficients to held-out player/event data instead of hand tuning |
| Tournament | Official group schedule and constrained R32 builder; stable named seeds | Correct simulation path | Encode the final FIFA bracket table once officially fixed |
| Randomness | Stable BLAKE2-derived subsystem streams for match, extra time, penalties, calibration | Strong contract | Migrate remaining social/legacy global RNG calls to injected streams |
| Persistence | Versioned, validated, atomic tournament checkpoint | Production-grade minimum | Add migration functions and content checksum |
| Macro scoring | Coupled intensity dynamics; xG fused before a single goal observation | Coherent research model | Replace Euler heuristic with fitted state-space point process |
| Micro engine | Spatial fields, action selection, pass/shot/aerial physics and affective coupling | Rich but heuristic | Establish explicit SI/normalized unit contract and fit jointly to event data |
| World model | v7 deployed plus evidence-gated v8 frame candidates; v8.9 strict SkillCorner/StatsBomb temporal LODO | Modular and empirically guarded | Beat the continuous-time baseline before any v8 promotion |
| Cognition | Triggered System-2 plans, bounded schemas, validated replay cache | Good hybrid architecture | Measure intervention lift and remove plans that do not beat rules |
| Narrative | Generated events become bounded, persistent social signals with a facts ledger | Correct causal placement | Learn persistence/decay from longitudinal outcomes |
| Memory | Layered retrieval, provenance graph, contradiction links, delayed utility | Auditable baseline | Add retention ablation benchmarks |
| Meta-learning | LLM proposes bounded deltas with evidence and time-scale audit | Safe experimental meta-controller | Require delayed outcome credit before slow parameter updates |
| LLM transport | Shared provider gateway; role agents remain logically independent | Right LLM amount | Add request IDs, token/cost telemetry, circuit breaker, provider adapter |
| Calibration | Layer gates, ablations, external baselines, branch-specific WM quality | Good research hygiene | Automate confidence intervals and regression thresholds in CI |
| Public API | Stable `src.app`/CLI and immutable runtime environment boundary | Clean integration surface | Keep compatibility facade stable through v6 |

## Corrections From This Audit

1. Replaced process-random `hash()` seeds with stable named streams.
2. Routed regulation, extra time, and penalties through isolated reproducible RNGs.
3. Removed discarded macro goal samples before macro/micro fusion.
4. Made tournament checkpoints atomic and version/schema validated.
5. Validated cognitive cache hits, made cache writes atomic, and enriched player facts before LLM calls.
6. Removed nested transport retry multiplication from the cognitive layer.
7. Replaced ordinal circular position embeddings with football-semantic role axes.
8. Removed a redundant sigmoid that collapsed player archetype ability contrast.
9. Added raw-data schema checks, invalid-row filtering, and duplicate filtering.
10. Seeded the public single-match API before random agent construction.
11. Added matched-seed causal evaluation and transactional meta-learning.
12. Added memory provenance, contradiction links, and delayed utility attribution.
13. Added continuous-time event and hierarchical policy experiment modules.
14. Added dynamic graph world-model v6 code and uncertainty-driven active sampling.
15. Fixed away-team coordination reading the home-team internal state.
16. Replaced pass-only world-model probability adoption with a validated,
    feasible multi-action simplex; shot receives its own authority evidence,
    while unvalidated cross control remains closed.
17. Added a cross-specific grouped-heldout validation contract, independent
    online calibration, bounded planning path, replayable ball trajectory and
    exact product evidence link. The current checkpoint still has zero cross
    authority because it predates this evidence contract.
18. Removed direct action authority from the joint shot head. Its development
    quality remains diagnostic, while runtime shot probability falls back to
    physics xG unless an identity-bound frozen head wins on sealed groups.
19. Made hold a strict counterfactual reference rather than an implicit learned
    recommendation. Suppression-driven probability redistribution to hold is
    now measured separately and cannot enter direct action authority or direct
    adoption counts, even if stale evidence attempts to open its gate.
20. Closed the replay-to-manager execution gap for cross trajectories and
    projected direct, suppressive and hold-reference signal semantics into one
    identity-bound official action explanation chain.
21. Extended cross and V2 policy semantics through paired-world propagation and
    versioned future mechanism examples, while preserving V1 replay
    compatibility and the non-causal downstream-window boundary.
22. Preserved those action semantics through synchronized comparison totals,
    future-review execution traces, season-level decision summaries, manager
    world threads and Studio, without reclassifying legacy V1 evidence.
23. Extended the same semantics into identity-bound historical chapters,
    chronological season trajectories and influence-path totals, with legacy
    zero normalization and rehashed-tamper rejection.
24. Added an explicitly bounded official-action semantic ledger across live
    threads, historical chapters and Studio; sample partitions and truncation
    remain distinct from full retained-record totals.

## LLM Scope Decision

Do not add more LLM roles now. Keep the current role agents because they model
different institutions, but share transport, schemas, facts, audit, and replay.
The next quality gain should come from stronger environment data and causal
evaluation, not additional generations.

LLM influence is appropriate in three places:

- sparse high-level decisions under novel context;
- narrative events that are converted into explicit persistent state;
- bounded meta-learning proposals with evidence and delayed evaluation.

LLM influence is inappropriate for continuous physics, score sampling, data
cleaning, or direct unaudited state mutation.

## Priority Research Queue

1. Collect balanced cross, shot and goal transitions; produce a new checkpoint
   that passes the registered cross gate and recalibrate the shot branch before
   broader promotion. Do not retrofit authority onto the current v9 checkpoint.
2. Continue the gradual `SocietyAgent` facade split only when a behavior-preserving boundary is available.
3. Collect real failed-pass labels for an external completion benchmark.
4. Split `TournamentManager` into scheduler, match service, standings, narrative orchestration, and repository.
5. Add matched-seed intervention tests for cognition, narrative persistence, memory, and meta-learning.
6. Fit macro intensity and micro action parameters against held-out event data with uncertainty intervals.
7. Keep v7 deployed and v6 as rollback; v8.9 remains evaluated until continuous-time LODO beats the continuous baseline.

## Release Gate

A change is accepted only when deterministic replay holds, state is finite and
bounded, score ownership is singular, generated content cannot override facts,
checkpoints survive interruption, and matched-seed ablations show no unexplained
cross-layer changes.

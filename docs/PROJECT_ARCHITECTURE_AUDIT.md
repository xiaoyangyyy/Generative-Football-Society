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
| Randomness | Stable BLAKE2-derived identity streams across match, scoring, referee, social, wear, carryover, agents, brackets and legacy journeys | Strong runtime contract with machine-enforced global-draw ban | Add matched-seed intervention invariance checks as new stochastic subsystems appear |
| Persistence | Checkpoint V5 binds inputs/randomness, snapshots Agent/social state, receipts reflection, and embeds bounded external-state rollback images | Strong identity-gated, crash-resumable recovery for project-owned causal state; external paths remain fail-closed | Obtain provider idempotency guarantees and validate recovery under production filesystem faults |
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
25. Versioned official action evidence to V2 and added a separately validated
    full retained-record semantic aggregate. It propagates through decision
    ledgers, live and historical world views and Studio while preserving V1
    reads, source-truncation boundaries and the ban on outcome attribution.
26. Versioned official action evidence to V3 and added complete baseline-to-
    actual and locally attributable action-transition matrices. The matrices
    retain V1/V2 read compatibility, fail closed under rehashed tampering and
    make the world model's concrete simulator-action changes product-visible.
27. Stratified same-chapter world facts by locally attributable action
    transition, with explicit overlapping membership, V1/V2 absence, source
    coverage and hard bans on ranking, cross-stratum effects and outcome
    attribution.
28. Added an accessible, responsive 5-by-5 action-transition map to the manager
    world navigator. It distinguishes observed zeros from unavailable legacy
    evidence, publishes source coverage and renders only with safe DOM APIs.
29. Replaced the multi-action expected-change TV proxy with the exact mismatch
    probability of the simulator's shared-uniform inverse-CDF sampler. The
    estimator is versioned and legacy product evidence remains visibly labelled.
30. Versioned official action evidence to V4 and propagated exact shared-draw
    expectation through decision, world-thread, chapter, trajectory and season
    views, with strict V1–V3 compatibility and explicit missing-version coverage.
31. Fixed V4 inclusion in V3-compatible transition propagation and connected
    each evidenced action-matrix cell to its bounded same-chapter world facts
    and latest identity-verified chapter through an accessible drill-down.
32. Replaced the single-incident credential closure with a V2 exact incident
    registry for DeepSeek and GitHub PAT exposure, independent redacted receipts
    and multi-family zero-value scans; refreshing the evidence chain also
    invalidated the stale post-code-change formal paper-package pass.
33. Replaced the remaining runtime process-global random draws with
    identity-derived Python and NumPy streams. Tournament match identity now
    reaches scoring, referee, dialogue, agent settlement, physical wear and
    cross-match carryover; bracket, world-day and legacy journey draws are
    isolated as well. The architecture audit rejects future direct global draws
    in simulation and memory runtime modules.
34. Bound tournament resume to checkpoint V2's original random world. The
    public API now forwards seeds into world construction, checkpoint payloads
    carry a canonical content checksum plus RNG contract/root seed, explicit
    seed conflicts fail closed, and identity-less V1 resumes are rejected
    instead of guessed. Caller-provided project roots now own their checkpoint
    instead of leaking state into the installed source tree.
35. Upgraded run manifests to V2 and tournament checkpoints to V3. Resume now
    binds portable immutable source contents, data/model hashes, Python/platform,
    structured simulation settings, tactics, coaches, all base rosters,
    pre-run product state and secret-filtered runtime options. Evolving squad
    carryover is hashed by every checkpoint rather than frozen as static input. It
    hashes every supplemental runtime value, never persists credential fields, and rejects
    any drift before provider resolution or match mutation. Mutable carryover
    is verified against the exact checkpoint that references it.
36. Upgraded tournament checkpoints to V4 with a versioned snapshot of all
    initialized mutable Agent fields plus optional match fields, dynamic coach
    adaptations, social posts, dialogue topic market and narrative history.
    Restore validates every Agent/random/coach identity before mutating memory.
    Reflection now persists a parsed response receipt before applying a
    deterministic operation, reuses that receipt after a crash and records
    application exactly once. Completed finals reconstruct champion state on
    the skip path. Carryover, fusion history, counterfactual evidence and an
    enabled cognitive cache tree are checkpoint-bound. Roster loading removes
    non-finite observations and incomplete team dynamics fall back to the
    status-derived vector instead of entering match math as NaN or false zeros.
37. Upgraded tournament checkpoints to V5 with bounded, content-addressed
    rollback images for carryover, fusion history, counterfactual evidence and
    enabled cognitive caches. Resume verifies the complete run identity before
    mutation, preserves displaced state in an atomic recovery journal and
    converges after an interrupted restore. Relative-path validation,
    independent file hashes, snapshot/file-count limits, symlink rejection and
    project-root confinement prevent the checkpoint from becoming an
    arbitrary filesystem writer. Externally configured cache directories stay
    read-only and fail closed.

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

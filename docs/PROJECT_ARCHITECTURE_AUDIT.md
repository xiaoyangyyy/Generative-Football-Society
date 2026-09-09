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
| Persistence | Checkpoint V6 binds inputs/randomness, snapshots Agent/social state, receipts reflection, and shares one workspace lease across startup, resume and rollback-complete matches | Strong identity-gated, crash-resumable recovery for project-owned causal state; missing commits, concurrent lifecycle writers, and external paths fail closed | Obtain provider idempotency guarantees and validate recovery under production filesystem faults |
| Macro scoring | Coupled intensity dynamics; xG fused before a single goal observation | Coherent research model | Replace Euler heuristic with fitted state-space point process |
| Micro engine | Root-bound effective rosters with explicit synthetic fallback provenance; spatial fields, action selection, pass/shot/aerial physics and affective coupling | Rich, portable and evidence-visible, but still heuristic | Establish explicit SI/normalized unit contract and fit jointly to event data |
| World model | v7 deployed plus evidence-gated v8 frame candidates; v8.9 strict SkillCorner/StatsBomb temporal LODO | Modular and empirically guarded | Beat the continuous-time baseline before any v8 promotion |
| Cognition | Triggered System-2 plans, bounded schemas, validated replay cache | Good hybrid architecture | Measure intervention lift and remove plans that do not beat rules |
| Narrative | Generated events become bounded, persistent social signals with a facts ledger | Correct causal placement | Learn persistence/decay from longitudinal outcomes |
| Memory | Layered retrieval, provenance graph, contradiction links, delayed utility | Auditable baseline | Add retention ablation benchmarks |
| Meta-learning | LLM proposes bounded deltas with evidence and time-scale audit | Safe experimental meta-controller | Require delayed outcome credit before slow parameter updates |
| LLM transport | Shared adapter boundary with logical request IDs, bounded prompt-free token/cost telemetry and a thread-safe circuit breaker | Production-shaped, evidence-bearing transport | Validate live-provider request IDs/usage fields and freeze an explicit pricing snapshot before a paid pilot |
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
38. Hardened V5 resume against conflicting recovery journals even when live
    files already match, validated recovery/last-recovery path types before
    mutation, and changed the public tournament handoff to parse an existing
    checkpoint once. The same validated payload now drives seed, manifest,
    rollback and world restoration, while non-resume callers retain their
    established method signature.
39. Corrected release-readiness composition so the Linux target hash lock is
    evaluated from its own verified artifacts and checks rather than the
    aggregate paper result. `code_ready` now means the paper code contract
    (excluding only the authorized confirmatory result) and all designated
    infrastructure/protocol gates are complete; `release_ready` still requires
    every external result and review. CLI and Web now expose this distinction
    consistently. Current state is 10 passed/11 open gates with the code
    contract ready and external evidence still open.
40. Joined the M2 terminal preflight, standalone checkpoint qualification,
    formal progress and final decision into one identity-bound five-stage
    Studio workflow. Recorded preflight and candidate receipts fail closed
    after dependency drift, rejected sealed candidates cannot become tuning
    evidence, interrupted runs can only resume the same identity, and the Web
    surface displays but never executes the single valid next command.
41. Replaced the opaque shared LLM client call with an explicit provider
    adapter and one logical request identity preserved across retries. Added
    bounded prompt-free attempt telemetry, token and configured-price cost
    accounting, plus a thread-safe closed/open/half-open circuit breaker.
    Cognitive match reports and the prospective pilot now fail closed when the
    telemetry delta is unavailable, truncated, secret-boundary-unsafe or
    inconsistent with the independently observed per-scope successful-call
    count. ContextVar-bound match scopes prevent concurrent cognitive matches
    from claiming each other's provider evidence.
42. Propagated the caller's project root through the macro-to-micro squad
    boundary, so installed or isolated workspaces no longer read rosters from
    the source checkout by accident. Missing roster files retain an explicitly
    labelled synthetic status fallback; an existing malformed or XI-incomplete
    roster now fails closed instead of being silently replaced. Exact roster
    identity, player counts and source travel through the micro summary and
    product report, and research/cognitive integrity rejects a real fallback.
43. Upgraded tournament checkpoints to V6 and placed every public tournament
    match behind one workspace-level file lease. A pre-match checkpoint now
    covers the in-memory world and project-owned carryover, fusion,
    counterfactual, cognitive, ball-log and narrative-debug state. Success
    requires exactly one matching in-memory result and durable checkpoint;
    exceptions and missing commits restore every bounded surface, including
    the durable pre-match checkpoint, and produce a redacted rollback receipt.
    Concurrent writers and configured external
    output targets fail before match execution. Fault-injection tests cover
    successful commit, memory/file rollback, missing commit, rollback failure,
    external paths and lock contention.
44. Extended the same operating-system lease across the public tournament
    lifecycle. Checkpoint loading, seed selection, run-identity verification,
    external-state recovery, manifest writing, world construction and all
    matches now execute under one project-root lock. Context-local ownership
    lets nested per-match transactions reuse that lease without weakening the
    standalone match boundary. A competing process fails before startup state
    is read or mutated; behavioral tests prove ownership, reuse and release.
45. Made Studio multi-file restore crash-convergent. A persistent digest-bound
    journal is activated before the first switch; startup or explicit recovery
    either proves the complete new target set or reconstructs the exact old
    set. Active transaction cleanup is atomically detached before deletion,
    unsafe or unverifiable journals fail closed, and real child-process exit
    tests cover partial and fully switched states without a match or training.
46. Made the frozen production-operations protocol executable from the shipped
    image without weakening the read-only runtime. A profile-scoped offline
    validation service owns dedicated output, persistence, backup, evidence and
    restore-scratch volumes. It shares the Web writer lease, adds a second
    validator lease, records the claimed task before match execution, and splits
    workload execution, operator attestation and immutable finalization. The
    exact operator statement now binds the final progress payload by SHA-256;
    the recovered task must retain its task ID and idempotency digest and finish
    on exactly the next attempt. Restore staging must remain outside the source
    workspace. These are
    zero-match contract and local substitute checks, not evidence that the
    prospective 100-match deployment drill has run.
47. Replaced self-reported human-study authority with one shared product-layer
    registration and evidence contract. Registrations are atomic and
    OS-serialized, bind protocol/case/time identity, and allocate comparative
    AB/BA order through role-stratified permuted blocks. Final analysis requires
    exact registry coverage and verifies non-symlink evidence bytes in a
    content-addressed archive. Comparative structured receipts are re-scored
    from frozen numeric case contracts instead of trusting `correct=true`.
    The amendments record zero prior participants and remain preparation, not
    user-value evidence.
48. Physically separated comparative participant cases from the moderator
    scoring seal. The seal binds the exact case-manifest SHA-256, remains
    analysis-only and is excluded from the participant evidence kit. Studio CLI
    and authenticated Web now provide aggregate status, consent-bound
    registration and deterministic blinded packet download. Packet writes are
    confined, symlink-safe and atomic no-overwrite; Web telemetry normalizes
    registration IDs. Live status stops asserting zero observations when a
    measurement record file appears. This closes delivery leakage and manual
    handoff gaps but records no participant or product-value result.
49. Added a physically separate product-value participant service. Provisioned
    state and receipt archives must live outside the repository; only a
    high-entropy capability hash is stored. The server gates measurement behind
    an unscored tutorial, reveals one condition at a time, owns the deadline,
    commits idempotent structured receipts and never mounts Studio routes or
    reads the scoring seal. A moderator-attested import revalidates the complete
    byte and identity chain before repository mutation. This closes the study
    execution handoff without claiming that a participant was observed.
50. Extracted the coherent latent-psychology boundary from `SocietyAgent` into
    `AgentPsychologyMixin`. Historical initialization, latent projection,
    appraisal, emotion, coping, salience and compatibility state views retain
    AST-equivalent definitions, while structural and numeric-invariant tests
    enforce the ownership. The prospective M2 import closure now binds the new
    module through the real runtime path; its refreshed receipt remains an
    explicit zero-training preflight rather than model or outcome evidence.
51. Completed the remaining behavior-preserving `SocietyAgent` facade split.
    Governance/referee response, tactical controls, reflection/cognitive
    ingestion and physical condition now live in four dedicated mixins with 22
    AST-equivalent methods. The facade is 127 lines and owns only construction,
    finite-value normalization and identity-derived region/style helpers. Tests
    and an AST-based architecture gate enforce inheritance, exclusive method
    ownership and the closed facade surface; the zero-training M2 import closure
    now binds all four modules without creating a model or outcome result.
52. Split the remaining `TournamentManager` facade debt into two cohesive
    layers instead of additional feature fragments. The lifecycle layer owns
    scheduling, advancement, knockout and receipted reflection; the state layer
    owns run identity, checkpoint persistence and world restoration. Ten moved
    methods remain AST-equivalent, direct tests enforce exact ownership, and the
    machine audit keeps the 130-line facade limited to construction plus six
    cross-layer adapters. The zero-training M2 identity now binds both modules.
53. Corrected the real process-kill recovery drill's evidence boundary. A
    bounded 45-second readiness timeout still fails closed, but the observed
    15-second local objective is reported separately from functional recovery
    correctness and cannot authorize a production RTO. Service restart, orphan
    reconciliation, committed-file identity, clean shutdown and lease release
    remain hard gates; the full suite passes with 1,411 tests and three skips.
54. Closed the cross-match society continuity gap without adding another
    product subsystem. The authoritative squad carryover now stores bounded,
    versioned, team-bound and content-addressed cognitive, social and
    psychological state. Settlement captures it after idempotent cognitive
    ingestion; the next match restores it before coach support and includes it
    in cache identity. Both live-provider routing and deterministic fallback
    consume the same prompt-safe context. Product world-state V2 projects only
    content-free counts and state transitions inside the existing six-stage
    manager thread, while V1 evidence remains replay-compatible. Matched-seed,
    privacy, tamper, compaction and next-decision tests establish the software
    mechanism but do not claim outcome improvement or real-world causality.
    The complete 197-file regression passes with 1,422 tests and three skips.
55. Replaced the production meta-learning shortcut with a delayed authority
    lifecycle. Reflection now persists a content-addressed shadow proposal and
    cannot mutate parameters. Later settlements attach idempotent descriptive
    observations without treating them as effect evidence. Authorization
    requires at least eight identity-bound matched-seed rows; the validator
    independently replays the source identity, paired effects, ATE and 95%
    interval before a positive lower bound may commit. Utility rows are bound
    to the runtime `[-5, 5]` scale, caller thresholds cannot be negative, and
    no-effect reflections remain audit-only rather than creating pending
    authority. Negative intervals, stale parameters, duplicate rows, boolean
    adjustment/lifecycle values, operation reuse and nested/receipt tampering
    fail closed.
    Proposal status survives through the existing
    reflection/society/checkpoint state and is projected without private text
    into world-state V3 and the existing six-stage manager thread. V1/V2 remain
    replay-compatible, and no formal effect result is claimed. The final
    198-file regression passes 1,433 tests with three declared skips, and every
    Python file modified by this stage passes Ruff.
56. Made that delayed authority legible without creating another product
    subsystem. World-state V4 projects content-free proposal identity,
    bounded parameter direction and time scale, observation progress,
    evaluation summary, authority state and next required evidence. Reflection
    text, memory evidence and matched rows remain private. Identity-bound
    per-match updates appear inside the existing six-stage manager world
    thread, which has no authorization control and preserves V1/V2/V3 replay.
    Mixed phase versions, nested rehashing, invalid authority/status pairs and
    order-dependent season summaries fail closed. This is product-visible
    governance, not evidence that meta-learning improves simulator outcomes.
    The complete 198-file regression passes 1,434 tests with three declared
    skips; touched Python files pass Ruff, the architecture audit is green,
    and zero-training recovery and M2 identity receipts are current.
57. Extracted the 885-line `ProductWorkspace._session` implementation into a
    dedicated stateless session-integrity repository. The facade keeps one
    six-line compatibility delegation and the same authoritative JSON file;
    the new function receives only root, path and two replay callbacks. A
    normalized comparison proves its body is otherwise exact. Direct tests
    cover valid loading, registry tamper rejection and callback forwarding,
    while the machine audit pins all registry validators, forbids a reverse
    workspace import, caps module size and rejects return of inline loading.
    Recovery evidence also binds the new implementation file. No schema,
    workflow, simulator behavior or research authority changes. The complete
    199-file regression passes 1,437 tests with three declared skips.
58. Hardened authoritative human-study registration against transient Windows
    sharing violations during same-directory `os.replace`. The writer retries
    only `PermissionError` with eight bounded attempts; permanent denial is
    re-raised and temporary state is removed. Injected transient/permanent
    tests and ten consecutive 24-registration stress runs pass without
    weakening the existing process lease or atomic-read contract.
59. Extracted workspace research-evidence assembly, formal-result identity and
    mode-specific readiness into a dedicated read-only repository. The facade
    forwards only root, mode, one current evidence value and the confined
    artifact resolver, retains the historical helper import, and has no reverse
    workspace dependency. Direct tests and the architecture audit enforce
    fail-closed authority, exact forwarding, single evidence evaluation and
    recovery identity coverage without changing schemas or research gates. The
    complete 200-file regression passes 1,442 tests with three declared skips.
60. Added one same-chapter world-model contribution story to the existing
    matchday command center. It compresses intervention, reviewed futures,
    official action adoption, persistent world transition and descriptive
    result evidence without combining independent aggregate populations. The
    latest source season, navigator, chapter, fixture and matchday remain
    explicit; pre-match stages never synthesize future evidence; outcome
    improvement and causal authority remain false. Direct, end-to-end, HTTP and
    accessibility checks cover the projection and responsive UI. The complete
    200-file V4.17 regression passes 1,444 tests with three declared skips, and
    the refreshed architecture audit remains fully green.
61. Moved the world-model contribution story out of the HTTP monolith into a
    dedicated pure product read model. The Web boundary now builds, validates,
    attaches and renders it, while exact replay rejects semantic or authority
    drift. The existing manager-journey gate rejects reverse Web dependencies,
    an oversized module or a returning inline implementation. Product recovery
    identity binds the new source directly. Fourteen direct tests cover every
    action state, review-use state, missing evidence, count bound, immutability,
    fresh unavailable state and tamper rejection without adding another state
    authority. The complete 201-file regression passes 1,456 tests with three
    declared skips; architecture, HTTP, accessibility, recovery and static
    checks remain green.
62. Closed the gap between the M2 two-step state objective and its one-step-only
    policy-utility objective. Training now supervises actor-centred utility at
    the final state of observed two-action rollouts with independent ensemble
    bootstrap routing, records separate resumable counters, and emits explicit
    grouped development evidence. Sealed validation rebuilds the same report;
    runtime and candidate qualification require both one-step and two-step pass
    utility gates in addition to two-step state skill. The pre-candidate,
    zero-training protocol amendment adds the new sealed gate and the refreshed
    preflight passes fourteen checks without opening sealed model-selection
    data. Eligible receipts now replay all internal gates instead of trusting a
    top-level boolean. The complete 202-file regression passes 1,460 tests with
    three declared skips, and the architecture audit enforces the complete path
    while making no model-effect or outcome claim.
63. Closed the remaining gap between M2 sequence qualification and runtime action
    control. A dedicated runtime API now executes exactly two explicit cleaned
    actions through member-consistent dynamics and reports sequence identity,
    actor-centred utility and calibrated uncertainty. High-level pass and cross
    comparisons marginalize every first action under one common pass/hold
    continuation policy derived from pre-model utilities and the simulator's
    actual temperature. One-step, sequence and two-step state gates must all be
    open; missing interfaces, malformed outputs or incomplete continuation
    support fail closed without a one-step fallback. Direct runtime and planner
    tests cover explicit sequence calls, common weights, leakage removal and all
    rejection paths. Candidate qualification and receipt replay require the
    predictor before formal execution. No checkpoint was trained and no match
    was executed. The complete 202-file regression passes 1,467 tests with three
    declared skips.
64. Closed the branch-support gap inside M2 explicit sequence planning. Training,
    development preflight and sealed validation now record grouped support,
    empirical mass and leakage-cleaned action prototypes for every observed
    second action. Runtime recomputes the gate and consumes only the normalized
    development-supported policy; current `pass` continuation support retains
    `pass` and `shot` while excluding five development holds and four crosses.
    M2 compares that supported sequence value with its registered exact-zero
    persistence baseline and no longer predicts unsupported `hold -> *`
    branches. Candidate qualification and receipt replay independently validate
    policy weights and prototypes. Protocol amendment V3 occurred before any
    candidate binding, training or formal execution, and the refreshed
    zero-training preflight passes all 16 checks without loading sealed rows.
65. Closed the remaining mixed-depth M2 execution path. The high-level pass
    decision already used the support-qualified two-action utility, but concrete
    receiver/target ranking still called the one-step predictor. M2 target
    ranking now uses the identical sequence authority and continuation policy;
    absent sequence capability fails closed to the unchanged rule distribution.
    The unused legacy pass-bonus helper was removed. Protocol amendment V4
    records zero prior candidates, training runs and formal runs, and the
    architecture audit prevents either one-step bypass or helper restoration.
    The complete 202-file regression passes 1,471 tests with three declared
    skips.
66. Removed home/away evidence leakage from the prospective mirrored M2
    controller. Absolute-coordinate observations and action targets now produce
    separate perspective policy-utility profiles, continuation support and
    action prototypes. Runtime passes the acting perspective through one-step,
    explicit two-step and pass-target planning; candidate qualification,
    sealed evaluation and receipt replay require both sides and reject
    pooled-only legacy checkpoints. The gate replays sample totals,
    sample-weighted pooled MSE, MSE-derived skill, perspective labels, support
    mass and action prototypes. On frozen development data, home supports pass
    and shot at 99.46% mass, while away supports pass only at 97.20%; away
    cannot borrow home shots to cross the branch threshold. Protocol amendment
    V5 occurred before any candidate, training or formal run. The zero-training
    preflight passes 18/18 checks with a 250-file identity closure, architecture
    audit passes 92/92, and the complete 202-file regression passes 1,474 tests
    with three declared skips.
67. Closed the chronology break in the manager-facing world-model contribution
    story. The old projection preferred any completed history row over the
    active next fixture, so the current five-stage workflow disappeared after
    matchday one. Story V2 now makes the active chapter primary, binds its
    current identity, keeps official-world stages pending, and retains one
    completed predecessor with same-chapter action, persistence and descriptive
    result evidence. Studio opens only a completed navigable source; completed
    seasons fall back to their latest world and empty seasons invent nothing.
    Replay validation and non-causal boundaries remain unchanged. No match,
    future generation, training or external call is executed by the change.
    Focused product regression passes 110 tests, CI-equivalent regression
    passes 1,475 with three skips and one registered deselection, and local
    HTTP, accessibility, recovery and all 92 architecture checks are green.

68. Replaced the manager-world Studio's implicit renderer override chain with
    one frozen, named 13-stage composition. The exact historical order is now
    visible at one boundary, including reviewed-future semantic enrichments,
    retained-record semantics, transition propagation, the action matrix,
    exact shared-draw expectation and drill-down. Tests require one root
    renderer and reject both legacy wrapper families, so a declaration move
    cannot silently detach a product evidence layer. DOM ownership, chapter
    restoration and all evidence semantics remain unchanged. The change runs
    no match, future generation, training, formal study or external call.
    Focused Web regression passes 60 tests, CI-equivalent regression passes
    1,476 with three skips and one registered deselection, and local HTTP,
    accessibility, recovery and all 92 architecture checks are green.

69. Replaced twelve implicit manager decision-ledger overrides with a frozen,
    named 13-stage renderer. Advisor evidence, runtime execution, reviewed
    futures, scenario mechanisms, official actions, world evolution, retained
    semantics and exact action-change expectation now have one visible order.
    Tests and architecture gates require one root renderer and reject the full
    legacy ledger-wrapper family. Entry ordering, DOM ownership and evidence
    boundaries are unchanged, and the refactor runs no match, future,
    training, formal study, participant session or external call. Focused Web
    regression passes 61 tests, CI-equivalent regression passes 1,477 with
    three skips and one registered deselection, and local HTTP, accessibility,
    recovery and all 92 architecture checks are green.

70. Replaced eight implicit action-adoption view overrides with a frozen,
    named nine-stage renderer. Runtime signals, cross and shot authority,
    hold-reference semantics, sealed historical evidence, current-code
    applicability, the manager protocol and prospective M2 control now have one
    reviewable order. A dedicated architecture gate requires one root renderer
    and rejects every legacy action-adoption wrapper. Evidence wording,
    identity boundaries and model authority remain unchanged; no match,
    future, training, study, participant session or external call is executed.
    Focused Web regression passes 62 tests, CI-equivalent regression passes
    1,478 with three skips and one registered deselection, and local HTTP,
    accessibility, recovery and all 93 architecture checks are green.

71. Replaced seven implicit unified world-evolution thread overrides with a
    frozen, named eight-stage composition. The product now has one visible
    order from reviewed futures and action semantics through official execution
    to persistent society state and retained records. Meta-learning governance
    remains a separate read-only helper rather than hidden composition state.
    Tests and a dedicated architecture gate require one root appender and
    reject every legacy thread wrapper. No world state, evidence, authority or
    claim changes, and no match, future, training, study, participant session
    or external call is executed. Focused Web regression passes 63 tests,
    CI-equivalent regression passes 1,479 with three skips and one registered
    deselection, and local HTTP, accessibility, recovery and all 94
    architecture checks are green.

72. Replaced four implicit manager-intelligence overrides with one frozen,
    named five-stage renderer. Base matchday evidence, club support, tactical
    runtime binding, official world-model action execution and the unified
    world-evolution thread now have one visible manager-facing order. Direct
    tests and a dedicated architecture gate require one root renderer and
    reject every legacy intelligence wrapper. The refactor preserves all DOM
    targets, early returns, evidence identities and non-causal wording; it
    changes no manager decision, action authority, world state or claim and
    executes no match, future, training, study, participant session or external
    call. Focused Web regression passes 64 tests, CI-equivalent regression
    passes 1,480 with three skips and one registered deselection, and local
    HTTP, accessibility, recovery and all 95 architecture checks are green.

73. Replaced three implicit manager future-set overrides with one frozen,
    named four-stage renderer. Base identity-bound cards, explicit manager
    review controls, per-timepoint scenario evidence and bounded action
    mechanisms now have one reviewable order, including the existing
    interrupted-task recovery control. Direct tests and a dedicated
    architecture gate require one root pipeline and reject every legacy future
    wrapper. Binding semantics, DOM ownership and non-causal wording remain
    unchanged; the refactor generates no future, changes no review receipt or
    world state and executes no match, training, study, participant session or
    external call. Focused Web regression passes 65 tests, CI-equivalent
    regression passes 1,481 with three skips and one registered deselection,
    and local HTTP, accessibility, recovery and all 96 architecture checks are
    green.

74. Replaced three implicit post-season recruitment overrides with one frozen,
    named four-stage configuration. The base transfer market, player-contract
    lifecycle, global free-agent market and sporting-director plan now share
    one visible order while retaining independent stale-state resets and
    team-bound loads. Direct tests and a dedicated architecture gate require
    one root pipeline and reject every legacy recruitment wrapper. No market,
    shortlist, contract, budget, season or club state changes, and no match,
    recruitment transaction, future, training, study, participant session or
    external call is executed. Focused Web regression passes 66 tests,
    CI-equivalent regression passes 1,482 with three skips and one registered
    deselection, and local HTTP, accessibility, recovery and all 97
    architecture checks are green.

75. Replaced three implicit season-render overrides with one frozen, named
    four-stage composition. Base season state, matchday command and manager
    career, club situation/timeline/decision evidence, and counterfactual
    futures plus world navigation now share one visible order and one input
    flow. Direct tests and a dedicated architecture gate require one root
    pipeline and reject every legacy season wrapper. DOM ownership, advice
    reset, preview scheduling and claim boundaries remain unchanged; no match,
    future, training, study, participant session or external call is executed.
    Focused Web regression passes 67 tests, CI-equivalent regression passes
    1,483 with three skips and one registered deselection, and local HTTP,
    accessibility, recovery and all 98 architecture checks are green.

76. Replaced three implicit evidence-library overrides with one frozen, named
    four-stage composition. Base match/pair/fork/study cards, deterministic
    branch identity, unified counterfactual-future summaries and
    multi-timepoint future sets now share one visible order and input
    projection. Direct tests and a dedicated architecture gate require one root
    pipeline and reject every legacy library wrapper. Filters, list ordering,
    artifact links, resume controls and non-causal boundaries remain unchanged;
    no match, future, training, study, participant session or external call is
    executed. Focused Web regression passes 68 tests, CI-equivalent regression
    passes 1,484 with three skips and one registered deselection, and local
    HTTP, accessibility, recovery and all 99 architecture checks are green.

77. Replaced two implicit future-mechanism example overrides with one frozen,
    named three-stage composition. Base action/window evidence, independent
    cross metrics and V2 policy-signal semantics now share one visible order.
    A per-call context binds only the details node created for the current
    examples, preventing an empty scenario set from enhancing an older sibling.
    Direct tests and a dedicated architecture gate require one root, exact
    order, null context initialization and explicit node binding while rejecting
    every legacy wrapper. No match, future, training, study, participant
    session or external call is executed. Focused Web regression passes 69
    tests, CI-equivalent regression passes 1,485 with three skips and one
    registered deselection, and local HTTP, accessibility, recovery and all 100
    architecture checks are green.

78. Replaced two implicit official-action execution overrides with one frozen,
    named three-stage composition. Base retained-action and direct ball-event
    identity evidence, V2 policy-signal semantics and the full retained-record
    summary now share one visible order across matchday intelligence and the
    decision ledger. A per-call context binds only the details node created for
    the current evidence, so unavailable evidence cannot mutate an older card.
    Direct tests and a dedicated architecture gate require one root, exact
    order, null context initialization and two bound semantic stages while
    rejecting every legacy wrapper. No match, future, training, study,
    participant session or external call is executed. Focused Web regression
    passes 70 tests, CI-equivalent regression passes 1,486 with three skips and
    one registered deselection, and local HTTP, accessibility, recovery and all
    101 architecture checks are green.

79. Replaced two implicit retained-action text reassignments with one frozen,
    named three-stage pure-text composition. V2 action/signal distribution,
    V3 locally attributable transitions and V4 exact shared-sampler expectation
    now share one order across official cards, decision ledgers, world
    trajectories and history chapters. A single text context also fixes the
    previous missing-input path that could return an orphaned V4 legacy marker.
    Direct tests and a dedicated architecture gate require one root, exact
    order, empty-input guards and final context return while rejecting every
    legacy wrapper. No match, future, training, study, participant session or
    external call is executed. Focused Web regression passes 71 tests,
    CI-equivalent regression passes 1,487 with three skips and one registered
    deselection, and local HTTP, accessibility, recovery and all 102
    architecture checks are green.

80. Replaced two implicit manager-world action-adoption ledger reassignments
    with one frozen, named three-stage renderer. Base adoption metrics and
    state navigation, descriptive downstream world propagation, and bounded
    official-action semantics now share one visible order. Direct tests and a
    dedicated architecture gate require one root and reject the complete legacy
    wrapper family. Missing evidence remains fail-closed, and downstream state
    remains descriptive rather than causal. No match, future, training, study,
    participant session or external call is executed. Focused Web regression
    passes 72 tests, CI-equivalent regression passes 1,488 with three skips and
    one registered deselection, and local HTTP, accessibility, recovery and all
    103 architecture checks are green.

81. Replaced the implicit manager decision-preview and world-model advice
    reassignments with two frozen, named two-stage pipelines. Deterministic
    manager-impact facts now precede the bounded world-model comparison in one
    visible order; advice status reset precedes identity-bound advice content
    in another. Direct tests, two architecture gates and accessibility
    verification reject both legacy wrapper families. Authority, adoption and
    non-causal wording are unchanged. No match, future, training, study,
    participant session or external call is executed. Focused Web regression
    passes 74 tests, CI-equivalent regression passes 1,490 with three skips and
    one registered deselection, and local HTTP, accessibility, recovery and all
    105 architecture checks are green.

82. Replaced the implicit manager squad, matchday command and career
    reassignments with three frozen, named pipelines. Roster state now precedes
    in-match rules; the command center precedes intelligence and club strategy;
    career/history state precedes recruitment configuration and archived
    transactions. Direct tests and three architecture gates require one root
    per surface and reject all three legacy wrapper families. Existing
    unavailable paths and product boundaries are unchanged. No match, future,
    recruitment transaction, training, study, participant session or external
    call is executed. Focused Web regression passes 77 tests, CI-equivalent
    regression passes 1,493 with three skips and one registered deselection,
    and local HTTP, accessibility, recovery and all 108 architecture checks are
    green.

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

1. When training is explicitly authorized, produce one frozen M2 checkpoint
   with the joint changing-action utility objective and support-qualified
   continuation runtime contract across both high-level action choice and
   receiver/target ranking, then require every registered development and sealed
   gate before the 360-run study. Do not retrofit the new authority onto the
   current v9 checkpoint.
2. Keep the completed `SocietyAgent` facade closed: new governance, tactics,
   reflection, condition, psychology, memory, social or match behavior belongs
   to its existing dedicated layer and must not re-enter the facade.
3. Collect real failed-pass labels for an external completion benchmark.
4. Keep the completed `TournamentManager` boundaries closed: lifecycle owns
   scheduling/advancement/reflection, state owns checkpoint/restore, the match
   service owns execution stages, and standings/bracket/referee logic stays in
   its existing domain modules.
5. Keep the matched-seed cognition, narrative, memory and meta-learning
   authority tests as release invariants. Do not promote any meta proposal
   without a replayable current-identity evaluation receipt.
6. Fit macro intensity and micro action parameters against held-out event data with uncertainty intervals.
7. Keep v7 deployed and v6 as rollback; v8.9 remains evaluated until continuous-time LODO beats the continuous baseline.

## Release Gate

A change is accepted only when deterministic replay holds, state is finite and
bounded, score ownership is singular, generated content cannot override facts,
checkpoints survive interruption, and matched-seed ablations show no unexplained
cross-layer changes.

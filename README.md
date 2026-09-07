# Generative Football Society

Football world-model research platform with a stable World Cup-scale simulator.

The project is research-first: learned world models and LLM cognition are
evidence-gated, default-off advisory layers. The deployed simulator remains
v7.0.0 until an explicit release manifest changes the production pointer. See
`docs/RESEARCH_DIRECTION.md` and
`data/evaluation/research_evidence_v1.json` for the frozen evidence boundary.
The completed formal baseline, seven-variant ablation, retrained-candidate
decision, and final verification are summarized in
`docs/FORMAL_RESEARCH_VALIDATION_2026-08.md`.

The historical pass-only action-policy controller closed one narrower gap: its
learned signal changed 25 sampled micro-actions in a fixed 24-run mechanism
study and changed behavior in all 30 pairs of a 60-run full-match study. It did
**not** establish outcome improvement. V3.63 replaced that controller with a
validated feasible multi-action simplex, so those sealed results are retained
as history but fail current-code identity verification and are not evidence for
the new controller. V3.64 adds a distinct cross-transition validation contract,
cross-specific runtime authority and replayable cross trajectories. The current
v9 checkpoint contains no such cross evidence, so cross authority correctly
remains zero until a future checkpoint passes the registered grouped-heldout
gate. M1 remains default-off. See `docs/ACTION_OUTCOME_RESULT_V1.md` and
`docs/ARCHITECTURE_V2.md` sections 81--92. V3.65 also makes the joint
world-model shot head diagnostic-only: direct shot authority requires an
independently sealed frozen head that beats the physics xG prior.
V3.66 makes hold a strict counterfactual reference action. Probability shifted
to hold after another validated action is suppressed is reported as indirect
redistribution, never as an independently learned hold recommendation or
direct adoption.
V3.67 closes the product replay contract: identity-bound cross trajectories
are accepted as direct official-match evidence, while hold alone remains
trajectory-free. The same official execution record now carries direct
preference, suppression-only and hold-reference semantics into match reports
and the manager timeline.
V3.68 extends the same contract through counterfactual propagation: cross is
an identity-matchable local action, future windows count crosses separately,
and V2 mechanism examples retain policy-signal and hold-reference semantics
while legacy V1 examples remain readable.
V3.69 closes the remaining aggregate gap: synchronized comparison totals,
season ledgers and manager world threads now retain cross, direct-preference,
suppression-only and hold-reference counts instead of collapsing them into a
generic mechanism-example total.
V3.70 carries those counts through identity-checked historical chapters and
the chronological season trajectory, with explicit zero normalization for
legacy records and fail-closed disagreement checks for current evidence.
V3.71 adds a separate bounded semantic ledger for official-match examples.
It exposes cross, direct, suppressive and hold-reference evidence while keeping
sample truncation visible and refusing to treat at most five examples as the
full distribution of official action decisions.
V3.72 adds the distinct full retained-record semantic aggregate. It partitions
all retained official actions and signal modes independently of the bounded
examples, preserves V1 evidence, exposes upstream truncation explicitly and
carries the identity-bound totals through decision ledgers, world history,
season trajectories and Studio without authorizing outcome attribution.
V3.73 adds the missing counterfactual transition structure: complete baseline
to actual action matrices and their strictly locally attributable subset. This
makes concrete changes such as hold-to-pass visible throughout the manager
product, preserves V1/V2 history, and fixes the decision summary's retained-
record count without converting simulator-local changes into outcome claims.
V3.74 connects each locally attributable transition to same-chapter descriptive
match and persistent-state facts. Transition strata may overlap, preserve
source coverage and historical absence, and are explicitly barred from effect
comparison, ranking or outcome attribution.
V3.75 turns those transition totals into an accessible, horizontally responsive
5-by-5 manager-facing map. It exposes V3 and legacy coverage beside the matrix,
uses an explicit unavailable state instead of treating missing evidence as zero,
and keeps every cell a descriptive count rather than an effect estimate.
V3.76 corrects the expected action-change estimator itself. Instead of treating
total-variation distance as the change probability, it computes exact interval
overlap under the simulator's shared inverse-CDF draw; TV remains a separate
distribution diagnostic and legacy product evidence is labelled as a proxy.
V3.77 closes the projection gap: current exact-estimator runs produce V4
official evidence, carry exact expectation through the decision ledger, world
thread and season navigator, and report legacy/missing coverage without
upgrading historical V1–V3 records or substituting zero for absence.
V3.78 closes the product drill-down gap: V4 chapters remain eligible for
V3-compatible transition propagation, and every evidenced non-zero action-map
cell can open its same-chapter downstream world facts and latest source chapter
with keyboard and screen-reader semantics. These remain descriptive,
overlapping strata rather than cross-cell effect estimates.
V3.79 upgrades credential closure from a DeepSeek-only attestation to a V2
all-known-incident contract covering both DeepSeek and the exposed GitHub
Classic PAT. Each incident requires its own redacted receipt; partial closure,
receipt reuse, stale commit scans and three known token families fail closed.
Refreshing the evidence chain also correctly invalidates the old paper-package
pass because post-experiment action-policy code changed; no result is relabelled
or regenerated.
V3.80 isolates runtime stochasticity by identity. The root seed now derives a
match seed and named referee, scoring, dialogue, physical-wear, carryover,
agent and bracket streams. The legacy journey simulator follows the same
contract. A machine audit rejects direct process-global random draws across
simulation and memory runtime modules, so an unrelated extra draw cannot
silently change another subsystem's future.
V3.81 extends that identity across process restarts. Tournament checkpoint V2
stores the random-world contract and root seed under a canonical content hash.
Resume reconstructs the world with the stored seed, rejects an explicitly
conflicting seed, and refuses unsafe V1 checkpoints that never recorded their
random-world identity. Public tournament and micro APIs now forward their seed
into world construction instead of only reseeding process-global generators;
custom project roots also reach the manager's checkpoint repository.
V3.82 upgrades that boundary to checkpoint V3 and manifest V2. Resume now
binds the portable identity of Python source contents, immutable data and model
artifacts, interpreter/platform, structured simulation configuration and a
secret-filtered, value-hashed runtime option set. Identity drift fails before provider
resolution or match mutation; V1/V2 checkpoints remain inspectable but cannot
be guessed into a current run. Evolving carryover is bound separately by each
checkpoint; complete Agent/social-state snapshots remain a later gate.
V3.83 closes that in-memory gate with checkpoint V4: mutable Agent psychology,
memory, beliefs, tactics, coach adaptations, the social feed, topic market and
narrative history are versioned and restored only after a full identity
preflight. Reflection is a receipted two-phase operation, so a persisted model
response is replayed without a second provider call and cannot be applied
twice; skipped finals also reconstruct their champion record. Carryover,
fusion history, counterfactual evidence and enabled cognitive caches are bound
as evolving external state. A legacy roster `NaN` can no longer enter match
math: non-finite observations become missing fields and incomplete team
dynamics use the status-derived fallback.
V3.84 upgrades tournament persistence to checkpoint V5. Every checkpoint now
contains a bounded, integrity-checked rollback image of its causal external
files and enabled cognitive cache. On explicit resume, the application first
reconstructs and verifies the code/data/model/configuration identity, then
idempotently restores internal external-state targets before rebuilding the
world. A durable recovery journal preserves the displaced state and lets an
interrupted rollback converge on the next attempt. Snapshot paths cannot
escape the project, symlinks and malformed payloads fail closed, and a cache
configured outside the project is never overwritten automatically.
V3.85 hardens that recovery path: an existing recovery journal is validated
against the requested checkpoint even when live files already match, so a
conflicting interrupted transaction cannot be silently archived. The public
tournament entry parses an existing V5 checkpoint once and passes the same
validated object through seed selection, manifest verification, external
rollback and in-memory restore; ordinary non-resume integrations retain their
previous call signature.
V3.86 corrects release-readiness semantics. The verified Linux target hash lock
is no longer made false merely because the separate formal paper result is
pending. The control plane now reports `code_ready=true` only when every
infrastructure/protocol gate passes and every paper-package check except the
authorized confirmatory-result cell passes; final `release_ready` remains
false until all external evidence exists. Studio CLI and Web expose the same
distinction, open-gate count and next action.
V3.87 aligns the M2 training, validation, planning and realized-outcome target
around actor-centred transition utility, while keeping the historical M1
result immutable and default-off. V3.88 exposes the frozen M2 preflight,
candidate qualification, 360-match budget, resumable progress and decision as
one identity-bound five-stage control plane; it is currently waiting for a new
candidate and has executed zero formal M2 matches.
V3.89 makes LLM transport evidence-bearing through explicit provider adapters,
retry-stable request IDs, prompt-free token/cost telemetry and a
closed/open/half-open circuit breaker. Match-scoped counters prevent concurrent
cognitive runs from claiming each other's calls, and provider evidence is now
visible in reports and required by the prospective pilot.
V3.90 binds effective squad construction to the caller's project root. Missing
rosters use a visible deterministic fallback, while malformed or
XI-incomplete rosters fail closed; exact roster provenance reaches match and
product reports.
V3.91 completes that root boundary across tournament regulation, extra time,
micro replay and affective replay. A shared finite, bounded signal contract now
keeps each team's coordination and conflict inputs side-specific; the away
team can no longer inherit the home team's conflict state.
V3.92 replaces prospective M2's manually bounded code identity with a
deterministic local-import closure. The zero-training preflight now binds 241
Python dependencies, including the real match runner and its side-isolation
contract, while completed historical studies retain their original identities.
V3.93 makes a selected physics-official score path an enforceable contract:
regulation and extra time now reject missing, non-finite, supplemented or
source-mismatched micro evidence and stop before score commit. Macro scoring
remains available only when the caller explicitly selects a macro score path.
V3.94 upgrades tournament persistence to checkpoint V6 and makes each public
`play_match` call one workspace-serialized transaction. It persists a complete
pre-match checkpoint, verifies exactly one matching in-memory and durable
result on success, and restores the durable checkpoint, manager world and
project-owned carryover, fusion, counterfactual, cognitive, ball-log and
narrative-debug state on any
catchable failure. Missing commits are failures, concurrent writers and
external output targets fail before match execution, and rollback receipts
record bounded error types without exception messages or credentials.
V3.95 extends that same operating-system lease across public tournament
startup, identity verification, resume recovery, world construction and the
complete tournament run. Per-match transactions reuse the context-owned lease
without reacquiring it; a second process fails before it can read evolving
state, restore files or write a run manifest.
V3.96 makes Studio restore crash-convergent. A durable recovery journal binds
the verified backup, displaced live state and replacement boundary so an
interrupted restore resumes or rolls back without inventing a second world.
V3.97 ships the production-validation runner in the read-only container and
separates fixed workload execution, deployment attestation and finalization.
The 100-match restart drill remains preregistered and unexecuted.
V3.98 replaces manually trusted human-study rows with an authoritative
pseudonymous session registry, role-stratified permuted-block allocation,
concrete scored product-value cases and byte-verified content-addressed
evidence. Comparative correctness is recomputed from frozen structured
receipts; no participant session has been executed.
V3.99 physically separates participant case content from a case-hash-bound
moderator scoring seal and connects deterministic blinded packet delivery to
both the Studio CLI and authenticated Web evidence workspace. Packets exclude
moderator identity and scoring material, use confined atomic no-overwrite
materialization, and expose only aggregate status. The evidence kit excludes
the seal. This is still zero-participant preparation, not user-value evidence.
V4.00 adds a separate capability-bound participant service. It requires an
external session directory, runs an unscored tutorial, starts the 900-second
clock only when the current case is revealed, withholds every future case and
correctness result, writes content-addressed structured receipts, and requires
an explicit moderator attestation before importing a completed record. The
participant application mounts no Studio or repository routes. This makes the
study end-to-end executable but does not execute it or create user-value
evidence. Completion is state-first and restart-recoverable, submission-ID
retries require exact receipt-backed answers, and remote delivery remains
loopback-bound behind a same-host trusted HTTPS proxy.
V4.01 makes the match action-adoption boundary structurally explicit. The
previous monolithic action step is now a small clock/context/sample/execute
orchestrator, while pass, shot, cross and hold resolution and policy evidence
recording have separate internal stages. The probability equations, one shared
uniform draw, feasible-action masking, scheduled override semantics and
world-model adoption fields are unchanged; this is a behavior-preserving code
quality increment, not new model or outcome evidence.
V4.02 applies the same discipline to the Web security gateway. Liveness,
Host/HTTPS enforcement, authentication, GET and POST routing, CSRF-gated
mutation, task lookup, readiness and method-error classification now have
separate boundaries. The endpoint set and response contracts are unchanged;
this is product maintainability evidence, not a production-deployment claim.
V4.03 decomposes the matching Studio status response into safe artifact links,
typed evidence-library projections and the manager future/world-navigation
assembly. Malformed task plans retain their prior fail-soft behavior, causal
claim flags remain false, and the returned JSON schema is unchanged.
V4.04 closes the remaining focused complexity debt in the Web product. Match,
paired-match, single-fork and fork-set requests now share one strict fixture
validator, while public result links, tactical-study progress and fork-set
progress have separate fail-soft projections. The whole Web product and the
action-adoption core pass the focused complexity, branch and statement gate;
CI and the immutable reproduction verifier require that exact gate. This is a
maintainability and reviewability increment only, not new model, match, causal,
user-value or deployment evidence.
V4.05 hardens the current-code action-adoption experiment boundary without
executing it. One progress validator now protects status, authorized resume,
analysis and reviewer export. It requires the exact frozen fixture/sample
prefix and arm order, finite analysis metrics, nested action-count bounds,
consistent completion markers and the fixed 24-run budget. Historical evidence
remains structurally valid but identity-stale after controller refactoring, so
it cannot be replayed as a current result. An identity-matched completed ledger
returns without rebuilding a match environment, and an incomplete arm cannot
advance into the next arm.
V4.06 applies the same fail-closed standard to the downstream 60-run full-match
outcome experiment. Its ledger accepts only the frozen M0-then-M1 schedule,
finite calibration and action metrics, zero world-model activity in M0, exact
arm completion and the fixed budget. Execution identity now includes the
observable calibration contract and both external baseline inputs. Starting
the simulations or writing their decision requires the exact explicit
authorization phrase declared in the reproduction manifest; status and result
verification remain read-only. Historical rows pass structural validation but
remain identity-stale and cannot support a current promotion claim.
V4.07 closes three remaining credential-evidence loopholes before any user
receipt is accepted. Distinct incident paths must also have distinct content
hashes, each non-empty receipt is capped at 10 MiB, and timezone-aware
revocation times must not follow the signature or lie in the future. The
protocol audit still performs zero provider calls and cannot assert that any
credential was revoked without genuine redacted provider receipts.
V4.08 continues the gradual `SocietyAgent` facade split at a behavior-preserving
boundary. Latent psychology initialization, state projection, appraisal,
emotion, coping, salience and the compatibility state views now belong to a
dedicated mixin. The 15 moved definitions are AST-equivalent to their previous
implementations, while structural and finite/probability-invariant tests keep
the ownership explicit. The prospective M2 preflight was refreshed without
training so its transitive code identity includes the new runtime module.
V4.09 completes that facade decomposition. Governance/referee behavior,
tactical controls, reflection/cognitive ingestion and physical condition now
belong to four dedicated mixins containing the remaining 22 behavior methods.
The 127-line facade owns only construction, finite-value normalization and
identity-derived region/style helpers. A machine AST ownership gate prevents
those responsibilities from drifting back into the facade, and the refreshed
M2 receipt remains zero-training readiness evidence only.
V4.10 applies the same cohesion rule to `TournamentManager` without creating
more feature fragments. Full-tournament scheduling, advancement and receipted
reflection now live in one lifecycle layer; run identity, checkpoint persistence
and world restoration live in one state layer. The 10 moved methods are AST-
equivalent, the facade shrinks from 404 to 130 lines, and its existing match,
referee and score adapters remain compatible. The refreshed M2 identity is
again zero-training readiness evidence, not a model or tournament result. The
full regression exposed and closed a shared-host timing flaw in the real
process-kill recovery drill: functional recovery remains fail-closed under a
bounded 45-second readiness timeout, while the observed 15-second local target
is now reported separately and cannot masquerade as data-integrity failure.
The complete suite passes with 1,411 tests and three declared skips.
V4.19--V4.22 align prospective M2 training, grouped development evidence,
sealed qualification and runtime control at the same explicit two-action
depth. Runtime continuation branches now come only from grouped development
support (minimum 32 samples, four matches and 95% covered mass) and carry
leakage-cleaned action prototypes. On the current development split this keeps
`pass -> pass` and `pass -> shot`, while excluding the five `pass -> hold`
and four `pass -> cross` examples. M2 compares the supported sequence against
the exact zero-transition utility baseline and cannot fall back to one-step or
unsupported hold rollouts. The same sequence gate now controls both high-level
pass utility and executable receiver/target ranking; M2 cannot silently use
one-step values after selecting pass. The V4 protocol amendment and 16-check
preflight are still zero-training readiness evidence: no M2 candidate or
efficacy result exists yet. The complete regression passes 1,471 tests with
three declared skips.
V4.23 closes a direction-of-play leakage gap in that prospective controller.
Observations and action targets use absolute pitch coordinates plus an
`attacking_home` flag, so pooled home/away continuation prototypes were not
valid runtime controls. Development evidence is now partitioned before
qualification: the home perspective retains 886 passes and 38 shots, covering
99.46% of 929 pass continuations, while the away perspective retains only 763
passes, covering 97.20% of 785; its 18 shots are below the registered
32-sample gate. The supported pass target x-coordinate is 0.2272 for home and
0.4532 for away instead of one pooled prototype. Training, sealed validation,
runtime planning, candidate qualification and receipt replay all bind the same
side. Pooled metrics remain diagnostic, both sides must independently pass,
and legacy checkpoints without perspective evidence receive zero M2
authority. Protocol amendment V5 precedes candidate binding, training and all
formal runs. The refreshed zero-training preflight passes 18/18 checks, binds
250 Python files, loads zero sealed rows and writes no checkpoint. The complete
202-file regression passes 1,474 tests with three declared skips.
V4.24 fixes the remaining chronology break in the manager-facing contribution
story. Once a season had one completed fixture, the earlier projection always
preferred that history row and could hide the active next-fixture intervention.
The V2 story now keeps the active fixture and its five-stage workflow in the
primary rail, binds its current chapter identity, and retains exactly one latest
completed predecessor as a compact same-chapter evidence link. Studio opens the
predecessor only when it is a completed navigable chapter; an active identity is
never sent to the historical navigator. Completed seasons still show their
latest completed world, while an empty completed season fails closed. This is a
read-only product projection change: it executes no match, future generation,
training, participant session or provider call and creates no causal claim.
Focused story, navigator and Web regression passes 110 tests; CI-equivalent
regression passes 1,475 tests with three declared skips and one registered
training-test deselection. Local HTTP, accessibility, recovery and all 92
architecture checks pass.

V4.25 replaces the manager-world page's implicit renderer override chain with
one frozen, named 13-stage composition. Base navigation, the active-first
story, action-adoption evidence, trajectory, reviewed futures, archive,
certificates, retained semantics, transition propagation, the action matrix,
exact shared-draw expectation and drill-down now execute in one reviewable
order. The DOM contract and stage order are unchanged, while tests and the
architecture audit reject either legacy wrapper family or a second root
renderer. This is a behavior-preserving product maintainability increment. It
executes no match, future generation, training, participant session or provider
call and creates no model, causal or outcome evidence. Focused Web regression
passes 60 tests; CI-equivalent regression passes 1,476 tests with three
declared skips and one registered training-test deselection. Local HTTP,
accessibility, recovery and all 92 architecture checks pass.

V4.26 applies the same explicit composition boundary to the manager decision
ledger. Twelve successive root overrides become named stages in one frozen
13-stage order spanning advisor evidence, runtime execution, reviewed futures,
local mechanisms, official actions, world evolution, retained semantics and
exact shared-draw expectation. The base ledger, entry order, DOM ownership and
evidence wording remain unchanged. Web, accessibility and architecture gates
require exactly one root ledger renderer and reject every legacy ledger
wrapper. This is a behavior-preserving product cohesion increment with zero
match, future, training, study, participant or provider execution. Focused Web
regression passes 61 tests; CI-equivalent regression passes 1,477 tests with
three declared skips and one registered training-test deselection. Local HTTP,
accessibility, recovery and all 92 architecture checks pass.

Current deployed simulator: **v7.0.0**. Frozen rollback: **v6.0.0**. See
`data/releases/current.json` for the only authoritative deployment pointer.

GFS combines:

- team status and historical memory
- multi-agent psychology, beliefs, media pressure, and tactics
- macro xG dynamics
- optional 6-second micro match physics
- optional LLM cognition through an OpenAI-compatible API
- optional latent world model planning

## Project Layout

```text
gfs.py                 Unified CLI entrypoint
src/app.py             Stable Python integration API
src/cli.py             CLI command implementations
src/simulation/        Tournament, agents, fusion, LLM, match pipeline
src/match_engine/      Micro physics, spatial, passing, shooting, cognition, WM
src/memory_engine/     Status, macro xG, Poisson, tournament probability
src/data_engine/       Raw data, roster, coach, FM import helpers
data/                  Required bundled datasets and model checkpoint
docs/                  Design and technical references
scripts/               Maintenance, calibration, benchmark, and legacy scripts
tests/                 Unit and smoke tests
```

Generated outputs are ignored by git:

- `outputs/`
- `reports/`
- `data/persistence/`
- `data/cache/`
- `data/world_model/traces/`

## Install

```bash
pip install -r requirements.txt
```

Recommended Python: 3.10+.

The Python wheel contains code, while datasets, release manifests, checkpoints,
and generated reports remain in a GFS workspace. When using the installed
`gfs` console command outside the repository, set `GFS_PROJECT_ROOT` or pass
`--base-dir` explicitly. Running inside the workspace is discovered
automatically.

## Unified CLI

For the cohesive product workflow, the normal path is one command:

```bash
python gfs.py studio run --name "My World Cup" --mode stable --seed 42 --home Brazil --away Argentina --fast
```

Studio modes make the product boundary explicit: `stable` uses the deployed
simulator, `research` adds the accepted calibrated world-model candidate, and
`cognitive` additionally requires real LLM credentials. Each match produces a
single JSON audit and HTML dashboard. The persisted workflow exposes one of
`blocked`, `ready_to_run`, `running`, or `review`, including the next
valid action. The lower-level `init`, `status`, and `match` commands remain
available for operators. See `docs/PRODUCT_STUDIO.md`.

Run the same cohesive workflow in the loopback-only Web Beta:

```bash
python gfs.py studio web
```

Open `http://127.0.0.1:8765`. The Web layer adds no second business path: it
uses the persisted Studio state machine and evidence gates directly. See
`docs/WEB_BETA_ACCEPTANCE.md` for its verified security/accessibility scope and
the production capabilities that are still deliberately unclaimed.

The focused accessibility contract is reproducible without training, a match,
or provider access:

```bash
python scripts/verify_web_accessibility.py --out data/evaluation/web_accessibility_verification_v1.json
```

Matches submitted from the Web page run as persistent, idempotent background
tasks. Refreshing the page recovers task observation instead of duplicating a
match; interrupted process state remains visible and is not automatically
replayed.

The Research Lab also exposes one cohesive world-model causal fork. A single
request atomically runs a prediction-only baseline (`MATCH_WM_PLAN=0`) and a
quality-gated action-policy treatment (`MATCH_WM_PLAN=1`) with the fixture,
tactics, checkpoint, fast/full configuration, and seed held fixed. It returns
both match reports and one synchronized comparison dashboard through the same
recoverable task and Evidence Library. This is simulator-local intervention
evidence for one fixture/seed, not real-football causality, population-level
efficacy, or permission to promote M1 beyond research-only status.

The comparison dashboard separates four evidence stages: isolated policy
assignment, realized sampled-action changes, direct runtime-identity links,
and 30/120-second shared-clock event windows. Only a uniquely linked local
action can inherit simulator-policy attribution. Later pass, shot, goal, and
turnover deltas remain descriptive because the trajectories have diverged and
additional policy changes may occur inside a window.

Authenticated remote deployment is explicit and separate from provider
credentials. The supplied Compose contract exposes only a Caddy TLS edge and
keeps GFS on an internal network:

```bash
docker compose --env-file deploy/.env.deploy -f deploy/compose.yaml up -d --build
```

Docker is not available in the current verification environment, so this is a
statically verified deployment contract rather than a claimed running service.
See `deploy/README.md`.

Create and verify an integrity-checked Studio recovery bundle:

```bash
python gfs.py studio backup --out backups/studio.zip
python gfs.py studio verify-backup backups/studio.zip
python gfs.py studio restore backups/studio.zip --replace
python gfs.py studio recover
```

Restore refuses to replace any existing product artifact unless `--replace`
is explicit. An activated multi-file restore owns a durable journal; Web
startup resolves it before task recovery, and `studio recover` provides the
same fail-closed operator path after an ungraceful CLI interruption.
See `docs/STUDIO_RECOVERY.md` for scope, trust boundaries, and the still-
unmeasured production RPO/RTO.

Exercise real local process death and restart without a match:

```bash
python scripts/verify_process_recovery.py --out data/evaluation/process_recovery_verification_v1.json
```

This verifies application recovery on a temporary filesystem; it is not a
container/deployment-volume RPO/RTO claim.

The historical confirmatory paper experiment is frozen separately from
exploratory work. Its default command is read-only:

```bash
python scripts/run_formal_experiment.py
```

Its frozen design compares only sealed M0 and M1 and caps compute at 30 matched
pairs / 60 runs. It must not be executed for V3.63; the changed controller
requires a new preregistered protocol and explicit authority. See
`docs/FORMAL_EXPERIMENT_PROTOCOL_V2.md`.

The historical paper package can be audited without training or simulation:

```bash
python scripts/verify_paper_package.py --out data/evaluation/paper_package_verification_v1.json
python scripts/verify_reproduction_release.py --out data/evaluation/reproduction_release_verification_v1.json
```

After the V3.63 controller change these commands intentionally fail the formal
result identity gate until a new protocol is executed; they must not be made to
pass by rewriting the old experiment identity. See `docs/PAPER_DRAFT.md` and
`docs/REPRODUCTION_GUIDE.md`. The release audit verifies two 64-package hashed
reference
profiles (Python 3.12 Linux deployment and Python 3.13 Windows development),
deterministic CycloneDX 1.6 SBOMs, an imported local runtime, immutable Python
and Caddy OCI digests, and a default-deny registry for all known external data
sources. A deterministic, content-addressed supplement now approves only the
CC BY 4.0 IDSSE/Sportec-derived subset and excludes the other four reviewed
sources. A real container build, confirmatory results, and independent
reproduction remain open.

Show top team status rankings:

```bash
python gfs.py status --top 20
```

Run a fast single micro match:

```bash
python gfs.py micro --home Brazil --away Argentina --fast --seed 42
```

Run lightweight Monte Carlo:

```bash
python gfs.py monte-carlo -n 1000 --seed 42
```

Run full tournament:

```bash
python gfs.py tournament --seed 42
```

Run full tournament with micro physics as official score:

```bash
python gfs.py tournament --micro --micro-score --seed 42
```

Resume from checkpoint:

```bash
python gfs.py tournament --resume
```

## Python API

```python
from src.app import load_status_table, run_micro_match, run_monte_carlo

stats, team_matches, data = load_status_table()
summary = run_micro_match("Brazil", "Argentina", fast=True, seed=42)
rankings = run_monte_carlo(1000, seed=42)
```

## LLM Configuration

LLM features require a valid OpenAI-compatible endpoint. For current DeepSeek
V4, use a newly rotated key only; never put a real credential in this file:

```env
DEEPSEEK_API_KEY=replace_with_rotated_key
BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-v4-flash
GFS_SEED=42
```

The gateway also accepts `API_KEY` or `OPENAI_API_KEY` for other compatible
providers. Studio preflight never sends a provider request and reports only
the credential variable name, never its value. Without a real key, non-LLM
commands and deterministic tests still run.

Product and academic maturity are governed by the evidence-gated roadmap in
`docs/EXCELLENCE_ROADMAP.md`. Architecture quality alone does not count as a
finished product or completed paper.

Product operations use a bounded, privacy-allowlisted telemetry stream shared
by Web requests, persistent tasks, and recovery. Aggregate metrics are visible
inside Studio; raw events and credentials are never exposed through the Web
API. Run the zero-match control-plane soak with:

```bash
python scripts/verify_product_operations.py --tasks 100
```

This verifies task operations only and does not count as the required
100-match product soak. See `docs/PRODUCT_OPERATIONS.md`.

Authenticated remote Studio sessions use the `__Host-` Cookie contract,
absolute/idle expiry, bounded capacity, selective logout, and client plus global
login limits. These controls are process-local and currently support the
single-application deployment contract, not horizontal replicas.

Backup creation, verification, and explicitly confirmed transactional restore
are integrated into the same Studio Web surface. The browser works only with
server-generated backup IDs and never supplies a filesystem path. Validate the
no-match recovery workflow with `python scripts/verify_web_recovery.py`.

### LLM Architecture

LLM roles remain independent, but share one provider gateway for connection reuse,
retry policy, and call accounting:

```text
coach / player / media / critic roles
                 |
           shared LLMGateway
                 |
       GenerationPipeline + audit
                 |
   tactics / narrative / meta-learning
```

Generated atmosphere is a causal `NarrativeEvent`, not disposable prose. Its
bounded signals affect coach pressure, crowd hostility, player anxiety, social
feedback, memory, and later matches. Raw model output and the publishable version
are stored separately in generation audits.

Reflection remains a meta-learning proposal mechanism, but reflection never
grants parameter authority by itself. `MetaLearningController` first stores a
bounded, identity-bound shadow proposal at the parameter's fast, medium, or
slow time scale. Later matches add descriptive observations without treating
them as effect evidence. A parameter change requires a separately replayable
matched-seed evaluation receipt with at least eight unique paired units and a
95% lower bound above the registered non-negative minimum effect.

Proposal state persists across matches with the existing society state and is
shown in the manager's existing world-evolution thread. The public projection
contains only proposal identity, bounded parameter direction, observation
progress, evaluation summary, authority state, and next required evidence.
Reflection text, memory evidence, and matched rows remain private, and the Web
view cannot authorize a change. These contracts establish software behavior;
they do not establish improved actions, match outcomes, or real-football
causality.

## Legacy Entrypoints

Older root scripts were moved to `scripts/legacy/`. Prefer `python gfs.py ...` for new usage.

## Tests

```bash
python -m pytest tests -q
```

Slow calibration and benchmark scripts live under `scripts/`.

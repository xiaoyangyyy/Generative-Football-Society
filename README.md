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
deterministic local-import closure. The zero-training preflight now binds 240
Python dependencies, including the real match runner and its side-isolation
contract, while completed historical studies retain their original identities.
V3.93 makes a selected physics-official score path an enforceable contract:
regulation and extra time now reject missing, non-finite, supplemented or
source-mismatched micro evidence and stop before score commit. Macro scoring
remains available only when the caller explicitly selects a macro score path.

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
```

Restore refuses to replace an existing session unless `--replace` is explicit.
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

Reflection remains a meta-learning mechanism. `MetaLearningController` applies
LLM proposals at separate time scales: tactical controls are fast variables,
role dynamics are medium variables, and recurrent weights are slow variables.
Every update records its proposed delta, applied delta, before/after values, and
verified memory evidence.

## Legacy Entrypoints

Older root scripts were moved to `scripts/legacy/`. Prefer `python gfs.py ...` for new usage.

## Tests

```bash
python -m pytest tests -q
```

Slow calibration and benchmark scripts live under `scripts/`.

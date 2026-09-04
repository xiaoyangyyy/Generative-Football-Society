# GFS Studio Product Surface

GFS Studio turns the simulator, world model, cognition layer, and evidence
reports into one persistent user workflow. The lower-level research scripts
remain available, but product users work through a studio session.

## Product loop

```text
configure/load -> verify evidence -> start persistent season
    -> freeze season/player commitments -> submit matchday decision
    -> advance/resume matchday -> debrief -> repeat
    -> board review + sporting plan -> next season
```

This is a derived state machine over persisted product evidence, not just
documentation and not a second mutable season tracker. `studio status` returns
one authoritative next action, a five-step season journey, completed and failed
counts, and progress flags. The normal guided route is the persistent season;
an isolated match remains available as an explicitly labeled alternative for
observation or tactical laboratory work.

The Web surface progressively discloses that complete system through four
stable work areas rather than one unbounded page:

- **Manager career** contains the season, matchday, lineup, club and long-term
  career loop.
- **Match laboratory** contains isolated observation, paired comparisons and
  fixed-budget tactical studies.
- **Evidence center** contains world-model adoption evidence, retained
  artifacts and release gates.
- **Operations and recovery** contains backups, restore controls and raw
  runtime state.

The authoritative workflow chooses the relevant area automatically until the
user explicitly chooses another area. That choice is retained only as one of
four allowlisted identifiers in browser session storage. Choosing “go to next
step” reveals the owning area before scrolling and moving keyboard focus.

## Causal world lab loop

The Match laboratory is one connected intervention workflow, not a collection
of unrelated demos:

```text
freeze fixture + shared seed
    -> choose exactly one tactical intervention
    -> run baseline and treatment worlds
    -> align both replays on a shared timeline
    -> inspect local action, chance and score-path divergence
    -> repeat over a fixed seed budget
    -> expose the final paired estimate only after completion
```

The one-pair view answers “where did these two simulated futures first
diverge?” The fixed-budget tactical study answers the narrower aggregate
question for the registered fixture and seed set. The Evidence center then
separates three claims that the interface never merges: the world model entered
the action policy, simulated behavior changed, and outcome improvement remains
unresolved. Manager advice can be reviewed or adopted, but its lifecycle ledger
records the choice as user behavior rather than converting it into causal or
performance evidence.

Before a manager freezes a matchday decision, Studio now recalculates a
zero-persistence impact preview whenever the tactic, rotation, lineup, club
situation response, or in-match rule changes. The preview and the actual
submission use the same authoritative backend normalization path. It shows the
exact automatic/manual lineup that would be frozen, fixed rotation mechanics,
in-match rule count, club-situation resolution, opponent preparation, and
hypothetical one-decision progress for season and named-player promises. It
does not predict the score, win probability, or causal effect, and it never
counts the hypothetical decision as completed evidence in the saved season.
Submission stays disabled until the current preview succeeds and carries its
season revision; the atomic submit rejects that preview if another tab or
process changed the season meanwhile.

After submission, the matchday command center keeps the decision in one
continuous lifecycle ledger. Every entry separates the frozen plan, direct
runtime execution, descriptive score, and season/player-promise accounting.
The ledger is replayed from authoritative season and report evidence rather
than persisted as another mutable journal. Full execution detail is loaded for
the most recent eight managed matches; older entries retain their frozen
decision, result, long-term accounting, identities, and full-report link. If a
report is missing or its match/team/decision identity disagrees, execution is
shown as unavailable instead of borrowing data or inferring success.

For each completed manager-controlled fixture, that same lifecycle now carries
an identity-bound world-state transition: simulated squad carryover immediately
before the match, immediately after settlement, and after the matchday recovery
transaction. The derived view exposes fatigue, morale, media pressure, injuries,
suspensions and bounded per-player carryover changes; recovery is kept separate
from the match delta. It is replayed from canonical snapshots and retained in
season history, so rehashed derived-field tampering still fails validation.
This evidence says what persisted simulated state changed. It does not say that
the decision caused the score, and injury fields are not real medical judgments.
Legacy completed fixtures without a pre-match snapshot remain explicitly
unavailable rather than receiving reconstructed evidence.

In `research` and `cognitive` modes, the manager can explicitly request a local
world-model advisory before submitting the next fixture decision. It evaluates
all seven playable tactics, including the club's native identity, as bounded
short-horizon action mixtures on one deterministic representative pre-match
state. The screen shows ranking, confidence, uncertainty, fatigue proxy,
recommendation margin and historical trust. Nothing is selected automatically:
the player may adopt the recommendation or review it and choose differently.
That interaction is bound to the advice identity and final frozen tactic in the
same lifecycle ledger. Advice expires after any season revision. Stable mode
does not load or imitate the research model, and no advice is described as a
score forecast, win probability, causal effect or real-football recommendation.

Three explicit modes prevent research flags from leaking into stable use:

| Mode | Stable simulator | Calibrated world model | Real LLM |
|---|---:|---:|---:|
| `stable` | yes | no | no |
| `research` | yes | yes | no |
| `cognitive` | yes | yes | yes, credentials required |

The stable mode is the default. Research mode requires the accepted candidate
checkpoint. Cognitive mode additionally refuses to run without a valid
`DEEPSEEK_API_KEY`, `API_KEY`, or `OPENAI_API_KEY`, and rejects an invalid
provider/model configuration before simulation.

## Quick start

```bash
python gfs.py studio run --name "My World Cup" --mode stable --seed 42 --home Brazil --away Argentina --fast
```

`studio run` is the shortest isolated-match entry: it creates or loads the
workspace, verifies readiness before spending simulation time, reserves the
match, runs it, and returns both report paths. The Web Studio is the guided
multi-season product entry. If a persisted Studio exists,
configuration overrides must match it. Changing name, mode, or seed requires
the explicit destructive boundary `--replace`.

Operators can still drive the same state machine step by step:

```bash
python gfs.py studio init --name "My World Cup" --mode stable --seed 42
python gfs.py studio status
python gfs.py studio match --home Brazil --away Argentina --fast
```

## Local Web Beta

The same state machine is available through one dependency-free local Web
surface:

```bash
python gfs.py studio web
```

Open `http://127.0.0.1:8765`. The page creates a Studio, shows evidence-aware
readiness, runs an authorized match, and opens the generated dashboard. It
does not duplicate simulation rules: all mutations call `ProductWorkspace`,
and control-plane status comes from `ProductControlPlane`.

The safe default remains loopback-only. Non-loopback serving is available only
through the separately authenticated remote boundary described below and a
trusted TLS-terminating reverse proxy. Mutations require a per-process CSRF
token; requests are size limited, concurrent mutations fail fast, report paths
are confined to Studio HTML outputs, and responses use restrictive browser
security headers. Verify the actual HTTP boundary without running a match or
making an external call:

```bash
python scripts/verify_product_web.py --out data/evaluation/web_beta_verification_v1.json
```

See `docs/WEB_BETA_ACCEPTANCE.md` for accepted scope and remaining Stage 1
limits.

The Web surface has a repeatable code-level accessibility gate:

```bash
python scripts/verify_web_accessibility.py --out data/evaluation/web_accessibility_verification_v1.json
```

It verifies both Studio and login semantics, focus and asynchronous status
contracts, mobile/forced-color styles, declared contrast, and the real local
HTTP surface without running a match. External assistive-technology and
target-user validation remain separate release gates.

Web match submissions are persistent background tasks, not long synchronous
HTTP requests. Each browser submission carries an idempotency key and receives
a task ID immediately. The page polls `queued`, `running`, `completed`,
`failed`, or `interrupted`, resumes observation after refresh, and opens the
dashboard only from a completed task. A workspace-level server lease prevents
two local Web workers from consuming the same queue. On process restart, a
previously running task becomes visible as `interrupted`; it is never silently
requeued because the underlying match transaction may already have spent
compute or provider calls.

Task history is available from `/api/v1/tasks` and individual status from
`/api/v1/tasks/<task_id>`. Internal worker identity and the hashed idempotency
key are never returned. The queue retains at most 1,000 records and prunes only
the oldest terminal history.

## Authenticated remote deployment

Local loopback remains the safe default. Non-loopback binding fails unless the
operator explicitly enables remote mode, supplies an exact public Host, and
sets an independent product access token of at least 32 random characters:

```bash
export GFS_WEB_ACCESS_TOKEN="<independent-random-product-token>"
python gfs.py studio web --host 0.0.0.0 --allow-remote --allowed-host studio.example.com
```

Do not expose that command directly to the Internet. Remote mode requires an
internal trusted reverse proxy that terminates HTTPS and sets
`X-Forwarded-Proto=https`. All non-health routes enforce the Host allowlist,
HTTPS proxy marker, product-token login, HttpOnly Secure SameSite Cookie, and
the existing CSRF/CSP boundaries. The product access token must never be an
LLM/provider key.

`deploy/compose.yaml` provides the intended Caddy topology: only Caddy exposes
80/443, while GFS stays on an internal network with a non-root user, read-only
root filesystem, dropped capabilities, persistent product volumes, and health
checks. See `deploy/README.md`. The contract is statically verified but remains
unbuilt because Docker is unavailable in the current environment.

The Docker-host handoff is implemented as a separately attested CI job. Its
third-party actions are locked to exact commits recorded in
`data/evaluation/ci_action_lock_v1.json`; a passing artifact must still match
the commit and verifier hashes before it can close the container gate.

## Backup and recovery

Back up the persisted Studio and every artifact referenced by its match
journal, verify the archive independently, and restore only through an
explicit replacement boundary:

```bash
python gfs.py studio backup --out backups/studio.zip
python gfs.py studio verify-backup backups/studio.zip
python gfs.py studio restore backups/studio.zip
python gfs.py studio restore backups/studio.zip --replace
```

The archive uses relative paths, size and SHA-256 checks, strict member
whitelists, semantic session validation, staging, session-last switching, and
rollback on failure. It excludes provider credentials, frozen releases,
research checkpoints, and training state. See `docs/STUDIO_RECOVERY.md`.

The local cross-process recovery gate is also executable without a match:

```bash
python scripts/verify_process_recovery.py --out data/evaluation/process_recovery_verification_v1.json
```

It force-kills only its own temporary child process, then checks OS-lease
release, startup task reconciliation, committed-file hashes, backup
verification, and health recovery in a fresh process. It does not claim
deployment-volume RPO/RTO.

Every match is reserved in the session before simulation starts. Match IDs and
seeds are monotonic even after a failure. The run journal distinguishes
`running`, `finalizing`, `completed`, `failed`, and recovered `interrupted`
attempts, so provider calls or completed simulation work cannot disappear just
because dashboard rendering fails.

Mode configuration is context-local and never mutates process `os.environ`.
Research and cognitive modes verify the exact checkpoint SHA-256 before the
match and compare it again with the runtime-reported signature afterward.
Stable mode verifies the active release pointer plus every artifact in the
frozen release manifest.

## Prospective cognitive pilot

Validate provider configuration before creating a cognitive Studio:

```bash
python gfs.py studio provider
```

This command creates no provider client and makes no network request. It emits
only a redacted provider summary.

The live-provider experiment is part of Studio rather than a detached research
script. First create a clean cognitive session and run the zero-cost preflight:

```bash
python gfs.py studio init --name "Cognitive Pilot v1" --mode cognitive --seed 42
python gfs.py studio pilot
```

Preflight never calls an external provider. It freezes two fixtures, their
seeds, a maximum of two coach calls per match, and the acceptance rule. It
blocks if the checkpoint, credentials, cognitive mode, or empty-session
requirement is missing. Only the explicit command below can spend provider
tokens:

```bash
python gfs.py studio pilot --execute
```

Each accepted match must record at least one successful provider response.
Rule-only fallback is rejected. The composed match JSON, HTML dashboard, and
cognitive audit log are linked from one report; the aggregate pilot decision is
written to `data/evaluation/llm_prospective_pilot_v1.json`. Passing this small
pilot supplies prospective evidence but does not authorize production
promotion by itself.

The session is persisted under `data/persistence/product_session.json`.
Every match creates both an auditable JSON report and a human-readable HTML
dashboard under `outputs/studio/<studio>/matches/`. The report composes score,
xG, possession, passing, shooting, psychology, tactical drift, world-model
calibration, decision adoption, cognitive plans, timeline, artifacts, and an
evidence snapshot in one envelope.

GFS Studio does not promote a research checkpoint. It reads the frozen release
and evaluation artifacts and exposes their status; the release manifest remains
the only production authority.

## Unified control plane

Training and research decisions are visible from the same product surface:

```bash
python gfs.py studio jobs
python gfs.py studio stop-job <run_id>
```

`jobs` reports both the persisted state and the observed state. A local job
whose recorded PID no longer exists is shown as `stale`, not as healthy
`running`. `stop-job` only creates a cooperative stop request; the trainer saves
progress and exits at an epoch boundary. Resume it with the original training
command plus `--resume --run-dir <directory>`.

The world model, frozen-backbone shot head, and live-LLM pilot are optional
adapters authorized by decision artifacts. Their existence never changes the
stable product automatically. See `docs/ARCHITECTURE_V2.md` for the complete
system boundary and lifecycle.

The same control plane now exposes a unified `release` object and a visible
"产品与论文发布门禁" panel. It combines the product and academic scores with
paper-package, transitive-lock, SBOM, data-license, local-runtime,
cross-platform-lock, image-digest, container-build, licensed external-data,
frozen human-validation protocols, target-user outcomes, independent
accessibility/security review, production soak/recovery, comparative user
value, confirmatory results, and independent reproduction. `code_ready` means
the ten code-controlled gates pass; it never aliases `release_ready`, which
remains false while any real-world gate is open. The full status currently has
18 gates. Each gate carries an evidence path and the response identifies the
next unresolved action.

## Integrated operations view

Studio status and the authenticated `GET /api/v1/operations` route expose
aggregate request latency/status, authentication outcomes, task lifecycle,
recovery activity, telemetry corruption, and write degradation. The underlying
bounded JSONL schema cannot accept credentials, request content, network
identifiers, fixture names, idempotency keys, local paths, or exception text.

Run the no-match control-plane verifier with:

```bash
python scripts/verify_product_operations.py --tasks 100
```

See `docs/PRODUCT_OPERATIONS.md` for retention, privacy, and claim boundaries.

The real production gate is separately frozen in
`data/evaluation/production_validation_protocol_v1.json`. Its runner executes
only with an exact authorization token and requires 100 fixed stable matches,
an observed ungraceful restart on the same named deployment volume, preserved
task identity, zero lost or duplicate logical transactions, zero provider
calls, and an integrity-equivalent staged restore. A normal invocation of
`python scripts/run_production_validation.py` is a zero-match protocol audit.
See `docs/PRODUCTION_VALIDATION.md`; the gate remains open until real deployment
evidence exists.

Comparative product value is separately preregistered instead of being inferred
from SUS. `scripts/product_value_study.py` audits or analyzes a 24-participant,
role-balanced and sequence-balanced crossover against the manual baseline. The
primary bound requires at least a 20% penalized-time improvement with accuracy
noninferiority and zero GFS critical errors. See
`docs/PRODUCT_VALUE_STUDY.md`. No participants have yet been observed, so this
gate also remains open.

The Evidence workspace now connects that frozen protocol to a moderator-facing
delivery boundary. It shows aggregate role and AB/BA registration counts,
registers only consented pseudonymous participants under CSRF protection, and
downloads a deterministic identity-bound packet. Participant cases contain no
answer key; final scoring uses a separate case-hash-bound moderator seal. The
same flow is available under `studio value-study status|register|packet`.
Generated packets exclude moderator identity and scoring material, while the
source repository and scoring seal remain explicitly outside the participant
delivery channel. This is operational preparation, not a claim that any user
study or product-value result exists.

Remote sessions are independent and process-local. Each has an eight-hour
absolute lifetime, a thirty-minute idle lifetime, and can be revoked with the
Studio “安全退出” control. At most 64 sessions are retained; creating another
evicts the oldest. Login failures are limited per salted in-memory client
identity and by a global fallback window. A limited request receives HTTP 429
and `Retry-After`; client addresses and salts never enter telemetry.

## Integrated recovery center

The Web surface lists only server-managed backups under `backups/studio/`.
Users can create an integrity-checked backup, verify it again, and prepare a
restore without entering a filesystem path. Restore requires the exact backup
ID to be typed, an explicit replacement checkbox, CSRF, and an idle match task
queue. The existing staged, fsynced, session-last transaction performs the
restore and resets cross-session task history.

The managed catalog is capped at 50 entries. Archive older bundles through the
deployment storage workflow before creating more. The Web API deliberately
does not stream backup archives; exporting them remains an authenticated
deployment/storage operation.

## Manager advisor adoption evidence

The manager decision ledger now summarizes how often world-model advice was
available, whether the manager explicitly adopted it or reviewed it and chose
another tactic, whether advice was left unlinked, and whether the frozen tactic
has direct execution evidence. The summary is derived from the authoritative
fixture ledger and is identity-bound; it is not separate mutable telemetry.

Use `python scripts/manager_advisor_study.py --status` to audit the frozen
collection protocol. Analysis of an existing Studio is read-only:
`python scripts/manager_advisor_study.py --analyze --base-dir <studio-dir>`.
The result measures evidence completeness only. It does not estimate score or
win effects and cannot promote the product or research candidate.

Advisor inference uses a two-phase optimistic transaction. Studio freezes the
request identity under a short session lease, releases that lease while the
local checkpoint evaluates the seven tactics, and reacquires it only for the
atomic commit. If the manager submits a decision, the fixture advances, team
carryover changes, or the checkpoint changes during inference, the advice is
discarded with a retryable stale-input result. No concurrent decision is
overwritten and no partial advice is persisted.

Repeated requests for the same current advice are semantically idempotent.
Studio verifies the persisted advice against the live fixture revision,
checkpoint file SHA-256 and both teams' world-state source identities before
returning it with `reused: true`. This retry path performs zero inference and
zero session writes. Concurrent duplicate requests converge on the same advice
identity and advance the revision only once; changed world state forces a new
evaluation instead of reusing stale advice.

Completed managed fixtures now expose a world-model advisor execution chain:
advice, explicit interaction, frozen selection and direct tactical runtime
binding. The micro engine records the exact 22-dimensional opening tactical
vector and its final state. Studio distinguishes a named locked preset from the
native `team_identity` vector, shows selected control values and in-match
changes, and identity-binds the result into the replayable decision ledger.
This proves that the selected tactical input reached the engine; it does not
prove that the tactic caused the observed score or result.

The submit preview also compares the recommended tactic with the manager's
current selection. It shows rank, proxy-value, confidence, uncertainty,
fatigue, structural-risk and event-distribution deltas computed by the server.
Low confidence, weak historical trust, a narrow top-two margin or insufficient
history marks the recommendation as exploratory and changes the action wording
accordingly. The thresholds disclose evidence weakness; they are not win-rate
calibration and never authorize automatic adoption.

## Unified manager journey

The career workspace exposes one five-stage lifecycle:

1. club planning;
2. pre-match decision and optional world-model advice;
3. match execution;
4. post-match evidence and replay;
5. long-term club and career operations.

The highlighted stage is derived from the canonical season command center.
The existing next-step control moves focus to the authoritative form or action
for that state, so the lifecycle rail is navigation and explanation, not
another mutable workflow.

Use the four-run diagnostic before any fixed-budget study:

    python scripts/action_adoption_study.py --smoke

It compares no-advisor, deterministic rule fallback, prediction-only and
direct-action paths using one shared short fixture per arm. It does not train,
call a provider, modify formal progress or support an effectiveness claim.
The explicitly authorized 24-run mechanism study for the historical pass-only
controller completed on 2026-08-31. Its frozen identity-bound decision was
`mechanism_confirmed`: 1951 of 1954 action
opportunities received non-zero policy influence, 25.246 counterfactual
changes were expected, 25 were realized, and all 12 matched pairs changed.
The historical decision JSON, 24 flat CSV rows and Markdown summary were
originally replayed through:

    python scripts/verify_action_adoption_result.py

After V3.63 the command correctly reports code-identity drift. This confirmed
that historical controller's micro-action adoption path; it is not current
evidence. The separate 60-run full-match
outcome protocol, `data/evaluation/action_outcome_protocol_v1.json`, also
completed on 2026-08-31. All 30 matched pairs changed behavior, but the M1-minus-
M0 external-loss estimate was -0.031893 with a 95% paired interval of
[-26.061955, 5.137430]. The external-validity and minimum-effect gates failed,
so the sealed decision is `inconclusive_keep_research_only` and promotion is
not supported. Its historical replay command now also fails current-code
identity by design:

    python scripts/verify_action_outcome_result.py

Evidence Center reports the mechanism and outcome layers separately. The
product may say that M1 changes simulated actions and full-match behavior; it
must not claim reliable outcome improvement or real-football causality.

## Season world navigation and action adoption

The matchday command center keeps one current intervention chapter and a
replayable history of completed official-world chapters. Its influence path
shows fixture-level evidence coverage, while the adjacent whole-season action
adoption ledger exposes the underlying official manager-team decision counts:
retained opportunities, non-zero probability influence, shared-randomness
local action changes, exact ball-event links, unresolved links and record
coverage. Probability influence and realized action changes are deliberately
separate states.

Each evidence-break or adoption-state diagnostic links to its most recent
identity-bound chapter. Studio verifies the current diagnostic binding and the
completed ledger entry before updating the season-scoped permalink, rebuilding
the projections and expanding the complete chapter. These controls are
read-only; none repairs evidence, reruns a match or turns mechanism
observability into an outcome-effect claim.

The same action-state rows now continue into the facts recorded later in their
completed chapters: descriptive win/draw/loss and points, plus post-match
fatigue, morale, media pressure, availability and injury/suspension changes.
Recent chapter cards show the same continuation before the full evidence
timeline is opened. Values are raw coverage and totals; missing chapters are
not imputed, states are not ranked, and differences between adoption states
are not treatment effects. The purpose is to make one simulated-world history
legible, not to claim that the world model caused its result or carryover.

The same panel also provides a chronological season world trajectory. Every
visible row is bound to its source fixture and canonical world-chapter
identity, and combines that match's adoption state, result, persistent-state
delta and deterministic change markers with season-to-date known points,
state coverage and local action changes. The window is bounded to 64 recent
chapters, while cumulative values still begin at the first completed chapter.
Missing evidence is never interpolated. Change markers identify recorded
events and gaps only; they are not inferred turning points, effect estimates
or causal explanations. Opening a row uses the same stale-safe permalink and
full-ledger expansion as the existing diagnostics.

Reviewed multi-timepoint futures now remain visible after the official world
advances. The prematch stage and its season-trajectory row show whether the
manager kept or revised the frozen selection, how many registered branch times
were usable, how many produced action divergence and local attribution, and
whether descriptive timing sensitivity was observed. The same values are
identity-bound from the review receipt through the execution trace, world
thread and historical chapter. They explain the decision context; they are not
compared with the observed score as forecast accuracy, tactic quality or a
treatment effect.

Each reviewed trajectory row can now expand the exact bounded fork dossier
that produced those totals. The dossier preserves the reviewed time, semantic
evidence state, action/local/descriptive counts, anchor verification and
shortened source/archive identities. The same dossier appears inside the
prematch stage of the complete official-world thread, so it remains one
product flow rather than a detached analysis page. These are simulated roads
not taken: the UI does not rank them or compare them with the observed score.

The complete world thread and each trajectory row now also expose a separate
review-to-official-world certificate. “Complete” means the reviewed final
decision identity reached an official tactical binding and same-match world-
model action evidence. Missing review, a later unreviewed edit, missing
tactical binding and missing action evidence remain distinct states. The
certificate never claims that a simulated branch opportunity is the same
action later observed in the official match, and it does not measure outcome
improvement.

## Action-specific world-model authority

Research mode now reports world-model action control on the same four-action
surface used by the match engine. Pass and shot may affect the sampling
distribution only when their own validation gate is open and the action is
currently feasible. Cross remains visibly unavailable until cross-specific
validation exists; hold remains the continuation reference rather than an
implicitly learned recommendation.

The latest-match influence panel adds one compact card per admitted action.
Each card shows signal opportunities, probability increases versus decreases,
mean probability movement and mean applied authority. The underlying evidence
retains the normalized baseline and treatment distributions and the shared
random draw, so a realized action change can still be isolated locally. These
cards describe simulator runtime behavior only and do not imply better scores,
better tactics or real-football causality.

Formal mechanism and full-match evidence is displayed only when its frozen
protocol, checkpoint and declared code-file hashes reproduce against the
current checkout. After a controller change, Studio marks the old result as
`stale_current_code_identity`, keeps its existence visible as history, and
suppresses its statistics and promotion fields. A new preregistered run is
required before the changed controller can inherit a formal conclusion.

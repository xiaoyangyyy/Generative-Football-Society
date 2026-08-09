# Product and Academic Excellence Roadmap

GFS reaches 10/10 only when both the product and academic tracks score 100
from auditable evidence. A strong architecture, a passing test suite, or an
interesting model is necessary but cannot substitute for deployment, users,
confirmatory results, external validity, a reproducibility package, and
independent review.

The machine-readable authority is
`data/evaluation/excellence_roadmap_v1.json`. Current scores are readiness
scores, not claims about universal football realism or scientific superiority.

## Security boundary

Any API key pasted into chat, source, logs, or screenshots is considered
compromised. Revoke it in the provider console and create a replacement. The
replacement belongs only in a local process environment or ignored local
`.env`; it must never enter Git, test fixtures, reports, or command history.

For the current DeepSeek OpenAI-compatible API:

```powershell
$env:DEEPSEEK_API_KEY="<new rotated key>"
$env:BASE_URL="https://api.deepseek.com"
$env:MODEL_NAME="deepseek-v4-flash"
python gfs.py studio provider
```

`studio provider` is a zero-call preflight. It reports the
provider, model, base URL, and credential variable name but never the value.

## Stage 0 — secure and freeze

Exit only after the exposed key is revoked, a replacement is configured
locally, the current baseline is backed up remotely and tagged, all tests and
architecture gates pass, and zero-call DeepSeek preflight is ready. No paid
call or formal match belongs in this stage.

## Stage 1 — product Beta

Build one task-oriented Web surface over the existing Studio state machine:
create/load, readiness, match execution, live status, report review, history,
and recovery. Add authentication for non-local deployment, health/readiness
endpoints, structured telemetry without secrets, backup/restore, packaging,
and a one-command deployment. Do not add new football intelligence modules.

Exit gates include accessibility checks, cross-platform packaging, automated
deployment verification, failure recovery, and a 100-match deterministic soak
with zero lost or duplicated transactions.

Current code-level progress: the loopback Web workflow, health/readiness
boundary, browser hardening, and automated accessibility landmarks pass a real
HTTP verifier in `data/evaluation/web_beta_verification_v1.json`. This raises
the evidence-backed product score from the frozen 58 baseline to 64 after
removing two unsupported user-value points from the original rubric. It does
not close Stage 1: authenticated deployment, background/recovery UX,
cross-platform packaging, production telemetry, the 100-match soak, and an
external accessibility audit remain open.

The integrity-checked Studio backup and transactional restore verifier adds
three reliability points, bringing the current product score to 67. This is a
local recovery proof, not the timed deployment-storage drill required to close
Stage 1/3.

The persistent idempotent background-match queue, single-server lease, task
history, refresh-safe polling, bounded retention, and interrupted-task
reconciliation add three more product points, bringing the current score to
70. The isolated verifier executes no real match; multi-host execution remains
unclaimed, while a later cross-process drill covers local process death.

The authenticated socket boundary and static container/TLS deployment
contract add one security point and two operations points, bringing the
product score to 73. Docker is unavailable in the current environment: no
image was built, no Compose service started, and no public certificate,
firewall, penetration test, or production volume drill is claimed.

Privacy-bounded persistent telemetry now covers normalized Web requests,
authentication outcomes, task transitions, worker lifecycle, and successful
backup/restore operations. A 100-task control-plane soak executes 200
concurrent submission attempts, exact idempotency checks, terminal failures,
and restart recovery. This adds one reliability point and one operations point,
bringing the product score to 75. It executes zero matches and therefore does
not close the deterministic 100-match or long-duration deployment soak gates.

Remote authentication now issues independent `__Host-` sessions with absolute
and idle expiry, bounded session capacity, current-session logout, per-client
in-memory salted rate limits, and a global anti-bypass limit with standard 429
and `Retry-After` responses. The real socket verifier proves these boundaries,
adding one security point and bringing the product score to 76. The remaining
security point still requires an independent review/penetration test against a
real deployed TLS edge; process-local sessions and rate limits are not claimed
to support horizontal replicas.

The recovery center is now part of the same authenticated Studio surface. It
uses a locked, 50-entry managed catalog with server-generated IDs; Web clients
cannot supply filesystem paths. Create performs verification, verify is
explicit, and restore requires CSRF, `replace=true`, an exact typed ID, no
active match tasks, and the existing transactional rollback implementation. A
real local socket verifier passes all 13 gates without running a match. This
closes the final functional-completeness point and adds one usability point,
bringing the product score to 78. Deployment-volume RPO/RTO, recovery during
an in-flight filesystem replacement, Web export, and target-user evidence
remain unclaimed.

The Studio and login surfaces now expose atomic polite/assertive status
regions, explicit asynchronous busy state, terminal-result focus, 44 px
controls, narrow-screen single-column reflow, forced-colors support, and an
Escape/correct-trigger focus return path for destructive recovery. The
dependency-free verifier parses both templates, checks semantic relationships,
tests six declared color pairs, and confirms the audited page over a real local
socket. All 32 checks pass and the lowest measured declared contrast is
6.92:1. This adds two usability/accessibility points, bringing the product
score to 80. It remains a code-level contract: no browser accessibility tree,
screen reader, zoom/reflow interoperability, external WCAG audit, or
target-user study is claimed.

A real cross-process drill now starts a loopback Studio service, commits a
Studio and managed backup over HTTP, persists a synthetic owned task only
after stopping the match worker, and force-kills that exact child process.
The next fresh process reacquires the OS lease, reconciles the orphaned task to
`interrupted`, serves health, reads the Studio, and verifies the original
backup. All 20 checks pass with zero match artifacts and zero loss among files
committed before the kill; the recorded local recovery time is 0.311 seconds.
This adds one reliability point, bringing the product score to 81. The run uses
a temporary local filesystem and kills between committed requests, so
deployment-volume failure injection, kill-during-transaction behavior, and a
production RPO/RTO or SLA remain open.

## Stage 2 — confirmatory academic experiment

Freeze the commit and run only
`formal_experiment_protocol_v2.json`: M0 versus sealed M1, thirty matched
pairs and sixty total runs. No interim analysis or optional stopping is
allowed. Analyze only when identity, pairing, budget, and completeness gates
pass.

The result may support promotion, equivalence, or an inconclusive
research-only decision. A negative result is valid evidence and must not be
reframed as success.

Before executing Stage 2, the project now has a registered-report draft,
eight-claim evidence registry, model and data cards, references, a
side-effect-classified reproduction manifest, an observed environment
snapshot, and an independent-review guide. A zero-simulation verifier passes
all 26 gates, including protocol/checkpoint identity, absence of confirmatory
outputs, freshness of existing evidence, unique claim markers, resolved
citations, and explicit non-claims. This adds two reproducibility points and
three manuscript/readiness points, bringing the academic score from 62 to 67.
It does not complete Stage 2 or Stage 5: the environment is not transitively
hash-locked, provider license review is incomplete, the results manuscript is
pending, and no independent reproduction exists.

## Stage 3 — product validation

Run a preregistered usability study with at least ten target users completing
the core workflow. Required evidence includes task completion, error rate,
time-on-task, SUS, qualitative failure themes, and issue resolution. Run an
operational soak and recovery drill on the deployment candidate.

Exit gates: at least 90% core-task completion, SUS at least 80, no unresolved
critical security/accessibility defects, and documented recovery objectives.

## Stage 4 — academic replication and mechanism

Choose the branch dictated by Stage 2 before changing code. If M1 improves,
perform a preregistered mechanism ablation and independent-seed replication.
If equivalent, investigate the prediction-to-action adoption path without
claiming benefit. If inconclusive, diagnose variance before registering any
new sample. Add an external provider/dataset replication and competitive
baselines.

## Stage 5 — paper and release candidate

Freeze a containerized, one-command reproduction package with manifests,
licenses, environment lock, data provenance, expected hashes, and an artifact
review guide. Complete the manuscript, limitations, negative results,
statistical appendix, model/data cards, threat model, and independent
reproduction by someone other than the implementer.

Only after every critical gate passes may the product and academic tracks be
reported as 100/100. Production promotion remains a separate signed release
decision.

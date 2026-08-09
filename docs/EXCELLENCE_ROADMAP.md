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

## Stage 2 — confirmatory academic experiment

Freeze the commit and run only
`formal_experiment_protocol_v2.json`: M0 versus sealed M1, thirty matched
pairs and sixty total runs. No interim analysis or optional stopping is
allowed. Analyze only when identity, pairing, budget, and completeness gates
pass.

The result may support promotion, equivalence, or an inconclusive
research-only decision. A negative result is valid evidence and must not be
reframed as success.

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

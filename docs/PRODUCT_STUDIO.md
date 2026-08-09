# GFS Studio Product Surface

GFS Studio turns the simulator, world model, cognition layer, and evidence
reports into one persistent user workflow. The lower-level research scripts
remain available, but product users work through a studio session.

## Product loop

```text
configure/load -> verify evidence -> reserve and run -> persist report -> review
    blocked          ready_to_run          running              review
```

This is a persisted state machine, not just documentation. `studio status`
returns the current state, completed and failed counts, progress flags, and one
next action derived from the session journal.

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

`studio run` is the normal end-to-end product entry: it creates or loads the
workspace, verifies readiness before spending simulation time, reserves the
match, runs it, and returns both report paths. If a persisted Studio exists,
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

The Beta is intentionally loopback-only until authenticated TLS deployment is
implemented. Mutations require a per-process CSRF token; requests are size
limited, concurrent mutations fail fast, report paths are confined to Studio
HTML outputs, and responses use restrictive browser security headers. Verify
the actual HTTP boundary without running a match or making an external call:

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

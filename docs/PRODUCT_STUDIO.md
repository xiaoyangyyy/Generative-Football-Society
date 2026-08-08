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
checkpoint. Cognitive mode additionally refuses to run without `API_KEY` or
`OPENAI_API_KEY`.

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

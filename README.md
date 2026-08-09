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

The next confirmatory paper experiment is frozen separately from exploratory
work. Its default command is read-only:

```bash
python scripts/run_formal_experiment.py
```

It compares only sealed M0 and M1, caps compute at 30 matched pairs / 60 runs,
and requires explicit `--execute`. See
`docs/FORMAL_EXPERIMENT_PROTOCOL_V2.md`.

The pre-execution paper package is independently auditable without training or
simulation:

```bash
python scripts/verify_paper_package.py --out data/evaluation/paper_package_verification_v1.json
python scripts/verify_reproduction_release.py --out data/evaluation/reproduction_release_verification_v1.json
```

See `docs/PAPER_DRAFT.md` and `docs/REPRODUCTION_GUIDE.md`. The draft keeps
all confirmatory result cells unavailable and does not claim independent
reproduction. The release audit verifies two 64-package hashed reference
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

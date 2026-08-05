# Generative Football Society

Football world-model research platform with a stable World Cup-scale simulator.

The project is research-first: learned world models and LLM cognition are
evidence-gated, default-off advisory layers. The deployed simulator remains
v7.0.0 until an explicit release manifest changes the production pointer. See
`docs/RESEARCH_DIRECTION.md` and
`data/evaluation/research_evidence_v1.json` for the frozen evidence boundary.

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

## Unified CLI

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

LLM features require a valid OpenAI-compatible endpoint.

```env
API_KEY=your_api_key
BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4-turbo-preview
GFS_SEED=42
```

Without a real key, non-LLM commands and most deterministic smoke tests can still run.

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

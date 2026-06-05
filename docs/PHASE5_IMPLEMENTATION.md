# Phase 5 — Narrative Isolation + Layer Gates

## Goal

Keep **L0 calibration** (M0 physics + System-1 affective ODE) reproducible and StatsBomb-gated, while **M1 (world model)** and **C1 (cognitive)** are validated as **additive layers** that must not materially shift L0 metrics.

| Layer | What it is | Gate type |
|-------|------------|-----------|
| **M0 / T0** | Micro physics + affective ODE | Full StatsBomb gate (`run_calibration_gate.py`) |
| **M1** | M0 + `MATCH_WORLD_MODEL=1` | Additive Δ vs M0 (`run_layer_gate.py`) |
| **C1** | M0 + `MATCH_COGNITIVE=1` | Additive Δ vs M0 (`run_layer_gate.py`) |
| **Tournament LLM** | Coach/media/reflection JSON | Off when `CALIBRATION_MODE=1` or `NARRATIVE_MODE=0` |

System-1 affective dynamics (crowd ψ, coach stress, ref strictness) **remain ON in M0** — they are not “narrative layers” for Phase 5.

## Environment isolation

`src/match_engine/calibration/narrative_isolation.py`:

| Helper | Role |
|--------|------|
| `expected_layer_flags(pipeline)` | Canonical env for M0/M1/C1/T0 |
| `apply_benchmark_isolation()` | Set + assert isolated env for benchmarks |
| `assert_isolated_env()` | Block WM/cognitive leak into M0/T0 |
| `narrative_layers_enabled()` | Tournament LLM on/off |
| `resolve_tournament_llm()` | `SimulationLLM` or `NullSimulationLLM` |

Benchmark env always sets:

```
CALIBRATION_MODE=1  MATCH_MICRO=1  MATCH_MICRO_SCORE=1
XG_SUPPLEMENT=0     MATCH_SCHEDULED_SHOTS=0
MATCH_LEGACY_POISSON=0
```

## Layer gate thresholds

From `data/calibration/observable_contract.json` → `layer_gates`:

| Layer | max Δloss | max \|Δz\| (primary) | min score ratio vs M0 |
|-------|-----------|----------------------|------------------------|
| M1 | 5.0 | 0.20 | 0.95 |
| C1 | 8.0 | 0.35 | 0.90 |

M1/C1 **do not** require absolute StatsBomb pass — only that they stay close to M0.

## Run

```powershell
$env:PYTHONPATH="."

# Prerequisite: M0 full gate
python scripts/run_calibration_gate.py --pipeline M0

# Additive layer gates (M1 + C1 vs M0)
python scripts/run_layer_gate.py
python scripts/run_layer_gate.py --quick          # 1 fixture, 20 min, 1 sample

# Full Phase 5 orchestrator
python scripts/run_phase5_validate.py
python scripts/run_phase5_validate.py --quick --skip-m0-gate
```

Reports: `reports/narrative_gates/{M1,C1,index}.json`

## Tournament narrative skip

```powershell
$env:CALIBRATION_MODE="1"   # auto during gate scripts
# or
$env:NARRATIVE_MODE="0"     # skip LLM without calibration benchmarks
python run_world_cup_2026_full.py
```

Uses `NullSimulationLLM` — no API calls, deterministic coach/media stubs.

## Tests

```powershell
python -m pytest tests/test_narrative_isolation.py tests/test_layer_gate.py -q
```

## Files

| File | Change |
|------|--------|
| `narrative_isolation.py` | Env isolation + tournament LLM resolver |
| `layer_gate.py` | Δloss / Δz / score-ratio gate vs M0 |
| `ablation.py` | `ablation_context` applies full benchmark isolation |
| `benchmark_core.py` | `assert_isolated_env` on L0 runs |
| `llm_engine.py` | `NullSimulationLLM` stub |
| `tournament_2026.py` | `resolve_tournament_llm()` |
| `observable_contract.json` | C1 pipeline + `layer_gates` |
| `scripts/run_layer_gate.py` | M1/C1 gate runner |
| `scripts/run_phase5_validate.py` | Phase 5 orchestrator |

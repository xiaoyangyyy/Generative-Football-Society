# Micro calibration vs StatsBomb WC 2022

## Observable contract (Phase 0)

Hard marginal bands + soft joint constraints live in:

- `data/calibration/observable_contract.json`
- `data/calibration/statsbomb_match_baselines.json`

Parameter registry (Phase 1):

- `data/calibration/param_registry.json` — profile `v5_physics_first`

## Pipelines (M0 / M1 / T0)

```powershell
$env:PYTHONPATH="."
python scripts/run_calibration_baselines.py --quick
python scripts/run_calibration_baselines.py --samples 3
```

Reports: `reports/calibration_baselines/{M0,M1,T0}.json`

| Pipeline | Meaning |
|----------|---------|
| M0 | Pure micro; cognitive + WM off |
| M1 | M0 + `MATCH_WORLD_MODEL=1` |
| T0 | Agent `status_score` instead of fixed eff_status |

## Ablation matrix (Phase 2/3)

Continuous **Δz vs M0** — no p10/p90 band thresholds.

```powershell
python scripts/run_ablation_matrix.py --samples 3
python scripts/run_ablation_matrix.py --quick
```

Reports: `reports/ablation/matrix.json`, `matrix.md`

**Subtractive** (M0 → OFF): `no_affective`, `no_spatial`, `no_blend`, `no_discipline_tick`, `no_tactical_bias`

**Additive** (M0 → ON): `M1` (world model), `C1` (cognitive)

Gate uses **continuous score** (`exp(-0.5·z²)`), not p10/p90 bands.

## Regenerate baselines

```powershell
$env:PYTHONPATH="."
python scripts/fetch_pass_calibration_statsbomb.py
python scripts/fetch_match_baselines_statsbomb.py
```

## Phase 3 — Joint calibration

Export gate-passed parameters, validate full gate, run ablation:

```powershell
python scripts/joint_calibrate.py --validate-only
```

Optional staged coordinate search (quick sim → full gate):

```powershell
python scripts/joint_calibrate.py --optimize --quick
python scripts/joint_calibrate.py --optimize
```

Outputs:
- `data/calibration/param_registry.json` → `profiles.v5_physics_first.overrides`
- `reports/joint_calibration/v5_physics_first.json`
- `reports/calibration_gate.json`
- `reports/ablation/matrix.json`

## Phase 4 — Single score path

Official goals = micro physics only (`MATCH_MICRO=1`, default `MATCH_MICRO_SCORE=1`). Macro λ is narrative-only.

See [`PHASE4_IMPLEMENTATION.md`](PHASE4_IMPLEMENTATION.md).

## Phase 5 — Narrative isolation + layer gates

L0 gate (M0) stays the StatsBomb absolute gate. **M1** and **C1** use **additive Δ vs M0** — they do not need to pass absolute StatsBomb bands.

See [`PHASE5_IMPLEMENTATION.md`](PHASE5_IMPLEMENTATION.md).

```powershell
python scripts/run_calibration_gate.py --pipeline M0
python scripts/run_layer_gate.py
python scripts/run_phase5_validate.py --quick
```

Reports: `reports/narrative_gates/{M1,C1,index}.json`

Tournament LLM is skipped when `CALIBRATION_MODE=1` or `NARRATIVE_MODE=0` (`NullSimulationLLM`).

## Full calibration gate (required before Phase 3+)

```powershell
python scripts/run_calibration_gate.py
python scripts/run_calibration_gate.py --pipeline M0 --samples 3 --match-seconds 5400
```

Exit code **0** only when all hard + soft constraints pass. Report: `reports/calibration_gate.json`.

Regenerate joint soft bands from StatsBomb:

```powershell
python scripts/fetch_match_baselines_statsbomb.py
```

## Full benchmark (multi-fixture)

```powershell
python scripts/benchmark_sim_vs_open_data.py --samples 3 --pipeline M0
python scripts/benchmark_sim_vs_open_data.py --quick   # 20-min smoke (not a calibration pass)
```
python scripts/benchmark_sim_vs_open_data.py --with-style-contrast
```

## Style contrast (gegenpress vs tiki-taka)

```powershell
python scripts/benchmark_style_contrast.py --samples 4 --match-seconds 5400
```

## Tournament / roster / world model

```powershell
python scripts/benchmark_tournament_smoke.py --group-matches 12
python scripts/benchmark_roster_calibration.py
python scripts/validate_world_model.py
```

## CI slow tests

```powershell
$env:RUN_SLOW_TESTS=1
pytest tests/test_calibration_suite.py tests/test_pass_calibration.py -m slow
```

## Key continuous knobs

| Area | Location |
|------|----------|
| Shots / passes / crosses | `MicroMatchConfig` action_* |
| Pass completion | `data/calibration/statsbomb_pass_rates.json`, `passing_engine` |
| Tick fouls / cards | `discipline_schedule.py`, `discipline_*` in `MicroMatchConfig` |
| Press vs possession fouls | `discipline_schedule.team_foul_weight` |

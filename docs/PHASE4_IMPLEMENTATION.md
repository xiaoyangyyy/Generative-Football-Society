# Phase 4 — Single Official Score Path

## Goal

One authoritative path for **official match goals** when micro is enabled:

| Component | Role |
|-----------|------|
| **Micro physics** | Official goals (shots + aerial) |
| **Macro λ ODE** | Narrative xG prior, event-calendar density, agent feedback |
| **Poisson / xG supplement** | **Disabled** on physics-official path |

## Score path modes

Resolved by `src/simulation/score_path.py` → `resolve_score_path_mode()`:

| Mode | Env | Official goals from |
|------|-----|---------------------|
| `physics_official` | `MATCH_MICRO=1` (default `MATCH_MICRO_SCORE=1`) | Shot/aerial physics |
| `micro_replay` | `MATCH_MICRO=1`, `MATCH_MICRO_SCORE=0` | Macro Poisson; micro replays |
| `macro_unified` | `MATCH_MICRO=0` | `resolve_unified_score()` Poisson |
| `poisson_legacy` | `MATCH_LEGACY_POISSON=1` | Status-vector Poisson |

## Phase 4 guards

| Feature | Physics-official |
|---------|------------------|
| `XG_SUPPLEMENT=1` | **Ignored** (no Poisson top-up) |
| `MATCH_SCHEDULED_SHOTS` | Default **off**; set explicitly to enable |
| Calendar `GOAL_SCORED` | Stripped from affective feed when `use_micro_goals` |
| Macro λ sample | Used for **xG prior / narrative only** |

## Tournament flow (default)

```
FusionController → expected_match_xg()     # macro prior (narrative)
                → run_physics_first_micro() # micro sim
                → finalize_official_score_from_micro()  # official score
```

Extra time and penalties unchanged: ET uses micro when `physics_official`; pens remain separate.

## Run

```powershell
$env:PYTHONPATH="."
$env:MATCH_MICRO="1"
$env:MATCH_MICRO_SCORE="1"   # default
python run_world_cup_2026_full.py
```

Legacy / replay (not recommended for calibration):

```powershell
$env:MATCH_MICRO_SCORE="0"   # micro_replay
$env:MATCH_LEGACY_POISSON="1"  # poisson_legacy
```

## Tests

```powershell
python -m pytest tests/test_score_path.py -q
```

## Files

| File | Change |
|------|--------|
| `src/simulation/score_path.py` | Mode resolution + `finalize_official_score_from_micro` |
| `src/simulation/match_pipeline.py` | Delegates to score_path |
| `src/memory_engine/macro_micro_fusion.py` | `macro_narrative_xg()`; physics early-return |
| `src/simulation/tournament_2026.py` | Single path via score_path |
| `src/match_engine/match_micro_runner.py` | XG supplement gated |
| `src/match_engine/event_schedule.py` | Scheduled shots gated |

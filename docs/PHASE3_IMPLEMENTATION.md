# Phase 3 — Ball Physics, Shots, GK, Micro xG → Poisson λ

Phase 3 adds **continuous ball trajectory** (drag, Magnus curve, knuckle flutter), **shot type selection** (softmax over driven / curved / knuckle / power / header), **goalkeeper saves**, **crosses & aerial duels**, and **micro xG integrals** blended into Poisson lambdas for optional match scoring.

## Modules

| Module | Role |
|--------|------|
| `ball_physics.py` | Trajectory ODE; `sample_shot_params`, `integrate_trajectory` |
| `shot_engine.py` | Shot resolution, GK model, xG, goal/save events |
| `aerial_duel.py` | Crosses, headers, second-ball contests |
| `action_engine.py` | Softmax: pass / shot / cross / hold |
| `goal_generator.py` | `lambdas_from_micro`, `simulate_match_score_from_micro` |
| `adapter.py` | `simulate_match_score_micro` — tournament-facing API |
| `match_micro_runner.py` | Unified tick loop (1b + 2a/2b + 3) |

## Environment variables

| Variable | Effect |
|----------|--------|
| `MATCH_MICRO=1` | Run full micro sim **in parallel** with legacy Poisson score (logging only) |
| `MATCH_MICRO_SCORE=1` | **Replace** Poisson score with micro λ sample; auto-enables micro logging |
| `MATCH_AFFECTIVE=1` | Phase 1b only (no spatial/passing/shots) |
| `GFS_SEED` | RNG seed |

## Quick demo

```bash
cd c:\Users\K\Desktop\Xiao_project\sim
python run_match_micro.py
```

Micro scoring in tournament:

```powershell
$env:MATCH_MICRO_SCORE="1"
$env:GFS_SEED="42"
python -m src.simulation.world_cup_runner
```

## Design notes

- **No discrete action labels** — shot “types” are softmax utilities over continuous parameters (ω, elevation, knuckle η).
- **Scheduled vs organic shots** — calendar `SHOT_ON/OFF` events are physics-resolved; calendar goals are stripped when `use_micro_goals=True`.
- **Official score** — `MATCH_MICRO_SCORE` samples goals from blended λ (status + ∫μxG); physics goals in `goals_micro_*` are diagnostic.
- **Performance** — use `MicroMatchConfig.fast_demo()` for tests (coarser grid, 30s ticks).

## Phase 3b (curved passes)

See [PHASE3B_IMPLEMENTATION.md](PHASE3B_IMPLEMENTATION.md) — Magnus ground passes in `passing_engine` + `integrate_pass_trajectory`.

## Tests

```bash
python -m unittest tests.test_match_phase3 tests.test_match_phase3b tests.test_match_micro_phase2 tests.test_affective_phase1b -v
```

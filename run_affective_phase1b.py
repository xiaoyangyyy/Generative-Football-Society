#!/usr/bin/env python3
"""
Phase 1b demo — affective coupling without full tournament / LLM.

Example:
  python run_affective_phase1b.py --home Brazil --away Argentina --seed 42
  python run_affective_phase1b.py --fast
"""

from __future__ import annotations

import argparse
import os
import sys

# project root on path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.simulation.world_cup_runner import build_world_and_tournament
from src.match_engine.config import AffectiveConfig
from src.match_engine.match_affective_runner import run_match_affective_simulation
from src.memory_engine.poisson_simulator import simulate_match_score


def main():
    parser = argparse.ArgumentParser(description="GFS Phase 1b affective coupling demo")
    parser.add_argument("--home", default="Brazil")
    parser.add_argument("--away", default="Argentina")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast", action="store_true", help="30s ticks, shorter output")
    parser.add_argument("--goals-home", type=int, default=None)
    parser.add_argument("--goals-away", type=int, default=None)
    args = parser.parse_args()

    os.environ.setdefault("GFS_SEED", str(args.seed))

    world, _, _ = build_world_and_tournament(ROOT, require_tactics=False)
    if args.home not in world.agents or args.away not in world.agents:
        print(f"Unknown team. Available sample: {list(world.agents.keys())[:8]}...")
        sys.exit(1)

    a1, a2 = world.agents[args.home], world.agents[args.away]
    eff1 = float(a1.status_score)
    eff2 = float(a2.status_score)
    gh, ga, xg1, xg2 = simulate_match_score(eff1, eff2, is_knockout=False)
    if args.goals_home is not None:
        gh = args.goals_home
    if args.goals_away is not None:
        ga = args.goals_away

    cfg = AffectiveConfig.fast_demo() if args.fast else AffectiveConfig()
    internal_1 = a1.simulate_internal_game(stage_pressure=0.4)
    internal_2 = a2.simulate_internal_game(stage_pressure=0.4)
    referee = {
        "profile_name": "demo",
        "strictness": 0.58,
        "bias_t1": 0.05,
    }

    print(f"=== Phase 1b Affective Simulation ===")
    print(f"Match: {args.home} vs {args.away} | Poisson prior {gh}-{ga} xG {xg1:.2f}-{xg2:.2f}")
    print(f"dt={cfg.dt_default}s ticks≈{int(cfg.match_seconds / cfg.dt_default)}")

    summary = run_match_affective_simulation(
        a1,
        a2,
        goals_home=gh,
        goals_away=ga,
        xg_home=xg1,
        xg_away=xg2,
        referee=referee,
        stage_pressure=0.4,
        drama_score=0.55,
        internal_home=internal_1,
        internal_away=internal_2,
        config=cfg,
        seed=args.seed,
        writeback_agents=True,
    )

    print(f"\n--- Crowd / Coach / Ref ---")
    print(f"  ψ_final = {summary.final_psi:+.3f}")
    print(f"  coach stress = {summary.home_coach_stress:.3f} / {summary.away_coach_stress:.3f}")
    print(f"  ref strictness (mean) = {summary.ref_strictness_mean:.3f}")
    print(f"  controversy ∫ = {summary.controversy_integral:.3f}")

    print(f"\n--- Team emotion (mean on pitch) ---")
    for label, emo in ((args.home, summary.home_emotion_mean), (args.away, summary.away_emotion_mean)):
        print(
            f"  {label}: pride={emo.get('pride', 0):.3f} anger={emo.get('anger', 0):.3f} "
            f"fear={emo.get('fear', 0):.3f} det={emo.get('determination', 0):.3f}"
        )

    print(f"\n--- Tactical drift (L1 vs kickoff) ---")
    print(f"  {args.home}: {summary.tactical_drift_home:.3f}")
    print(f"  {args.away}: {summary.tactical_drift_away:.3f}")
    print(f"  {args.home} controls now: {a1.tactical_controls}")

    if summary.timeline_snippet:
        print(f"\n--- Key micro-events ---")
        for line in summary.timeline_snippet:
            print(f"  {line}")

    print(f"\n--- Sample player modulators (home ST) ---")
    from src.match_engine.affective_coupling import AffectiveSpatialCoupling
    from src.match_engine.macro_bridge import build_match_affective_state
    import numpy as np

    st = build_match_affective_state(a1, a2, referee=referee, rng=np.random.default_rng(args.seed))
    eng = AffectiveSpatialCoupling(cfg)
    st_player = next(p for p in st.home.players if p.role == "ST")
    mod = eng.compute_modulators(st_player, st)
    print(
        f"  ST τ_dec={mod.tau_dec:.3f} vision={mod.vision_scale:.3f} "
        f"move_α={mod.move_alpha:.3f} shot_bias={mod.shot_utility_bias:+.3f} foul_ι={mod.foul_impulse:.3f}"
    )
    print("\nDone.")


if __name__ == "__main__":
    main()

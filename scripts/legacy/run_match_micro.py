#!/usr/bin/env python3
"""Phase 1b+2a+2b micro match demo."""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.simulation.world_cup_runner import build_world_and_tournament
from src.match_engine.micro_config import MicroMatchConfig
from src.match_engine.match_micro_runner import run_match_micro_simulation
from src.memory_engine.poisson_simulator import simulate_match_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--home", default="Brazil")
    p.add_argument("--away", default="Argentina")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fast", action="store_true")
    args = p.parse_args()

    world, _, _ = build_world_and_tournament(ROOT)
    a1, a2 = world.agents[args.home], world.agents[args.away]
    gh, ga, xg1, xg2 = simulate_match_score(a1.status_score, a2.status_score)

    cfg = MicroMatchConfig.fast_demo() if args.fast else MicroMatchConfig()
    i1 = a1.simulate_internal_game(0.4)
    i2 = a2.simulate_internal_game(0.4)

    print(f"=== Micro Match (1b+2a+2b) {args.home} vs {args.away} ===")
    print(f"Poisson prior: {gh}-{ga} xG {xg1:.2f}-{xg2:.2f} | dt={cfg.dt_default}s")

    s = run_match_micro_simulation(
        a1,
        a2,
        goals_home=gh,
        goals_away=ga,
        xg_home=xg1,
        xg_away=xg2,
        referee={"strictness": 0.58, "bias_t1": 0.04},
        internal_home=i1,
        internal_away=i2,
        config=cfg,
        seed=args.seed,
    )

    print(f"\nPossession home: {s.possession_home:.1%}")
    print(f"Passes: {s.passes_home} ({s.pass_completion_home:.1%} ok) / {s.passes_away} ({s.pass_completion_away:.1%} ok)")
    print(f"Through balls: {s.through_balls_home} / {s.through_balls_away}")
    print(f"Phi mean: {s.phi_integral_home:.3f} / {s.phi_integral_away:.3f}")
    print(f"Micro xG: {s.micro_xg_home:.2f} - {s.micro_xg_away:.2f}")
    print(f"Crowd psi: {s.final_psi:+.3f}")
    print(f"Coach stress: {s.home_coach_stress:.3f} / {s.away_coach_stress:.3f}")
    eh = s.home_emotion_mean
    print(f"Home emotion: pride={eh['pride']:.2f} anger={eh['anger']:.2f} fear={eh['fear']:.2f}")


if __name__ == "__main__":
    main()

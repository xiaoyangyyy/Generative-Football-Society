#!/usr/bin/env python3
"""Run fast micro matches (no LLM) and record world-model transitions."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("MATCH_MICRO", "1")
os.environ.setdefault("MATCH_MICRO_SCORE", "1")
os.environ.setdefault("MATCH_SCHEDULED_SHOTS", "0")
os.environ.setdefault("MATCH_WM_RECORD", "1")
os.environ.setdefault("MATCH_COGNITIVE", "0")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=24, help="Number of random pair matches")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import numpy as np

    from src.simulation.tournament_2026 import WORLD_CUP_2026_GROUPS
    from src.simulation.world_cup_runner import build_world_and_tournament
    from src.match_engine.match_micro_runner import run_match_micro_simulation
    from src.match_engine.micro_config import MicroMatchConfig
    from src.match_engine.world_model.recorder import TransitionRecorder
    from src.match_engine.world_model.config import default_trace_dir
    from src.match_engine.world_model.observation import encode_observation
    from src.match_engine.world_model.action_codec import zero_action
    from src.match_engine.world_model.config import WorldModelConfig

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    teams = [t for group in WORLD_CUP_2026_GROUPS.values() for t in group]
    rng = np.random.default_rng(args.seed)
    engine, _, _ = build_world_and_tournament(base_dir, require_tactics=False)
    cfg = MicroMatchConfig()
    cfg.use_micro_goals = True
    wm_cfg = WorldModelConfig.from_env()
    trace_root = default_trace_dir(base_dir)
    total = 0

    for i in range(args.pairs):
        t1, t2 = rng.choice(teams, size=2, replace=False)
        a1, a2 = engine.agents[t1], engine.agents[t2]
        rec = TransitionRecorder.for_match(base_dir, t1, t2, f"trace_{i}")
        seed = args.seed + i * 17

        # Tick-level recording via lightweight loop inside runner hook:
        # attach recorder on agents for micro runner
        a1._wm_recorder = rec  # noqa: SLF001
        a2._wm_recorder = rec

        run_match_micro_simulation(
            a1,
            a2,
            goals_home=0,
            goals_away=0,
            xg_home=1.1,
            xg_away=1.0,
            seed=seed,
            config=cfg,
            writeback_agents=False,
            stage_name="wm_collect",
        )
        rec.close()
        total += rec.count
        print(f"[{i+1}/{args.pairs}] {t1} vs {t2} → {rec.count} transitions ({rec.path})")

    print(f"\nDone. {total} transitions under {trace_root}")


if __name__ == "__main__":
    main()

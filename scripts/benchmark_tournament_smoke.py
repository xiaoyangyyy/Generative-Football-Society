#!/usr/bin/env python3
"""Smoke-check macro tournament goal rate vs WC 2022 open data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "data" / "calibration" / "statsbomb_match_baselines.json"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--group-matches", type=int, default=12, help="Simulated group matches (6 pairs x 2)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    goals_band = baseline["metrics"]["goals_per_team_match"]
    target_lo = float(goals_band["p10"])
    target_hi = float(goals_band["p90"])
    target_mean = float(goals_band["mean"])

    from src.match_engine.match_micro_runner import run_match_micro_simulation
    from src.match_engine.micro_config import MicroMatchConfig
    from src.simulation.world_cup_runner import build_world_and_tournament

    pairs = [
        ("Mexico", "South Korea"),
        ("Brazil", "Germany"),
        ("France", "England"),
        ("Argentina", "Netherlands"),
        ("Spain", "Morocco"),
        ("Portugal", "Uruguay"),
    ]
    engine, _, _ = build_world_and_tournament(str(ROOT), require_tactics=False)
    cfg = MicroMatchConfig()
    cfg.use_micro_goals = True

    goals_per_match: list[float] = []
    n = 0
    for home, away in pairs:
        if n >= args.group_matches:
            break
        if home not in engine.agents or away not in engine.agents:
            continue
        for a1, a2 in ((home, away), (away, home)):
            if n >= args.group_matches:
                break
            s = run_match_micro_simulation(
                engine.agents[a1],
                engine.agents[a2],
                goals_home=0,
                goals_away=0,
                xg_home=0.75,
                xg_away=0.75,
                seed=args.seed + n,
                config=cfg,
                writeback_agents=False,
                match_seconds=5400.0,
            )
            goals_per_match.append(float(s.goals_micro_home + s.goals_micro_away))
            n += 1

    avg_goals = float(sum(goals_per_match) / max(1, len(goals_per_match)))
    avg_per_team = avg_goals / 2.0
    ok = target_lo <= avg_per_team <= target_hi
    print(
        json.dumps(
            {
                "matches": len(goals_per_match),
                "avg_goals_per_match": avg_goals,
                "avg_goals_per_team": avg_per_team,
                "target_mean_per_team": target_mean,
                "target_p10_p90": [target_lo, target_hi],
                "in_band": ok,
            },
            indent=2,
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

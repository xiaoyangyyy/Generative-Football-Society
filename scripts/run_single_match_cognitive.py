#!/usr/bin/env python3
"""Run one full micro match with optional cognitive (System 2) layer."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description="Single match: micro + cognitive dual-system")
    p.add_argument("--home", default="Brazil")
    p.add_argument("--away", default="Argentina")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--micro", type=int, default=None, help="1/0 override MATCH_MICRO")
    p.add_argument("--cognitive", type=int, default=None, help="1/0 override MATCH_COGNITIVE")
    p.add_argument("--out", default="", help="JSON report path")
    args = p.parse_args()

    if args.micro is not None:
        os.environ["MATCH_MICRO"] = "1" if args.micro else "0"
    if args.cognitive is not None:
        os.environ["MATCH_COGNITIVE"] = "1" if args.cognitive else "0"
    os.environ.setdefault("GFS_SEED", str(args.seed))
    os.environ.setdefault("MATCH_COGNITIVE_CACHE", str(ROOT / "data" / "cache" / "cognitive"))

    from src.simulation.world_cup_runner import build_world_and_tournament
    from src.simulation.match_pipeline import prepare_match_agents, run_micro_layer
    from src.memory_engine.macro_goal_dynamics import simulate_match_score_dynamics
    from src.memory_engine.macro_micro_fusion import resolve_unified_score

    base = str(ROOT)
    engine, tournament, _ = build_world_and_tournament(base, load_coaches=True)
    home_name, away_name = args.home, args.away
    if home_name not in engine.agents:
        home_name = list(engine.agents.keys())[0]
    if away_name not in engine.agents:
        away_name = list(engine.agents.keys())[1]

    ah, aa = engine.agents[home_name], engine.agents[away_name]
    ah.recover(rest_units=1.2)
    aa.recover(rest_units=1.2)
    prepare_match_agents(ah, aa, base)

    rng = np.random.default_rng(args.seed)
    eff_h = ah.get_effective_status()
    eff_a = aa.get_effective_status()

    ph, pa, xh, xa, _ = simulate_match_score_dynamics(
        ah, aa, eff_h, eff_a, is_knockout=False, stage_pressure=0.35, rng=rng
    )

    micro = run_micro_layer(
        ah,
        aa,
        goals_home=ph,
        goals_away=pa,
        xg_home=xh,
        xg_away=xa,
        referee={"profile_name": "balanced", "strictness": 0.52, "bias_t1": 0.0},
        stage_pressure=0.35,
        drama_score=0.45,
        internal_home={"coordination": 0.62, "conflict_heat": 0.12},
        internal_away={"coordination": 0.58, "conflict_heat": 0.14},
        seed=args.seed,
    )

    gh, ga, xgf, xga, meta = resolve_unified_score(
        ah, aa, eff_h, eff_a, micro_summary=micro, rng=rng
    )

    from src.simulation.match_pipeline import _save_cognitive_match_log

    _save_cognitive_match_log(base, ah, aa, micro, stage_name="single_match_cognitive")

    report = {
        "home": home_name,
        "away": away_name,
        "seed": args.seed,
        "score": f"{gh}-{ga}",
        "xg": f"{xgf:.2f}-{xga:.2f}",
        "macro_preview": f"{ph}-{pa}",
        "micro_xg": f"{micro.micro_xg_home:.2f}-{micro.micro_xg_away:.2f}",
        "fusion_meta": meta,
        "substitutions": micro.substitutions,
        "cognitive_triggers": micro.cognitive_triggers,
        "cognitive_plans": micro.cognitive_plans,
        "cognitive_tier_usage": micro.cognitive_tier_usage,
        "timeline": micro.timeline_snippet,
        "player_stats_teams": list(micro.player_stats.keys()),
        "cognitive_log_path": str(
            ROOT / "data" / "persistence" / "cognitive_log"
            / f"{home_name}_vs_{away_name}_single_match_cognitive.json"
        ),
    }

    out_path = args.out or str(ROOT / "reports" / f"match_{home_name}_vs_{away_name}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

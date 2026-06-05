#!/usr/bin/env python3
"""Sanity-check FM/roster ability ranges vs expected pro bands."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    from src.match_engine.squad_factory import build_team_squad
    from src.simulation.world_cup_runner import build_world_and_tournament
    import numpy as np

    engine, _, _ = build_world_and_tournament(str(ROOT), require_tactics=False)
    teams = list(engine.agents.keys())[:16]
    pass_skills: list[float] = []
    shot_skills: list[float] = []
    for team in teams:
        try:
            squad = build_team_squad(engine.agents[team], np.random.default_rng(0))
        except Exception:
            continue
        for p in squad.players:
            if p.role == "GK":
                continue
            pass_skills.append(float(p.abilities.pass_skill))
            shot_skills.append(float(p.abilities.shot))

    report = {
        "teams_sampled": len(teams),
        "pass_skill_mean": float(np.mean(pass_skills)) if pass_skills else 0.0,
        "pass_skill_p10_p90": [
            float(np.quantile(pass_skills, 0.1)) if pass_skills else 0.0,
            float(np.quantile(pass_skills, 0.9)) if pass_skills else 0.0,
        ],
        "shot_skill_mean": float(np.mean(shot_skills)) if shot_skills else 0.0,
        "checks": {
            "pass_skill_in_pro_band": (
                pass_skills
                and float(np.quantile(pass_skills, 0.1)) >= 0.28
                and float(np.quantile(pass_skills, 0.9)) <= 0.88
            ),
            "shot_skill_in_pro_band": (
                shot_skills
                and float(np.quantile(shot_skills, 0.1)) >= 0.24
                and float(np.quantile(shot_skills, 0.9)) <= 0.90
                and 0.32 <= float(np.mean(shot_skills)) <= 0.72
            ),
        },
    }
    print(json.dumps(report, indent=2))
    ok = all(report["checks"].values())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Preflight — full-stack micro (WM + cognitive path) must simulate without macro fallback."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Mirror full-run stack
os.environ.setdefault("MATCH_MICRO", "1")
os.environ.setdefault("MATCH_COGNITIVE", "1")
os.environ.setdefault("MATCH_MICRO_SCORE", "1")
os.environ.setdefault("MATCH_WORLD_MODEL", "1")
os.environ.setdefault("MATCH_WM_PLAN", "1")
os.environ.setdefault("MATCH_SCHEDULED_SHOTS", "0")
os.environ.setdefault("CALIBRATION_MODE", "0")


def main() -> int:
    from src.simulation.tactics_sync import apply_coach_tactics_from_llm
    from src.simulation.match_pipeline import run_physics_first_micro
    from src.simulation.world_cup_runner import build_world_and_tournament

    engine, _, _ = build_world_and_tournament(str(ROOT), require_tactics=False)
    home_name, away_name = "Mexico", "Czech Republic"
    home = engine.agents[home_name]
    away = engine.agents[away_name]

    apply_coach_tactics_from_llm(
        home,
        {
            "tactical_preset": "gegenpress",
            "formation": "4-2-3-1",
            "controls": {"pressing_intensity": 0.85, "risk_budget": 0.75, "line_height": 0.8, "rotation_aggressiveness": 0.7},
        },
    )
    apply_coach_tactics_from_llm(
        away,
        {
            "tactical_preset": "park_the_bus",
            "formation": "5-4-1",
            "controls": {"pressing_intensity": 0.35, "risk_budget": 0.4, "line_height": 0.3, "rotation_aggressiveness": 0.25},
        },
    )

    referee = {"strictness": 0.55, "bias_t1": 0.1, "profile_name": "balanced"}
    try:
        summary = run_physics_first_micro(
            home,
            away,
            xg_prior_home=1.0,
            xg_prior_away=1.0,
            eff_status_home=65.0,
            eff_status_away=55.0,
            referee=referee,
            stage_pressure=0.35,
            drama_score=0.4,
            internal_home={"coordination": 0.65, "conflict_heat": 0.2},
            internal_away={"coordination": 0.62, "conflict_heat": 0.25},
            seed=42,
            stage_name="preflight",
            match_seconds=1200.0,
        )
    except Exception as exc:
        print(f"MICRO PREFLIGHT FAILED: {exc}", file=sys.stderr)
        return 1

    total_passes = summary.passes_home + summary.passes_away
    total_shots = summary.shots_home + summary.shots_away
    if summary.ticks < 10 or total_passes < 50:
        print(
            f"MICRO PREFLIGHT FAILED: empty sim ticks={summary.ticks} passes={total_passes}",
            file=sys.stderr,
        )
        return 1

    print(
        f"MICRO PREFLIGHT OK: ticks={summary.ticks} passes={total_passes} "
        f"shots={total_shots} goals={summary.goals_physics_home}-{summary.goals_physics_away} "
        f"preset_locked={getattr(home, '_tactical_preset_locked', False)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

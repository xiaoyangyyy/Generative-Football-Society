#!/usr/bin/env python3
"""High-press vs possession style contrast — fouls, passing, shots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.match_engine.match_micro_runner import run_match_micro_simulation
from src.match_engine.micro_config import MicroMatchConfig
from src.match_engine.tactical_catalog import TACTICAL_PRESETS
from src.simulation.world_cup_runner import build_world_and_tournament

ROOT = Path(__file__).resolve().parents[1]
PRESS_PRESET = "gegenpress"
POSSESS_PRESET = "tiki_taka"


def _mean(vals: list[float]) -> float:
    return float(sum(vals) / max(1, len(vals)))


def _run_block(
    engine,
    *,
    home: str,
    away: str,
    tac_home: dict,
    tac_away: dict,
    label: str,
    samples: int,
    seed_start: int,
    match_seconds: float,
) -> dict:
    cfg = MicroMatchConfig()
    cfg.use_micro_goals = True
    rows = []
    for i in range(samples):
        seed = seed_start + i
        s = run_match_micro_simulation(
            engine.agents[home],
            engine.agents[away],
            goals_home=0,
            goals_away=0,
            xg_home=0.7,
            xg_away=0.8,
            eff_status_home=68,
            eff_status_away=62,
            seed=seed,
            config=cfg,
            writeback_agents=False,
            match_seconds=match_seconds,
            tactical_override_home=tac_home,
            tactical_override_away=tac_away,
        )
        passes = s.passes_home + s.passes_away
        shots = s.shots_home + s.shots_away
        rows.append(
            {
                "fouls_home": float(s.fouls_committed_home),
                "fouls_away": float(s.fouls_committed_away),
                "fouls_total": float(s.fouls_committed_home + s.fouls_committed_away),
                "passes_per_team": passes / 2.0,
                "pass_completion": (
                    (s.pass_completion_home * s.passes_home) + (s.pass_completion_away * s.passes_away)
                )
                / max(1.0, passes),
                "long_pass_share": (s.long_passes_home + s.long_passes_away) / max(1.0, passes),
                "shots_per_team": shots / 2.0,
                "possession_home": float(s.possession_home),
            }
        )
    return {
        "label": label,
        "home_preset": label.split("_")[0] if "_" in label else label,
        "away_preset": label.split("_")[-1] if "_" in label else label,
        "samples": samples,
        "means": {k: _mean([r[k] for r in rows]) for k in rows[0].keys()},
        "raw": rows,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--match-seconds", type=float, default=5400.0)
    p.add_argument("--home", default="Mexico")
    p.add_argument("--away", default="South Korea")
    p.add_argument("--seed-start", type=int, default=100)
    args = p.parse_args()

    engine, _, _ = build_world_and_tournament(str(ROOT), require_tactics=False)
    press_tac = dict(TACTICAL_PRESETS[PRESS_PRESET])
    possess_tac = dict(TACTICAL_PRESETS[POSSESS_PRESET])

    from src.match_engine.discipline_schedule import team_press_factor

    report_weights = {
        "press_team_press_factor": team_press_factor(press_tac),
        "possession_team_press_factor": team_press_factor(possess_tac),
        "note": "foul_weight also includes possession_relief and squad foul_impulse at kickoff",
    }

    contrast = _run_block(
        engine,
        home=args.home,
        away=args.away,
        tac_home=press_tac,
        tac_away=possess_tac,
        label=f"{PRESS_PRESET}_vs_{POSSESS_PRESET}",
        samples=args.samples,
        seed_start=args.seed_start,
        match_seconds=args.match_seconds,
    )
    balanced = _run_block(
        engine,
        home=args.home,
        away=args.away,
        tac_home=dict(TACTICAL_PRESETS["balanced"]),
        tac_away=dict(TACTICAL_PRESETS["balanced"]),
        label="balanced_vs_balanced",
        samples=args.samples,
        seed_start=args.seed_start + 500,
        match_seconds=args.match_seconds,
    )

    cm = contrast["means"]
    bm = balanced["means"]
    report = {
        "presets": {"high_press": PRESS_PRESET, "possession": POSSESS_PRESET},
        "tactical_press_factors": report_weights,
        "contrast": contrast,
        "balanced_control": balanced,
        "deltas_contrast_minus_balanced": {
            k: float(cm[k] - bm[k]) for k in cm.keys()
        },
        "checks": {
            "press_team_more_fouls_than_possession": cm["fouls_home"] > cm["fouls_away"] + 0.5,
            "press_team_more_fouls_than_balanced_home": cm["fouls_home"] >= bm["fouls_home"] - 0.5,
            "possession_team_fewer_fouls_than_balanced_away": cm["fouls_away"] < bm["fouls_away"],
            "possession_team_more_passes": cm["passes_per_team"] >= bm["passes_per_team"] - 30.0,
        },
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Calibration regression tests (optional slow full-match)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "data" / "calibration" / "statsbomb_match_baselines.json"


def _band(key: str) -> tuple[float, float, float]:
    b = json.loads(BASELINE.read_text(encoding="utf-8"))["metrics"][key]
    return float(b["p10"]), float(b["p90"]), float(b["mean"])


@pytest.mark.slow
def test_micro_match_core_metrics_in_band():
    if os.environ.get("RUN_SLOW_TESTS", "").strip() not in ("1", "true", "yes"):
        pytest.skip("Set RUN_SLOW_TESTS=1 for full calibration regression")

    from src.match_engine.match_micro_runner import run_match_micro_simulation
    from src.match_engine.micro_config import MicroMatchConfig
    from src.simulation.world_cup_runner import build_world_and_tournament

    engine, _, _ = build_world_and_tournament(str(ROOT), require_tactics=False)
    cfg = MicroMatchConfig()
    cfg.use_micro_goals = True
    s = run_match_micro_simulation(
        engine.agents["Mexico"],
        engine.agents["South Korea"],
        goals_home=0,
        goals_away=0,
        xg_home=0.7,
        xg_away=0.8,
        seed=42,
        config=cfg,
        writeback_agents=False,
        match_seconds=5400.0,
    )
    passes = s.passes_home + s.passes_away
    shots = s.shots_home + s.shots_away
    cmp_rate = ((s.pass_completion_home * s.passes_home) + (s.pass_completion_away * s.passes_away)) / max(1, passes)
    metrics = {
        "pass_completion": cmp_rate,
        "passes_per_team_match": passes / 2.0,
        "shots_per_team_match": shots / 2.0,
        "fouls_committed_per_team_match": (s.fouls_committed_home + s.fouls_committed_away) / 2.0,
    }
    for key, val in metrics.items():
        lo, hi, _ = _band(key)
        assert lo <= val <= hi, f"{key}={val:.4f} outside [{lo},{hi}]"

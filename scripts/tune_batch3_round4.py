#!/usr/bin/env python3
"""Batch-3 r4 — xg_geom + gk pair; r3 shot pair fixed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.match_engine.calibration.ablation import PIPELINE_PRESETS, ablation_context
from src.match_engine.calibration.apply_params import apply_params_to_config
from src.match_engine.calibration.benchmark_core import run_micro_benchmark_rows
from src.match_engine.calibration.contract import evaluate_rows, load_contract, load_statsbomb_baselines
from src.match_engine.calibration.profile import load_profile
from src.match_engine.micro_config import MicroMatchConfig

# (xg_geom_base, gk_b_reflex)
CANDIDATES = [
    (0.50, 3.32),  # r3 baseline
    (0.52, 3.32),
    (0.52, 3.34),
    (0.54, 3.32),
    (0.50, 3.34),
    (0.50, 3.36),
    (0.52, 3.36),
    (0.54, 3.34),
]


def focus_loss(ev: dict) -> float:
    hard = ev["hard_metrics"]
    soft = ev["soft_constraints"]
    goals = float(hard["goals_per_team_match"]["sim_mean"])
    shots = float(hard["shots_per_team_match"]["sim_mean"])
    gxg = float(soft["goals_to_micro_xg_ratio"]["sim_mean"])
    loss = 1.5 * ((goals - 1.5234375) / 1.48) ** 2
    loss += 1.2 * ((gxg - 1.2488266677293303) / 1.19) ** 2
    loss += 0.8 * ((shots - 9.0) / 6.24) ** 2
    return loss


def main() -> int:
    profile = load_profile()
    spec = PIPELINE_PRESETS["M0"]
    contract = load_contract()
    baselines = load_statsbomb_baselines()
    fixtures = [("Mexico", "South Korea"), ("Brazil", "Germany")]

    best = None
    rows_out = []
    for i, (xg, gk) in enumerate(CANDIDATES):
        cfg = MicroMatchConfig()
        cfg.use_micro_goals = True
        merged = dict(profile.overrides)
        merged["shot.xg_geom_base"] = xg
        merged["shot.gk_b_reflex"] = gk
        apply_params_to_config(cfg, merged, profile)
        cfg.action_shot_dist_bonus = 2.95
        cfg.action_shot_cooldown_sec = 10.0

        with ablation_context(spec) as _:
            rows = run_micro_benchmark_rows(
                root=str(ROOT),
                fixtures=fixtures,
                samples=1,
                match_seconds=5400.0,
                spec=spec,
                cfg=cfg,
            )
        ev = evaluate_rows(rows, contract=contract, baselines=baselines)
        hard = ev["hard_metrics"]
        soft = ev["soft_constraints"]
        row = {
            "id": i,
            "shot.xg_geom_base": xg,
            "shot.gk_b_reflex": gk,
            "focus_loss": focus_loss(ev),
            "all_pass": ev["all_pass"],
            "passes": hard["passes_per_team_match"]["sim_mean"],
            "shots": hard["shots_per_team_match"]["sim_mean"],
            "goals": hard["goals_per_team_match"]["sim_mean"],
            "gxg": soft["goals_to_micro_xg_ratio"]["sim_mean"],
            "micro_xg": soft["micro_xg_per_team_match"]["sim_mean"],
        }
        rows_out.append(row)
        print(json.dumps(row), flush=True)
        if best is None or row["focus_loss"] < best["focus_loss"]:
            best = row

    path = ROOT / "reports" / "batch3_r4_tune.json"
    path.write_text(json.dumps({"candidates": rows_out, "best": best}, indent=2) + "\n", encoding="utf-8")
    print("\nBEST:", json.dumps(best, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Batch-3 round 2 — goals/xG pair (xg_geom + gk) + light pass bump."""

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

TARGETS = {
    "goals_per_team_match": 1.5234375,
    "passes_per_team_match": 535.27,
    "shots_per_team_match": 11.67,
}
SOFT = {"goals_to_micro_xg_ratio": 1.2488266677293303}

CANDIDATES = [
    {"shot.xg_geom_base": 0.48, "shot.gk_b_reflex": 3.28},
    {"shot.xg_geom_base": 0.50, "shot.gk_b_reflex": 3.30},
    {"shot.xg_geom_base": 0.48, "shot.gk_b_reflex": 3.32, "action.action_pass_base": 1.64},
    {"shot.xg_geom_base": 0.50, "shot.gk_b_reflex": 3.32, "action.action_pass_base": 1.62},
    {"shot.xg_geom_base": 0.46, "shot.gk_b_reflex": 3.35, "action.action_pass_base": 1.62},
    {"shot.xg_geom_base": 0.52, "shot.gk_b_reflex": 3.28, "action.action_pass_base": 1.62},
    {"shot.xg_geom_base": 0.48, "shot.gk_b_reflex": 3.30, "action.action_pass_base": 1.62, "action.action_shot_dist_bonus": 2.80},
]


def focus_loss(ev: dict) -> float:
    hard = ev["hard_metrics"]
    loss = 0.0
    for key, tgt in TARGETS.items():
        sim = float(hard[key]["sim_mean"])
        scale = max(1e-6, float(hard[key]["scale"]))
        loss += 1.2 * ((sim - tgt) / scale) ** 2
    gxg = ev["soft_constraints"]["goals_to_micro_xg_ratio"]
    z = abs(float(gxg["sim_mean"]) - SOFT["goals_to_micro_xg_ratio"]) / max(1e-6, float(gxg["scale"]))
    loss += 1.5 * z * z
    return loss


def main() -> int:
    profile = load_profile()
    spec = PIPELINE_PRESETS["M0"]
    contract = load_contract()
    baselines = load_statsbomb_baselines()
    fixtures = [("Mexico", "South Korea"), ("Brazil", "Germany")]

    best = None
    rows_out = []
    for i, ov in enumerate(CANDIDATES):
        cfg = MicroMatchConfig()
        cfg.use_micro_goals = True
        merged = dict(profile.overrides)
        merged.update(ov)
        apply_params_to_config(cfg, merged, profile)
        if "action.action_shot_dist_bonus" in ov:
            cfg.action_shot_dist_bonus = float(ov["action.action_shot_dist_bonus"])
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
            **ov,
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

    out = {"candidates": rows_out, "best": best}
    path = ROOT / "reports" / "batch3_r2_tune.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("\nBEST:", json.dumps(best, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

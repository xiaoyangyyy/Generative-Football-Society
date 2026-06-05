#!/usr/bin/env python3
"""Batch-3 round 1 — evaluate a small candidate set (fast)."""

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

TARGETS = {"passes_per_team_match": 535.27, "shots_per_team_match": 11.67}

CANDIDATES = [
    {"action.action_pass_base": 1.72, "action.action_shot_base": 0.50, "action.action_shot_cooldown_sec": 10.0},
    {"action.action_pass_base": 1.76, "action.action_shot_base": 0.54, "action.action_shot_cooldown_sec": 9.5},
    {"action.action_pass_base": 1.68, "action.action_shot_base": 0.56, "action.action_shot_cooldown_sec": 9.0},
    {"action.action_pass_base": 1.72, "action.action_shot_base": 0.58, "action.action_shot_cooldown_sec": 8.5},
    {"action.action_pass_base": 1.80, "action.action_shot_base": 0.52, "action.action_shot_cooldown_sec": 9.5},
    {"action.action_pass_base": 1.72, "action.action_shot_base": 0.62, "action.action_shot_cooldown_sec": 8.0},
]


def focus_loss(ev: dict) -> float:
    hard = ev["hard_metrics"]
    loss = 0.0
    for key, tgt in TARGETS.items():
        sim = float(hard[key]["sim_mean"])
        scale = max(1e-6, float(hard[key]["scale"]))
        loss += ((sim - tgt) / scale) ** 2
    soft = ev["soft_constraints"]
    for sk in ("goals_to_micro_xg_ratio", "micro_xg_per_team_match"):
        row = soft.get(sk, {})
        if row and not row.get("skipped"):
            z = abs(float(row["sim_mean"]) - float(row["target_mean"])) / max(1e-6, float(row["scale"]))
            loss += 0.4 * z * z
    return loss


def main() -> int:
    profile = load_profile()
    spec = PIPELINE_PRESETS["M0"]
    contract = load_contract()
    baselines = load_statsbomb_baselines()
    fixtures = [("Mexico", "South Korea"), ("Brazil", "Germany")]

    results = []
    best = None
    for i, ov in enumerate(CANDIDATES):
        cfg = MicroMatchConfig()
        cfg.use_micro_goals = True
        merged = dict(profile.overrides)
        merged.update(ov)
        apply_params_to_config(cfg, merged, profile)
        with ablation_context(spec) as _:
            rows = run_micro_benchmark_rows(
                root=str(ROOT),
                fixtures=fixtures,
                samples=1,
                match_seconds=1200.0,
                spec=spec,
                cfg=cfg,
            )
        ev = evaluate_rows(rows, contract=contract, baselines=baselines)
        hard = ev["hard_metrics"]
        row = {
            "id": i,
            **ov,
            "focus_loss": focus_loss(ev),
            "all_pass": ev["all_pass"],
            "passes": hard["passes_per_team_match"]["sim_mean"],
            "shots": hard["shots_per_team_match"]["sim_mean"],
            "goals": hard["goals_per_team_match"]["sim_mean"],
            "gxg": ev["soft_constraints"]["goals_to_micro_xg_ratio"]["sim_mean"],
            "micro_xg": ev["soft_constraints"]["micro_xg_per_team_match"]["sim_mean"],
        }
        results.append(row)
        print(json.dumps(row), flush=True)
        if best is None or row["focus_loss"] < best["focus_loss"]:
            best = row

    out = {"candidates": results, "best": best}
    path = ROOT / "reports" / "batch3_r1_tune.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("\nBEST:", json.dumps(best, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

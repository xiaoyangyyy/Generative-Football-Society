#!/usr/bin/env python3
"""Tune narrative anomaly knobs on low_block stress fixtures (M1 + WM)."""

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
from src.match_engine.calibration.narrative_gate import evaluate_narrative_rows
from src.match_engine.calibration.profile import load_profile
from src.match_engine.micro_config import MicroMatchConfig
from src.match_engine.tactical_catalog import TACTICAL_PRESETS

FIXTURES = [
    ("Canada", "Bosnia and Herzegovina"),
    ("Bosnia and Herzegovina", "Qatar"),
    ("Brazil", "Morocco"),
    ("United States", "Australia"),
]

CANDIDATES = [
    {},
    {"action_shot_cooldown_sec": 12.0, "action_shot_volume_decay": 0.065},
    {"action_shot_cooldown_sec": 13.0, "action_shot_volume_decay": 0.075, "action_shot_skew_decay": 0.12},
    {"action_shot_cooldown_sec": 12.0, "action_shot_volume_decay": 0.08, "shot_finish_xg_scale": 1.10},
    {"action_shot_cooldown_sec": 13.0, "action_shot_volume_decay": 0.08, "shot_finish_save_scale": 0.72},
    {"action_shot_cooldown_sec": 12.0, "action_shot_volume_decay": 0.07, "gk_b_reflex": 3.36},
]

CFG_FIELDS = (
    "action_shot_cooldown_sec",
    "action_shot_volume_decay",
    "action_shot_skew_decay",
    "shot_finish_xg_scale",
    "shot_finish_save_scale",
    "gk_b_reflex",
)


def apply_cfg_overrides(cfg: MicroMatchConfig, overrides: dict) -> None:
    apply_params_to_config(cfg, overrides, profile=load_profile())
    for k, v in overrides.items():
        if k in CFG_FIELDS and hasattr(cfg, k):
            setattr(cfg, k, float(v))


def focus_loss(narr: dict, cal: dict) -> float:
    loss = 8.0 * narr["anomaly_rate"]
    loss += 2.0 * max(0.0, narr["max_match_goals"] - 5.0)
    loss += 1.5 * max(0.0, narr["max_match_shots"] - 34.0)
    loss += 1.0 * max(0.0, narr["avg_gxg_ratio"] - 2.0)
    if cal.get("all_pass"):
        loss -= 0.5
    hard = cal.get("hard_metrics", {})
    goals = float(hard.get("goals_per_team_match", {}).get("sim_mean", 1.5))
    loss += 0.8 * ((goals - 1.52) / 1.5) ** 2
    return loss


def main() -> int:
    profile = load_profile()
    spec = PIPELINE_PRESETS["M1"]
    contract = load_contract()
    baselines = load_statsbomb_baselines()
    tac = dict(TACTICAL_PRESETS["low_block"])
    best = None
    rows_out = []

    for i, overrides in enumerate(CANDIDATES):
        cfg = MicroMatchConfig()
        cfg.use_micro_goals = True
        merged = dict(profile.overrides)
        apply_params_to_config(cfg, merged, profile)
        apply_cfg_overrides(cfg, overrides)

        with ablation_context(spec) as _:
            rows = run_micro_benchmark_rows(
                root=str(ROOT),
                fixtures=FIXTURES,
                samples=1,
                match_seconds=5400.0,
                spec=spec,
                cfg=cfg,
                seed_start=42,
                tactical_override_home=tac,
                tactical_override_away=tac,
            )
        narr = evaluate_narrative_rows(rows)
        cal = evaluate_rows(rows, contract=contract, baselines=baselines)
        row = {
            "id": i,
            "overrides": overrides,
            "focus_loss": focus_loss(narr, cal),
            "narrative_all_pass": narr["all_pass"],
            "calibration_all_pass": cal["all_pass"],
            "anomaly_rate": narr["anomaly_rate"],
            "max_goals": narr["max_match_goals"],
            "max_shots": narr["max_match_shots"],
            "avg_gxg": narr["avg_gxg_ratio"],
        }
        rows_out.append(row)
        print(json.dumps(row), flush=True)
        if best is None or row["focus_loss"] < best["focus_loss"]:
            best = row

    path = ROOT / "reports" / "narrative_anomaly_tune.json"
    path.write_text(json.dumps({"candidates": rows_out, "best": best}, indent=2) + "\n", encoding="utf-8")
    print("\nBEST:", json.dumps(best, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

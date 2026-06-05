#!/usr/bin/env python3
"""Quick grid for batch-2 knob pairs (Step 1 xG/GK, then report gate metrics)."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.match_engine.calibration.ablation import PIPELINE_PRESETS, ablation_context
from src.match_engine.calibration.apply_params import apply_params_to_config
from src.match_engine.calibration.benchmark_core import run_micro_benchmark_rows
from src.match_engine.calibration.contract import evaluate_rows, load_contract, load_statsbomb_baselines
from src.match_engine.calibration.objective import calibration_loss
from src.match_engine.calibration.profile import load_profile
from src.match_engine.micro_config import MicroMatchConfig

FOCUS = (
    "headers_per_team_match",
    "goals_per_team_match",
    "goals_to_micro_xg_ratio",
    "micro_xg_per_team_match",
    "passes_per_team_match",
    "crosses_per_team_match",
    "yellow_cards_per_team_match",
)


def _eval(overrides: dict, *, fixtures, samples: int, match_seconds: float) -> dict:
    profile = load_profile()
    spec = PIPELINE_PRESETS["M0"]
    cfg = MicroMatchConfig()
    cfg.use_micro_goals = True
    merged = dict(profile.overrides)
    merged.update(overrides)
    apply_params_to_config(cfg, merged, profile)
    with ablation_context(spec) as _:
        rows = run_micro_benchmark_rows(
            root=str(ROOT),
            fixtures=fixtures,
            samples=samples,
            match_seconds=match_seconds,
            spec=spec,
            cfg=cfg,
        )
    return evaluate_rows(rows, contract=load_contract(), baselines=load_statsbomb_baselines())


def main() -> int:
    fixtures = [("Mexico", "South Korea")]
    samples = 1
    match_seconds = 1200.0

    print("=== Step 1: xg_geom_base x gk_b_reflex ===")
    best = None
    for xg, gk in itertools.product([0.40, 0.44, 0.48], [3.10, 3.20, 3.25]):
        gk = min(gk, 3.35)  # registry upper bound
        ov = {"shot.xg_geom_base": xg, "shot.gk_b_reflex": gk}
        ev = _eval(ov, fixtures=fixtures, samples=samples, match_seconds=match_seconds)
        loss = calibration_loss(ev)
        hard = ev["hard_metrics"]
        soft = ev["soft_constraints"]
        row = {
            "xg_geom_base": xg,
            "gk_b_reflex": gk,
            "loss": loss,
            "all_pass": ev["all_pass"],
            "headers": hard.get("headers_per_team_match", {}).get("sim_mean"),
            "goals": hard.get("goals_per_team_match", {}).get("sim_mean"),
            "gxg_ratio": soft.get("goals_to_micro_xg_ratio", {}).get("sim_mean"),
            "micro_xg": soft.get("micro_xg_per_team_match", {}).get("sim_mean"),
        }
        print(json.dumps(row))
        if best is None or loss < best["loss"]:
            best = row

    print("BEST step1:", json.dumps(best, indent=2))
    step1 = {"shot.xg_geom_base": best["xg_geom_base"], "shot.gk_b_reflex": best["gk_b_reflex"]}

    print("\n=== Step 2: pass_base x cross_base x cross_tick (fixed step1) ===")
    best2 = None
    for pb, cb, ct in itertools.product([1.42, 1.46, 1.50], [0.58, 0.62, 0.66], [18.0, 20.0, 22.0]):
        ov = {
            **step1,
            "action.action_pass_base": pb,
            "action.action_cross_base": cb,
            "discipline.cross_tick_target": ct,
        }
        ev = _eval(ov, fixtures=fixtures, samples=samples, match_seconds=match_seconds)
        loss = calibration_loss(ev)
        hard = ev["hard_metrics"]
        row = {
            **ov,
            "loss": loss,
            "all_pass": ev["all_pass"],
            "passes": hard.get("passes_per_team_match", {}).get("sim_mean"),
            "crosses": hard.get("crosses_per_team_match", {}).get("sim_mean"),
            "headers": hard.get("headers_per_team_match", {}).get("sim_mean"),
            "pass_completion": hard.get("pass_completion", {}).get("sim_mean"),
        }
        print(json.dumps(row))
        if best2 is None or loss < best2["loss"]:
            best2 = row

    print("BEST step2:", json.dumps(best2, indent=2))
    step2 = {k: best2[k] for k in step1 | {
        "action.action_pass_base": best2["action.action_pass_base"],
        "action.action_cross_base": best2["action.action_cross_base"],
        "discipline.cross_tick_target": best2["discipline.cross_tick_target"],
    }}

    print("\n=== Step 3: yellow_on_foul (fixed step1+2) ===")
    best3 = None
    for yf in [0.016, 0.020, 0.024]:
        ov = {**step2, "discipline.discipline_yellow_on_foul_base": yf}
        ev = _eval(ov, fixtures=fixtures, samples=samples, match_seconds=match_seconds)
        loss = calibration_loss(ev)
        hard = ev["hard_metrics"]
        row = {
            **ov,
            "loss": loss,
            "all_pass": ev["all_pass"],
            "yellows": hard.get("yellow_cards_per_team_match", {}).get("sim_mean"),
            "score_combined": ev["score_combined"],
        }
        print(json.dumps(row))
        if best3 is None or loss < best3["loss"]:
            best3 = row

    print("\n=== FINAL OVERRIDES ===")
    final = {k: best3[k] for k in step2 | {"discipline.discipline_yellow_on_foul_base": best3["discipline.discipline_yellow_on_foul_base"]}}
    print(json.dumps(final, indent=2))
    out = ROOT / "reports" / "batch2_tune_result.json"
    out.write_text(json.dumps({"overrides": final, "eval": best3}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

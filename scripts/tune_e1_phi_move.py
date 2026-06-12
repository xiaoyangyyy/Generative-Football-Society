#!/usr/bin/env python3
"""E1 parameter scan — phi_move_scale × w_phi_move with M0 profile."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.match_engine.calibration.ablation import PIPELINE_PRESETS, ablation_context
from src.match_engine.calibration.apply_params import apply_params_to_config
from src.match_engine.calibration.benchmark_core import run_micro_benchmark_rows
from src.match_engine.calibration.contract import evaluate_rows, load_contract, load_statsbomb_baselines
from src.match_engine.calibration.profile import load_profile
from src.match_engine.micro_config import MicroMatchConfig

FIXTURES = [("Mexico", "South Korea"), ("Brazil", "Germany")]
MATCH_SECONDS = 5400.0

# scale 0.04 gave shots +4.3 vs off; scan lower band for ~11.7 target
CANDIDATES = [
    (0.010, 0.50),
    (0.015, 0.50),
    (0.020, 0.50),
    (0.025, 0.50),
    (0.030, 0.50),
    (0.035, 0.50),
    (0.040, 0.50),
    (0.025, 0.35),
    (0.025, 0.40),
    (0.025, 0.45),
]

TARGET_SHOTS = 11.671875
TARGET_GOALS = 1.5234375
TARGET_GXG = 1.2488266677293303
TARGET_PASSES = 535.2734375


def focus_loss(ev: dict) -> float:
    hard = ev["hard_metrics"]
    soft = ev["soft_constraints"]
    goals = float(hard["goals_per_team_match"]["sim_mean"])
    shots = float(hard["shots_per_team_match"]["sim_mean"])
    passes = float(hard["passes_per_team_match"]["sim_mean"])
    gxg = float(soft["goals_to_micro_xg_ratio"]["sim_mean"])
    loss = 1.4 * ((goals - TARGET_GOALS) / 1.48) ** 2
    loss += 1.0 * ((gxg - TARGET_GXG) / 1.19) ** 2
    loss += 1.2 * ((shots - TARGET_SHOTS) / 6.24) ** 2
    loss += 0.4 * ((passes - TARGET_PASSES) / 148.0) ** 2
    return loss


def _run(*, phi_on: bool, scale: float = 0.04, w_phi: float = 0.50) -> dict:
    profile = load_profile()
    spec = PIPELINE_PRESETS["M0"]
    cfg = MicroMatchConfig()
    cfg.use_micro_goals = True
    cfg.enable_phi_gradient_move = phi_on
    cfg.phi_move_scale = scale
    cfg.w_phi_move = w_phi
    apply_params_to_config(cfg, dict(profile.overrides), profile)

    t0 = time.time()
    with ablation_context(spec):
        rows = run_micro_benchmark_rows(
            root=str(ROOT),
            fixtures=FIXTURES,
            samples=1,
            match_seconds=MATCH_SECONDS,
            spec=spec,
            cfg=cfg,
        )
    ev = evaluate_rows(rows, contract=load_contract(), baselines=load_statsbomb_baselines())
    hard = ev["hard_metrics"]
    soft = ev["soft_constraints"]
    return {
        "phi_gradient_move": phi_on,
        "phi_move_scale": scale,
        "w_phi_move": w_phi,
        "elapsed_s": round(time.time() - t0, 1),
        "all_pass": ev["all_pass"],
        "loss": focus_loss(ev),
        "passes": hard["passes_per_team_match"]["sim_mean"],
        "shots": hard["shots_per_team_match"]["sim_mean"],
        "goals": hard["goals_per_team_match"]["sim_mean"],
        "pass_completion": hard["pass_completion"]["sim_mean"],
        "gxg": soft["goals_to_micro_xg_ratio"]["sim_mean"],
        "micro_xg": soft["micro_xg_per_team_match"]["sim_mean"],
    }


def main() -> int:
    results: list[dict] = []
    print("[E1] baseline off", flush=True)
    off = _run(phi_on=False)
    off["label"] = "e1_off"
    results.append(off)

    best = None
    for i, (scale, w_phi) in enumerate(CANDIDATES):
        label = f"scale={scale:.3f}_w={w_phi:.2f}"
        print(f"[E1] {i + 1}/{len(CANDIDATES)} {label}", flush=True)
        row = _run(phi_on=True, scale=scale, w_phi=w_phi)
        row["label"] = label
        results.append(row)
        if best is None or row["loss"] < best["loss"]:
            best = row

    out = {
        "fixtures": [f"{a}_vs_{b}" for a, b in FIXTURES],
        "match_seconds": MATCH_SECONDS,
        "baseline_off": off,
        "candidates": results[1:],
        "best": best,
        "targets": {
            "shots": TARGET_SHOTS,
            "goals": TARGET_GOALS,
            "gxg": TARGET_GXG,
            "passes": TARGET_PASSES,
        },
    }
    path = ROOT / "reports" / "e1_phi_move_tune.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best": best, "baseline_off": off}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

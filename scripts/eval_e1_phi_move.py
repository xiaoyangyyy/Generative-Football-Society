#!/usr/bin/env python3
"""E1 smoke — compare M0 gate metrics with phi gradient movement on/off."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.match_engine.calibration.ablation import PIPELINE_PRESETS, ablation_context
from src.match_engine.calibration.benchmark_core import run_micro_benchmark_rows
from src.match_engine.calibration.contract import evaluate_rows, load_contract, load_statsbomb_baselines
from src.match_engine.micro_config import MicroMatchConfig

FIXTURES = [("Mexico", "South Korea"), ("Brazil", "Germany")]


def _run(label: str, *, phi_on: bool) -> dict:
    spec = PIPELINE_PRESETS["M0"]
    cfg = MicroMatchConfig()
    cfg.use_micro_goals = True
    cfg.enable_phi_gradient_move = phi_on
    with ablation_context(spec) as _:
        rows = run_micro_benchmark_rows(
            root=str(ROOT),
            fixtures=FIXTURES,
            samples=1,
            match_seconds=5400.0,
            spec=spec,
            cfg=cfg,
        )
    ev = evaluate_rows(rows, contract=load_contract(), baselines=load_statsbomb_baselines())
    hard = ev["hard_metrics"]
    soft = ev["soft_constraints"]
    return {
        "label": label,
        "phi_gradient_move": phi_on,
        "all_pass": ev["all_pass"],
        "passes": hard["passes_per_team_match"]["sim_mean"],
        "shots": hard["shots_per_team_match"]["sim_mean"],
        "goals": hard["goals_per_team_match"]["sim_mean"],
        "pass_completion": hard["pass_completion"]["sim_mean"],
        "gxg": soft["goals_to_micro_xg_ratio"]["sim_mean"],
        "micro_xg": soft["micro_xg_per_team_match"]["sim_mean"],
    }


def main() -> int:
    off = _run("e1_off", phi_on=False)
    on = _run("e1_on", phi_on=True)
    out = {"off": off, "on": on}
    path = ROOT / "reports" / "e1_phi_move_smoke.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

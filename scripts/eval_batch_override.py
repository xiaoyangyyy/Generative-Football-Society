#!/usr/bin/env python3
"""Single M0 benchmark eval with profile overrides (for batch-3 validation)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.match_engine.calibration.ablation import PIPELINE_PRESETS, ablation_context
from src.match_engine.calibration.apply_params import apply_params_to_config
from src.match_engine.calibration.benchmark_core import DEFAULT_FIXTURES, run_micro_benchmark_rows
from src.match_engine.calibration.contract import evaluate_rows, load_contract, load_statsbomb_baselines
from src.match_engine.calibration.profile import load_profile
from src.match_engine.micro_config import MicroMatchConfig

FOCUS = (
    "passes_per_team_match",
    "shots_per_team_match",
    "goals_per_team_match",
    "pass_completion",
    "goals_to_micro_xg_ratio",
    "micro_xg_per_team_match",
    "crosses_per_team_match",
    "yellow_cards_per_team_match",
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="run")
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--match-seconds", type=float, default=5400.0)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--override", action="append", default=[], help="param_id=value")
    p.add_argument("--set", dest="set_field", action="append", default=[], help="MicroMatchConfig field=value")
    args = p.parse_args()

    fixtures = [("Mexico", "South Korea")] if args.quick else DEFAULT_FIXTURES[:2]
    if args.quick:
        args.match_seconds = 1200.0

    overrides = {}
    for token in args.override:
        k, v = token.split("=", 1)
        overrides[k.strip()] = float(v.strip())

    profile = load_profile()
    spec = PIPELINE_PRESETS["M0"]
    cfg = MicroMatchConfig()
    cfg.use_micro_goals = True
    merged = dict(profile.overrides)
    merged.update(overrides)
    apply_params_to_config(cfg, merged, profile)
    for token in getattr(args, "set_field", []) or []:
        k, v = token.split("=", 1)
        if hasattr(cfg, k.strip()):
            setattr(cfg, k.strip(), float(v.strip()))

    with ablation_context(spec) as _:
        rows = run_micro_benchmark_rows(
            root=str(ROOT),
            fixtures=fixtures,
            samples=args.samples,
            match_seconds=args.match_seconds,
            spec=spec,
            cfg=cfg,
        )
    ev = evaluate_rows(rows, contract=load_contract(), baselines=load_statsbomb_baselines())
    hard = ev["hard_metrics"]
    soft = ev["soft_constraints"]
    summary = {"label": args.label, "all_pass": ev["all_pass"], "overrides": overrides}
    for k in FOCUS:
        if k in hard:
            summary[k] = hard[k]["sim_mean"]
        elif k in soft:
            summary[k] = soft[k]["sim_mean"]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

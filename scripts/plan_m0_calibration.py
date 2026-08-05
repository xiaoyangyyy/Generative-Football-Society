#!/usr/bin/env python3
"""Plan staged M0 calibration without tuning an already passing baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

METRIC_FAMILIES = {
    "passes_per_team_match": ["action_cadence", "passing"],
    "pass_completion": ["passing"],
    "interceptions_per_pass": ["passing", "spatial"],
    "shots_per_team_match": ["action_cadence", "shot_selection"],
    "shots_on_target_rate": ["shot_geometry", "goalkeeper"],
    "goals_per_team_match": ["shot_geometry", "goalkeeper"],
    "crosses_per_team_match": ["action_cadence", "cross_selection"],
    "headers_per_team_match": ["cross_selection", "aerial"],
    "tackles_per_team_match": ["pressure", "tackle"],
    "fouls_committed_per_team_match": ["pressure", "discipline"],
    "yellow_cards_per_team_match": ["discipline"],
    "red_cards_per_team_match": ["discipline"],
}


def build_plan(report: dict) -> dict:
    failed = list(report.get("failed_metrics", []))
    failed_soft = list(report.get("failed_soft", []))
    families = sorted({
        family for metric in failed for family in METRIC_FAMILIES.get(metric, [])
    })
    passing = bool(report.get("all_pass")) and not failed and not failed_soft
    return {
        "schema_version": 1,
        "source_protocol": report.get("protocol"),
        "source_samples": report.get("samples_total"),
        "baseline_pass": passing,
        "action": "no_op_freeze_parameters" if passing else "staged_family_search",
        "failed_metrics": failed,
        "failed_soft": failed_soft,
        "parameter_families": families,
        "guardrails": {
            "matched_seeds_required": True,
            "already_passing_metrics_may_not_regress": True,
            "formal_gate_required_before_persist": True,
            "single_fixture_results_cannot_persist": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline", default="data/evaluation/m0_formal_baseline_v1.json",
    )
    parser.add_argument(
        "--out", default="data/evaluation/m0_calibration_plan_v1.json",
    )
    args = parser.parse_args()
    report = json.loads((ROOT / args.baseline).read_text(encoding="utf-8"))
    plan = build_plan(report)
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Ablation matrix: continuous z/score deltas vs M0 (no p10/p90 band thresholds)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.match_engine.calibration.ablation import (
    ADDITIVE_LAYERS,
    PIPELINE_PRESETS,
    SUBTRACTIVE_ABLATIONS,
    ablation_context,
)
from src.match_engine.calibration.benchmark_core import DEFAULT_FIXTURES, run_micro_benchmark_rows
from src.match_engine.calibration.contract import evaluate_rows, load_contract, load_statsbomb_baselines
from src.match_engine.calibration.objective import calibration_loss, calibration_score

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "ablation"

PRIMARY_METRICS = [
    "pass_completion",
    "passes_per_team_match",
    "interceptions_per_pass",
    "shots_per_team_match",
    "goals_per_team_match",
    "goals_to_micro_xg_ratio",
    "fouls_committed_per_team_match",
    "possession_passes_correlation",
    "shots_goals_correlation",
]


def _z_from_eval(evaluation: dict, metric_id: str) -> float | None:
    if metric_id in evaluation.get("hard_metrics", {}):
        return evaluation["hard_metrics"][metric_id].get("z_abs")
    if metric_id in evaluation.get("soft_constraints", {}):
        row = evaluation["soft_constraints"][metric_id]
        if row.get("skipped"):
            return None
        return row.get("z_abs")
    return None


def _score_from_eval(evaluation: dict, metric_id: str) -> float | None:
    if metric_id in evaluation.get("hard_metrics", {}):
        return evaluation["hard_metrics"][metric_id].get("score")
    if metric_id in evaluation.get("soft_constraints", {}):
        row = evaluation["soft_constraints"][metric_id]
        if row.get("skipped"):
            return None
        return row.get("score")
    return None


def _metric_z_delta(baseline_eval: dict, variant_eval: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in set(baseline_eval.get("hard_metrics", {})) | set(baseline_eval.get("soft_constraints", {})):
        bz = _z_from_eval(baseline_eval, key)
        vz = _z_from_eval(variant_eval, key)
        if bz is None or vz is None:
            continue
        out[key] = float(vz - bz)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Run ablation matrix vs M0 (continuous scoring only).")
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--match-seconds", type=float, default=5400.0)
    p.add_argument("--seed-start", type=int, default=42)
    p.add_argument("--quick", action="store_true", help="One fixture, one sample, 20-min")
    args = p.parse_args()

    os.environ.setdefault("PYTHONPATH", str(ROOT))
    contract = load_contract()
    baselines = load_statsbomb_baselines()
    fixtures = [("Mexico", "South Korea")] if args.quick else DEFAULT_FIXTURES
    samples = 1 if args.quick else args.samples
    match_seconds = 1200.0 if args.quick else args.match_seconds

    REPORTS.mkdir(parents=True, exist_ok=True)
    baseline_spec = PIPELINE_PRESETS["M0"]

    with ablation_context(baseline_spec) as cfg:
        baseline_rows = run_micro_benchmark_rows(
            root=str(ROOT),
            fixtures=fixtures,
            samples=samples,
            seed_start=args.seed_start,
            match_seconds=match_seconds,
            spec=baseline_spec,
            cfg=cfg,
        )
    baseline_eval = evaluate_rows(baseline_rows, contract=contract, baselines=baselines)

    matrix: dict = {
        "method": "continuous_zscore",
        "baseline": {
            "name": "M0",
            "loss": calibration_loss(baseline_eval),
            "score": calibration_score(baseline_eval),
            "all_pass": baseline_eval["all_pass"],
            "failed": baseline_eval["failed_hard"] + baseline_eval["failed_soft"],
            "evaluation": {
                "score_hard_mean": baseline_eval.get("score_hard_mean"),
                "score_soft_mean": baseline_eval.get("score_soft_mean"),
            },
        },
        "subtractive": {},
        "additive": {},
    }

    exit_code = 0

    for key, spec in SUBTRACTIVE_ABLATIONS.items():
        with ablation_context(spec) as cfg:
            rows = run_micro_benchmark_rows(
                root=str(ROOT),
                fixtures=fixtures,
                samples=samples,
                seed_start=args.seed_start,
                match_seconds=match_seconds,
                spec=spec,
                cfg=cfg,
            )
        evaluation = evaluate_rows(rows, contract=contract, baselines=baselines)
        z_delta = _metric_z_delta(baseline_eval, evaluation)
        primary = {m: z_delta.get(m) for m in PRIMARY_METRICS if m in z_delta}
        matrix["subtractive"][key] = {
            "description": spec.description,
            "loss": calibration_loss(evaluation),
            "score": calibration_score(evaluation),
            "delta_loss": calibration_loss(evaluation) - calibration_loss(baseline_eval),
            "all_pass": evaluation["all_pass"],
            "failed": evaluation["failed_hard"] + evaluation["failed_soft"],
            "delta_z_primary": primary,
            "delta_z_all": z_delta,
        }
        if not evaluation["all_pass"]:
            exit_code = 1

    for key, spec in ADDITIVE_LAYERS.items():
        with ablation_context(spec) as cfg:
            rows = run_micro_benchmark_rows(
                root=str(ROOT),
                fixtures=fixtures,
                samples=samples,
                seed_start=args.seed_start,
                match_seconds=match_seconds,
                spec=spec,
                cfg=cfg,
            )
        evaluation = evaluate_rows(rows, contract=contract, baselines=baselines)
        z_delta = _metric_z_delta(baseline_eval, evaluation)
        primary = {m: z_delta.get(m) for m in PRIMARY_METRICS if m in z_delta}
        matrix["additive"][key] = {
            "description": spec.description,
            "loss": calibration_loss(evaluation),
            "score": calibration_score(evaluation),
            "delta_loss": calibration_loss(evaluation) - calibration_loss(baseline_eval),
            "all_pass": evaluation["all_pass"],
            "failed": evaluation["failed_hard"] + evaluation["failed_soft"],
            "delta_z_primary": primary,
            "delta_z_all": z_delta,
        }
        if not evaluation["all_pass"]:
            exit_code = 1

    out_path = REPORTS / "matrix.json"
    out_path.write_text(json.dumps(matrix, indent=2), encoding="utf-8")

    md_lines = [
        "# Ablation matrix (continuous z, Δ vs M0)",
        "",
        f"Baseline M0: score={matrix['baseline']['score']:.4f} loss={matrix['baseline']['loss']:.3f} pass={matrix['baseline']['all_pass']}",
        "",
        "## Subtractive (turn OFF vs M0)",
        "",
        "| Ablation | score | Δloss | pass | top Δz |",
        "|----------|-------|-------|------|--------|",
    ]
    for key, data in matrix["subtractive"].items():
        dz = data.get("delta_z_primary", {})
        top = sorted(dz.items(), key=lambda x: abs(x[1] or 0), reverse=True)[:3]
        top_s = ", ".join(f"{k}:{v:+.2f}" for k, v in top if v is not None)
        md_lines.append(
            f"| {key} | {data['score']:.4f} | {data['delta_loss']:+.2f} | {data['all_pass']} | {top_s} |"
        )

    md_lines.extend([
        "",
        "## Additive (turn ON vs M0)",
        "",
        "| Layer | score | Δloss | pass | top Δz |",
        "|-------|-------|-------|------|--------|",
    ])
    for key, data in matrix["additive"].items():
        dz = data.get("delta_z_primary", {})
        top = sorted(dz.items(), key=lambda x: abs(x[1] or 0), reverse=True)[:3]
        top_s = ", ".join(f"{k}:{v:+.2f}" for k, v in top if v is not None)
        md_lines.append(
            f"| {key} | {data['score']:.4f} | {data['delta_loss']:+.2f} | {data['all_pass']} | {top_s} |"
        )

    md_path = REPORTS / "matrix.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(json.dumps({"report": str(out_path.relative_to(ROOT)), "baseline_pass": matrix["baseline"]["all_pass"]}, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

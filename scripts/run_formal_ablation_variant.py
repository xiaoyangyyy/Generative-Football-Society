#!/usr/bin/env python3
"""Run one formal matched-seed ablation variant against frozen M0."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_sim_vs_open_data import _bootstrap_mean_intervals
from src.match_engine.calibration.ablation import (
    ADDITIVE_LAYERS,
    SUBTRACTIVE_ABLATIONS,
    ablation_context,
)
from src.match_engine.calibration.benchmark_core import (
    DEFAULT_FIXTURES,
    run_micro_benchmark_rows,
)
from src.match_engine.calibration.contract import (
    evaluate_rows,
    load_contract,
    load_statsbomb_baselines,
)
from src.match_engine.calibration.objective import calibration_loss, calibration_score

VARIANTS = {**SUBTRACTIVE_ABLATIONS, **ADDITIVE_LAYERS}


def _evaluation_from_frozen(report: dict) -> dict:
    return {
        "hard_metrics": report["report"],
        "soft_constraints": report["soft_constraints"],
        "failed_hard": report["failed_metrics"],
        "failed_soft": report["failed_soft"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=sorted(VARIANTS))
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", help="Explicit checkpoint for the M1 candidate.")
    parser.add_argument("--resume", action="store_true", help="Resume from atomic per-match progress.")
    args = parser.parse_args()
    if args.checkpoint and args.variant != "M1":
        raise SystemExit("--checkpoint is only valid for M1")
    checkpoint = None
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        checkpoint = (checkpoint if checkpoint.is_absolute() else ROOT / checkpoint).resolve()
        if not checkpoint.is_file():
            raise SystemExit(f"missing M1 checkpoint: {checkpoint}")
    spec = VARIANTS[args.variant]
    output = ROOT / args.out
    progress_path = output.with_suffix(output.suffix + ".progress.json")
    progress_rows = []
    if args.resume and progress_path.is_file():
        progress_rows = json.loads(progress_path.read_text(encoding="utf-8")).get("rows", [])
    completed = {
        (str(row["fixture"]), int(row["sample_index"]))
        for row in progress_rows
    }

    def save_progress(row: dict) -> None:
        progress_rows.append(row)
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = progress_path.with_suffix(progress_path.suffix + ".tmp")
        temporary.write_text(json.dumps({
            "schema_version": 1, "variant": args.variant,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "rows": progress_rows,
        }, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, progress_path)
        print(f"[formal] completed {len(progress_rows)}/18", flush=True)

    started = time.perf_counter()
    with ablation_context(spec) as cfg:
        previous_checkpoint = os.environ.get("MATCH_WM_CHECKPOINT")
        if checkpoint is not None:
            os.environ["MATCH_WM_CHECKPOINT"] = str(checkpoint)
        try:
            new_rows = run_micro_benchmark_rows(
                root=str(ROOT), fixtures=DEFAULT_FIXTURES, samples=3,
                seed_start=42, match_seconds=5400.0, spec=spec, cfg=cfg,
                completed_keys=completed, row_callback=save_progress,
            )
        finally:
            if checkpoint is not None:
                if previous_checkpoint is None:
                    os.environ.pop("MATCH_WM_CHECKPOINT", None)
                else:
                    os.environ["MATCH_WM_CHECKPOINT"] = previous_checkpoint
    rows = progress_rows if progress_rows else new_rows
    evaluation = evaluate_rows(
        rows, contract=load_contract(), baselines=load_statsbomb_baselines(),
    )
    frozen = json.loads(
        (ROOT / "data/evaluation/m0_formal_baseline_v1.json").read_text(
            encoding="utf-8"
        )
    )
    baseline_evaluation = _evaluation_from_frozen(frozen)
    baseline_loss = calibration_loss(baseline_evaluation)
    loss = calibration_loss(evaluation)
    payload = {
        "schema_version": 1,
        "protocol": "six_fixtures_three_seeds_full_90min_matched_to_m0",
        "variant": args.variant,
        "description": spec.description,
        "samples_total": len(rows),
        "loss": loss,
        "score": calibration_score(evaluation),
        "baseline_loss": baseline_loss,
        "delta_loss": loss - baseline_loss,
        "all_pass": evaluation["all_pass"],
        "failed_metrics": evaluation["failed_hard"],
        "failed_soft": evaluation["failed_soft"],
        "uncertainty": {
            "method": "deterministic_match_bootstrap",
            "draws": 2000,
            "seed": 42,
            "metrics": _bootstrap_mean_intervals(rows, seed=42, draws=2000),
        },
        "report": evaluation["hard_metrics"],
        "soft_constraints": evaluation["soft_constraints"],
        "raw_rows": rows,
        "candidate": {
            "checkpoint": (
                checkpoint.relative_to(ROOT).as_posix()
                if checkpoint is not None else None
            ),
            "explicit_checkpoint": checkpoint is not None,
        },
        "runtime": {"elapsed_seconds": time.perf_counter() - started},
    }
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if progress_path.exists():
        progress_path.unlink()
    print(json.dumps({
        "variant": args.variant,
        "samples_total": len(rows),
        "all_pass": evaluation["all_pass"],
        "delta_loss": payload["delta_loss"],
        "output": str(output.relative_to(ROOT)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Phase 3 — staged joint calibration (continuous objective, no p10/p90 band clipping).

Modes:
  --validate-only   Export profile + run full gate + ablation (default after Phase 2 pass)
  --optimize        Coordinate search on registry params (quick sim), then full gate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.match_engine.calibration.ablation import PIPELINE_PRESETS, ablation_context
from src.match_engine.calibration.apply_params import (
    apply_params_to_config,
    export_profile_overrides,
    persist_pass_rates,
)
from src.match_engine.calibration.benchmark_core import DEFAULT_FIXTURES, run_micro_benchmark_rows
from src.match_engine.calibration.contract import evaluate_rows, load_contract, load_statsbomb_baselines
from src.match_engine.calibration.objective import calibration_loss, calibration_score
from src.match_engine.calibration.profile import CalibrationProfile, load_param_registry, load_profile
from src.match_engine.micro_config import MicroMatchConfig

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "joint_calibration"
REGISTRY_PATH = ROOT / "data" / "calibration" / "param_registry.json"

# Staged joint fit: fix earlier stages before later ones.
STAGES: list[tuple[str, list[int], list[str]]] = [
    ("A_pass", [0, 1], ["passing"]),
    ("B_shot", [1], ["shot", "action"]),
    ("C_discipline", [1], ["discipline"]),
    ("D_macro", [2], ["macro", "affective"]),
]


def _stage_params(profile: CalibrationProfile, stage: tuple[str, list[int], list[str]]) -> list:
    _name, tiers, owners = stage
    return [p for p in profile.parameters if p.tier in tiers and p.owner in owners]


def _eval_params(
    overrides: dict[str, float],
    profile: CalibrationProfile,
    *,
    root: str,
    fixtures: list[tuple[str, str]],
    samples: int,
    match_seconds: float,
    seed_start: int,
) -> dict[str, Any]:
    spec = PIPELINE_PRESETS["M0"]
    cfg = MicroMatchConfig()
    cfg.use_micro_goals = True
    apply_params_to_config(cfg, overrides, profile)
    with ablation_context(spec) as _:
        rows = run_micro_benchmark_rows(
            root=root,
            fixtures=fixtures,
            samples=samples,
            seed_start=seed_start,
            match_seconds=match_seconds,
            spec=spec,
            cfg=cfg,
        )
    return evaluate_rows(rows, contract=load_contract(), baselines=load_statsbomb_baselines())


def _coordinate_stage(
    base_overrides: dict[str, float],
    profile: CalibrationProfile,
    stage_params: list,
    *,
    root: str,
    fixtures: list[tuple[str, str]],
    samples: int,
    match_seconds: float,
    grid_points: int,
    seed_start: int,
) -> dict[str, float]:
    current = dict(base_overrides)
    best_loss = float("inf")

    for _round in range(2):
        for pspec in stage_params:
            lo, hi = pspec.bounds
            grid = np.linspace(lo, hi, grid_points)
            local_best = best_loss
            local_val = current.get(pspec.id, pspec.default)
            for v in grid:
                trial = dict(current)
                trial[pspec.id] = float(v)
                ev = _eval_params(
                    trial,
                    profile,
                    root=root,
                    fixtures=fixtures,
                    samples=samples,
                    match_seconds=match_seconds,
                    seed_start=seed_start,
                )
                loss = calibration_loss(ev)
                if loss < local_best:
                    local_best = loss
                    local_val = float(v)
            current[pspec.id] = local_val
            if local_best < best_loss:
                best_loss = local_best
    return current


def _save_profile(profile_name: str, overrides: dict[str, float], report: dict[str, Any]) -> Path:
    data = load_param_registry()
    profiles = data.setdefault("profiles", {})
    entry = profiles.setdefault(profile_name, {})
    entry["description"] = entry.get(
        "description", "Joint-calibrated profile (Phase 3)."
    )
    entry["overrides"] = {k: round(v, 6) for k, v in sorted(overrides.items())}
    entry["last_calibration"] = report
    REGISTRY_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    out = REPORTS / f"{profile_name}.json"
    REPORTS.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"profile": profile_name, "overrides": entry["overrides"], **report}, indent=2), encoding="utf-8")
    return out


def _run_full_gate(overrides: dict[str, float], profile: CalibrationProfile) -> dict[str, Any]:
    ev = _eval_params(
        overrides,
        profile,
        root=str(ROOT),
        fixtures=DEFAULT_FIXTURES,
        samples=3,
        match_seconds=5400.0,
        seed_start=42,
    )
    gate_path = ROOT / "reports" / "calibration_gate.json"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "gate": "calibration_full",
        "pipeline": "M0",
        "profile": profile.name,
        "samples_total": ev["samples_total"],
        "all_pass": ev["all_pass"],
        "failed_hard": ev["failed_hard"],
        "failed_soft": ev["failed_soft"],
        "score_hard_mean": ev.get("score_hard_mean"),
        "score_soft_mean": ev.get("score_soft_mean"),
        "hard_metrics": ev["hard_metrics"],
        "soft_constraints": ev["soft_constraints"],
    }
    gate_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return ev


def main() -> int:
    p = argparse.ArgumentParser(description="Phase 3 joint calibration")
    p.add_argument("--profile", default="v5_physics_first")
    p.add_argument("--validate-only", action="store_true", help="Export profile + gate + ablation")
    p.add_argument("--optimize", action="store_true", help="Run staged coordinate search then gate")
    p.add_argument("--quick", action="store_true", help="2 fixtures, 1 sample, 20-min matches for search")
    p.add_argument("--grid", type=int, default=5, help="Grid points per parameter in search")
    args = p.parse_args()

    os.environ.setdefault("PYTHONPATH", str(ROOT))
    profile = load_profile(args.profile)
    cfg = MicroMatchConfig()
    cfg.use_micro_goals = True

    overrides = export_profile_overrides(cfg, profile)
    for pid, val in profile.overrides.items():
        overrides[pid] = float(val)

    if not args.validate_only and not args.optimize:
        args.validate_only = True

    if args.optimize:
        fixtures = [("Mexico", "South Korea"), ("Brazil", "Germany")] if args.quick else DEFAULT_FIXTURES
        samples = 1 if args.quick else 2
        match_seconds = 1200.0 if args.quick else 2700.0
        print(f"[Phase3] Staged search: {len(fixtures)} fixtures x {samples} x {match_seconds}s", file=sys.stderr)
        for stage in STAGES:
            params = _stage_params(profile, stage)
            if not params:
                continue
            print(f"  stage {stage[0]}: {len(params)} params", file=sys.stderr)
            overrides = _coordinate_stage(
                overrides,
                profile,
                params,
                root=str(ROOT),
                fixtures=fixtures,
                samples=samples,
                match_seconds=match_seconds,
                grid_points=args.grid,
                seed_start=42,
            )
        persist_pass_rates(overrides, profile)

    ev_quick = _eval_params(
        overrides,
        profile,
        root=str(ROOT),
        fixtures=[("Mexico", "South Korea")],
        samples=1,
        match_seconds=1200.0,
        seed_start=99,
    )
    report = {
        "loss": calibration_loss(ev_quick),
        "score": calibration_score(ev_quick),
        "all_pass_quick": ev_quick["all_pass"],
        "failed_hard": ev_quick["failed_hard"],
        "failed_soft": ev_quick["failed_soft"],
    }
    out_path = _save_profile(args.profile, overrides, report)
    print(f"Wrote profile overrides to {out_path}", file=sys.stderr)

    print("[Phase3] Full gate (6x3x90min)...", file=sys.stderr)
    ev_full = _run_full_gate(overrides, profile)
    if not ev_full["all_pass"]:
        print("GATE FAILED after Phase 3", file=sys.stderr)
        print(json.dumps({"failed_hard": ev_full["failed_hard"], "failed_soft": ev_full["failed_soft"]}, indent=2))
        return 1

    print("[Phase3] Ablation matrix...", file=sys.stderr)
    import subprocess

    ab_args = [sys.executable, str(ROOT / "scripts" / "run_ablation_matrix.py")]
    if args.quick:
        ab_args.append("--quick")
    else:
        ab_args.extend(["--samples", "3"])
    r = subprocess.run(ab_args, cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT)})
    if r.returncode != 0:
        return r.returncode

    print("\nPHASE 3 COMPLETE — profile saved, gate passed, ablation done.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

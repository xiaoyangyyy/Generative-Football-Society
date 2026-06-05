"""Phase 5 — additive layer gates (M1 / C1) relative to M0 baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.match_engine.calibration.contract import evaluate_rows, load_contract
from src.match_engine.calibration.objective import calibration_loss, calibration_score

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PRIMARY = [
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


def metric_z_delta(baseline_eval: dict, variant_eval: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    keys = set(baseline_eval.get("hard_metrics", {})) | set(baseline_eval.get("soft_constraints", {}))
    for key in keys:
        bz = _z_from_eval(baseline_eval, key)
        vz = _z_from_eval(variant_eval, key)
        if bz is None or vz is None:
            continue
        out[key] = float(vz - bz)
    return out


def load_layer_gate_spec(contract: dict[str, Any] | None = None, layer: str = "M1") -> dict[str, Any]:
    contract = contract or load_contract()
    gates = contract.get("layer_gates", {})
    spec = dict(gates.get(layer, {}))
    spec.setdefault("max_delta_loss", 8.0)
    spec.setdefault("max_abs_delta_z", 0.35)
    spec.setdefault("min_score_ratio_vs_m0", 0.90)
    spec.setdefault("primary_metrics", list(DEFAULT_PRIMARY))
    spec.setdefault("require_m0_absolute_pass", True)
    return spec


def evaluate_layer_gate(
    m0_eval: dict[str, Any],
    layer_eval: dict[str, Any],
    *,
    layer: str,
    contract: dict[str, Any] | None = None,
    m0_absolute_passed: bool | None = None,
) -> dict[str, Any]:
    """
    Pass if additive layer does not materially shift L0 metrics vs M0.
    Does NOT require absolute StatsBomb pass for M1/C1.
    """
    contract = contract or load_contract()
    spec = load_layer_gate_spec(contract, layer)
    m0_loss = calibration_loss(m0_eval)
    layer_loss = calibration_loss(layer_eval)
    delta_loss = float(layer_loss - m0_loss)
    m0_score = calibration_score(m0_eval)
    layer_score = calibration_score(layer_eval)
    score_ratio = float(layer_score / max(1e-9, m0_score))

    dz = metric_z_delta(m0_eval, layer_eval)
    primary = list(spec.get("primary_metrics", DEFAULT_PRIMARY))
    max_dz = float(spec.get("max_abs_delta_z", 0.35))
    max_loss = float(spec.get("max_delta_loss", 8.0))
    min_ratio = float(spec.get("min_score_ratio_vs_m0", 0.90))

    primary_delta = {m: dz[m] for m in primary if m in dz}
    violations = [m for m, v in primary_delta.items() if abs(v) > max_dz]
    failed: list[str] = []
    if abs(delta_loss) > max_loss:
        failed.append(f"delta_loss>{max_loss}")
    if score_ratio < min_ratio:
        failed.append(f"score_ratio<{min_ratio}")
    failed.extend(violations)

    require_m0 = bool(spec.get("require_m0_absolute_pass", True))
    m0_ok = m0_absolute_passed if m0_absolute_passed is not None else bool(m0_eval.get("all_pass", False))
    if require_m0 and not m0_ok:
        failed.append("m0_baseline_not_passed")

    return {
        "layer": layer,
        "reference": "M0",
        "m0_loss": m0_loss,
        "layer_loss": layer_loss,
        "delta_loss": delta_loss,
        "m0_score": m0_score,
        "layer_score": layer_score,
        "score_ratio_vs_m0": score_ratio,
        "delta_z_primary": primary_delta,
        "delta_z_all": dz,
        "max_abs_delta_z": max_dz,
        "max_delta_loss": max_loss,
        "min_score_ratio_vs_m0": min_ratio,
        "violations": violations,
        "failed": failed,
        "all_pass": len(failed) == 0,
        "layer_absolute_pass": layer_eval.get("all_pass", False),
    }


def evaluate_layer_rows(
    m0_rows: list[dict[str, Any]],
    layer_rows: list[dict[str, Any]],
    *,
    layer: str,
    contract: dict[str, Any] | None = None,
    baselines: dict[str, Any] | None = None,
    m0_absolute_passed: bool | None = None,
) -> dict[str, Any]:
    contract = contract or load_contract()
    m0_eval = evaluate_rows(m0_rows, contract=contract, baselines=baselines)
    layer_eval = evaluate_rows(layer_rows, contract=contract, baselines=baselines)
    report = evaluate_layer_gate(
        m0_eval,
        layer_eval,
        layer=layer,
        contract=contract,
        m0_absolute_passed=m0_absolute_passed,
    )
    report["m0_evaluation"] = {
        "all_pass": m0_eval.get("all_pass"),
        "score_combined": m0_eval.get("score_combined"),
        "failed_hard": m0_eval.get("failed_hard"),
        "failed_soft": m0_eval.get("failed_soft"),
    }
    report["layer_evaluation"] = {
        "all_pass": layer_eval.get("all_pass"),
        "score_combined": layer_eval.get("score_combined"),
        "failed_hard": layer_eval.get("failed_hard"),
        "failed_soft": layer_eval.get("failed_soft"),
    }
    return report

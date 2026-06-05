"""Load and evaluate the observable calibration contract."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.match_engine.calibration.benchmark_core import team_level_rows

ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = ROOT / "data" / "calibration" / "observable_contract.json"
BASELINE_PATH = ROOT / "data" / "calibration" / "statsbomb_match_baselines.json"
JOINT_PATH = ROOT / "data" / "calibration" / "joint_baselines.json"

SPARSE_METRICS = frozenset({"yellow_cards_per_team_match", "red_cards_per_team_match"})


def load_contract(path: Path | None = None) -> dict[str, Any]:
    p = path or CONTRACT_PATH
    contract = json.loads(p.read_text(encoding="utf-8"))
    if JOINT_PATH.is_file():
        joint = json.loads(JOINT_PATH.read_text(encoding="utf-8"))
        soft_bands = joint.get("soft_constraints", {})
        corr_ref = joint.get("correlations_reference", {})
        for spec in contract.get("soft_constraints", []):
            stype = spec.get("type", "range")
            cid = str(spec.get("id", ""))
            if stype == "range":
                band = soft_bands.get(cid)
                if not band:
                    continue
                target = float(band.get("mean", 0.0))
                p25 = float(band.get("p25", target))
                p75 = float(band.get("p75", target))
                iqr = max(p75 - p25, 1e-6)
                spec["target_mean"] = target
                spec["scale"] = float(iqr / 1.349)
                spec["source"] = "joint_baselines.json"
            elif stype == "correlation" and cid in corr_ref:
                ref = corr_ref[cid]
                if isinstance(ref, dict):
                    if "target_mean" not in spec:
                        spec["target_mean"] = float(
                            ref.get("target_mean", ref.get("statsbomb_mean", 0.0))
                        )
                    if "scale" not in spec:
                        spec["scale"] = float(ref.get("scale", 0.25))
                    if "statsbomb_reference" not in spec and "statsbomb_mean" in ref:
                        spec["statsbomb_reference"] = float(ref["statsbomb_mean"])
                else:
                    if "target_mean" not in spec:
                        spec["target_mean"] = float(ref)
                    if "scale" not in spec:
                        spec["scale"] = 0.15
                spec.setdefault("source", "joint_baselines.json")
    return contract


def load_statsbomb_baselines(path: Path | None = None) -> dict[str, Any]:
    p = path or BASELINE_PATH
    return json.loads(p.read_text(encoding="utf-8"))


def _mean(vals: list[float]) -> float:
    return float(sum(vals) / max(1, len(vals)))


def _correlation(xs: list[float], ys: list[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 3:
        return 0.0
    mx = sum(xs[:n]) / n
    my = sum(ys[:n]) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    den_x = math.sqrt(sum((xs[i] - mx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ys[i] - my) ** 2 for i in range(n)))
    if den_x < 1e-12 or den_y < 1e-12:
        return 0.0
    return float(num / (den_x * den_y))


def _robust_scale_from_quantiles(q: dict[str, Any], fallback: float = 0.1) -> float:
    """
    Robust scale estimate from quantiles (continuous, no hard clipping rules).
    Prefer IQR-based sigma, fallback to p10-p90 spread.
    """
    p25 = float(q.get("p25", 0.0))
    p75 = float(q.get("p75", 0.0))
    p10 = float(q.get("p10", 0.0))
    p90 = float(q.get("p90", 0.0))
    iqr_sigma = (p75 - p25) / 1.349 if p75 > p25 else 0.0
    pspan_sigma = (p90 - p10) / 2.563 if p90 > p10 else 0.0
    scale = max(iqr_sigma, pspan_sigma, float(fallback))
    return float(scale)


def _gaussian_score(sim: float, target: float, scale: float) -> tuple[float, float]:
    """Return (z_abs, score in (0,1]) using Gaussian-like continuous penalty."""
    s = max(1e-8, float(scale))
    z_abs = abs((float(sim) - float(target)) / s)
    score = math.exp(-0.5 * z_abs * z_abs)
    return float(z_abs), float(score)


def evaluate_rows(
    rows: list[dict[str, Any]],
    *,
    contract: dict[str, Any] | None = None,
    baselines: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate match rows against hard marginal bands and soft joint constraints."""
    contract = contract or load_contract()
    baselines = baselines or load_statsbomb_baselines()
    bands = baselines["metrics"]

    hard_cfg = contract["hard_metrics"]
    hard_metrics = list(hard_cfg["metrics"])
    supported = set(rows[0].keys()) - {"fixture"} if rows else set()

    gate_cfg = contract.get("gate", {})
    min_combined = float(gate_cfg.get("min_combined_score", 0.52))
    min_metric = float(gate_cfg.get("min_metric_score", 0.30))

    hard_report: dict[str, Any] = {}
    hard_fails: list[str] = []
    hard_scores: list[float] = []
    for key in hard_metrics:
        if key not in supported or key not in bands:
            continue
        sim_mean = _mean([float(r[key]) for r in rows])
        target = float(bands[key].get("mean", 0.0))
        if key in SPARSE_METRICS:
            scale = max(math.sqrt(max(target, 1e-8)), _robust_scale_from_quantiles(bands[key], fallback=0.05))
        else:
            scale = _robust_scale_from_quantiles(bands[key], fallback=0.1)
        z_abs, score = _gaussian_score(sim_mean, target, scale)
        ok = score >= min_metric
        hard_report[key] = {
            "sim_mean": sim_mean,
            "target_mean": target,
            "scale": scale,
            "z_abs": z_abs,
            "score": score,
            "pass": ok,
        }
        hard_scores.append(score)
        if not ok:
            hard_fails.append(key)

    soft_report: dict[str, Any] = {}
    soft_fails: list[str] = []
    soft_scores: list[float] = []
    for spec in contract.get("soft_constraints", []):
        cid = str(spec["id"])
        stype = spec.get("type", "range")
        if stype == "range":
            if cid not in supported:
                continue
            sim_mean = _mean([float(r[cid]) for r in rows])
            target = float(spec.get("target_mean", 0.0))
            scale = float(spec.get("scale", max(abs(target) * 0.25, 0.1)))
            z_abs, score = _gaussian_score(sim_mean, target, scale)
            ok = score >= min_metric
            soft_report[cid] = {
                "sim_mean": sim_mean,
                "target_mean": target,
                "scale": scale,
                "z_abs": z_abs,
                "score": score,
                "pass": ok,
            }
            soft_scores.append(score)
            if not ok:
                soft_fails.append(cid)
        elif stype == "correlation":
            m1, m2 = spec["metrics"]
            use_team = bool(spec.get("use_team_rows", False))
            corr_rows = team_level_rows(rows) if use_team else rows
            if m1 not in (corr_rows[0].keys() if corr_rows else set()) or m2 not in (
                corr_rows[0].keys() if corr_rows else set()
            ):
                continue
            min_samples = int(spec.get("min_samples", 3))
            if len(corr_rows) < min_samples:
                soft_report[cid] = {
                    "correlation": None,
                    "metrics": [m1, m2],
                    "pass": True,
                    "score": None,
                    "z_abs": None,
                    "skipped": True,
                    "reason": f"need>={min_samples} team-rows, got {len(corr_rows)}",
                }
                continue
            corr = _correlation(
                [float(r[m1]) for r in corr_rows],
                [float(r[m2]) for r in corr_rows],
            )
            target = float(spec.get("target_mean", 0.0))
            scale = float(spec.get("scale", 0.15))
            z_abs, score = _gaussian_score(corr, target, scale)
            ok = score >= min_metric
            entry: dict[str, Any] = {
                "correlation": corr,
                "metrics": [m1, m2],
                "target_mean": target,
                "scale": scale,
                "z_abs": z_abs,
                "score": score,
                "pass": ok,
            }
            if spec.get("statsbomb_reference") is not None:
                entry["statsbomb_reference"] = float(spec["statsbomb_reference"])
            soft_report[cid] = entry
            soft_scores.append(score)
            if not ok:
                soft_fails.append(cid)

    combined = _mean(hard_scores + soft_scores) if (hard_scores or soft_scores) else 0.0

    return {
        "profile": contract.get("profile", ""),
        "samples_total": len(rows),
        "hard_metrics": hard_report,
        "soft_constraints": soft_report,
        "failed_hard": hard_fails,
        "failed_soft": soft_fails,
        "score_hard_mean": _mean(hard_scores) if hard_scores else 0.0,
        "score_soft_mean": _mean(soft_scores) if soft_scores else 0.0,
        "score_combined": combined,
        "all_hard_pass": len(hard_fails) == 0,
        "all_soft_pass": len(soft_fails) == 0,
        "all_pass": len(hard_fails) == 0 and len(soft_fails) == 0 and combined >= min_combined,
    }

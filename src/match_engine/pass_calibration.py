"""
Pass completion calibration vs StatsBomb open-data baselines.

Loads data/calibration/statsbomb_pass_rates.json (WC 2022 defaults) and maps
sim pass kinds → target completion for log-odds bias in PassingEngine.

No hard probability floors — only continuous log-odds shifts toward empirical rates.
"""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from src.simulation.runtime import environment_snapshot, env_bool

_DEFAULT: Dict[str, Any] = {
    "overall_completion": 0.8224,
    "sim_targets": {
        "overall": 0.8224,
        "short": 0.853,
        "through": 0.74,
        "long": 0.6221,
        "intercept_share_max": 0.05,
    },
}


def _logit(p: float) -> float:
    p = max(1e-4, min(1.0 - 1e-4, float(p)))
    return math.log(p / (1.0 - p))


@lru_cache(maxsize=8)
def load_pass_calibration(base_dir: str = ".") -> Dict[str, Any]:
    path = Path(base_dir) / "data" / "calibration" / "statsbomb_pass_rates.json"
    if not path.is_file():
        return dict(_DEFAULT)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        targets = data.get("sim_targets", {})
        overall = float(data.get("overall_completion", targets.get("overall", 0.8224)))
        targets.setdefault("overall", overall)
        by_len = data.get("by_length_m", {})
        targets.setdefault("short", float(by_len.get("short_lt_15", 0.853)))
        targets.setdefault("long", float(by_len.get("long_gte_30", 0.6221)))
        targets.setdefault("through", float(targets.get("through", 0.74)))
        data["sim_targets"] = targets
        data["overall_completion"] = overall
        return data
    except (OSError, json.JSONDecodeError):
        return dict(_DEFAULT)


def overall_completion_rate(*, base_dir: str = ".") -> float:
    cal = load_pass_calibration(base_dir)
    return float(cal.get("overall_completion", 0.8224))


def overall_logit_anchor(*, base_dir: str = ".") -> float:
    """Log-odds center matching StatsBomb overall pass completion."""
    return _logit(overall_completion_rate(base_dir=base_dir))


def target_completion(kind: str, dist_norm: float, *, base_dir: str = ".") -> float:
    """Map micro pass kind + normalized distance to a StatsBomb-informed target rate."""
    cal = load_pass_calibration(base_dir)
    t = cal.get("sim_targets", _DEFAULT["sim_targets"])
    short_max = 0.22
    long_min = 0.38
    short_t = float(t.get("short", 0.853))
    through_t = float(t.get("through", 0.74))
    long_t = float(t.get("long", 0.6221))
    if kind == "long" or dist_norm >= long_min:
        return long_t
    if kind == "through":
        return through_t
    if kind == "short" or dist_norm <= short_max:
        return short_t
    # interpolate short → through by distance (StatsBomb medium band)
    w = (float(dist_norm) - short_max) / max(1e-6, long_min - short_max)
    w = max(0.0, min(1.0, w))
    return (1.0 - w) * short_t + w * through_t


def target_logit(kind: str, dist_norm: float, *, base_dir: str = ".") -> float:
    return _logit(target_completion(kind, dist_norm, base_dir=base_dir))


def intercept_share_cap(base_dir: str = ".") -> float:
    cal = load_pass_calibration(base_dir)
    return float(cal.get("sim_targets", {}).get("intercept_share_max", 0.05))


def logit_target_blend(*, base_dir: str = ".") -> float:
    """
    Blend micro physics log-odds with StatsBomb per-kind target (0..1, no clipping).
  0.88 ≈ pull full-match completion toward WC2022 overall 82.24%.
    """
    values = environment_snapshot()
    override = values.get("CALIBRATION_LOGIT_BLEND", "").strip()
    if override:
        return float(override)
    if env_bool(values, "CALIBRATION_NO_BLEND", False):
        return 0.0
    cal = load_pass_calibration(base_dir)
    tuning = cal.get("sim_tuning", {})
    legacy = tuning.get("logit_residual_uplift")
    if legacy is not None and "logit_target_blend" not in tuning:
        return min(0.95, 0.55 + float(legacy))
    return float(tuning.get("logit_target_blend", 0.88))


def intercept_risk_scale(*, base_dir: str = ".") -> float:
    """Scale geometric intercept risk → ~2% intercept/pass (StatsBomb WC2022)."""
    override = environment_snapshot().get("CALIBRATION_INTERCEPT_RISK_SCALE", "").strip()
    if override:
        return float(override)
    cal = load_pass_calibration(base_dir)
    tuning = cal.get("sim_tuning", {})
    return float(tuning.get("intercept_risk_scale", 0.028))


def pass_mix_log_bias(kind: str, *, base_dir: str = ".") -> float:
    """
    Log-prior bias for pass kind selection to match open-data event mix.
    Positive => sampled more often by softmax utility.
    """
    cal = load_pass_calibration(base_dir)
    mix = cal.get("sim_tuning", {}).get(
        "pass_mix_targets",
        {"short": 0.72, "through": 0.16, "long": 0.09, "wall": 0.03},
    )
    p_kind = float(mix.get(kind, 0.25))
    # neutral prior among non-wall choices
    p_ref = 0.25 if kind == "wall" else (1.0 - float(mix.get("wall", 0.03))) / 3.0
    return math.log(max(1e-4, p_kind) / max(1e-4, p_ref))

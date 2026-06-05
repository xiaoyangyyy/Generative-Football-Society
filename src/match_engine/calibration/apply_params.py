"""Apply parameter-registry overrides to MicroMatchConfig and pass JSON tuning."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.match_engine.calibration.profile import CalibrationProfile, ParamSpec
from src.match_engine.micro_config import MicroMatchConfig

ROOT = Path(__file__).resolve().parents[3]
PASS_RATES_PATH = ROOT / "data" / "calibration" / "statsbomb_pass_rates.json"


def _set_nested(d: dict, dotted: str, value: float) -> None:
    parts = dotted.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = float(value)


def apply_params_to_config(
    cfg: MicroMatchConfig,
    overrides: dict[str, float],
    profile: CalibrationProfile,
) -> MicroMatchConfig:
    """Apply registry overrides; JSON pass tuning via env for runtime sims."""
    id_to_spec = {p.id: p for p in profile.parameters}
    pass_json_dirty = False
    pass_patch: dict[str, float] = {}

    for param_id, val in overrides.items():
        spec = id_to_spec.get(param_id)
        if spec is None:
            continue
        field = spec.field
        if not field:
            continue
        if "statsbomb_pass_rates.json" in spec.file:
            pass_patch[field.split(".")[-1] if "." in field else field] = float(val)
            pass_json_dirty = True
            continue
        attr = field.split(".")[-1] if "." in field else field
        if hasattr(cfg, attr):
            setattr(cfg, attr, float(val))

    if pass_patch.get("logit_target_blend") is not None:
        os.environ["CALIBRATION_LOGIT_BLEND"] = str(pass_patch["logit_target_blend"])
    if pass_patch.get("intercept_risk_scale") is not None:
        os.environ["CALIBRATION_INTERCEPT_RISK_SCALE"] = str(pass_patch["intercept_risk_scale"])

    if pass_json_dirty and PASS_RATES_PATH.is_file():
        from src.match_engine import pass_calibration

        pass_calibration.load_pass_calibration.cache_clear()

    return cfg


def persist_pass_rates(overrides: dict[str, float], profile: CalibrationProfile) -> None:
    """Write pass tuning keys into statsbomb_pass_rates.json (for reproducible runs)."""
    if not PASS_RATES_PATH.is_file():
        return
    data = json.loads(PASS_RATES_PATH.read_text(encoding="utf-8"))
    tuning = data.setdefault("sim_tuning", {})
    for pspec in profile.parameters:
        if pspec.id not in overrides:
            continue
        if "statsbomb_pass_rates.json" not in pspec.file:
            continue
        key = (pspec.field or "").split(".")[-1]
        if key:
            tuning[key] = float(overrides[pspec.id])
    PASS_RATES_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    from src.match_engine import pass_calibration

    pass_calibration.load_pass_calibration.cache_clear()
    os.environ.pop("CALIBRATION_LOGIT_BLEND", None)
    os.environ.pop("CALIBRATION_INTERCEPT_RISK_SCALE", None)


def export_profile_overrides(cfg: MicroMatchConfig, profile: CalibrationProfile) -> dict[str, float]:
    """Snapshot current config values into registry param ids."""
    out: dict[str, float] = {}
    for p in profile.parameters:
        if not p.field:
            continue
        if "statsbomb_pass_rates.json" in p.file and PASS_RATES_PATH.is_file():
            data = json.loads(PASS_RATES_PATH.read_text(encoding="utf-8"))
            key = p.field.split(".")[-1]
            val = data.get("sim_tuning", {}).get(key)
            if val is not None:
                out[p.id] = float(val)
            continue
        attr = p.field.split(".")[-1]
        if hasattr(cfg, attr):
            out[p.id] = float(getattr(cfg, attr))
    return out

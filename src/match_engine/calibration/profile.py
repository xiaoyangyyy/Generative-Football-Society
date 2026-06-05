"""Parameter registry and calibration profiles."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
REGISTRY_PATH = ROOT / "data" / "calibration" / "param_registry.json"


@dataclass
class ParamSpec:
    id: str
    owner: str
    affects: list[str]
    bounds: tuple[float, float]
    default: float
    file: str
    field: str | None
    tier: int = 1


@dataclass
class CalibrationProfile:
    name: str
    description: str = ""
    overrides: dict[str, float] = field(default_factory=dict)
    parameters: list[ParamSpec] = field(default_factory=list)

    def get(self, param_id: str) -> float:
        if param_id in self.overrides:
            return float(self.overrides[param_id])
        for p in self.parameters:
            if p.id == param_id:
                return float(p.default)
        raise KeyError(param_id)

    def tier_params(self, tier: int) -> list[ParamSpec]:
        return [p for p in self.parameters if p.tier == tier]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "overrides": dict(self.overrides),
            "parameter_count": len(self.parameters),
            "tiers": {
                str(t): len(self.tier_params(t))
                for t in sorted({p.tier for p in self.parameters})
            },
        }


def load_param_registry(path: Path | None = None) -> dict[str, Any]:
    p = path or REGISTRY_PATH
    return json.loads(p.read_text(encoding="utf-8"))


def load_profile(name: str | None = None, *, registry_path: Path | None = None) -> CalibrationProfile:
    data = load_param_registry(registry_path)
    profile_name = name or data.get("profile_default", "v5_physics_first")
    profiles = data.get("profiles", {})
    if profile_name not in profiles:
        raise KeyError(f"Unknown calibration profile: {profile_name}")
    prof = profiles[profile_name]
    params = [
        ParamSpec(
            id=str(p["id"]),
            owner=str(p["owner"]),
            affects=[str(a) for a in p.get("affects", [])],
            bounds=(float(p["bounds"][0]), float(p["bounds"][1])),
            default=float(p["default"]),
            file=str(p["file"]),
            field=p.get("field"),
            tier=int(p.get("tier", 1)),
        )
        for p in data.get("parameters", [])
    ]
    return CalibrationProfile(
        name=profile_name,
        description=str(prof.get("description", "")),
        overrides={str(k): float(v) for k, v in prof.get("overrides", {}).items()},
        parameters=params,
    )


def apply_profile_to_config(cfg: Any, profile: CalibrationProfile) -> Any:
    """Apply registry overrides onto MicroMatchConfig fields (tier 0–2 only)."""
    field_map = {
        "match.dt_default": "dt_default",
        "action.action_pass_base": "action_pass_base",
        "action.action_shot_base": "action_shot_base",
        "action.action_shot_cooldown_sec": "action_shot_cooldown_sec",
        "action.action_cross_base": "action_cross_base",
        "shot.shot_sot_z_in_goal": "shot_sot_z_in_goal",
        "shot.shot_sot_z_wide": "shot_sot_z_wide",
        "shot.gk_b_reflex": "gk_b_reflex",
        "shot.xg_geom_base": "xg_geom_base",
        "discipline.discipline_foul_target_base": "discipline_foul_target_base",
        "discipline.discipline_yellow_on_foul_base": "discipline_yellow_on_foul_base",
        "discipline.cross_tick_target": "cross_tick_target",
        "possession.possession_strength_beta": "possession_strength_beta",
        "pass.pass_b_land_err": "pass_b_land_err",
        "pass.pass_intercept_b_dist": "pass_intercept_b_dist",
        "macro.lambda0": "lambda0",
        "macro.gamma_xg": "gamma_xg",
        "affective.z_emo_decay": "z_emo_decay",
    }
    for param_id, attr in field_map.items():
        if param_id not in profile.overrides:
            continue
        if hasattr(cfg, attr):
            setattr(cfg, attr, profile.get(param_id))
    return cfg

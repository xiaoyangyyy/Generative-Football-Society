"""Load real coach profiles and attach to SocietyAgent (dynamics-based priors)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.data_engine.entity_dynamics import (
    COACH_PRESET_HINTS,
    coach_authority_dynamics,
    coach_exogenous_inputs,
    coach_mental_equilibrium,
    coach_reputation_prior,
    dominant_preset,
    preset_affinities,
)


@dataclass
class CoachProfile:
    team_id: str
    name: str
    nationality: str = ""
    in_charge_since: Optional[int] = None
    previous_role: str = ""
    source: str = "manual"
    preferred_preset: str = "balanced"
    mental: Dict[str, float] = field(default_factory=dict)
    preset_affinities: Dict[str, float] = field(default_factory=dict)
    dynamics_input: Dict[str, float] = field(default_factory=dict)
    team_meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "team_id": self.team_id,
            "name": self.name,
            "nationality": self.nationality,
            "in_charge_since": self.in_charge_since,
            "previous_role": self.previous_role,
            "source": self.source,
            "preferred_preset": self.preferred_preset,
            "mental": self.mental,
            "preset_affinities": self.preset_affinities,
            "dynamics_input": self.dynamics_input,
            "team_meta": self.team_meta,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CoachProfile":
        return cls(
            team_id=str(d["team_id"]),
            name=str(d.get("name", "Unknown")),
            nationality=str(d.get("nationality", "")),
            in_charge_since=d.get("in_charge_since"),
            previous_role=str(d.get("previous_role", "")),
            source=str(d.get("source", "manual")),
            preferred_preset=str(d.get("preferred_preset", "balanced")),
            mental={k: float(v) for k, v in (d.get("mental") or {}).items()},
            preset_affinities={k: float(v) for k, v in (d.get("preset_affinities") or {}).items()},
            dynamics_input={k: float(v) for k, v in (d.get("dynamics_input") or {}).items()},
            team_meta=dict(d.get("team_meta") or {}),
        )


def derive_mental_attributes(
    *,
    in_charge_since: Optional[int],
    fifa_ranking: Optional[float],
    style_desc: str = "",
    coach_name: str = "",
    current_year: int = 2026,
) -> Dict[str, float]:
    tenure = 0.0
    if in_charge_since:
        tenure = max(0.0, float(current_year - int(in_charge_since)))
    rank = float(fifa_ranking) if fifa_ranking is not None else 50.0
    rep = coach_reputation_prior(coach_name, COACH_PRESET_HINTS)
    u = coach_exogenous_inputs(
        tenure_years=tenure,
        fifa_ranking=rank,
        reputation_prior=rep,
        style_desc=style_desc,
    )
    mental = coach_mental_equilibrium(u)
    aff = preset_affinities(mental, style_desc)
    preset = dominant_preset(aff)
    mental["_inferred_preset"] = preset
    mental["_dynamics_u"] = {f"u{i}": float(v) for i, v in enumerate(u)}
    return mental


def infer_preferred_preset(mental: Dict[str, float], style_desc: str, coach_name: str) -> str:
    aff = preset_affinities(mental, style_desc)
    if coach_name in COACH_PRESET_HINTS:
        hinted = COACH_PRESET_HINTS[coach_name]
        if hinted in aff:
            aff = dict(aff)
            aff[hinted] = aff.get(hinted, 0.0) + 0.25
            s = sum(aff.values())
            aff = {k: v / s for k, v in aff.items()}
    return dominant_preset(aff)


def build_coach_profile_payload(
    team_id: str,
    name: str,
    *,
    nationality: str = "",
    in_charge_since: Optional[int] = None,
    previous_role: str = "",
    source: str = "dynamics",
    fifa_ranking: Optional[float] = None,
    style_desc: str = "",
    team_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    mental_raw = derive_mental_attributes(
        in_charge_since=in_charge_since,
        fifa_ranking=fifa_ranking,
        style_desc=style_desc,
        coach_name=name,
    )
    u_meta = mental_raw.pop("_dynamics_u", {})
    mental_raw.pop("_inferred_preset", "balanced")
    aff = preset_affinities(mental_raw, style_desc)
    preset = infer_preferred_preset(mental_raw, style_desc, name)
    return {
        "team_id": team_id,
        "name": name,
        "nationality": nationality,
        "in_charge_since": in_charge_since,
        "previous_role": previous_role,
        "source": source,
        "preferred_preset": preset,
        "mental": mental_raw,
        "preset_affinities": aff,
        "dynamics_input": u_meta,
        "team_meta": team_meta or {},
    }


def coach_authority_from_profile(profile: CoachProfile) -> float:
    return coach_authority_dynamics(profile.mental)


def load_coaches_json(path: str) -> Dict[str, CoachProfile]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[str, CoachProfile] = {}
    for team_id, payload in raw.items():
        if isinstance(payload, dict) and "team_id" not in payload:
            payload = {"team_id": team_id, **payload}
        out[team_id] = CoachProfile.from_dict(payload)
    return out


def default_coaches_path(base_dir: str) -> str:
    return os.path.join(base_dir, "data", "coaches", "wc2026_coaches.json")


def attach_coaches_to_agents(agents: Dict[str, Any], coaches: Dict[str, CoachProfile]) -> int:
    n = 0
    for team_name, agent in agents.items():
        prof = coaches.get(team_name)
        if not prof:
            continue
        agent.coach_profile = prof
        agent.coach_name = prof.name
        agent.roles["Manager"]["name"] = prof.name
        agent.coach_authority = coach_authority_from_profile(prof)
        agent.semantic_memory["coach"] = prof.name
        agent.semantic_memory["coach_nationality"] = prof.nationality
        agent.semantic_memory["coach_preset"] = prof.preferred_preset
        agent.semantic_memory["coach_preset_affinities"] = prof.preset_affinities
        n += 1
    return n


def merge_coach_into_tactical_map(tactical_map: Dict[str, Any], coaches: Dict[str, CoachProfile]) -> Dict[str, Any]:
    merged = dict(tactical_map)
    for team, prof in coaches.items():
        entry = dict(merged.get(team, {}))
        entry["coach"] = prof.to_dict()
        merged[team] = entry
    return merged

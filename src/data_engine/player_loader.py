"""Player profiles with the same dynamics specification as coaches."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.data_engine.entity_dynamics import (
    PLAYER_CHANNEL_NAMES,
    PLAYER_CONDITION_KEYS,
    build_player_dynamics_payload,
    dominant_channel,
)


@dataclass
class PlayerProfile:
    player_id: str
    name: str
    team_id: str
    role: str
    condition: Dict[str, float] = field(default_factory=dict)
    channel_affinities: Dict[str, float] = field(default_factory=dict)
    primary_channel: str = "progressive_passer"
    dynamics_input: Dict[str, float] = field(default_factory=dict)
    role_embedding: List[float] = field(default_factory=list)
    abilities: Dict[str, float] = field(default_factory=dict)
    dynamics_model: str = "entity_dynamics_v1"
    # identity / provenance
    shirt_number: Optional[int] = None
    age: Optional[int] = None
    club: str = ""
    market_value_eur: float = 0.0
    international_caps: int = 0
    squad_role: str = "starter"
    source: str = "transfermarkt"
    transfermarkt_url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "player_id": self.player_id,
            "name": self.name,
            "team_id": self.team_id,
            "role": self.role,
            "condition": self.condition,
            "channel_affinities": self.channel_affinities,
            "primary_channel": self.primary_channel,
            "dynamics_input": self.dynamics_input,
            "role_embedding": self.role_embedding,
            "abilities": self.abilities,
            "dynamics_model": self.dynamics_model,
            "shirt_number": self.shirt_number,
            "age": self.age,
            "club": self.club,
            "market_value_eur": self.market_value_eur,
            "international_caps": self.international_caps,
            "squad_role": self.squad_role,
            "source": self.source,
            "transfermarkt_url": self.transfermarkt_url,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PlayerProfile":
        return cls(
            player_id=str(d["player_id"]),
            name=str(d.get("name", "")),
            team_id=str(d.get("team_id", "")),
            role=str(d.get("role", "CM")),
            condition={k: float(v) for k, v in (d.get("condition") or {}).items()},
            channel_affinities={
                k: float(v) for k, v in (d.get("channel_affinities") or {}).items()
            },
            primary_channel=str(d.get("primary_channel", "progressive_passer")),
            dynamics_input={k: float(v) for k, v in (d.get("dynamics_input") or {}).items()},
            role_embedding=[float(x) for x in (d.get("role_embedding") or [])],
            abilities={k: float(v) for k, v in (d.get("abilities") or {}).items()},
            dynamics_model=str(d.get("dynamics_model", "entity_dynamics_v1")),
            shirt_number=d.get("shirt_number"),
            age=d.get("age"),
            club=str(d.get("club", "")),
            market_value_eur=float(d.get("market_value_eur", 0)),
            international_caps=int(d.get("international_caps", 0)),
            squad_role=str(d.get("squad_role", "starter")),
            source=str(d.get("source", "")),
            transfermarkt_url=str(d.get("transfermarkt_url", "")),
        )


def build_player_profile_from_observables(
    *,
    player_id: str,
    name: str,
    team_id: str,
    role: str,
    market_value_eur: float,
    international_caps: int,
    height_cm: float,
    squad_max_mv: float,
    age: Optional[int] = None,
    shirt_number: Optional[int] = None,
    club: str = "",
    squad_role: str = "starter",
    transfermarkt_url: str = "",
) -> Dict[str, Any]:
    import math

    dyn = build_player_dynamics_payload(
        role=role,
        log_market_value=math.log1p(max(0.0, market_value_eur)),
        international_caps=float(international_caps),
        height_cm=height_cm,
        squad_max_mv=squad_max_mv,
        age=float(age) if age is not None else None,
    )
    return {
        "player_id": player_id,
        "name": name,
        "team_id": team_id,
        "role": role,
        "shirt_number": shirt_number,
        "age": age,
        "club": club,
        "market_value_eur": market_value_eur,
        "international_caps": international_caps,
        "squad_role": squad_role,
        "source": "transfermarkt+dynamics",
        "transfermarkt_url": transfermarkt_url,
        **dyn,
    }


def condition_composure(profile: PlayerProfile) -> float:
    return float(profile.condition.get("composure", profile.abilities.get("mental", 0.5)))


__all__ = [
    "PlayerProfile",
    "PLAYER_CONDITION_KEYS",
    "PLAYER_CHANNEL_NAMES",
    "build_player_profile_from_observables",
    "dominant_channel",
    "condition_composure",
]

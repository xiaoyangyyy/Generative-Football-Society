"""Build 11-player squads for affective simulation from SocietyAgent priors."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, List

import numpy as np

from src.match_engine.formation import interpolate_anchors, roles_for_formation
from src.match_engine.tactical_catalog import normalize_formation_key
from src.match_engine.state import PlayerAbilities, PlayerAffectiveState, TeamAffectiveState

if TYPE_CHECKING:
    from src.simulation.agent import SocietyAgent

# emotion logit indices: pride, anger, fear, determination
_E_P, _E_A, _E_F, _E_D = 0, 1, 2, 3
ROLE_LABELS = {
    "GK": "Goalkeeper",
    "LB": "Left Back",
    "LWB": "Left Wing-Back",
    "CB": "Centre Back",
    "RB": "Right Back",
    "RWB": "Right Wing-Back",
    "DM": "Defensive Mid",
    "CM": "Central Mid",
    "LM": "Left Mid",
    "RM": "Right Mid",
    "LW": "Left Winger",
    "RW": "Right Winger",
    "AM": "Attacking Mid",
    "ST": "Striker",
}


def _stable_seed(team_name: str, salt: str) -> int:
    h = hashlib.md5(f"{team_name}:{salt}".encode()).hexdigest()
    return int(h[:8], 16)


def _emotion_logits_from_agent(agent: "SocietyAgent") -> np.ndarray:
    emo = getattr(agent, "emotion_profile", {}) or {}
    pride = float(emo.get("pride", 0.2))
    anger = float(emo.get("anger", 0.1))
    fear = float(emo.get("fear", 0.2))
    det = float(emo.get("determination", 0.4))
    # map probs to logits (avoid log(0))
    eps = 1e-6
    arr = np.array([pride, anger, fear, det], dtype=float) + eps
    arr = arr / arr.sum()
    return np.log(arr)


def build_team_squad(
    agent: "SocietyAgent",
    rng: np.random.Generator,
    base_dir: str | Path | None = None,
    *,
    match_eff_status: float | None = None,
) -> TeamAffectiveState:
    team_id = agent.team_name
    if base_dir is None:
        import os

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.data_engine.roster_loader import build_team_squad_from_roster
    from src.simulation.squad_registry import load_effective_roster, roster_identity

    roster = getattr(agent, "_roster_carryover_snapshot", None)
    roster_source = "prepared_carryover_snapshot"
    if roster is None:
        roster = load_effective_roster(base_dir, team_id)
        roster_source = "effective_roster"
    if roster is not None:
        real = build_team_squad_from_roster(agent, roster, rng)
        if real is None:
            raise ValueError(
                f"effective roster for {team_id!r} cannot form an available XI"
            )
        real.squad_provenance = {
            "schema_version": 1,
            "team_id": team_id,
            "source": roster_source,
            "roster_source": str(roster.get("source") or "unknown"),
            "roster_identity": roster_identity(roster),
            "roster_player_count": len(roster.get("players") or []),
            "simulation_player_count": len(real.players),
            "fallback_used": False,
            "fallback_reason": None,
        }
        return real

    seed = _stable_seed(team_id, "squad")
    local_rng = np.random.default_rng(seed)

    fatigue = float(np.clip(getattr(agent, "fatigue", 0.0), 0.0, 1.0))
    stamina_logit = float(np.arctanh(np.clip(1.0 - fatigue * 0.85, 0.05, 0.95)))
    psychology_profile = getattr(agent, "psychology_profile", {}) or {}
    morale_base = float(np.clip(psychology_profile.get("confidence", 0.5), 0.05, 0.98))
    morale_logit = float(np.arctanh(np.clip(morale_base * 2 - 1, -0.95, 0.95)))
    team_emo = _emotion_logits_from_agent(agent)

    media = float(np.clip(getattr(agent, "media_exposure", 0.5), 0.05, 1.0))
    icon_idx = int(np.argmax([0.3, 0.5, 0.5, 0.5, 0.5, 0.6, 0.75, 0.8, 0.95, 1.0, 0.95]))  # ST default icon

    from src.match_engine.tactical_profile import build_tactical_vector_for_agent

    coach_tactical_pre = build_tactical_vector_for_agent(agent)
    anchor_status = max(12.0, float(getattr(agent, "status_score", 50.0)))
    eff = float(match_eff_status if match_eff_status is not None else anchor_status)
    eff_ratio = float(np.clip(eff / anchor_status, 0.88, 1.10))
    tier = float(np.clip(anchor_status / 100.0 * eff_ratio, 0.25, 0.95))
    width = float(coach_tactical_pre.get("rotation_aggressiveness", 0.5))
    line_h = float(coach_tactical_pre.get("line_height", 0.5))
    fkey = normalize_formation_key(getattr(agent, "formation", "4-3-3"))
    role_list = roles_for_formation(fkey)

    players: List[PlayerAffectiveState] = []
    cb_count = 0
    for i, role in enumerate(role_list):
        pid = f"{team_id.lower().replace(' ', '_')}_{role.lower()}_{i}"
        is_icon = i == icon_idx
        mental = float(np.clip(0.45 + local_rng.normal(0, 0.08) + (0.12 if is_icon else 0), 0.1, 0.98))
        affinity = float(
            np.clip(
                0.35 + 0.45 * media + (0.25 if is_icon else 0) + local_rng.uniform(-0.05, 0.05),
                0.1,
                1.0,
            )
        )
        z = team_emo + local_rng.normal(0, 0.15, size=4)
        if role == "GK":
            z[_E_F] -= 0.1
            z[_E_P] += 0.05
        cb_wide = 0.0
        if role == "CB":
            cb_wide = 0.35 if cb_count == 0 else 0.65
            cb_count += 1
        ab = _role_abilities(role, tier, mental, is_icon)
        pos = interpolate_anchors(
            role,
            0.45,
            line_height=line_h,
            width=width,
            attacks_high_x=True,
            cb_wide=cb_wide,
            formation_key=fkey,
        )
        players.append(
            PlayerAffectiveState(
                player_id=pid,
                name=f"{team_id} {ROLE_LABELS.get(role, role)}",
                role=role,
                team_id=team_id,
                is_icon=is_icon,
                fan_affinity=affinity,
                mental=mental,
                z_emo=z,
                morale_logit=morale_logit + local_rng.normal(0, 0.08),
                cognitive_load=float(np.clip(0.12 + fatigue * 0.35, 0, 1)),
                stamina_logit=stamina_logit + local_rng.normal(0, 0.05),
                spatial_cognition_logit=float(np.arctanh(np.clip(mental * 1.1 - 0.5, -0.9, 0.9))),
                position=pos.copy(),
                abilities=ab,
                cb_wide=cb_wide,
            )
        )

    icon_id = players[icon_idx].player_id
    peer = _build_peer_matrix(players, local_rng)

    coach_tactical = dict(coach_tactical_pre)

    from src.match_engine.state import CoachAffectiveState

    conflict = float(getattr(agent, "conflict_heat", 0.12))
    cohesion = float(getattr(agent, "team_cohesion", 0.6))
    coach_name = str(getattr(agent, "coach_name", "") or "")
    coach = CoachAffectiveState(
        team_id=team_id,
        coach_name=coach_name,
        z_stress=float(np.arctanh(np.clip(conflict * 1.2, 0.01, 0.9))),
        z_trust=float(np.arctanh(np.clip(cohesion * 1.4 - 0.5, -0.9, 0.9))),
        z_rage=float(np.arctanh(np.clip(conflict * 0.8, 0, 0.85))),
        tactical_base=dict(coach_tactical),
        tactical_current=dict(coach_tactical),
    )

    return TeamAffectiveState(
        team_id=team_id,
        players=players,
        coach=coach,
        peer_matrix=peer,
        icon_player_id=icon_id,
        formation_key=fkey,
        squad_provenance={
            "schema_version": 1,
            "team_id": team_id,
            "source": "synthetic_status_fallback",
            "roster_source": None,
            "roster_identity": None,
            "roster_player_count": 0,
            "simulation_player_count": len(players),
            "fallback_used": True,
            "fallback_reason": "roster_unavailable",
        },
    )


def _role_abilities(role: str, tier: float, mental: float, is_icon: bool) -> PlayerAbilities:
    base = tier * 0.65 + mental * 0.35 + (0.08 if is_icon else 0)
    if role == "GK":
        return PlayerAbilities(
            tech=base * 0.75,
            pass_skill=base * 0.7,
            vision=base * 0.8,
            spatial=base * 0.85,
            pace=base * 0.55,
            press=0.15,
            curve=0.2,
            shot=0.15,
            power=0.4,
            knuckle=0.1,
            aerial=base * 0.9,
            heading=base * 0.85,
            gk_reflex=base * 1.05,
            gk_aerial=base * 1.0,
        )
    if role in ("CB", "LB", "RB"):
        return PlayerAbilities(
            tech=base * 0.85,
            pass_skill=base * 0.8,
            vision=base * 0.75,
            spatial=base * 0.9,
            pace=base * 0.85,
            press=base * 0.95,
            curve=base * 0.55,
            shot=base * 0.45,
            power=base * 0.7,
            knuckle=base * 0.35,
            aerial=base * 0.95,
            heading=base * 0.9,
        )
    if role in ("LW", "RW", "ST"):
        return PlayerAbilities(
            tech=base * 1.05,
            pass_skill=base * 0.9,
            vision=base * 0.95,
            spatial=base * 1.0,
            pace=base * 1.1,
            press=base * 0.7,
            curve=base * 1.05,
            shot=base * 1.08,
            power=base * 0.95,
            knuckle=base * 0.75,
            aerial=base * 0.75,
            heading=base * 0.85,
        )
    return PlayerAbilities(
        tech=base,
        pass_skill=base * 1.05,
        vision=base * 1.05,
        spatial=base * 1.0,
        pace=base,
        press=base * 0.85,
        curve=base * 0.85,
        shot=base * 0.8,
        power=base * 0.85,
        knuckle=base * 0.5,
        aerial=base * 0.7,
        heading=base * 0.65,
    )


def init_away_positions(team: TeamAffectiveState) -> None:
    from src.match_engine.formation import mirror_for_away

    for p in team.players:
        p.position = mirror_for_away(p.position)


def _build_peer_matrix(players: List[PlayerAffectiveState], rng: np.random.Generator) -> np.ndarray:
    n = len(players)
    mat = np.zeros((n, n), dtype=float)
    role_group = {
        "GK": 0,
        "LB": 1,
        "CB": 1,
        "RB": 1,
        "DM": 2,
        "CM": 2,
        "LW": 3,
        "RW": 3,
        "ST": 3,
    }
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            gi = role_group.get(players[i].role, 2)
            gj = role_group.get(players[j].role, 2)
            same_line = float(gi == gj)
            adjacent = float(abs(gi - gj) == 1)
            mat[i, j] = 0.08 * same_line + 0.04 * adjacent + 0.01
    return mat


def pick_random_outfield_player(team: TeamAffectiveState, rng: np.random.Generator, exclude_gk: bool = True) -> PlayerAffectiveState:
    pool = [p for p in team.players if p.on_pitch and (not exclude_gk or p.role != "GK")]
    if not pool:
        return team.players[0]
    weights = np.array([1.4 if p.role in ("ST", "LW", "RW") else 1.0 for p in pool], dtype=float)
    weights /= weights.sum()
    idx = int(rng.choice(len(pool), p=weights))
    return pool[idx]

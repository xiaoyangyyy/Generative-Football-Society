"""Load per-team roster JSON into match engine player states."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from src.match_engine.formation import interpolate_anchors, roles_for_formation
from src.match_engine.state import PlayerAbilities, PlayerAffectiveState, TeamAffectiveState
from src.match_engine.tactical_catalog import normalize_formation_key
from src.match_engine.tactical_profile import build_tactical_vector_for_agent

if TYPE_CHECKING:
    from src.simulation.agent import SocietyAgent


def roster_path_for_team(base_dir: str, team_name: str) -> str:
    safe = re.sub(r"[^\w\-]+", "_", team_name)
    return os.path.join(base_dir, "data", "rosters", f"{safe}.json")


def load_roster_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _abilities_from_dict(d: Dict[str, float]) -> PlayerAbilities:
    gk_val = float(d.get("gk", 0.0))
    phys = float(d.get("phys", 0.5))
    return PlayerAbilities(
        tech=float(d.get("tech", 0.5)),
        pass_skill=float(d.get("pass_skill", d.get("tech", 0.5))),
        vision=float(d.get("vision", 0.5)),
        spatial=float(d.get("spatial", d.get("vision", 0.5))),
        pace=float(d.get("pace", 0.5)),
        press=float(d.get("press", 0.5)),
        curve=float(d.get("curve", 0.4)),
        shot=float(d.get("shot", d.get("tech", 0.5) * 0.9)),
        power=float(d.get("power", phys)),
        aerial=float(d.get("aerial", 0.5)),
        heading=float(d.get("heading", d.get("aerial", 0.5))),
        gk_reflex=float(d.get("gk_reflex", gk_val if gk_val > 0 else 0.58)),
        gk_aerial=float(d.get("gk_aerial", gk_val * 0.95 if gk_val > 0 else 0.55)),
    )


def _player_state_from_pdata(
    pdata: Dict[str, Any],
    *,
    role: str,
    team_id: str,
    on_pitch: bool,
    position: np.ndarray,
    team_emo: np.ndarray,
    morale_logit: float,
    fatigue: float,
    media: float,
    cb_wide: float,
    rng: np.random.Generator,
) -> PlayerAffectiveState:
    from src.data_engine.entity_dynamics import latent_logit_from_unit_interval
    from src.match_engine.math_utils import sigmoid

    cond = pdata.get("condition") or {}
    mental = float(cond.get("composure", pdata.get("abilities", {}).get("mental", 0.55)))
    ab = _abilities_from_dict(pdata.get("abilities", {}))
    ch_aff = {k: float(v) for k, v in (pdata.get("channel_affinities") or {}).items()}
    avail = float(pdata.get("availability", 1.0))
    if pdata.get("squad_role") == "suspended":
        avail = 0.0
    return PlayerAffectiveState(
        player_id=str(pdata.get("player_id", "")),
        name=str(pdata.get("name", "")),
        role=role,
        team_id=team_id,
        is_icon=False,
        fan_affinity=float(sigmoid(-0.4 + 0.9 * media)),
        mental=mental,
        z_emo=team_emo + rng.normal(0, 0.12, size=4),
        morale_logit=morale_logit + rng.normal(0, 0.06),
        cognitive_load=float(sigmoid(-1.8 + 2.5 * fatigue)),
        stamina_logit=latent_logit_from_unit_interval(1.0 - fatigue * 0.85),
        spatial_cognition_logit=latent_logit_from_unit_interval(mental),
        position=position.copy(),
        abilities=ab,
        cb_wide=cb_wide,
        channel_affinities=ch_aff,
        primary_channel=str(pdata.get("primary_channel", "")),
        availability=avail,
        squad_role=str(pdata.get("squad_role", "starter")),
        on_pitch=on_pitch and avail > 0.15,
    )


def build_team_squad_from_roster(
    agent: "SocietyAgent",
    roster: Dict[str, Any],
    rng: np.random.Generator,
) -> Optional[TeamAffectiveState]:
    players_data = roster.get("players") or []
    starters = [
        p
        for p in players_data
        if p.get("squad_role") == "starter" and float(p.get("availability", 1.0)) > 0.15
    ]
    if len(starters) < 11:
        starters = [
            p for p in players_data if float(p.get("availability", 1.0)) > 0.15
        ][:11]
    if len(starters) < 11:
        return None
    bench_pool = [
        p
        for p in players_data
        if p not in starters
        and p.get("squad_role") != "suspended"
        and float(p.get("availability", 1.0)) > 0.15
    ]

    team_id = agent.team_name
    from src.data_engine.entity_dynamics import latent_logit_from_unit_interval
    from src.match_engine.math_utils import sigmoid

    fatigue = float(sigmoid(-2.5 + 2.2 * (1.0 - float(getattr(agent, "fatigue", 0.0)))))
    stamina_logit = latent_logit_from_unit_interval(1.0 - fatigue * 0.85)
    morale_base = float(agent.psychology_profile.get("confidence", 0.5))
    morale_logit = latent_logit_from_unit_interval(morale_base)

    coach_tactical_pre = build_tactical_vector_for_agent(agent)
    width = float(coach_tactical_pre.get("rotation_aggressiveness", 0.5))
    line_h = float(coach_tactical_pre.get("line_height", 0.5))
    fkey = normalize_formation_key(getattr(agent, "formation", roster.get("formation", "4-3-3")))
    role_list = roles_for_formation(fkey)

    from src.match_engine.squad_factory import _build_peer_matrix, _emotion_logits_from_agent
    from src.match_engine.state import CoachAffectiveState

    team_emo = _emotion_logits_from_agent(agent)
    media = float(sigmoid(2.0 * float(getattr(agent, "media_exposure", 0.5)) - 1.0))

    by_role: Dict[str, List[Dict]] = {}
    for p in starters:
        by_role.setdefault(str(p.get("role", "CM")), []).append(p)
    used_ids: set = set()

    def _pop_role(role: str, fallbacks: Optional[List[str]] = None) -> Optional[Dict]:
        for r in [role] + (fallbacks or []):
            pool = [x for x in by_role.get(r, []) if id(x) not in used_ids]
            if pool:
                return pool[0]
        for p in starters:
            if id(p) not in used_ids:
                return p
        return None

    players: List[PlayerAffectiveState] = []
    icon_idx = 0
    best_mv = -1.0
    cb_count = 0

    for i, role in enumerate(role_list):
        fallbacks = None
        if role in ("LM", "RM"):
            fallbacks = ["LW", "RW", "CM", "AM"]
        elif role in ("LW", "RW"):
            fallbacks = ["LM", "RM", "ST", "AM"]
        elif role == "DM":
            fallbacks = ["CM", "CB"]
        pdata = _pop_role(role, fallbacks)
        if pdata is None:
            continue
        used_ids.add(id(pdata))
        mv = float(pdata.get("market_value_eur") or 0)
        if mv > best_mv:
            best_mv = mv
            icon_idx = i

        icon_bump = 0.15 * (1.0 if i == icon_idx else 0.0)
        affinity = float(sigmoid(-0.4 + 0.9 * media + icon_bump))
        cb_wide = 0.0
        if role == "CB":
            cb_wide = 0.35 if cb_count == 0 else 0.65
            cb_count += 1

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
            _player_state_from_pdata(
                pdata,
                role=role,
                team_id=team_id,
                on_pitch=True,
                position=pos,
                team_emo=team_emo,
                morale_logit=morale_logit,
                fatigue=fatigue,
                media=media,
                cb_wide=cb_wide,
                rng=rng,
            )
        )
        players[-1].fan_affinity = affinity
        players[-1].stamina_logit = stamina_logit

    if len(players) < 11:
        return None

    # Bench (up to 12) — off pitch, available for substitutions
    bench_y = 0.04
    for j, pdata in enumerate(bench_pool[:12]):
        brole = str(pdata.get("role", "CM"))
        bpos = interpolate_anchors(
            brole,
            bench_y + 0.01 * (j % 6),
            line_height=0.5,
            width=0.5,
            attacks_high_x=False,
            formation_key=fkey,
        )
        bp = _player_state_from_pdata(
            pdata,
            role=brole,
            team_id=team_id,
            on_pitch=False,
            position=bpos,
            team_emo=team_emo,
            morale_logit=morale_logit,
            fatigue=fatigue * 0.85,
            media=media,
            cb_wide=0.0,
            rng=rng,
        )
        bp.squad_role = "bench"
        players.append(bp)

    players[icon_idx].is_icon = True
    players[icon_idx].fan_affinity = float(sigmoid(latent_logit_from_unit_interval(players[icon_idx].fan_affinity) + 0.35))

    conflict = float(getattr(agent, "conflict_heat", 0.12))
    cohesion = float(getattr(agent, "team_cohesion", 0.6))
    coach = CoachAffectiveState(
        team_id=team_id,
        coach_name=str(getattr(agent, "coach_name", "")),
        z_stress=latent_logit_from_unit_interval(conflict * 1.05),
        z_trust=latent_logit_from_unit_interval(cohesion),
        z_rage=latent_logit_from_unit_interval(conflict * 0.85),
        tactical_base=dict(coach_tactical_pre),
        tactical_current=dict(coach_tactical_pre),
    )

    return TeamAffectiveState(
        team_id=team_id,
        players=players,
        coach=coach,
        peer_matrix=_build_peer_matrix(players, rng),
        icon_player_id=players[icon_idx].player_id,
        formation_key=fkey,
    )

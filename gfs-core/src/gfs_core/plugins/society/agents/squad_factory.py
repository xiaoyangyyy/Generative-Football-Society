from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np

from gfs_core.contracts.micro_math import latent_logit, sigmoid
from gfs_core.contracts.micro_state import (
    CoachAffectiveState,
    PlayerAbilities,
    PlayerAffectiveState,
    TeamAffectiveState,
)
from gfs_core.contracts.society import TeamProfile

ROLE_LABELS = {
    "GK": "Goalkeeper", "LB": "Left Back", "LWB": "Left Wing-Back",
    "CB": "Centre Back", "RB": "Right Back", "RWB": "Right Wing-Back",
    "DM": "Defensive Mid", "CM": "Central Mid", "LM": "Left Mid",
    "RM": "Right Mid", "LW": "Left Winger", "RW": "Right Winger",
    "AM": "Attacking Mid", "ST": "Striker",
}


class SquadFactory:
    def __init__(self, rosters, spatial) -> None:
        self.rosters = rosters
        self.spatial = spatial

    def build(
        self,
        profile: TeamProfile,
        rng: np.random.Generator,
        *,
        match_eff_status: float | None = None,
        roster_override: Mapping[str, Any] | None = None,
    ) -> TeamAffectiveState:
        roster = self.rosters.load(profile.team_name, roster_override)
        if roster is not None:
            team = self._from_roster(profile, roster, rng)
            if team is None:
                raise ValueError(
                    f"effective roster for {profile.team_name!r} cannot form an available XI"
                )
            team.squad_provenance = {
                "schema_version": 1,
                "team_id": profile.team_name,
                "source": "prepared_carryover_snapshot" if roster_override is not None else "effective_roster",
                "roster_source": str(roster.get("source") or "unknown"),
                "roster_identity": self.rosters.identity(roster),
                "roster_player_count": len(roster.get("players") or []),
                "simulation_player_count": len(team.players),
                "fallback_used": False,
                "fallback_reason": None,
            }
            return team
        return self._from_synthetic(profile, match_eff_status)

    def _from_roster(self, profile, roster, rng):
        players_data = roster.get("players") or []
        starters = [
            player for player in players_data
            if player.get("squad_role") == "starter"
            and float(player.get("availability", 1.0)) > 0.15
        ]
        if len(starters) < 11:
            starters = [
                player for player in players_data
                if float(player.get("availability", 1.0)) > 0.15
            ][:11]
        if len(starters) < 11:
            return None
        bench = [
            player for player in players_data
            if player not in starters
            and player.get("squad_role") == "bench"
            and float(player.get("availability", 1.0)) > 0.15
        ]

        fatigue = float(sigmoid(-2.5 + 2.2 * (1.0 - profile.fatigue)))
        stamina = latent_logit(1.0 - fatigue * 0.85)
        morale = latent_logit(profile.psychology_profile.get("confidence", 0.5))
        tactical = dict(profile.tactical)
        width = float(tactical.get("rotation_aggressiveness", 0.5))
        line_height = float(tactical.get("line_height", 0.5))
        formation = self.spatial.normalize_formation_key(
            profile.formation or roster.get("formation", "4-3-3")
        )
        roles = self.spatial.roles_for_formation(formation)
        emotions = _emotion_logits(profile)
        media = float(sigmoid(2.0 * profile.media_exposure - 1.0))
        by_role = {}
        for player in starters:
            by_role.setdefault(str(player.get("role", "CM")), []).append(player)
        used = set()
        players = []
        icon_index = 0
        best_value = -1.0
        cb_count = 0

        for index, role in enumerate(roles):
            fallbacks = None
            if role in {"LM", "RM"}:
                fallbacks = ["LW", "RW", "CM", "AM"]
            elif role in {"LW", "RW"}:
                fallbacks = ["LM", "RM", "ST", "AM"]
            elif role == "DM":
                fallbacks = ["CM", "CB"]
            pdata = _pop_role(starters, by_role, used, role, fallbacks)
            if pdata is None:
                continue
            used.add(id(pdata))
            market_value = float(pdata.get("market_value_eur") or 0)
            if market_value > best_value:
                best_value = market_value
                icon_index = index
            cb_wide = 0.0
            if role == "CB":
                cb_wide = 0.35 if cb_count == 0 else 0.65
                cb_count += 1
            position = self.spatial.interpolate_anchors(
                role,
                0.45,
                line_height=line_height,
                width=width,
                attacks_high_x=True,
                cb_wide=cb_wide,
                formation_key=formation,
            )
            player = _player_from_data(
                pdata,
                role=role,
                team_id=profile.team_name,
                on_pitch=True,
                position=position,
                emotions=emotions,
                morale=morale,
                fatigue=fatigue,
                media=media,
                cb_wide=cb_wide,
                rng=rng,
            )
            affinity = float(sigmoid(
                -0.4 + 0.9 * media + 0.15 * float(index == icon_index)
            ))
            player.fan_affinity = affinity
            player.stamina_logit = stamina
            players.append(player)
        if len(players) < 11:
            return None

        for index, pdata in enumerate(bench[:12]):
            role = str(pdata.get("role", "CM"))
            position = self.spatial.interpolate_anchors(
                role,
                0.04 + 0.01 * (index % 6),
                line_height=0.5,
                width=0.5,
                attacks_high_x=False,
                formation_key=formation,
            )
            player = _player_from_data(
                pdata,
                role=role,
                team_id=profile.team_name,
                on_pitch=False,
                position=position,
                emotions=emotions,
                morale=morale,
                fatigue=fatigue * 0.85,
                media=media,
                cb_wide=0.0,
                rng=rng,
            )
            player.squad_role = "bench"
            players.append(player)

        players[icon_index].is_icon = True
        players[icon_index].fan_affinity = float(sigmoid(
            latent_logit(players[icon_index].fan_affinity) + 0.35
        ))
        coach = CoachAffectiveState(
            team_id=profile.team_name,
            coach_name=profile.coach_name,
            z_stress=latent_logit(profile.conflict_heat * 1.05),
            z_trust=latent_logit(profile.team_cohesion),
            z_rage=latent_logit(profile.conflict_heat * 0.85),
            tactical_base=tactical,
            tactical_current=dict(tactical),
        )
        return TeamAffectiveState(
            team_id=profile.team_name,
            players=players,
            coach=coach,
            peer_matrix=_peer_matrix(players),
            icon_player_id=players[icon_index].player_id,
            formation_key=formation,
        )

    def _from_synthetic(self, profile, match_eff_status):
        seed = int(hashlib.md5(
            f"{profile.team_name}:squad".encode()
        ).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        fatigue = float(np.clip(profile.fatigue, 0.0, 1.0))
        stamina = float(np.arctanh(np.clip(1.0 - fatigue * 0.85, 0.05, 0.95)))
        confidence = float(np.clip(
            profile.psychology_profile.get("confidence", 0.5), 0.05, 0.98,
        ))
        morale = float(np.arctanh(np.clip(confidence * 2 - 1, -0.95, 0.95)))
        emotions = _emotion_logits(profile)
        media = float(np.clip(profile.media_exposure, 0.05, 1.0))
        tactical = dict(profile.tactical)
        anchor_status = max(12.0, profile.status_score)
        effective = float(
            match_eff_status if match_eff_status is not None else anchor_status
        )
        ratio = float(np.clip(effective / anchor_status, 0.88, 1.10))
        tier = float(np.clip(anchor_status / 100.0 * ratio, 0.25, 0.95))
        width = float(tactical.get("rotation_aggressiveness", 0.5))
        line_height = float(tactical.get("line_height", 0.5))
        formation = self.spatial.normalize_formation_key(profile.formation)
        roles = self.spatial.roles_for_formation(formation)
        icon_index = 9
        players = []
        cb_count = 0
        for index, role in enumerate(roles):
            player_id = f"{profile.team_name.lower().replace(' ', '_')}_{role.lower()}_{index}"
            is_icon = index == icon_index
            mental = float(np.clip(
                0.45 + rng.normal(0, 0.08) + (0.12 if is_icon else 0), 0.1, 0.98,
            ))
            affinity = float(np.clip(
                0.35 + 0.45 * media + (0.25 if is_icon else 0)
                + rng.uniform(-0.05, 0.05),
                0.1,
                1.0,
            ))
            emotion = emotions + rng.normal(0, 0.15, size=4)
            if role == "GK":
                emotion[2] -= 0.1
                emotion[0] += 0.05
            cb_wide = 0.0
            if role == "CB":
                cb_wide = 0.35 if cb_count == 0 else 0.65
                cb_count += 1
            position = self.spatial.interpolate_anchors(
                role,
                0.45,
                line_height=line_height,
                width=width,
                attacks_high_x=True,
                cb_wide=cb_wide,
                formation_key=formation,
            )
            players.append(PlayerAffectiveState(
                player_id=player_id,
                name=f"{profile.team_name} {ROLE_LABELS.get(role, role)}",
                role=role,
                team_id=profile.team_name,
                is_icon=is_icon,
                fan_affinity=affinity,
                mental=mental,
                z_emo=emotion,
                morale_logit=morale + rng.normal(0, 0.08),
                cognitive_load=float(np.clip(0.12 + fatigue * 0.35, 0, 1)),
                stamina_logit=stamina + rng.normal(0, 0.05),
                spatial_cognition_logit=float(np.arctanh(
                    np.clip(mental * 1.1 - 0.5, -0.9, 0.9)
                )),
                position=position.copy(),
                abilities=_role_abilities(role, tier, mental, is_icon),
                cb_wide=cb_wide,
            ))
        coach = CoachAffectiveState(
            team_id=profile.team_name,
            coach_name=profile.coach_name,
            z_stress=float(np.arctanh(np.clip(profile.conflict_heat * 1.2, 0.01, 0.9))),
            z_trust=float(np.arctanh(np.clip(profile.team_cohesion * 1.4 - 0.5, -0.9, 0.9))),
            z_rage=float(np.arctanh(np.clip(profile.conflict_heat * 0.8, 0, 0.85))),
            tactical_base=tactical,
            tactical_current=dict(tactical),
        )
        return TeamAffectiveState(
            team_id=profile.team_name,
            players=players,
            coach=coach,
            peer_matrix=_peer_matrix(players),
            icon_player_id=players[icon_index].player_id,
            formation_key=formation,
            squad_provenance={
                "schema_version": 1,
                "team_id": profile.team_name,
                "source": "synthetic_status_fallback",
                "roster_source": None,
                "roster_identity": None,
                "roster_player_count": 0,
                "simulation_player_count": len(players),
                "fallback_used": True,
                "fallback_reason": "roster_unavailable",
            },
        )


def _pop_role(starters, by_role, used, role, fallbacks):
    for candidate_role in [role] + (fallbacks or []):
        pool = [
            player for player in by_role.get(candidate_role, [])
            if id(player) not in used
        ]
        if pool:
            return pool[0]
    for player in starters:
        if id(player) not in used:
            return player
    return None


def _player_from_data(
    pdata,
    *,
    role,
    team_id,
    on_pitch,
    position,
    emotions,
    morale,
    fatigue,
    media,
    cb_wide,
    rng,
):
    condition = pdata.get("condition") or {}
    mental = _finite(
        condition.get("composure"),
        _finite(pdata.get("abilities", {}).get("mental"), 0.55),
    )
    availability = _finite(pdata.get("availability"), 1.0)
    if pdata.get("squad_role") == "suspended":
        availability = 0.0
    return PlayerAffectiveState(
        player_id=str(pdata.get("player_id", "")),
        name=str(pdata.get("name", "")),
        role=role,
        team_id=team_id,
        fan_affinity=float(sigmoid(-0.4 + 0.9 * media)),
        mental=mental,
        z_emo=emotions + rng.normal(0, 0.12, size=4),
        morale_logit=morale + rng.normal(0, 0.06),
        cognitive_load=float(sigmoid(-1.8 + 2.5 * fatigue)),
        stamina_logit=latent_logit(1.0 - fatigue * 0.85),
        spatial_cognition_logit=latent_logit(mental),
        position=position.copy(),
        abilities=_abilities(pdata.get("abilities", {})),
        cb_wide=cb_wide,
        channel_affinities={
            key: _finite(value, 0.0)
            for key, value in (pdata.get("channel_affinities") or {}).items()
        },
        primary_channel=str(pdata.get("primary_channel", "")),
        availability=availability,
        squad_role=str(pdata.get("squad_role", "starter")),
        on_pitch=on_pitch and availability > 0.15,
    )


def _abilities(values):
    goalkeeper = _finite(values.get("gk"), 0.0)
    physical = _finite(values.get("phys"), 0.5)
    technical = _finite(values.get("tech"), 0.5)
    vision = _finite(values.get("vision"), 0.5)
    aerial = _finite(values.get("aerial"), 0.5)
    return PlayerAbilities(
        tech=technical,
        pass_skill=_finite(values.get("pass_skill"), technical),
        vision=vision,
        spatial=_finite(values.get("spatial"), vision),
        pace=_finite(values.get("pace"), 0.5),
        press=_finite(values.get("press"), 0.5),
        curve=_finite(values.get("curve"), 0.4),
        shot=_finite(values.get("shot"), technical * 0.9),
        power=_finite(values.get("power"), physical),
        aerial=aerial,
        heading=_finite(values.get("heading"), aerial),
        gk_reflex=_finite(
            values.get("gk_reflex"), goalkeeper if goalkeeper > 0 else 0.58,
        ),
        gk_aerial=_finite(
            values.get("gk_aerial"), goalkeeper * 0.95 if goalkeeper > 0 else 0.55,
        ),
    )


def _role_abilities(role, tier, mental, is_icon):
    base = tier * 0.65 + mental * 0.35 + (0.08 if is_icon else 0)
    if role == "GK":
        return PlayerAbilities(
            tech=base * 0.75, pass_skill=base * 0.7, vision=base * 0.8,
            spatial=base * 0.85, pace=base * 0.55, press=0.15, curve=0.2,
            shot=0.15, power=0.4, knuckle=0.1, aerial=base * 0.9,
            heading=base * 0.85, gk_reflex=base * 1.05, gk_aerial=base,
        )
    if role in {"CB", "LB", "RB"}:
        return PlayerAbilities(
            tech=base * 0.85, pass_skill=base * 0.8, vision=base * 0.75,
            spatial=base * 0.9, pace=base * 0.85, press=base * 0.95,
            curve=base * 0.55, shot=base * 0.45, power=base * 0.7,
            knuckle=base * 0.35, aerial=base * 0.95, heading=base * 0.9,
        )
    if role in {"LW", "RW", "ST"}:
        return PlayerAbilities(
            tech=base * 1.05, pass_skill=base * 0.9, vision=base * 0.95,
            spatial=base, pace=base * 1.1, press=base * 0.7,
            curve=base * 1.05, shot=base * 1.08, power=base * 0.95,
            knuckle=base * 0.75, aerial=base * 0.75, heading=base * 0.85,
        )
    return PlayerAbilities(
        tech=base, pass_skill=base * 1.05, vision=base * 1.05,
        spatial=base, pace=base, press=base * 0.85, curve=base * 0.85,
        shot=base * 0.8, power=base * 0.85, knuckle=base * 0.5,
        aerial=base * 0.7, heading=base * 0.65,
    )


def _emotion_logits(profile):
    emotions = profile.emotion_profile
    values = np.array([
        float(emotions.get("pride", 0.2)),
        float(emotions.get("anger", 0.1)),
        float(emotions.get("fear", 0.2)),
        float(emotions.get("determination", 0.4)),
    ]) + 1e-6
    return np.log(values / values.sum())


def _peer_matrix(players):
    matrix = np.zeros((len(players), len(players)), dtype=float)
    groups = {
        "GK": 0, "LB": 1, "CB": 1, "RB": 1, "DM": 2,
        "CM": 2, "LW": 3, "RW": 3, "ST": 3,
    }
    for row, first in enumerate(players):
        for column, second in enumerate(players):
            if row == column:
                continue
            first_group = groups.get(first.role, 2)
            second_group = groups.get(second.role, 2)
            matrix[row, column] = (
                0.08 * float(first_group == second_group)
                + 0.04 * float(abs(first_group - second_group) == 1)
                + 0.01
            )
    return matrix


def _finite(value, default):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)

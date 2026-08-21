"""Replayable season identities and fixture tactics for every club."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from src.product.match_plan import NATIVE_TACTIC, PLAYABLE_TACTICS
from src.simulation.squad_registry import roster_identity


CLUB_STRATEGY_SCHEMA_VERSION = 1
_TACTICS = (
    "balanced", "tiki_taka", "gegenpress", "counter_attack",
    "low_block_counter", "direct_vertical",
)
_STYLE_ABILITIES = {
    "balanced": (
        "tech", "pass_skill", "vision", "spatial", "pace", "press",
        "shot", "power", "aerial", "mental",
    ),
    "tiki_taka": ("tech", "pass_skill", "vision", "spatial"),
    "gegenpress": ("press", "pace", "mental", "power"),
    "counter_attack": ("pace", "vision", "shot", "mental"),
    "low_block_counter": ("press", "power", "aerial", "mental"),
    "direct_vertical": ("power", "aerial", "shot", "pace"),
}
_SNAPSHOT_FIELDS = {
    "schema_version", "season_id", "source_season_id",
    "source_archive_identity", "ecosystem_transition_identity", "profiles",
    "claim_boundary",
}
_PROFILE_FIELDS = {
    "schema_version", "team", "control", "roster_identity",
    "evidence_quality", "source_season_id", "previous_primary_tactic",
    "previous_position", "risk_posture", "style_scores", "primary_tactic",
    "home_tactic", "away_tactic", "recruitment_move_count", "reasons",
}


def strategy_identity(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ability_mean(roster: Mapping[str, Any], keys: Sequence[str]) -> float:
    values = []
    for player in roster.get("players") or []:
        if not isinstance(player, Mapping):
            continue
        abilities = player.get("abilities") or {}
        if not isinstance(abilities, Mapping):
            continue
        for key in keys:
            try:
                value = float(abilities.get(key, 0.5))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(min(1.0, max(0.0, value)))
    return sum(values) / len(values) if values else 0.5


def _position_for_team(
    source_archive: Mapping[str, Any] | None, team: str,
) -> tuple[int | None, int | None]:
    if source_archive is None:
        return None, None
    standings = source_archive.get("final_standings")
    rows = [
        row for row in standings or []
        if isinstance(row, Mapping) and row.get("team") == team
    ]
    if len(rows) != 1 or not isinstance(standings, list):
        return None, None
    position = rows[0].get("position")
    if isinstance(position, bool) or not isinstance(position, int):
        return None, None
    return position, len(standings)


def _risk_posture(position: int | None, league_size: int | None) -> str:
    if position is None or league_size is None:
        return "balanced"
    if position <= math.ceil(league_size / 3):
        return "proactive"
    if position > math.ceil(league_size * 2 / 3):
        return "conservative"
    return "balanced"


def build_season_club_strategies(
    *, season_id: str, teams: Sequence[str], manager_team: str | None,
    rosters: Mapping[str, Mapping[str, Any] | None],
    source_archive: Mapping[str, Any] | None = None,
    previous_snapshot: Mapping[str, Any] | None = None,
    ecosystem_transition: Mapping[str, Any] | None = None,
    recruitment_moves: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Build one deterministic strategy snapshot after the window closes."""
    clean_teams = sorted(str(team) for team in teams)
    if (
        not clean_teams or len(set(clean_teams)) != len(clean_teams)
        or any(not team.strip() for team in clean_teams)
    ):
        raise ValueError("club strategy teams are invalid")
    if manager_team is not None and manager_team not in clean_teams:
        raise ValueError("club strategy manager identity mismatch")
    source_season_id = (
        str(source_archive.get("season_id") or "")
        if source_archive is not None else None
    )
    previous_profiles = {
        str(profile.get("team")): profile
        for profile in (previous_snapshot or {}).get("profiles") or []
        if isinstance(profile, Mapping)
    }
    moves = dict(recruitment_moves or {})
    profiles = []
    for team in clean_teams:
        roster = rosters.get(team)
        previous = previous_profiles.get(team)
        previous_tactic = (
            str(previous.get("primary_tactic"))
            if previous is not None else None
        )
        position, league_size = _position_for_team(source_archive, team)
        posture = _risk_posture(position, league_size)
        scores = {
            tactic: _ability_mean(roster, keys) if roster is not None else 0.5
            for tactic, keys in _STYLE_ABILITIES.items()
        }
        if posture == "proactive":
            posture_adjustments = {
                "gegenpress": 0.030, "tiki_taka": 0.018,
                "direct_vertical": 0.010,
            }
        elif posture == "conservative":
            posture_adjustments = {
                "low_block_counter": 0.035, "counter_attack": 0.022,
            }
        else:
            posture_adjustments = {"balanced": 0.020}
        for tactic, adjustment in posture_adjustments.items():
            scores[tactic] += adjustment
        if previous_tactic in scores:
            scores[previous_tactic] += 0.018
        frozen_scores = {
            tactic: round(scores[tactic], 6) for tactic in _TACTICS
        }
        primary = sorted(
            _TACTICS, key=lambda tactic: (-frozen_scores[tactic], tactic),
        )[0]
        away = primary
        if posture == "conservative":
            away = sorted(
                ("low_block_counter", "counter_attack"),
                key=lambda tactic: (-frozen_scores[tactic], tactic),
            )[0]
        move_count = moves.get(team, 0)
        if (
            isinstance(move_count, bool) or not isinstance(move_count, int)
            or not 0 <= move_count <= 2
        ):
            raise ValueError("club strategy recruitment evidence is invalid")
        reasons = [f"roster_fit:{primary}", f"risk_posture:{posture}"]
        if previous_tactic is not None:
            reasons.append(f"identity_inertia:{previous_tactic}")
        if move_count:
            reasons.append(f"post_window_roster:{move_count}_moves")
        profiles.append({
            "schema_version": CLUB_STRATEGY_SCHEMA_VERSION,
            "team": team,
            "control": "manager" if team == manager_team else "ai_club",
            "roster_identity": roster_identity(roster) if roster is not None else None,
            "evidence_quality": "roster_derived" if roster is not None else "team_baseline",
            "source_season_id": source_season_id,
            "previous_primary_tactic": previous_tactic,
            "previous_position": position,
            "risk_posture": posture,
            "style_scores": frozen_scores,
            "primary_tactic": primary,
            "home_tactic": primary,
            "away_tactic": away,
            "recruitment_move_count": move_count,
            "reasons": reasons,
        })
    snapshot = {
        "schema_version": CLUB_STRATEGY_SCHEMA_VERSION,
        "season_id": season_id,
        "source_season_id": source_season_id,
        "source_archive_identity": (
            strategy_identity(source_archive) if source_archive is not None else None
        ),
        "ecosystem_transition_identity": (
            strategy_identity(ecosystem_transition)
            if ecosystem_transition is not None else None
        ),
        "profiles": profiles,
        "claim_boundary": (
            "deterministic season gameplay policy from bounded roster and "
            "simulated-standing evidence; not learned or real-world coaching advice"
        ),
    }
    validate_club_strategy_snapshot(snapshot, teams=clean_teams)
    return snapshot


def validate_club_strategy_snapshot(
    snapshot: Mapping[str, Any], *, teams: Sequence[str] | None = None,
) -> None:
    if not isinstance(snapshot, Mapping) or set(snapshot) != _SNAPSHOT_FIELDS:
        raise ValueError("invalid club strategy snapshot")
    if snapshot.get("schema_version") != 1 or not str(snapshot.get("season_id") or ""):
        raise ValueError("invalid club strategy season identity")
    for field in ("source_archive_identity", "ecosystem_transition_identity"):
        value = snapshot.get(field)
        if value is not None and (not isinstance(value, str) or len(value) != 64):
            raise ValueError("invalid club strategy source identity")
    profiles = snapshot.get("profiles")
    if not isinstance(profiles, list):
        raise ValueError("invalid club strategy profiles")
    profile_teams = []
    for profile in profiles:
        if not isinstance(profile, Mapping) or set(profile) != _PROFILE_FIELDS:
            raise ValueError("invalid club strategy profile")
        team = profile.get("team")
        scores = profile.get("style_scores")
        if (
            profile.get("schema_version") != 1
            or not isinstance(team, str) or not team.strip()
            or profile.get("control") not in {"manager", "ai_club"}
            or profile.get("evidence_quality") not in {"roster_derived", "team_baseline"}
            or profile.get("risk_posture") not in {"proactive", "balanced", "conservative"}
            or profile.get("primary_tactic") not in _TACTICS
            or profile.get("home_tactic") not in _TACTICS
            or profile.get("away_tactic") not in _TACTICS
            or not isinstance(scores, Mapping) or set(scores) != set(_TACTICS)
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) or not 0 <= float(value) <= 1.2
                for value in scores.values()
            )
            or not isinstance(profile.get("reasons"), list)
            or not all(isinstance(reason, str) for reason in profile["reasons"])
        ):
            raise ValueError("invalid club strategy profile evidence")
        profile_teams.append(team)
    if profile_teams != sorted(profile_teams) or len(set(profile_teams)) != len(profile_teams):
        raise ValueError("invalid club strategy profile ordering")
    if teams is not None and profile_teams != sorted(str(team) for team in teams):
        raise ValueError("club strategy competition coverage mismatch")


def resolve_fixture_club_strategy(
    snapshot: Mapping[str, Any], *, home: str, away: str,
    manager_decision: Mapping[str, Any] | None = None,
    opponent_preparation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve actual fixture tactics and immutable attribution evidence."""
    validate_club_strategy_snapshot(snapshot)
    profiles = {profile["team"]: profile for profile in snapshot["profiles"]}
    if home not in profiles or away not in profiles or home == away:
        raise ValueError("club strategy fixture identity mismatch")
    home_tactic = str(profiles[home]["home_tactic"])
    away_tactic = str(profiles[away]["away_tactic"])
    manager_team = None
    adaptation = False
    if manager_decision is not None:
        manager_team = str(manager_decision.get("team") or "")
        manager_tactic = str(manager_decision.get("tactic") or "")
        if manager_team not in {home, away} or manager_tactic not in PLAYABLE_TACTICS:
            raise ValueError("club strategy manager override mismatch")
        if not isinstance(opponent_preparation, Mapping):
            raise ValueError("club strategy managed fixture lacks preparation")
        selected = str(opponent_preparation.get("selected_tactic") or NATIVE_TACTIC)
        if selected not in PLAYABLE_TACTICS:
            raise ValueError("club strategy opponent tactic is invalid")
        opponent = away if manager_team == home else home
        opponent_identity = (
            str(profiles[opponent]["away_tactic"])
            if opponent == away else str(profiles[opponent]["home_tactic"])
        )
        opponent_tactic = opponent_identity if selected == NATIVE_TACTIC else selected
        adaptation = selected != NATIVE_TACTIC
        if manager_team == home:
            home_tactic, away_tactic = manager_tactic, opponent_tactic
        else:
            away_tactic, home_tactic = manager_tactic, opponent_tactic
    return {
        "schema_version": CLUB_STRATEGY_SCHEMA_VERSION,
        "season_id": snapshot["season_id"],
        "snapshot_identity": strategy_identity(snapshot),
        "home": {
            "team": home, "profile_identity": strategy_identity(profiles[home]),
            "season_identity_tactic": profiles[home]["home_tactic"],
            "applied_tactic": home_tactic,
        },
        "away": {
            "team": away, "profile_identity": strategy_identity(profiles[away]),
            "season_identity_tactic": profiles[away]["away_tactic"],
            "applied_tactic": away_tactic,
        },
        "manager_override_team": manager_team,
        "opponent_adaptation_applied": adaptation,
        "home_tactic": home_tactic,
        "away_tactic": away_tactic,
        "claim_boundary": "frozen season gameplay tactics; no causal claim",
    }


__all__ = [
    "build_season_club_strategies", "resolve_fixture_club_strategy",
    "strategy_identity", "validate_club_strategy_snapshot",
]

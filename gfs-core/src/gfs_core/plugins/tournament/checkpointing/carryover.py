from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace

from gfs_core.contracts.society import TeamProfile


@dataclass
class PlayerCarryover:
    player_id: str
    name: str = ""
    matches_played: int = 0
    minutes_ema: float = 0.0
    goals: int = 0
    assists: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    suspension_matches_left: int = 0
    injury_matches_left: int = 0
    injury_severity: float = 0.0
    form_ema: float = 0.55
    media_sentiment: float = 0.0

    @classmethod
    def from_dict(cls, data):
        return cls(
            player_id=str(data["player_id"]),
            name=str(data.get("name", "")),
            matches_played=int(data.get("matches_played", 0)),
            minutes_ema=float(data.get("minutes_ema", 0.0)),
            goals=int(data.get("goals", 0)),
            assists=int(data.get("assists", 0)),
            yellow_cards=int(data.get("yellow_cards", 0)),
            red_cards=int(data.get("red_cards", 0)),
            suspension_matches_left=int(data.get("suspension_matches_left", 0)),
            injury_matches_left=int(data.get("injury_matches_left", 0)),
            injury_severity=float(data.get("injury_severity", 0.0)),
            form_ema=float(data.get("form_ema", 0.55)),
            media_sentiment=float(data.get("media_sentiment", 0.0)),
        )


@dataclass
class TeamCarryover:
    team_id: str
    players: dict[str, PlayerCarryover] = field(default_factory=dict)
    fatigue_ema: float = 0.0
    morale_ema: float = 0.55
    media_pressure: float = 0.0
    settled_match_ids: list[str] = field(default_factory=list)
    recovery_ids: list[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "team_id": self.team_id,
            "players": {key: asdict(value) for key, value in self.players.items()},
            "fatigue_ema": self.fatigue_ema,
            "morale_ema": self.morale_ema,
            "media_pressure": self.media_pressure,
            "settled_match_ids": self.settled_match_ids[-256:],
            "recovery_ids": self.recovery_ids[-256:],
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            team_id=str(data["team_id"]),
            players={
                key: PlayerCarryover.from_dict(value)
                for key, value in (data.get("players") or {}).items()
            },
            fatigue_ema=float(data.get("fatigue_ema", 0.0)),
            morale_ema=float(data.get("morale_ema", 0.55)),
            media_pressure=float(data.get("media_pressure", 0.0)),
            settled_match_ids=list(data.get("settled_match_ids", []))[-256:],
            recovery_ids=list(data.get("recovery_ids", []))[-256:],
        )


class CrossMatchService:
    def __init__(self, *, fatigue_status_penalty: float = 12.0, morale_status_scale: float = 6.0, memory=None, memory_status_scale: float = 1.0) -> None:
        if fatigue_status_penalty < 0 or morale_status_scale < 0 or memory_status_scale < 0:
            raise ValueError("carryover status coefficients cannot be negative")
        self.fatigue_status_penalty = float(fatigue_status_penalty)
        self.morale_status_scale = float(morale_status_scale)
        self.memory_status_scale = float(memory_status_scale)
        self.memory = memory

    def settle_outcome(self, carryover: TeamCarryover, *, won: bool, lost: bool, match_id: str, opponent: str = "", score=(0, 0), xg=(0.0, 0.0), engine: str = "unknown", emotion=None, coach_stress: float | None = None) -> None:
        if match_id in carryover.settled_match_ids:
            return
        carryover.fatigue_ema = _bounded(carryover.fatigue_ema + 0.08)
        carryover.morale_ema = _bounded(
            carryover.morale_ema + (0.06 if won else -0.04 if lost else 0.01),
            lower=0.05,
            upper=0.98,
        )
        carryover.media_pressure = _bounded(carryover.media_pressure + (0.02 if lost else 0.005))
        carryover.settled_match_ids.append(match_id)
        carryover.settled_match_ids = carryover.settled_match_ids[-256:]
        if self.memory is not None:
            self.memory.record_match_result(
                team=carryover.team_id,
                opponent=opponent,
                match_id=match_id,
                match_index=len(carryover.settled_match_ids),
                won=won,
                lost=lost,
                score=score,
                xg=xg,
                engine=engine,
                emotion=emotion,
                coach_stress=coach_stress,
            )

    def settle(self, carryover: TeamCarryover, summary, match_id: str) -> None:
        if match_id in carryover.settled_match_ids:
            return
        home_win = summary.goals_micro_home > summary.goals_micro_away
        away_win = summary.goals_micro_away > summary.goals_micro_home
        is_home = summary.home_team == carryover.team_id
        won, lost = (home_win, away_win) if is_home else (away_win, home_win)
        self.settle_outcome(
            carryover,
            won=won,
            lost=lost,
            match_id=match_id,
            opponent=summary.away_team if is_home else summary.home_team,
            score=(
                summary.goals_micro_home if is_home else summary.goals_micro_away,
                summary.goals_micro_away if is_home else summary.goals_micro_home,
            ),
            xg=(
                summary.micro_xg_home if is_home else summary.micro_xg_away,
                summary.micro_xg_away if is_home else summary.micro_xg_home,
            ),
            engine="micro",
            emotion=summary.home_emotion_mean if is_home else summary.away_emotion_mean,
            coach_stress=summary.home_coach_stress if is_home else summary.away_coach_stress,
        )
        stats = summary.player_stats.get(carryover.team_id, {})
        for player_id, row in stats.items():
            player = carryover.players.setdefault(
                player_id, PlayerCarryover(player_id)
            )
            player.matches_played += 1
            minutes = float(row.get("minutes", 0.0))
            player.minutes_ema = (
                minutes if player.matches_played == 1
                else 0.8 * player.minutes_ema + 0.2 * minutes
            )
            player.goals += int(row.get("goals", 0))
            player.assists += int(row.get("assists", 0))
            player.yellow_cards += int(row.get("yellow_cards", 0))
            player.red_cards += int(row.get("red_cards", 0))
            if int(row.get("red_cards", 0)):
                player.suspension_matches_left = max(
                    1, player.suspension_matches_left,
                )
            player.form_ema = _bounded(
                0.8 * player.form_ema + 0.2 * (0.65 if won else 0.40 if lost else 0.52),
                lower=0.05,
                upper=0.98,
            )
        if self.memory is not None:
            self.memory.record_match_summary(
                summary, match_id, len(carryover.settled_match_ids),
            )

    def recover(self, carryover: TeamCarryover, recovery_id: str, rest_units: float = 1.0) -> None:
        if recovery_id in carryover.recovery_ids:
            return
        if not math.isfinite(rest_units) or not 0.0 <= rest_units <= 4.0:
            raise ValueError("rest_units must be finite and between 0 and 4")
        carryover.fatigue_ema *= math.exp(-0.42 * rest_units)
        carryover.media_pressure *= math.exp(-0.18 * rest_units)
        for player in carryover.players.values():
            player.suspension_matches_left = max(
                0, player.suspension_matches_left - int(rest_units),
            )
            player.injury_matches_left = max(
                0, player.injury_matches_left - int(rest_units),
            )
            player.injury_severity *= math.exp(-0.25 * rest_units)
        carryover.recovery_ids.append(recovery_id)
        carryover.recovery_ids = carryover.recovery_ids[-256:]

    def apply_to_profile(self, profile: TeamProfile, carryover: TeamCarryover) -> TeamProfile:
        effects = self.memory.effects(carryover.team_id) if self.memory is not None else {
            "valence": 0.0, "pressure": 0.0, "emotion": {},
        }
        psychology = dict(profile.psychology_profile)
        psychology["confidence"] = carryover.morale_ema
        psychology["memory_confidence"] = _bounded(0.5 + 0.5 * effects["valence"])
        emotion = dict(profile.emotion_profile)
        for key in set(emotion) | set(effects["emotion"]):
            emotion[key] = (
                0.7 * float(emotion.get(key, 0.0))
                + 0.3 * float(effects["emotion"].get(key, 0.0))
            )
        effective_status = (
            profile.status_score
            - self.fatigue_status_penalty * carryover.fatigue_ema
            + self.morale_status_scale * (carryover.morale_ema - 0.55)
            + self.memory_status_scale * effects["valence"]
        )
        return replace(
            profile,
            status_score=float(min(100.0, max(1.0, effective_status))),
            fatigue=carryover.fatigue_ema,
            emotion_profile=emotion,
            conflict_heat=_bounded(profile.conflict_heat + 0.1 * effects["pressure"]),
            media_exposure=_bounded(profile.media_exposure + carryover.media_pressure),
            psychology_profile=psychology,
        )


def _bounded(value: float, *, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(upper, max(lower, float(value)))

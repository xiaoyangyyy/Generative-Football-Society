"""
Cross-match persistence: injuries, cards, form, media → next-match baselines.

Stored on SocietyAgent.squad_carryover and optional JSON under data/persistence/.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from src.data_engine.entity_dynamics import PLAYER_CONDITION_KEYS

TEAM_STATE_KEYS = ("attack", "defense", "press", "morale_field", "institutional_pressure")
from src.match_engine.math_utils import sigmoid, tanh_clip

if TYPE_CHECKING:
    from src.simulation.agent import SocietyAgent


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
    condition_delta: Dict[str, float] = field(default_factory=dict)
    last_rating: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PlayerCarryover":
        return cls(
            player_id=str(d["player_id"]),
            name=str(d.get("name", "")),
            matches_played=int(d.get("matches_played", 0)),
            minutes_ema=float(d.get("minutes_ema", 0)),
            goals=int(d.get("goals", 0)),
            assists=int(d.get("assists", 0)),
            yellow_cards=int(d.get("yellow_cards", 0)),
            red_cards=int(d.get("red_cards", 0)),
            suspension_matches_left=int(d.get("suspension_matches_left", 0)),
            injury_matches_left=int(d.get("injury_matches_left", 0)),
            injury_severity=float(d.get("injury_severity", 0)),
            form_ema=float(d.get("form_ema", 0.55)),
            media_sentiment=float(d.get("media_sentiment", 0)),
            condition_delta={k: float(v) for k, v in (d.get("condition_delta") or {}).items()},
            last_rating=float(d.get("last_rating", 0.5)),
        )


@dataclass
class TeamSquadCarryover:
    team_id: str
    players: Dict[str, PlayerCarryover] = field(default_factory=dict)
    team_media_pressure: float = 0.0
    squad_morale_ema: float = 0.55
    team_dynamics_delta: Dict[str, float] = field(default_factory=dict)
    last_match_stage: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "team_id": self.team_id,
            "players": {k: v.to_dict() for k, v in self.players.items()},
            "team_media_pressure": self.team_media_pressure,
            "squad_morale_ema": self.squad_morale_ema,
            "team_dynamics_delta": self.team_dynamics_delta,
            "last_match_stage": self.last_match_stage,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TeamSquadCarryover":
        players = {k: PlayerCarryover.from_dict(v) for k, v in (d.get("players") or {}).items()}
        return cls(
            team_id=str(d.get("team_id", "")),
            players=players,
            team_media_pressure=float(d.get("team_media_pressure", 0)),
            squad_morale_ema=float(d.get("squad_morale_ema", 0.55)),
            team_dynamics_delta={k: float(v) for k, v in (d.get("team_dynamics_delta") or {}).items()},
            last_match_stage=str(d.get("last_match_stage", "")),
        )


def _rating_from_performance(
    *,
    goals: int,
    xg_share: float,
    pass_cmp: float,
    won: bool,
    cards: int,
) -> float:
    base = 0.48 + 0.12 * goals + 0.35 * xg_share + 0.18 * pass_cmp
    base += 0.08 if won else -0.05
    base -= 0.15 * cards
    return float(sigmoid(3.2 * (base - 0.52)))


def ensure_team_carryover(agent: "SocietyAgent") -> TeamSquadCarryover:
    if not hasattr(agent, "squad_carryover") or agent.squad_carryover is None:
        agent.squad_carryover = TeamSquadCarryover(team_id=agent.team_name)
    return agent.squad_carryover


def sync_carryover_from_roster(agent: "SocietyAgent", roster: Dict[str, Any]) -> None:
    """Seed player IDs from roster JSON."""
    carry = ensure_team_carryover(agent)
    for p in roster.get("players", []):
        pid = str(p.get("player_id", ""))
        if not pid:
            continue
        if pid not in carry.players:
            carry.players[pid] = PlayerCarryover(player_id=pid, name=str(p.get("name", "")))


def apply_carryover_to_agent(agent: "SocietyAgent") -> None:
    """Team-level fatigue / morale / dynamics from squad carryover."""
    carry = ensure_team_carryover(agent)
    n_inj = sum(1 for p in carry.players.values() if p.injury_matches_left > 0)
    n_susp = sum(1 for p in carry.players.values() if p.suspension_matches_left > 0)
    squad_size = max(1, len(carry.players))
    agent.fatigue = float(
        tanh_clip(agent.fatigue + 0.08 * (n_inj / squad_size) + 0.05 * carry.team_media_pressure)
    )
    agent.injury_load = float(tanh_clip(0.35 * n_inj / squad_size + 0.15 * carry.team_media_pressure))
    agent.readiness = float(sigmoid(2.2 * (carry.squad_morale_ema - 0.45) - 0.4 * agent.fatigue - 0.35 * agent.injury_load))
    if hasattr(agent, "team_dynamics") and agent.team_dynamics:
        td = dict(agent.team_dynamics)
        for k in TEAM_STATE_KEYS:
            td[k] = float(tanh_clip(td.get(k, 0) + carry.team_dynamics_delta.get(k, 0)))
        agent.team_dynamics = td
    agent.semantic_memory["squad_morale_ema"] = carry.squad_morale_ema
    agent.semantic_memory["unavailable_players"] = n_inj + n_susp


def apply_carryover_to_roster_dict(roster: Dict[str, Any]) -> Dict[str, Any]:
    """Mutate roster player condition from per-player carryover (pre-match)."""
    team_id = roster.get("team_id", "")
    out = dict(roster)
    players_out = []
    for p in roster.get("players", []):
        rec = dict(p)
        pid = str(rec.get("player_id", ""))
        carry = _get_carryover_player(team_id, pid, rec.get("name", ""))
        if carry.suspension_matches_left > 0:
            rec["squad_role"] = "suspended"
            rec["availability"] = 0.0
        elif carry.injury_matches_left > 0:
            rec["availability"] = float(sigmoid(-1.5 + 0.4 * carry.injury_matches_left))
        else:
            rec["availability"] = 1.0
        cond = dict(rec.get("condition", {}))
        for k in PLAYER_CONDITION_KEYS:
            base = float(cond.get(k, 0.5))
            delta = float(carry.condition_delta.get(k, 0))
            inj_pen = 0.12 * carry.injury_severity if carry.injury_matches_left else 0
            form_boost = 0.08 * (carry.form_ema - 0.5)
            cond[k] = float(sigmoid(math.log(base / (1 - base + 1e-6) + 1e-6) + delta + form_boost - inj_pen))
        rec["condition"] = cond
        rec["form_ema"] = carry.form_ema
        rec["media_sentiment"] = carry.media_sentiment
        players_out.append(rec)
    out["players"] = players_out
    return out


def _get_carryover_player(team_id: str, pid: str, name: str) -> PlayerCarryover:
    # module-level cache via agent not available here; caller should use TeamSquadCarryover on agent
    return PlayerCarryover(player_id=pid, name=name)


def apply_carryover_to_roster_for_agent(agent: "SocietyAgent", roster: Dict[str, Any]) -> Dict[str, Any]:
    carry = ensure_team_carryover(agent)
    sync_carryover_from_roster(agent, roster)
    players_out = []
    for p in roster.get("players", []):
        rec = dict(p)
        pid = str(rec.get("player_id", ""))
        pc = carry.players.get(pid) or PlayerCarryover(player_id=pid, name=str(rec.get("name", "")))
        if pc.suspension_matches_left > 0:
            rec["squad_role"] = "suspended"
            rec["availability"] = 0.0
        elif pc.injury_matches_left > 0:
            rec["availability"] = float(sigmoid(-1.2 - 0.5 * pc.injury_severity))
        else:
            rec["availability"] = 1.0
        cond = dict(rec.get("condition", {}))
        for k in PLAYER_CONDITION_KEYS:
            v = float(cond.get(k, 0.5))
            v = float(tanh_clip(v + pc.condition_delta.get(k, 0) + 0.06 * (pc.form_ema - 0.5)))
            if pc.injury_matches_left > 0:
                v = float(v * (1.0 - 0.35 * pc.injury_severity))
            cond[k] = v
        rec["condition"] = cond
        rec["form_ema"] = pc.form_ema
        rec["media_sentiment"] = pc.media_sentiment
        players_out.append(rec)
    out = dict(roster)
    out["players"] = players_out
    return out


def ingest_match_result(
    agent: "SocietyAgent",
    *,
    roster: Optional[Dict[str, Any]],
    result: str,
    score_diff: int,
    xg_for: float,
    xg_against: float,
    prof_score: float,
    social_chaos: float,
    stage_name: str,
    micro_player_stats: Optional[Dict[str, Dict[str, float]]] = None,
    cards: Optional[Dict[str, int]] = None,
    new_injuries: Optional[List[str]] = None,
) -> None:
    """Update carryover after match; feeds next fixture."""
    carry = ensure_team_carryover(agent)
    if roster:
        sync_carryover_from_roster(agent, roster)
    won = result == "win"
    form_team = 0.55 + (0.12 if won else (-0.08 if result == "loss" else 0))
    form_team += 0.06 * tanh_clip(score_diff / 2.0) + 0.04 * tanh_clip(xg_for - xg_against)
    carry.squad_morale_ema = float(0.72 * carry.squad_morale_ema + 0.28 * form_team)
    carry.team_media_pressure = float(
        tanh_clip(0.65 * carry.team_media_pressure + 0.25 * abs(social_chaos) / 5.0 + 0.1 * abs(prof_score) / 2.0)
    )
    carry.last_match_stage = stage_name
    carry.team_dynamics_delta = {
        "attack": float(0.15 * tanh_clip(xg_for - 1.0)),
        "defense": float(-0.1 * tanh_clip(xg_against - 1.0)),
        "press": float(0.05 * (1 if won else -0.3)),
        "morale_field": float(0.2 * (carry.squad_morale_ema - 0.5)),
        "institutional_pressure": float(0.05 * carry.team_media_pressure),
    }
    cards = cards or {}
    new_injuries = new_injuries or []
    micro_player_stats = micro_player_stats or {}

    for pid, pc in list(carry.players.items()):
        stats = micro_player_stats.get(pid, {})
        yc = int(cards.get(pid, 0)) + int(stats.get("yellow_cards", 0))
        rc = int(stats.get("red_cards", 0))
        if yc:
            pc.yellow_cards += yc
            if pc.yellow_cards >= 2:
                pc.suspension_matches_left = max(pc.suspension_matches_left, 1)
                pc.yellow_cards = 0
        if rc:
            pc.red_cards += rc
            pc.suspension_matches_left = max(pc.suspension_matches_left, 1)
        if pid in new_injuries:
            pc.injury_matches_left = max(pc.injury_matches_left, int(1 + 2 * agent.injury_load))
            pc.injury_severity = float(tanh_clip(0.4 + agent.injury_load))
        pc.matches_played += 1
        pc.minutes_ema = 0.85 * pc.minutes_ema + 0.15 * float(stats.get("minutes", 70))
        pc.goals += int(stats.get("goals", 0))
        pc.assists += int(stats.get("assists", 0))
        if (
            float(stats.get("minutes", 0)) > 75
            and float(agent.injury_load) > 0.55
            and np.random.random() < 0.06 * float(agent.injury_load)
        ):
            if pid not in new_injuries:
                new_injuries.append(pid)
                pc.injury_matches_left = max(pc.injury_matches_left, int(1 + 2 * agent.injury_load))
                pc.injury_severity = float(tanh_clip(0.4 + agent.injury_load))
        rating = _rating_from_performance(
            goals=int(stats.get("goals", 0)),
            xg_share=float(stats.get("xg_share", 0.1)),
            pass_cmp=float(stats.get("pass_cmp", 0.5)),
            won=won,
            cards=yc,
        )
        pc.last_rating = rating
        pc.form_ema = float(0.7 * pc.form_ema + 0.3 * rating)
        pc.media_sentiment = float(
            tanh_clip(0.8 * pc.media_sentiment + 0.15 * prof_score / 2.0 + 0.05 * social_chaos / 5.0)
        )
        for k in PLAYER_CONDITION_KEYS:
            drift = 0.04 * (rating - 0.5)
            if k == "composure":
                drift += 0.03 * pc.media_sentiment
            pc.condition_delta[k] = float(tanh_clip(pc.condition_delta.get(k, 0) + drift))
        if pc.suspension_matches_left > 0:
            pc.suspension_matches_left = max(0, pc.suspension_matches_left - 1)
        if pc.injury_matches_left > 0:
            pc.injury_matches_left = max(0, pc.injury_matches_left - 1)
            pc.injury_severity *= 0.65

    agent.squad_carryover = carry


def load_persistence(base_dir: str, agents: Dict[str, "SocietyAgent"]) -> None:
    path = Path(base_dir) / "data" / "persistence" / "squad_carryover.json"
    if not path.exists():
        return
    raw = json.loads(path.read_text(encoding="utf-8"))
    for team, payload in raw.items():
        if team in agents:
            agents[team].squad_carryover = TeamSquadCarryover.from_dict(payload)


def save_persistence(base_dir: str, agents: Dict[str, "SocietyAgent"]) -> None:
    path = Path(base_dir) / "data" / "persistence" / "squad_carryover.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = {}
    for team, agent in agents.items():
        if hasattr(agent, "squad_carryover") and agent.squad_carryover:
            raw[team] = agent.squad_carryover.to_dict()
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

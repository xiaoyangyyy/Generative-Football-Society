"""15-minute meso windows → MicroAppraisalPacket for GFS recursive_update."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from src.match_engine.state import MatchAffectiveState, TeamAffectiveState


@dataclass
class MicroAppraisalPacket:
    team: str
    phase: str
    xg_for: float = 0.0
    xg_against: float = 0.0
    phi_integral: float = 0.0  # placeholder until spatial phase
    foul_controversy: float = 0.0
    icon_morale_shock: float = 0.0
    crowd_psi_mean: float = 0.0
    tactical_drift: float = 0.0
    emotion_mean: Dict[str, float] = field(default_factory=dict)

    def to_appraisal_event(self, stage_pressure: float = 0.3) -> Dict[str, Any]:
        """Map into SocietyAgent._appraise_event compatible keys."""
        xg_diff = self.xg_for - self.xg_against
        return {
            "score_diff": float(np.tanh(xg_diff * 0.8)),
            "result": "win" if xg_diff > 0.15 else ("loss" if xg_diff < -0.15 else "draw"),
            "stage_pressure": stage_pressure,
            "referee_controversy": float(np.clip(self.foul_controversy, 0.0, 1.0)),
            "upset_factor": 0.0,
            "social_chaos": float(np.clip(abs(self.crowd_psi_mean) * 0.5, 0.0, 3.0)),
        }


class MesoAggregator:
    def __init__(self, window_seconds: float = 15.0 * 60.0):
        self.window = float(window_seconds)
        self._windows: Dict[str, List[Dict[str, Any]]] = {}

    def reset(self):
        self._windows = {}

    def record_tick(
        self,
        state: MatchAffectiveState,
        *,
        team_id: str,
        xg_for: float,
        xg_against: float,
        controversy: float,
        psi: float,
        emotion: Dict[str, float],
        tactical: Dict[str, float],
        tactical_base: Dict[str, float],
        icon_shock: float,
    ) -> None:
        phase_idx = int(state.clock_seconds // self.window)
        phase = f"{int(phase_idx * self.window // 60)}-{int((phase_idx + 1) * self.window // 60)}"
        drift = sum(abs(tactical.get(k, 0.5) - tactical_base.get(k, 0.5)) for k in tactical) / max(1, len(tactical))
        row = {
            "phase": phase,
            "xg_for": xg_for,
            "xg_against": xg_against,
            "controversy": controversy,
            "psi": psi,
            "emotion": dict(emotion),
            "icon_shock": icon_shock,
            "tactical_drift": drift,
        }
        self._windows.setdefault(team_id, []).append(row)

    def flush_packets(self, home_id: str, away_id: str) -> List[MicroAppraisalPacket]:
        packets = []
        for tid in (home_id, away_id):
            rows = self._windows.get(tid, [])
            if not rows:
                continue
            # aggregate last window per phase label
            by_phase: Dict[str, List[Dict]] = {}
            for r in rows:
                by_phase.setdefault(r["phase"], []).append(r)
            for phase, chunk in by_phase.items():
                xgf = float(np.mean([c["xg_for"] for c in chunk]))
                xga = float(np.mean([c["xg_against"] for c in chunk]))
                packets.append(
                    MicroAppraisalPacket(
                        team=tid,
                        phase=phase,
                        xg_for=xgf,
                        xg_against=xga,
                        foul_controversy=float(np.mean([c["controversy"] for c in chunk])),
                        icon_morale_shock=float(np.mean([c["icon_shock"] for c in chunk])),
                        crowd_psi_mean=float(np.mean([c["psi"] for c in chunk])),
                        tactical_drift=float(np.mean([c["tactical_drift"] for c in chunk])),
                        emotion_mean=_mean_emotion([c["emotion"] for c in chunk]),
                    )
                )
        return packets


def _mean_emotion(emotions: List[Dict[str, float]]) -> Dict[str, float]:
    if not emotions:
        return {}
    keys = emotions[0].keys()
    return {k: float(np.mean([e.get(k, 0.0) for e in emotions])) for k in keys}


def icon_emotion_shock(team: TeamAffectiveState) -> float:
    if not team.icon_player_id:
        return 0.0
    for p in team.players:
        if p.player_id == team.icon_player_id:
            e = p.emotion_profile()
            return float(e.get("pride", 0) - e.get("fear", 0) + 0.5 * e.get("anger", 0))
    return 0.0

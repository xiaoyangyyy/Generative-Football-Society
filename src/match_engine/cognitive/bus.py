"""Match cognitive event bus: salience → queued triggers."""

from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

from src.match_engine.cognitive.config import CognitiveMatchConfig
from src.match_engine.cognitive.entity_registry import (
    build_facts_ledger,
    half_time_triggers,
    triggers_for_micro_event,
    xg_swing_triggers,
)
from src.match_engine.cognitive.events import CognitiveTriggerEvent
from src.match_engine.cognitive.routing import TierBudget
from src.match_engine.cognitive.salience import (
    base_salience_from_micro,
    crowd_surge_salience,
    half_time_salience,
    should_fire,
)
from src.match_engine.micro_events import MicroEvent

if TYPE_CHECKING:
    from src.match_engine.state import MatchAffectiveState


class MatchCognitiveBus:
    def __init__(self, cfg: CognitiveMatchConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng
        self.last_fire: Dict[str, float] = {}
        self.budget = TierBudget(cfg)
        self.pending: List[CognitiveTriggerEvent] = []
        self.fired_log: List[CognitiveTriggerEvent] = []
        self._prev_psi = 0.0
        self._half_time_fired = False
        self._second_half_fired = False

    def ingest_micro_event(self, ev: MicroEvent, state: "MatchAffectiveState") -> None:
        if not self.cfg.enabled:
            return
        sal = base_salience_from_micro(ev, state)
        facts = build_facts_ledger(state, ev.t_sec)
        candidates = triggers_for_micro_event(ev, state, sal, facts)
        for trig in candidates:
            self._maybe_queue(trig)

    def ingest_clock(
        self,
        t_sec: float,
        state: "MatchAffectiveState",
        *,
        xg_swing_home: float = 0.0,
    ) -> None:
        if not self.cfg.enabled:
            return
        facts = build_facts_ledger(state, t_sec)

        ht_sal = half_time_salience(t_sec, self.cfg)
        if ht_sal > 0:
            if abs(t_sec - self.cfg.half_time_sec) < self.cfg.half_time_window_sec and not self._half_time_fired:
                for trig in half_time_triggers(t_sec, state, ht_sal, facts):
                    self._maybe_queue(trig)
                self._half_time_fired = True
            if (
                abs(t_sec - self.cfg.second_half_sec) < self.cfg.half_time_window_sec
                and not self._second_half_fired
            ):
                for trig in half_time_triggers(t_sec, state, ht_sal * 0.9, facts):
                    self._maybe_queue(trig)
                self._second_half_fired = True

        c_sal = crowd_surge_salience(state, self._prev_psi)
        if c_sal > 0.5:
            from src.match_engine.cognitive.entity_registry import crowd_entity_id
            from src.match_engine.cognitive.events import ENTITY_TIER_CROWD

            self._maybe_queue(
                CognitiveTriggerEvent(
                    t_sec,
                    "crowd_surge",
                    ENTITY_TIER_CROWD,
                    crowd_entity_id(),
                    salience=c_sal,
                    facts={**facts, "event": "psi_surge"},
                )
            )

        for trig in xg_swing_triggers(t_sec, state, xg_swing_home, facts):
            self._maybe_queue(trig)

        self._prev_psi = float(state.crowd.psi)

    def ingest_substitution(
        self,
        t_sec: float,
        state: "MatchAffectiveState",
        team_id: str,
        off_id: str,
        on_id: str,
    ) -> None:
        if not self.cfg.enabled:
            return
        from src.match_engine.cognitive.entity_registry import players_tier_b_for_sub

        facts = build_facts_ledger(state, t_sec)
        for trig in players_tier_b_for_sub(state, off_id, on_id, team_id, t_sec, facts):
            self._maybe_queue(trig)

    def _maybe_queue(self, trig: CognitiveTriggerEvent) -> None:
        if not should_fire(
            trig.salience,
            trig.entity_id,
            trig.entity_tier,
            trig.t_sec,
            self.last_fire,
            self.cfg,
            self.rng,
        ):
            return
        if not self.budget.admit(trig):
            return
        self.pending.append(trig)
        self.last_fire[trig.entity_id] = trig.t_sec
        self.fired_log.append(trig)

    def drain_pending(self) -> List[CognitiveTriggerEvent]:
        out = list(self.pending)
        self.pending.clear()
        return out

    def tier_usage(self) -> Dict[str, int]:
        return self.budget.summary()

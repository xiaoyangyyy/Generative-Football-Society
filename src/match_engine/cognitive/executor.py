"""Execute System 2 LLM calls (or rule fallback) and apply plans."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from src.match_engine.cognitive.apply import apply_cognitive_plan
from src.match_engine.cognitive.config import CognitiveMatchConfig
from src.match_engine.cognitive.events import (
    ENTITY_TIER_ASSISTANT,
    ENTITY_TIER_COACH,
    ENTITY_TIER_CROWD,
    ENTITY_TIER_PLAYER,
    ENTITY_TIER_REFEREE,
    CognitivePlanRecord,
    CognitiveTriggerEvent,
)
from src.match_engine.cognitive.schemas import (
    validate_assistant_plan,
    validate_coach_plan,
    validate_crowd_plan,
    validate_player_plan,
    validate_referee_plan,
)

if TYPE_CHECKING:
    from src.match_engine.state import MatchAffectiveState
    from src.simulation.agent import SocietyAgent
    from src.simulation.llm_engine import SimulationLLM


def _cache_key(trig: CognitiveTriggerEvent) -> str:
    blob = json.dumps(trig.to_dict(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def _rule_fallback_plan(trig: CognitiveTriggerEvent) -> Dict[str, Any]:
    """Deterministic plan when LLM unavailable (tests / offline)."""
    kind = trig.kind
    if trig.entity_tier == ENTITY_TIER_COACH:
        press = 0.08 if kind in ("goal", "xg_swing") else (-0.06 if kind == "goal_conceded" else 0.0)
        return validate_coach_plan(
            {
                "reasoning": f"Rule fallback: react to {kind}",
                "confidence": 0.55,
                "controls_delta": {
                    "pressing_intensity": press,
                    "risk_budget": 0.05 if kind == "goal" else -0.03,
                },
            }
        )
    if trig.entity_tier == ENTITY_TIER_PLAYER:
        eb = {"pride": 0.12, "determination": 0.08} if kind == "goal" else {"fear": 0.06}
        return validate_player_plan(
            {"narrative": f"Focus after {kind}", "confidence": 0.5, "emotion_bias": eb, "risk_delta": 0.0}
        )
    if trig.entity_tier == ENTITY_TIER_REFEREE:
        return validate_referee_plan(
            {
                "reasoning": f"Manage {kind}",
                "strictness_delta": 0.05 if kind in ("red_card", "var") else 0.0,
                "bias_delta": 0.0,
                "var_recommendation": "review" if kind == "var" else "none",
                "calm_delta": -0.05,
            }
        )
    if trig.entity_tier == ENTITY_TIER_ASSISTANT:
        return validate_assistant_plan(
            {
                "signal": f"Flag for {kind}",
                "offside_strictness_delta": 0.04,
                "recommend_card_review": kind == "var",
            }
        )
    return validate_crowd_plan(
        {
            "chant_narrative": f"Crowd reacts to {kind}",
            "psi_pulse": 0.15 if kind in ("goal", "crowd_surge") else 0.05,
            "home_boost_delta": 0.03,
        }
    )


class CognitiveExecutor:
    def __init__(
        self,
        cfg: CognitiveMatchConfig,
        llm: Optional["SimulationLLM"] = None,
        *,
        use_llm: bool = True,
    ) -> None:
        self.cfg = cfg
        self.llm = llm
        self.use_llm = use_llm and llm is not None
        self.records: List[CognitivePlanRecord] = []
        if cfg.cache_dir:
            Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)

    def _load_cache(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.cfg.cache_dir:
            return None
        path = Path(self.cfg.cache_dir) / f"{key}.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def _save_cache(self, key: str, plan: Dict[str, Any]) -> None:
        if not self.cfg.cache_dir:
            return
        path = Path(self.cfg.cache_dir) / f"{key}.json"
        fd, temporary = tempfile.mkstemp(prefix=f".{key}-", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(plan, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)

    @staticmethod
    def _validate_plan(trig: CognitiveTriggerEvent, data: Any) -> Dict[str, Any]:
        if trig.entity_tier == ENTITY_TIER_COACH:
            return validate_coach_plan(data)
        if trig.entity_tier == ENTITY_TIER_PLAYER:
            return validate_player_plan(data)
        if trig.entity_tier == ENTITY_TIER_REFEREE:
            return validate_referee_plan(data)
        if trig.entity_tier == ENTITY_TIER_ASSISTANT:
            return validate_assistant_plan(data)
        return validate_crowd_plan(data)

    def _call_llm_for_trigger(self, trig: CognitiveTriggerEvent) -> Dict[str, Any]:
        facts = trig.facts
        kind = trig.kind
        if not self.use_llm:
            return _rule_fallback_plan(trig)

        # Provider retries belong to LLMGateway; this layer only retries malformed plans.
        from src.simulation.runtime import environment_snapshot, env_int
        attempts = max(1, env_int(environment_snapshot(), "COGNITIVE_PLAN_RETRIES", 2))
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                if trig.entity_tier == ENTITY_TIER_COACH:
                    raw = self.llm.coach_in_match_plan(trig.team_id or "team", facts, kind)
                elif trig.entity_tier == ENTITY_TIER_PLAYER:
                    name = facts.get("player_name", trig.entity_id)
                    raw = self.llm.player_in_match_reflection(name, trig.team_id or "", facts, kind)
                elif trig.entity_tier == ENTITY_TIER_REFEREE:
                    raw = self.llm.referee_in_match_judgment(facts, kind)
                elif trig.entity_tier == ENTITY_TIER_ASSISTANT:
                    side = trig.entity_id.split(":")[-1] if ":" in trig.entity_id else "left"
                    raw = self.llm.assistant_signal(side, facts, kind)
                else:
                    raw = self.llm.crowd_collective_reaction(facts, kind)
                data = json.loads(raw) if isinstance(raw, str) else raw
                return self._validate_plan(trig, data)
            except Exception as exc:
                last_exc = exc
                if attempt < attempts - 1:
                    wait = min(14.0, 1.4 * (2 ** attempt))
                    time.sleep(wait)
        print(
            f"  [COGNITIVE] LLM exhausted {attempts} retries ({last_exc}); using rule fallback."
        )
        return _rule_fallback_plan(trig)

    def process_trigger(
        self,
        trig: CognitiveTriggerEvent,
        state: "MatchAffectiveState",
        *,
        home_agent: Optional["SocietyAgent"] = None,
        away_agent: Optional["SocietyAgent"] = None,
    ) -> CognitivePlanRecord:
        # Enrichment must happen before cache-key construction and LLM execution.
        if trig.entity_tier == ENTITY_TIER_PLAYER:
            for player in state.home.players + state.away.players:
                if player.player_id == trig.entity_id:
                    trig.facts["player_name"] = player.name
                    break

        key = _cache_key(trig)
        cached = self._load_cache(key)
        plan: Dict[str, Any]
        from_cache = False
        if cached is not None:
            try:
                plan = self._validate_plan(trig, cached)
                from_cache = True
            except (TypeError, ValueError, KeyError):
                plan = self._call_llm_for_trigger(trig)
                self._save_cache(key, plan)
        else:
            plan = self._call_llm_for_trigger(trig)
            self._save_cache(key, plan)

        rec = CognitivePlanRecord(trigger=trig, plan=plan, cached=from_cache)
        try:
            applied = apply_cognitive_plan(
                trig,
                state,
                plan,
                home_agent=home_agent,
                away_agent=away_agent,
                crowd_numeric=self.cfg.crowd_numeric,
            )
            rec.applied = bool(applied)
            rec.plan["applied_fields"] = applied
        except Exception as exc:
            rec.error = str(exc)
            rec.applied = False
        self.records.append(rec)
        return rec

    def process_batch(
        self,
        triggers: List[CognitiveTriggerEvent],
        state: "MatchAffectiveState",
        *,
        home_agent: Optional["SocietyAgent"] = None,
        away_agent: Optional["SocietyAgent"] = None,
    ) -> List[CognitivePlanRecord]:
        out = []
        for trig in triggers:
            out.append(
                self.process_trigger(
                    trig,
                    state,
                    home_agent=home_agent,
                    away_agent=away_agent,
                )
            )
        return out

    def export_log(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.records]

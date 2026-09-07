"""Own audited reflection, cognitive-memory ingestion and LLM context."""

import json


class AgentReflectionMixin:
    """Own audited reflection, cognitive-memory ingestion and LLM context."""

    def _reflection_context_payload(self):
        memory_context = self.retrieve_memory_context_display(top_k=8)
        stage_hint = (
            self.decision_memory[-1].get("stage") if self.decision_memory else None
        )
        style_hint = (
            self.decision_memory[-1].get("opponent_style")
            if self.decision_memory
            else None
        )
        similar_decisions = self.retrieve_similar_decision_memories(
            self.tactical_controls,
            top_k=4,
            current_stage=stage_hint,
            current_opponent_style=style_hint,
        )
        return memory_context, {
            "high_salience_memory": memory_context,
            "similar_decision_outcomes": similar_decisions,
            "current_controls": self.tactical_controls,
        }

    def request_reflection_payload(self, llm):
        """Request and parse a reflection without mutating agent state."""
        print(f"\n  [V13 SUMMIT] {self.name}: Reflecting on the momentum and power...")
        memory_context, reflection_payload = self._reflection_context_payload()
        verdict_str = llm.perform_agent_reflection(
            self.name,
            self.momentum,
            self.hidden_state.tolist(),
            self.roles["Icon"]["patience"],
            memory_context=json.dumps(reflection_payload, ensure_ascii=False),
        )
        data = json.loads(verdict_str)
        if not isinstance(data, dict):
            raise ValueError("Reflection response must be a JSON object")
        return data

    def apply_reflection_payload(self, data, *, operation_id=None):
        """Apply one parsed reflection exactly once for a durable operation ID."""
        if not isinstance(data, dict):
            raise TypeError("Reflection payload must be a mapping")
        if operation_id:
            previous = next(
                (
                    record
                    for record in self.llm_reflection_audit
                    if isinstance(record, dict)
                    and record.get("operation_id") == operation_id
                ),
                None,
            )
            if previous is not None:
                return previous
        audit = self.apply_llm_reflection(data, operation_id=operation_id)
        self._register_memory_event(
            content=f"Reflection update: {self.reflection_diary[:160]}",
            layer="procedural",
            importance=8.0,
            emotion=self._emotion_scalar(),
            event_type="reflection_update",
            tags=["reflection"],
            write_temperature=1.25,
            metadata={"latest_reflection_audit": audit},
        )
        self.consolidate_memories()
        return audit

    def perform_reflection(self, llm, *, operation_id=None):
        data = self.request_reflection_payload(llm)
        return self.apply_reflection_payload(data, operation_id=operation_id)

    def ingest_micro_cognitive_memory(
        self,
        cognitive_plans: list,
        team_name: str = "",
        *,
        operation_id: str | None = None,
    ) -> None:
        """Absorb in-match System 2 narratives into episodic memory for cross-match continuity."""
        if not cognitive_plans:
            return
        for index, rec in enumerate(cognitive_plans):
            if not isinstance(rec, dict):
                continue
            trig = rec.get("trigger") or {}
            if team_name and trig.get("team_id") and trig.get("team_id") != team_name:
                if trig.get("entity_tier") == "coach" and team_name not in str(
                    trig.get("entity_id", "")
                ):
                    continue
            plan = rec.get("plan") or {}
            narrative = str(
                plan.get("narrative", "") or plan.get("reasoning", "")
            ).strip()
            if len(narrative) < 8:
                continue
            continuity_event_id = (
                f"{operation_id}:{index}" if operation_id else None
            )
            if continuity_event_id and any(
                isinstance(memory, dict)
                and isinstance(memory.get("metadata"), dict)
                and memory["metadata"].get("continuity_event_id")
                == continuity_event_id
                for memory in self.episodic_memory
            ):
                continue
            self._register_memory_event(
                content=f"In-match ({trig.get('kind', 'event')}): {narrative[:200]}",
                layer="episodic",
                importance=6.0 + 2.0 * float(trig.get("salience", 0)),
                emotion=self._emotion_scalar(),
                event_type="micro_cognitive",
                tags=["cognitive", str(trig.get("kind", ""))],
                write_temperature=1.1,
                metadata={
                    "trigger": trig,
                    "applied": rec.get("applied", False),
                    "continuity_event_id": continuity_event_id,
                },
            )

    def apply_llm_reflection(self, reflection, *, operation_id=None):
        from src.simulation.meta_learning import MetaLearningController

        audit_log = MetaLearningController().apply(
            self,
            reflection,
            operation_id=operation_id,
        )
        self.llm_reflection_audit.append(audit_log)
        return audit_log

    def get_context_for_llm(self):
        hs = self.hidden_state
        morale = "High" if hs[0] > 0.4 else "Low"
        top_beliefs = [
            b.get("claim", "")[:60]
            for b in sorted(
                self.beliefs, key=lambda x: x.get("confidence", 0.0), reverse=True
            )[:2]
        ]
        coach_line = f"Coach: {self.coach_name}"
        if self.coach_profile is not None:
            cp = self.coach_profile
            coach_line += f" ({cp.nationality}, preset={cp.preferred_preset}, exp={cp.mental.get('experience', 0):.2f})"
        return (
            f"{coach_line} | Psych: {morale}, Stability: {hs[1]:.2f}, Momentum: {self.momentum:.2f}, "
            f"Icon_Patience: {self.roles['Icon']['patience']:.2f}, Fatigue: {self.fatigue:.2f}, "
            f"InjuryLoad: {self.injury_load:.2f}, Readiness: {self.readiness:.2f}, Style: {self.style_archetype}, "
            f"CoachAuthority: {self.coach_authority:.2f}, IconInfluence: {self.icon_influence:.2f}, "
            f"Cohesion: {self.team_cohesion:.2f}, ConflictHeat: {self.conflict_heat:.2f}, RefTrust: {self.referee_trust:.2f}, "
            f"EmotionProfile: {self.emotion_profile}, Beliefs: {top_beliefs}, Controls: {self.tactical_controls}"
        )

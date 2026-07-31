"""Memory creation, decision records, utility attribution, and lifecycle."""

import datetime

import numpy as np

from src.simulation.memory_service import (
    attribute_delayed_utility,
    build_provenance,
    memory_stream_view,
)


class AgentMemoryWriteMixin:
    def _content_novelty(self, content, layer="episodic", lookback=25):
        if not content:
            return 0.5
        bank = self.episodic_memory if layer == "episodic" else self.procedural_memory
        if not bank:
            return 1.0
        tokens = set(str(content).lower().split())
        recent = bank[-lookback:]
        max_overlap = 0.0
        for rec in recent:
            past = set(str(rec.get("content", "")).lower().split())
            if not tokens or not past:
                continue
            overlap = len(tokens & past) / max(1, len(tokens | past))
            max_overlap = max(max_overlap, overlap)
        return float(np.clip(1.0 - max_overlap, 0.05, 1.0))

    def _memory_write_diagnostics(
        self, *, content, layer, importance, emotion, write_temperature,
    ):
        novelty = self._content_novelty(content, layer=layer)
        # Probabilistic write gate (continuous, no hard threshold rule).
        write_logit = (
            0.45 * float(importance)
            + 1.4 * float(emotion)
            + 0.9 * float(novelty)
            + 0.25 * float(write_temperature)
            - 3.6
        )
        p_write = self._bounded_sigmoid(write_logit)
        # Continuous framework: all events are written; p_write is retained as a soft write-intensity diagnostic.
        return novelty, p_write

    def _build_memory_record(
        self, *, content, layer, importance, emotion, event_type, match, stage,
        speaker, tags, opponent, affected_states, summary, appraisal,
        emotion_profile, coping, stage_pressure, metadata, source,
        causal_parent_ids, contradicts, valid_from, valid_until, novelty,
    ):
        self.memory_clock += 1
        record_appraisal = appraisal or dict(self.appraisal_state)
        rec = {
            "id": f"{self.team_name}-{self.memory_clock}",
            "layer": layer,
            "type": layer,
            "event": str(event_type),
            "content": content,
            "match": match,
            "stage": stage,
            "timestamp": self.memory_clock,
            "importance": float(importance),
            "emotion": float(emotion),
            "valence": float(record_appraisal.get("impact", 0.0)),
            "arousal": float(record_appraisal.get("novelty", 0.5)),
            "confidence": float(self._sigmoid(self.z_state["confidence"])),
            "novelty": float(novelty),
            "speaker": speaker,
            "tags": tags or [],
            "opponent": opponent,
            "affected_states": affected_states or {},
            "summary": summary or str(content),
            "appraisal": record_appraisal,
            "emotion_profile": emotion_profile or dict(self.emotion_profile),
            "coping": coping or dict(self.coping_profile),
            "metadata": metadata or {},
            "provenance": build_provenance(
                source, causal_parent_ids, contradicts,
                valid_from if valid_from is not None else self.memory_clock,
                valid_until,
            ),
            "retrieval_count": 0,
            "downstream_utility_sum": 0.0,
            "downstream_utility_count": 0,
            "created_step": self.memory_clock,
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
        }
        rec["memory_weight"] = self._memory_salience(
            rec["appraisal"],
            rec["emotion_profile"],
            stage_pressure=stage_pressure,
            importance=rec["importance"] / 10.0,
        )
        rec["salience"] = float(rec["memory_weight"])
        return rec

    def _store_memory_record(self, *, rec, layer, tags, content, p_write):
        if layer == "episodic":
            self.episodic_memory.append(rec)
        elif layer == "procedural":
            self.procedural_memory.append(rec)
        else:
            key = str(tags[0] if tags else f"semantic_{self.memory_clock}")
            self.semantic_memory[key] = content

        self.memory_event_log.append(
            {"step": self.memory_clock, "layer": layer, "p_write": float(p_write), "id": rec["id"]}
        )
        self._decay_and_prune_memory()

    def _register_memory_event(
        self,
        content,
        layer="episodic",
        importance=5,
        emotion=0.2,
        event_type="generic_event",
        match=None,
        stage=None,
        speaker=None,
        tags=None,
        opponent=None,
        write_temperature=1.0,
        affected_states=None,
        summary=None,
        appraisal=None,
        emotion_profile=None,
        coping=None,
        stage_pressure=0.3,
        metadata=None,
        force_write=False,
        source="simulation",
        causal_parent_ids=None,
        contradicts=None,
        valid_from=None,
        valid_until=None,
    ):
        novelty, p_write = self._memory_write_diagnostics(
            content=content, layer=layer, importance=importance,
            emotion=emotion, write_temperature=write_temperature,
        )
        rec = self._build_memory_record(
            content=content, layer=layer, importance=importance, emotion=emotion,
            event_type=event_type, match=match, stage=stage, speaker=speaker,
            tags=tags, opponent=opponent, affected_states=affected_states,
            summary=summary, appraisal=appraisal, emotion_profile=emotion_profile,
            coping=coping, stage_pressure=stage_pressure, metadata=metadata,
            source=source, causal_parent_ids=causal_parent_ids,
            contradicts=contradicts, valid_from=valid_from, valid_until=valid_until,
            novelty=novelty,
        )
        self._store_memory_record(
            rec=rec, layer=layer, tags=tags, content=content, p_write=p_write,
        )
        return rec

    @property
    def memory_stream(self):
        # Backward-compatible read view built from canonical layered memories.
        return memory_stream_view(self.episodic_memory, self.procedural_memory)

    def _control_vector(self, controls):
        c = controls or {}
        return np.array(
            [
                self._clip01(c.get("pressing_intensity", 0.5)),
                self._clip01(c.get("risk_budget", 0.5)),
                self._clip01(c.get("line_height", 0.5)),
                self._clip01(c.get("rotation_aggressiveness", 0.5)),
            ],
            dtype=float,
        )

    def record_decision_event(
        self,
        opponent,
        stage_name,
        controls,
        outcomes,
        opponent_style=None,
        stage_pressure=None,
        decision_audit=None,
    ):
        controls = controls or {}
        outcomes = outcomes or {}
        rec = {
            "step": self.memory_clock,
            "opponent": opponent,
            "stage": stage_name,
            "opponent_style": opponent_style or "unknown",
            "stage_pressure": float(stage_pressure) if stage_pressure is not None else None,
            "controls": {
                "pressing_intensity": self._clip01(controls.get("pressing_intensity", 0.5)),
                "risk_budget": self._clip01(controls.get("risk_budget", 0.5)),
                "line_height": self._clip01(controls.get("line_height", 0.5)),
                "rotation_aggressiveness": self._clip01(controls.get("rotation_aggressiveness", 0.5)),
            },
            "outcomes": {
                "goal_diff": float(outcomes.get("goal_diff", 0.0)),
                "xg_for": float(outcomes.get("xg_for", 0.0)),
                "xg_against": float(outcomes.get("xg_against", 0.0)),
                "fatigue_delta": float(outcomes.get("fatigue_delta", 0.0)),
                "chaos": float(outcomes.get("chaos", 0.0)),
                "result": str(outcomes.get("result", "draw")),
            },
            "world_model_fusion": dict(decision_audit or {}),
        }
        self.decision_memory.append(rec)
        if len(self.decision_memory) > 300:
            self.decision_memory = self.decision_memory[-300:]

        emotion = min(1.0, 0.18 * abs(rec["outcomes"]["goal_diff"]) + 0.08 * abs(rec["outcomes"]["chaos"]))
        mem_rec = self._register_memory_event(
            content=(
                f"Decision event vs {opponent} [{stage_name}]: controls={rec['controls']} "
                f"-> result={rec['outcomes']['result']}, gd={rec['outcomes']['goal_diff']:.1f}, "
                f"xG={rec['outcomes']['xg_for']:.2f}/{rec['outcomes']['xg_against']:.2f}, opp_style={rec['opponent_style']}, "
                f"fatigueΔ={rec['outcomes']['fatigue_delta']:.3f}"
            ),
            layer="episodic",
            importance=6.5 + abs(rec["outcomes"]["goal_diff"]) * 0.7,
            emotion=emotion,
            event_type="decision_event",
            match=f"{self.team_name} vs {opponent}",
            stage=stage_name,
            opponent=opponent,
            tags=["decision_event", stage_name],
            write_temperature=1.2,
            affected_states={
                "morale": float(self.hidden_state[0]),
                "stability": float(self.hidden_state[1]),
                "cohesion": float(self.team_cohesion),
            },
            metadata={
                "controls": rec["controls"],
                "outcomes": rec["outcomes"],
                "stage": stage_name,
                "opponent_style": rec["opponent_style"],
                "stage_pressure": rec["stage_pressure"],
                "world_model_fusion": rec["world_model_fusion"],
            },
            force_write=True,
        )
        if mem_rec is not None:
            rec["step"] = mem_rec["created_step"]
            rec["evidence_id"] = mem_rec.get("id")
            self.update_beliefs_from_match_event({"memory_rec": mem_rec})
        return rec

    def _decay_and_prune_memory(self):
        for rec in self.episodic_memory:
            age = self.memory_clock - rec["created_step"]
            sal = float(rec.get("memory_weight", rec.get("salience", 0.0)))
            rec["memory_weight"] = float(
                sal * np.exp(-self.lambda_m * age * (1.0 - self.beta_m * sal))
            )
            rec["salience"] = float(np.clip(rec["memory_weight"], 0.0, 1.0))

        for rec in self.procedural_memory:
            age = self.memory_clock - rec["created_step"]
            sal = float(rec.get("memory_weight", rec.get("salience", 0.0)))
            rec["memory_weight"] = float(
                sal * np.exp(-self.lambda_m * age * (1.0 - self.beta_m * sal))
            )
            rec["salience"] = float(np.clip(rec["memory_weight"], 0.0, 1.0))

        if len(self.episodic_memory) > self.memory_max_episodic:
            self.episodic_memory = sorted(
                self.episodic_memory, key=lambda r: (r.get("memory_weight", r.get("salience", 0.0)), r.get("created_step", 0))
            )[-self.memory_max_episodic:]
            self.episodic_memory.sort(key=lambda r: r["created_step"])

    def record_memory_utility(self, memory_ids, utility):
        """Attribute delayed downstream utility to retrieved evidence."""
        return attribute_delayed_utility(
            self.episodic_memory + self.procedural_memory,
            memory_ids or [], utility,
        )

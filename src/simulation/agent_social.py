"""Social feed, dialogue, and relationship behavior for society agents."""

import numpy as np
from src.simulation.random_control import named_py_rng


class SocialAgentMixin:
    def update_rivalry(self, opp_name, drama_score):
        current = self.rivalry_database.get(opp_name, 0.0)
        self.rivalry_database[opp_name] = current + (drama_score * 0.1)
    
    def decide_action(self, feed, *, rng=None):
        if not feed.posts:
            return "POST"
        rng = rng or named_py_rng(
            getattr(self, "random_root_seed", 42),
            "social_action", len(feed.posts),
        )
        base_post_prob = 0.4 + (self.personality["arrogance"] - 0.5) * 0.3
        if self.hidden_state[2] > 0.3:
            base_post_prob += 0.1
        return "POST" if rng.random() < np.clip(base_post_prob, 0.15, 0.85) else "REPLY"
    
    def generate_comment(self):
        tone = "confident" if self.hidden_state[0] > 0 else "defiant"
        if tone == "confident":
            return f"We trust our structure and intensity. {self.formation} stays aggressive."
        return f"Pressure builds, but {self.team_name} stands firm. We adapt and fight."
    
    def record_social_dialogue_event(self, content, stage_name, topic, signal, target=None):
        signal = signal or {}
        sentiment = float(signal.get("sentiment", 0.0))
        prov = float(signal.get("provocation_level", 0.0))
        appraisal = {
            "impact": float(np.tanh(0.9 * sentiment - 0.6 * prov)),
            "novelty": float(self._sigmoid(1.2 * prov)),
            "control": float(self._sigmoid(1.4 * self.team_cohesion - 0.8 * prov)),
            "certainty": float(self._sigmoid(0.5 + 0.6 * abs(sentiment))),
            "norm_violation": float(self._sigmoid(1.8 * prov)),
            "agency": float(np.tanh(0.4 - 1.1 * prov)),
        }
        emotion_profile = self._emotion_from_appraisal(appraisal)
        coping = self._coping_from_appraisal_emotion(appraisal, emotion_profile)
        self._register_memory_event(
            content=content,
            layer="episodic",
            importance=6.2 + 1.8 * prov,
            emotion=self._emotion_scalar(),
            event_type="social_dialogue",
            stage=stage_name,
            opponent=target,
            tags=["social_dialogue", stage_name, str(topic)],
            write_temperature=1.15,
            affected_states={
                "morale": float(self.morale),
                "unity": float(self.team_cohesion),
                "conflict_heat": float(self.conflict_heat),
            },
            metadata={
                "topic": topic,
                "signal": signal,
                "target": target,
                "stage": stage_name,
                "emotion_profile": self.emotion_profile,
            },
            appraisal=appraisal,
            emotion_profile=emotion_profile,
            coping=coping,
            stage_pressure=0.45 if stage_name in {"Semi-Finals", "Final"} else 0.25,
            force_write=True,
        )
    
    def apply_social_signal(self, signal, stage_pressure=0.2):
        signal = signal or {}
        pressure = float(np.clip(stage_pressure, 0.0, 1.0))
        sentiment = float(np.clip(signal.get("sentiment", 0.0), -1.0, 1.0))
        unity = float(np.clip(signal.get("unity_signal", 0.0), -1.0, 1.0))
        prov = float(np.clip(signal.get("provocation_level", 0.0), 0.0, 1.0))
        cred = float(np.clip(signal.get("credibility", 0.5), 0.0, 1.0))
        blame_ref = float(np.clip(signal.get("blame_referee", 0.0), 0.0, 1.0))
    
        morale_delta = 0.16 * sentiment * (0.65 + 0.35 * cred) + 0.08 * unity - 0.10 * prov
        cohesion_delta = 0.14 * unity * cred - 0.10 * prov - 0.06 * max(0.0, -sentiment)
        conflict_push = (0.22 + 0.30 * pressure) * prov + 0.10 * max(0.0, -sentiment) - 0.12 * max(0.0, unity)
        grievance_push = 0.17 * blame_ref * (0.50 + 0.50 * prov)
    
        self.z_state["morale"] = self.rho_s * self.z_state["morale"] + 0.45 * morale_delta
        self.z_state["unity"] = self.rho_s * self.z_state["unity"] + 0.55 * cohesion_delta
        self.conflict_heat = float(np.clip(self.conflict_heat * np.exp(-0.08 * max(0.0, unity)) + conflict_push, 0.0, 1.05))
        self.referee_grievance = float(np.clip(self.referee_grievance * np.exp(-0.09) + grievance_push, 0.0, 0.92))
    
        trust_idx = self.social_narrative_state["trust_index"]
        polar = self.social_narrative_state["polarization"]
        fatigue = self.social_narrative_state["narrative_fatigue"]
        trust_idx = float(np.clip(trust_idx + 0.08 * unity * cred - 0.06 * prov, 0.05, 0.99))
        polar = float(np.clip(polar * np.exp(-0.05 * max(0.0, unity)) + 0.20 * prov + 0.08 * max(0.0, -sentiment), 0.0, 1.5))
        fatigue = float(np.clip(fatigue * np.exp(-0.03) + 0.10 * prov + 0.04 * pressure, 0.0, 1.6))
        self.social_narrative_state["trust_index"] = trust_idx
        self.social_narrative_state["polarization"] = polar
        self.social_narrative_state["narrative_fatigue"] = fatigue
        self._project_latents_to_states()
    
        chaos_delta = float(np.clip(
            0.85 * prov + 0.55 * polar + 0.22 * self.referee_grievance - 0.35 * trust_idx - 0.25 * max(0.0, unity),
            -2.0,
            3.0,
        ))
        self._register_memory_event(
            content=(
                f"Social signal feedback: sent={sentiment:.2f}, unity={unity:.2f}, prov={prov:.2f}, "
                f"cred={cred:.2f}, chaosΔ={chaos_delta:.2f}"
            ),
            layer="episodic",
            importance=6.4 + 1.6 * prov,
            emotion=self._emotion_scalar(),
            event_type="social_feedback",
            tags=["social_feedback"],
            write_temperature=1.1 + 0.2 * pressure,
            affected_states={
                "morale": float(self.morale),
                "cohesion": float(self.team_cohesion),
                "conflict_heat": float(self.conflict_heat),
                "referee_grievance": float(self.referee_grievance),
            },
            metadata={"signal": signal, "chaos_delta": chaos_delta, "emotion_profile": self.emotion_profile},
            force_write=True,
        )
        return {"chaos_delta": chaos_delta}
    
    def compress_social_memory(self, stage_name, lookback=18, top_k=3):
        candidates = []
        for rec in self.episodic_memory[-max(8, int(lookback)) :]:
            tags = rec.get("tags", [])
            if "social_dialogue" not in tags and "social_feedback" not in tags:
                continue
            age = max(0, self.memory_clock - int(rec.get("created_step", self.memory_clock)))
            recency = float(np.exp(-0.12 * age))
            score = float(rec.get("salience", 0.0)) * (0.65 + 0.35 * recency)
            candidates.append((score, rec))
        if not candidates:
            return
        candidates.sort(key=lambda x: x[0], reverse=True)
        selected = [r for _, r in candidates[: max(1, int(top_k))]]
        themes = []
        for rec in selected:
            meta = rec.get("metadata", {})
            topic = str(meta.get("topic", "general_dialogue"))
            sig = meta.get("signal", {})
            themes.append(
                f"{topic}(prov={float(sig.get('provocation_level', 0.0)):.2f}, sent={float(sig.get('sentiment', 0.0)):.2f})"
            )
        summary = f"Social memory compression [{stage_name}]: " + "; ".join(themes[: max(1, int(top_k))])
        self._register_memory_event(
            content=summary,
            layer="procedural",
            importance=7.2,
            emotion=0.30,
            tags=["social_memory_compression", stage_name],
            write_temperature=1.2,
            force_write=True,
        )
    
    def add_memory(self, content, importance=5, speaker=None):
        emotion = min(1.0, float(importance) / 10.0)
        self._register_memory_event(
            content=content,
            layer="episodic",
            importance=float(importance),
            emotion=emotion,
            speaker=speaker,
            tags=["external_memory"],
            write_temperature=1.0,
            force_write=True,
        )
    
    def update_relationship(self, name, change):
        # Placeholder or link to rivalry
        current = self.rivalry_database.get(name, 0.0)
        self.rivalry_database[name] = max(0.0, current - (change / 100.0))
    

"""Belief induction, tactical application, and memory consolidation."""

import numpy as np

from src.simulation.random_control import named_rng


class AgentBeliefMemoryMixin:
    def update_beliefs_from_match_event(self, match_event):
        event = match_event or {}
        memory_rec = event.get("memory_rec")
        if memory_rec is None:
            return
        m_vec = self._memory_vector(memory_rec)
        memory_weight = float(memory_rec.get("memory_weight", memory_rec.get("salience", 0.0)))
        evidence_id = memory_rec.get("id")

        if not self.beliefs:
            seed_claims = [
                {
                    "claim": "High pressing after emotional losses increases fatigue and defensive errors.",
                    "policy_effect": {"pressing_intensity": -0.08, "rotation_aggressiveness": 0.08, "risk_budget": -0.05},
                },
                {
                    "claim": "Referee controversy increases emotional risk and tactical impatience.",
                    "policy_effect": {"risk_budget": 0.06, "referee_trust": -0.07},
                },
                {
                    "claim": "Collective planning after setbacks restores stability and confidence.",
                    "policy_effect": {"risk_budget": -0.04, "line_height": -0.03, "pressing_intensity": -0.02},
                },
            ]
            for i, sc in enumerate(seed_claims, 1):
                prototype_rng = named_rng(
                    getattr(self, "random_root_seed", 42),
                    "belief_prototype", self.team_name, i,
                )
                self.beliefs.append(
                    {
                        "id": f"{self.team_name}_BELIEF_{i:03d}",
                        "claim": sc["claim"],
                        "support": 0.20,
                        "contradiction": 0.20,
                        "confidence": 0.50,
                        "policy_effect": sc["policy_effect"],
                        "supporting_memories": [],
                        "contradicting_memories": [],
                        "prototype": prototype_rng.normal(
                            0.0, 0.05, size=m_vec.shape[0],
                        ).tolist(),
                    }
                )

        for belief in self.beliefs:
            proto = np.array(belief.get("prototype", np.zeros_like(m_vec)), dtype=float)
            alignment = self._cosine_similarity(m_vec, proto)
            support_gain = memory_weight * self._sigmoid(self.k_alignment * alignment)
            contradiction_gain = memory_weight * self._sigmoid(-self.k_alignment * alignment)

            belief["support"] = float(self.rho_b * float(belief.get("support", 0.0)) + support_gain)
            belief["contradiction"] = float(self.rho_b * float(belief.get("contradiction", 0.0)) + contradiction_gain)
            belief["confidence"] = float(
                self._sigmoid(self.alpha_conf * (belief["support"] - belief["contradiction"]))
            )

            if evidence_id:
                if support_gain >= contradiction_gain:
                    belief["supporting_memories"] = (belief.get("supporting_memories", []) + [evidence_id])[-80:]
                else:
                    belief["contradicting_memories"] = (belief.get("contradicting_memories", []) + [evidence_id])[-80:]

            # Continuous prototype update.
            proto = self.rho_b * proto + (1.0 - self.rho_b) * m_vec
            belief["prototype"] = proto.tolist()

    def apply_beliefs_to_tactics(self, stage_name=None, opponent_style=None):
        _ = stage_name
        _ = opponent_style
        if not self.beliefs:
            return
        aggregate = {k: 0.0 for k in self.tactical_controls.keys()}
        belief_context = 0.0
        for belief in self.beliefs:
            confidence = float(np.clip(belief.get("confidence", 0.0), 0.0, 1.0))
            activation = self._sigmoid(self.s_b * (confidence - self.c0))
            belief_context += activation * confidence
            for k, eff in belief.get("policy_effect", {}).items():
                if k in aggregate:
                    aggregate[k] += activation * confidence * float(eff)
        for k, v in aggregate.items():
            bounded = self.delta_max * np.tanh(float(v) / max(self.eps, self.delta_max))
            self.tactical_controls[k] = self._sigmoid(self._logit01(self.tactical_controls[k]) + bounded)
        # Beliefs affect experience-level risk latent, while decision_memory remains factual.
        self.z_state["risk_tolerance"] = (
            self.rho_s * self.z_state["risk_tolerance"] + 0.08 * np.tanh(belief_context)
        )
        self._project_latents_to_states()

    def consolidate_memories(self):
        # Stage-level consolidation: summarize top episodic events into procedural memory.
        top_events = self.retrieve_memory_context_display(top_k=6)
        if not top_events:
            return
        opponents = [e.get("opponent") for e in top_events if e.get("opponent")]
        brief = ", ".join([e["content"][:48] for e in top_events[:3]])
        summary = (
            f"Consolidated pattern after stage: key opponents={opponents[:3]}, "
            f"event_signature={brief}"
        )
        self._register_memory_event(
            content=summary,
            layer="procedural",
            importance=7.0,
            emotion=0.35,
            tags=["stage_consolidation"],
            write_temperature=1.2,
        )

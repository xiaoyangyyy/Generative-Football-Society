"""Post-match affective, belief, latent-state, and memory evolution."""

import numpy as np

from src.simulation.random_control import named_rng


class AgentMatchDynamicsMixin:
    def _build_match_update_context(
        self, *, match_result, score_diff, prof_score,
        social_chaos, opp_name, opp_status,
    ):
        # 1. Update Momentum (No thresholds, continuous growth)
        surprise_factor = max(0.0, (opp_status - self.status_score) / 100.0)
        if match_result == "win":
            self.momentum = self.momentum * 0.8 + (surprise_factor * 1.1) + (score_diff * 0.08)
        else:
            self.momentum *= 0.6 # Fades fast on loss
            
        # 2. Cinderella Flip Gamma: From Pressure to Power
        # As momentum grows for LOW EXPOSURE teams, Gamma approaches 1.0
        gamma = np.tanh(self.momentum * (1.0 - self.media_exposure)**2)
        
        # 3. Base Impact & Rivalry
        hate_score = self.rivalry_database.get(opp_name, 0.0)
        res_val = 1.4 if match_result == "win" else (-1.6 if match_result == "loss" else 0.2)
        if match_result == "loss": res_val *= (1.0 + hate_score * 1.5)
        
        # 4. Decoupled Social Feedback with Gamma-Inversion
        social_impact_base = social_chaos * (1.0 + self.media_exposure ** 3)
        # Flip: (1 - 2.5 * gamma) makes negative chaos POSITIVE for high-momentum weak teams
        final_social_contribution = social_impact_base * (1.0 - 1.6 * gamma)
        
        # 5. Internal Pressure Transmission
        pres_pressure = social_chaos * self.roles["President"]["media_sensitivity"]
        
        # 6. Legacy psych terms preserved as continuous drivers in latent updates.
        X_t = np.array([
            res_val + (score_diff * 0.1),
            prof_score * 0.4,
            -final_social_contribution * 0.5 - (pres_pressure * 0.3),
        ])
        
        upset = bool(match_result == "win" and (opp_status - self.status_score) > 12.0)
        referee_controversy = 1.0 if self.referee_grievance > 0.42 else 0.0

        appraisal = self._appraise_event(
            {
                "result": match_result,
                "score_diff": score_diff,
                "stage_pressure": 0.35,
                "referee_controversy": referee_controversy,
                "upset_factor": 1.0 if upset else 0.0,
                "social_chaos": social_chaos,
            }
        )
        emotion = self._emotion_from_appraisal(appraisal)
        coping = self._coping_from_appraisal_emotion(appraisal, emotion)

        # Belief context (continuous field, no hard activation).
        belief_context = 0.0
        for belief in self.beliefs:
            conf = float(np.clip(belief.get("confidence", 0.0), 0.0, 1.0))
            belief_context += self._sigmoid(self.s_b * (conf - self.c0)) * conf
        belief_context = float(np.tanh(belief_context))
        return gamma, appraisal, emotion, coping, belief_context


    def _update_match_latent_state(
        self, *, appraisal, emotion, coping, belief_context, social_chaos,
        rng,
    ):
        # Latent state update (continuous, bounded by tanh/sigmoid projection only).
        noise = rng.normal(0.0, self.state_noise_sigma, size=8)
        self.z_state["morale"] = (
            self.rho_s * self.z_state["morale"]
            + 0.55 * appraisal["impact"]
            + 0.32 * emotion["pride"]
            - 0.38 * emotion["shame"]
            - 0.26 * emotion["fear"]
            + 0.16 * belief_context
            + noise[0]
        )
        self.z_state["stability"] = (
            self.rho_s * self.z_state["stability"]
            + 0.30 * coping["planning"]
            + 0.24 * coping["self_correction"]
            - 0.28 * coping["external_blame"]
            - 0.22 * emotion["anger"]
            + 0.18 * belief_context
            + noise[1]
        )
        self.z_state["unity"] = (
            self.rho_s * self.z_state["unity"]
            + 0.30 * coping["planning"]
            + 0.22 * coping["self_correction"]
            - 0.30 * coping["external_blame"]
            + 0.14 * emotion["determination"]
            + noise[2]
        )
        self.z_state["confidence"] = (
            self.rho_s * self.z_state["confidence"]
            + 0.32 * emotion["pride"]
            - 0.28 * emotion["fear"]
            - 0.24 * emotion["shame"]
            + 0.20 * belief_context
            + noise[3]
        )
        self.z_state["risk_tolerance"] = (
            self.rho_s * self.z_state["risk_tolerance"]
            + 0.38 * coping["risk_shift"]
            + 0.16 * emotion["determination"]
            - 0.22 * emotion["fear"]
            + noise[4]
        )
        self.z_state["pressing_intensity"] = (
            self.rho_s * self.z_state["pressing_intensity"]
            + 0.24 * emotion["determination"]
            + 0.12 * coping["planning"]
            - 0.18 * emotion["fear"]
            + noise[5]
        )
        self.z_state["referee_trust"] = (
            self.rho_s * self.z_state["referee_trust"]
            - 0.42 * appraisal["norm_violation"]
            - 0.20 * emotion["anger"]
            + 0.12 * coping["planning"]
            + noise[6]
        )
        self.z_state["media_sensitivity"] = (
            self.rho_s * self.z_state["media_sensitivity"]
            + 0.20 * np.tanh(social_chaos / 5.0)
            + 0.14 * emotion["fear"]
            + 0.08 * emotion["pride"]
            + noise[7]
        )
        self._project_latents_to_states()


    def _record_match_outcome(
        self, *, match_result, score_diff, prof_score, social_chaos,
        opp_name, appraisal, emotion, coping, gamma,
    ):
        outcome_text = (
            f"Match vs {opp_name}: {match_result} (diff={score_diff}, prof={prof_score:.2f}, chaos={social_chaos:.2f})"
        )
        emotion_scalar = self._emotion_scalar()
        self._register_memory_event(
            content=outcome_text,
            layer="episodic",
            importance=6.0 + abs(score_diff),
            emotion=emotion_scalar,
            event_type=f"match_{match_result}",
            match=f"{self.team_name} vs {opp_name}",
            stage="matchday",
            opponent=opp_name,
            tags=["match", match_result],
            affected_states={
                "morale": float(self.morale),
                "stability": float(np.tanh(self.z_state["stability"])),
                "referee_trust": float(self.referee_trust),
                "unity": float(self.team_cohesion),
            },
            summary=f"{self.team_name} {match_result} vs {opp_name} (gd={score_diff})",
            appraisal=appraisal,
            emotion_profile=emotion,
            coping=coping,
            stage_pressure=0.35,
            write_temperature=1.15,
        )
        print(f"  [MOMENTUM: {self.momentum:.2f} | GAMMA: {gamma:.2f}] {self.name}")


    def recursive_update(self, match_result, score_diff, prof_score,
                         social_chaos, opp_name, opp_status, *, rng=None):
        """
        Final V13 Logic: 
        - Rivalry amplification persists.
        - Momentum field flips social impact for 'Cinderella' teams.
        """
        gamma, appraisal, emotion, coping, belief_context = (
            self._build_match_update_context(
                match_result=match_result, score_diff=score_diff,
                prof_score=prof_score, social_chaos=social_chaos,
                opp_name=opp_name, opp_status=opp_status,
            )
        )

        rng = rng or named_rng(getattr(self, "random_root_seed", 42),
                               "match_dynamics", self.memory_clock, opp_name,
                               match_result, score_diff)
        self._update_match_latent_state(
            appraisal=appraisal, emotion=emotion, coping=coping,
            belief_context=belief_context, social_chaos=social_chaos,
            rng=rng,
        )
        self._record_match_outcome(
            match_result=match_result, score_diff=score_diff,
            prof_score=prof_score, social_chaos=social_chaos,
            opp_name=opp_name, appraisal=appraisal, emotion=emotion,
            coping=coping, gamma=gamma,
        )

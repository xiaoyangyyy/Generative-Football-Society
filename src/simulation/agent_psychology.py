"""Latent appraisal, emotion and coping state for society agents."""

import numpy as np


class AgentPsychologyMixin:
    """Own the bounded psychological-state projection used across agent layers."""

    def _initialize_psychology_from_history(self, stats):
        c1 = self._finite(stats.get("c1_win_rate", 50.0), 50.0) / 100.0
        c3 = self._finite(stats.get("c3_major_exp", 45.0), 45.0) / 100.0
        c5 = self._finite(stats.get("c5_pressure", 45.0), 45.0) / 100.0
        modern = (
            self._finite(
                stats.get("modern_power", stats.get("final_status_score", 50.0)), 50.0
            )
            / 100.0
        )
        exposure = float(np.clip(self.media_exposure, 0.05, 1.0))
        profile = {
            "confidence": float(np.clip(0.58 * c1 + 0.42 * modern, 0.05, 0.98)),
            "resilience": float(
                np.clip(0.50 * c3 + 0.25 * c1 + 0.25 * (1.0 - c5), 0.05, 0.98)
            ),
            "stability": float(np.clip(0.60 * c3 + 0.40 * c1, 0.05, 0.98)),
            "media_sensitivity": float(
                np.clip(0.55 * exposure + 0.45 * c5, 0.05, 0.98)
            ),
            "penalty_anxiety": float(
                np.clip(0.35 + 0.50 * c5 * (1.0 - c3), 0.05, 0.98)
            ),
            "expectation_pressure": float(
                np.clip(0.55 * modern + 0.45 * exposure, 0.05, 0.99)
            ),
            "underdog_identity": float(
                np.clip((1.0 - modern) * (0.65 + 0.35 * c1), 0.0, 0.98)
            ),
        }
        return profile

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-float(x)))

    def _softmax_dict(self, logits, tau=1.0):
        tau = max(1e-6, float(tau))
        keys = list(logits.keys())
        arr = np.array([float(logits[k]) for k in keys], dtype=float) / tau
        arr = arr - np.max(arr)
        exps = np.exp(arr)
        denom = max(self.eps, float(np.sum(exps)))
        probs = exps / denom
        return {k: float(v) for k, v in zip(keys, probs)}

    def _logit01(self, x):
        x = float(np.clip(x, self.eps, 1.0 - self.eps))
        return float(np.log(x / (1.0 - x)))

    def _initialize_latent_states(self):
        return {
            "morale": float(
                np.arctanh(
                    np.clip(
                        (self.psychology_profile["confidence"] - 0.5) * 1.6,
                        -0.999,
                        0.999,
                    )
                )
            ),
            "stability": float(
                np.arctanh(
                    np.clip(
                        (self.psychology_profile["stability"] - 0.5) * 1.8,
                        -0.999,
                        0.999,
                    )
                )
            ),
            "unity": self._logit01(self.team_cohesion),
            "confidence": self._logit01(self.psychology_profile.get("confidence", 0.5)),
            "risk_tolerance": self._logit01(
                self.tactical_controls.get("risk_budget", 0.5)
            ),
            "pressing_intensity": self._logit01(
                self.tactical_controls.get("pressing_intensity", 0.5)
            ),
            "referee_trust": self._logit01(self.referee_trust),
            "media_sensitivity": self._logit01(
                self.roles["President"]["media_sensitivity"]
            ),
        }

    def _project_latents_to_states(self):
        self.team_cohesion = float(self._sigmoid(self.z_state["unity"]))
        self.referee_trust = float(self._sigmoid(self.z_state["referee_trust"]))
        self.roles["President"]["media_sensitivity"] = float(
            self._sigmoid(self.z_state["media_sensitivity"])
        )
        self.tactical_controls["risk_budget"] = float(
            self._sigmoid(self.z_state["risk_tolerance"])
        )
        self.tactical_controls["pressing_intensity"] = float(
            self._sigmoid(self.z_state["pressing_intensity"])
        )

    def _emotion_scalar(self):
        e = self.emotion_profile
        scalar = (
            0.24 * float(e.get("determination", 0.0))
            + 0.22 * float(e.get("anger", 0.0))
            + 0.18 * float(e.get("fear", 0.0))
            + 0.18 * float(e.get("shame", 0.0))
            + 0.18 * float(e.get("pride", 0.0))
        )
        return float(np.clip(scalar, 0.0, 1.0))

    def _appraise_event(self, event):
        e = event or {}
        score_diff = float(e.get("score_diff", 0.0))
        result = str(e.get("result", "draw"))
        stage_pressure = float(np.clip(e.get("stage_pressure", 0.3), 0.0, 1.0))
        referee_controversy = float(
            np.clip(e.get("referee_controversy", 0.0), 0.0, 1.0)
        )
        upset_factor = float(np.clip(e.get("upset_factor", 0.0), 0.0, 1.0))
        social_chaos = float(e.get("social_chaos", 0.0))

        if result == "win":
            base_impact = 0.45 + 0.12 * score_diff
            agency_raw = 0.40
        elif result == "loss":
            base_impact = -0.45 + 0.10 * score_diff
            agency_raw = -0.20 - 0.35 * referee_controversy
        else:
            base_impact = 0.04 * np.tanh(score_diff)
            agency_raw = -0.04 * referee_controversy

        z_impact = base_impact + 0.25 * upset_factor - 0.10 * referee_controversy
        z_novelty = (
            0.60 * abs(score_diff)
            + 1.10 * upset_factor
            + 0.40 * referee_controversy
            - 0.30
        )
        z_control = (
            0.90 * (self.coach_authority - self.icon_influence)
            + 0.70 * self.team_cohesion
            - 0.50 * self.conflict_heat
        )
        z_certainty = 1.20 * abs(score_diff) + 0.80 * stage_pressure - 0.60
        z_norm = (
            1.60 * referee_controversy
            + 0.22 * max(0.0, social_chaos)
            - 0.70 * self.referee_trust
        )
        z_agency = agency_raw + 0.35 * np.tanh(
            self.coach_authority - self.icon_influence
        )

        app = {
            "impact": float(np.tanh(z_impact)),
            "novelty": float(self._sigmoid(z_novelty)),
            "control": float(self._sigmoid(z_control)),
            "certainty": float(self._sigmoid(z_certainty)),
            "norm_violation": float(self._sigmoid(z_norm)),
            "agency": float(np.tanh(z_agency)),
        }
        self.appraisal_state = app
        return app

    def _emotion_from_appraisal(self, appraisal):
        impact = float(appraisal["impact"])
        novelty = float(appraisal["novelty"])
        control = float(appraisal["control"])
        certainty = float(appraisal["certainty"])
        norm_violation = float(appraisal["norm_violation"])
        agency = float(appraisal["agency"])

        logits = {
            "pride": 3.0 * impact + 1.5 * agency + 1.0 * control,
            "anger": -3.0 * impact
            - 2.0 * agency
            + 2.5 * norm_violation
            + 1.0 * novelty,
            "shame": -3.0 * impact + 2.0 * agency + 1.0 * certainty,
            "fear": -2.5 * impact + 1.8 * (1.0 - control) + 1.5 * (1.0 - certainty),
            "determination": -1.2 * impact + 2.0 * control + 1.2 * novelty,
        }
        emo = self._softmax_dict(logits, tau=self.tau_e)
        self.emotion_profile = emo
        return emo

    def _coping_from_appraisal_emotion(self, appraisal, emotion):
        control = float(appraisal["control"])
        certainty = float(appraisal["certainty"])
        norm_violation = float(appraisal["norm_violation"])
        agency = float(appraisal["agency"])
        pride = float(emotion["pride"])
        anger = float(emotion["anger"])
        shame = float(emotion["shame"])
        fear = float(emotion["fear"])
        determination = float(emotion["determination"])

        logits = {
            "planning": 1.5 * determination + 1.0 * control - 0.8 * fear,
            "self_correction": 1.5 * shame + 1.2 * agency + 0.8 * certainty,
            "external_blame": 1.8 * anger - 1.5 * agency + 1.2 * norm_violation,
        }
        base = self._softmax_dict(logits, tau=self.tau_c)
        risk_shift = float(
            np.tanh(1.2 * anger + 0.8 * determination - 1.0 * fear + 0.2 * pride)
        )
        coping = {
            "planning": base["planning"],
            "self_correction": base["self_correction"],
            "external_blame": base["external_blame"],
            "risk_shift": risk_shift,
        }
        self.coping_profile = coping
        return coping

    def _stage_weight(self, stage_pressure):
        return float(self._sigmoid(3.0 * float(stage_pressure) - 1.0))

    def _memory_salience(self, appraisal, emotion, stage_pressure, importance=1.0):
        impact = float(appraisal["impact"])
        novelty = float(appraisal["novelty"])
        max_emotion = float(max(emotion.values())) if emotion else 0.0
        stage_w = self._stage_weight(stage_pressure)
        z = (
            1.8 * abs(impact)
            + 1.2 * max_emotion
            + 1.0 * novelty
            + 1.4 * stage_w
            + 0.25 * float(importance)
            - 2.0
        )
        return float(self._sigmoid(z))

    @property
    def morale(self):
        return float(np.tanh(self.z_state["morale"]))

    @morale.setter
    def morale(self, value):
        self.z_state["morale"] = float(np.arctanh(np.clip(value, -0.999, 0.999)))
        self._project_latents_to_states()

    @property
    def hidden_state(self):
        # Compatibility read view: [morale, stability, media_pressure]
        morale = float(np.tanh(self.z_state["morale"]))
        stability = float(np.tanh(self.z_state["stability"]))
        media_pressure = float(
            2.0 * self._sigmoid(self.z_state["media_sensitivity"]) - 1.0
        )
        return np.array([morale, stability, media_pressure], dtype=float)

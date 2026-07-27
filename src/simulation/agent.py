import datetime
import random
import numpy as np
import json

from src.simulation.memory_service import (
    attribute_delayed_utility,
    build_provenance,
    memory_stream_view,
)

class SocietyAgent:
    @staticmethod
    def _finite(value, default=0.0):
        try:
            v = float(value)
        except Exception:
            return float(default)
        if not np.isfinite(v):
            return float(default)
        return v

    def __init__(self, name, stats, tactical_info=None):
        self.name = name
        self.team_name = name
        self.stats = stats
        self.tier = stats.get('tier', 'Semi-Core')
        raw_status = self._finite(stats.get('final_status_score', 50.0), 50.0)
        self.status_score = raw_status
        self.region = self._infer_region()
        self.personality = {
            "arrogance": min(0.95, max(0.1, self.status_score / 100.0)),
            "discipline": random.uniform(0.45, 0.95),
            "temper": random.uniform(0.2, 0.8),
        }

        gate = stats.get("global_exposure_gate")
        gate_to_exposure = {"pass": 0.9, "partial_pass": 0.6, "fail": 0.35}
        default_exposure = gate_to_exposure.get(gate, 0.5)
        self.media_exposure = self._finite(stats.get('global_exp', default_exposure * 100.0), default_exposure * 100.0) / 100.0
        self.media_exposure = float(np.clip(self.media_exposure, 0.05, 1.0))
        
        # --- HIERARCHICAL ROLES ---
        coach_payload = tactical_info.get("coach") if tactical_info else None
        self.coach_profile = None
        self.coach_name = "Head Coach"
        if coach_payload and isinstance(coach_payload, dict) and coach_payload.get("name"):
            self.coach_name = str(coach_payload["name"])

        self.roles = {
            "Manager": {"name": self.coach_name, "rationality": 0.9, "pressure": 0.0},
            "Icon": {"ego": random.uniform(0.6, 0.95), "patience": 0.8},
            "President": {"media_sensitivity": self.media_exposure}
        }
        
        # --- SYSTEM 1: RECURSIVE PSYCH (RNN) ---
        # Core state is stored in latent z_state (continuous, unbounded), then projected.
        self.W_h = 0.85 
        
        # --- TOURNAMENT MOMENTUM (Continuous Field) ---
        self.momentum = 0.0 # Cinderella Energy
        
        # --- RIVALRY & EXPOSURE ---
        self.rivalry_database = {} 
        self.media_filter = (stats.get('c5_pressure', 50) / 100.0) * 0.5
        self.W_x = 0.3 * (1.0 + self.media_exposure) 
        
        self.reflection_diary = ""
        self.psychology_profile = self._initialize_psychology_from_history(stats)
        self.formation = tactical_info.get('formation', '4-4-2') if tactical_info else "4-4-2"
        self.style_desc = tactical_info.get('style_desc', '') if tactical_info else ""
        self.style_archetype = self._infer_style_archetype()
        self.semantic_memory = {
            "identity": self.team_name,
            "tier": self.tier,
            "region": self.region,
            "style_archetype": self.style_archetype,
        }
        self.episodic_memory = []
        self.procedural_memory = []
        self.decision_memory = []
        self.memory_event_log = []
        self.memory_max_episodic = 420
        self.memory_decay_lambda = 0.10
        self.memory_clock = 0
        self.beliefs = []
        self.llm_reflection_audit = []
        self.fatigue = 0.0
        self.injury_list = []
        self.injury_load = 0.0
        self.readiness = 1.0
        self.tactical_controls = {
            "pressing_intensity": 0.50,
            "risk_budget": 0.50,
            "line_height": 0.50,
            "rotation_aggressiveness": 0.50,
        }
        # --- INTERNAL GAME: COACH vs PLAYER POWER BALANCE ---
        self.coach_authority = float(np.clip(0.52 + random.uniform(-0.08, 0.10), 0.20, 0.95))
        self.icon_influence = float(np.clip(self.roles["Icon"]["ego"] * 0.85, 0.20, 0.95))
        self.team_cohesion = float(np.clip(0.62 + random.uniform(-0.10, 0.12), 0.15, 0.98))
        self.conflict_heat = 0.12
        # --- REFEREE RELATION / PERCEIVED FAIRNESS ---
        self.referee_trust = float(np.clip(0.55 + random.uniform(-0.08, 0.08), 0.10, 0.98))
        self.referee_grievance = 0.05
        self.social_narrative_state = {
            "trust_index": 0.56,
            "polarization": 0.18,
            "narrative_fatigue": 0.10,
        }
        # --- Continuous Affective Appraisal Memory Framework params ---
        self.tau_e = 0.80
        self.tau_c = 0.85
        self.lambda_m = 0.08
        self.beta_m = 0.50
        self.gamma_retrieval = 4.0
        self.eta_retrieval = 1.0
        self.rho_b = 0.97
        self.k_alignment = 5.0
        self.alpha_conf = 2.0
        self.s_b = 8.0
        self.c0 = 0.50
        self.delta_max = 0.15
        self.rho_s = 0.85
        self.state_noise_sigma = 0.02
        self.eps = 1e-9

        self.appraisal_state = {
            "impact": 0.0,            # [-1, 1]
            "novelty": 0.5,           # [0, 1]
            "control": 0.5,           # [0, 1]
            "certainty": 0.5,         # [0, 1]
            "norm_violation": 0.2,    # [0, 1]
            "agency": 0.0,            # [-1, 1]
        }
        self.emotion_profile = {
            "pride": 0.2,
            "anger": 0.1,
            "shame": 0.1,
            "fear": 0.2,
            "determination": 0.4,
        }
        self.coping_profile = {
            "planning": 0.35,
            "self_correction": 0.25,
            "external_blame": 0.20,
            "risk_shift": 0.0,   # [-1, 1]
        }
        self.roles["Icon"]["patience"] = float(np.clip(0.45 + 0.55 * self.psychology_profile["resilience"], 0.1, 1.2))
        self.roles["President"]["media_sensitivity"] = float(np.clip(self.psychology_profile["media_sensitivity"], 0.05, 0.99))
        self.team_cohesion = float(np.clip(0.35 + 0.60 * self.psychology_profile["resilience"], 0.05, 0.99))
        self.z_state = self._initialize_latent_states()

    def _infer_region(self):
        europe = {
            "England", "Germany", "France", "Spain", "Portugal", "Netherlands", "Belgium",
            "Croatia", "Austria", "Switzerland", "Sweden", "Scotland", "Turkey",
            "Bosnia and Herzegovina", "Norway", "Czech Republic",
        }
        south_america = {
            "Argentina", "Brazil", "Uruguay", "Colombia", "Paraguay", "Chile", "Ecuador",
            "Peru", "Venezuela", "Bolivia",
        }
        north_america = {
            "United States", "Mexico", "Canada", "Panama", "Qatar", "Curaçao",
        }
        africa = {
            "Morocco", "Egypt", "Ivory Coast", "Senegal", "Ghana", "Algeria", "South Africa",
            "Tunisia", "DR Congo", "Cape Verde",
        }
        asia = {
            "Japan", "South Korea", "Saudi Arabia", "Iraq", "Iran", "Jordan", "Uzbekistan",
            "Australia",
        }
        oceania = {"New Zealand", "Haiti"}

        if self.team_name in europe:
            return "Europe"
        if self.team_name in south_america:
            return "South America"
        if self.team_name in north_america:
            return "North/Central America"
        if self.team_name in africa:
            return "Africa"
        if self.team_name in asia:
            return "Asia"
        if self.team_name in oceania:
            return "Oceania"
        return "Global"

    def _infer_style_archetype(self):
        from src.match_engine.tactical_catalog import infer_archetype_from_text

        return infer_archetype_from_text(self.style_desc, self.formation)

    def _initialize_psychology_from_history(self, stats):
        c1 = self._finite(stats.get("c1_win_rate", 50.0), 50.0) / 100.0
        c3 = self._finite(stats.get("c3_major_exp", 45.0), 45.0) / 100.0
        c5 = self._finite(stats.get("c5_pressure", 45.0), 45.0) / 100.0
        modern = self._finite(stats.get("modern_power", stats.get("final_status_score", 50.0)), 50.0) / 100.0
        exposure = float(np.clip(self.media_exposure, 0.05, 1.0))
        profile = {
            "confidence": float(np.clip(0.58 * c1 + 0.42 * modern, 0.05, 0.98)),
            "resilience": float(np.clip(0.50 * c3 + 0.25 * c1 + 0.25 * (1.0 - c5), 0.05, 0.98)),
            "stability": float(np.clip(0.60 * c3 + 0.40 * c1, 0.05, 0.98)),
            "media_sensitivity": float(np.clip(0.55 * exposure + 0.45 * c5, 0.05, 0.98)),
            "penalty_anxiety": float(np.clip(0.35 + 0.50 * c5 * (1.0 - c3), 0.05, 0.98)),
            "expectation_pressure": float(np.clip(0.55 * modern + 0.45 * exposure, 0.05, 0.99)),
            "underdog_identity": float(np.clip((1.0 - modern) * (0.65 + 0.35 * c1), 0.0, 0.98)),
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
            "morale": float(np.arctanh(np.clip((self.psychology_profile["confidence"] - 0.5) * 1.6, -0.999, 0.999))),
            "stability": float(np.arctanh(np.clip((self.psychology_profile["stability"] - 0.5) * 1.8, -0.999, 0.999))),
            "unity": self._logit01(self.team_cohesion),
            "confidence": self._logit01(self.psychology_profile.get("confidence", 0.5)),
            "risk_tolerance": self._logit01(self.tactical_controls.get("risk_budget", 0.5)),
            "pressing_intensity": self._logit01(self.tactical_controls.get("pressing_intensity", 0.5)),
            "referee_trust": self._logit01(self.referee_trust),
            "media_sensitivity": self._logit01(self.roles["President"]["media_sensitivity"]),
        }

    def _project_latents_to_states(self):
        self.team_cohesion = float(self._sigmoid(self.z_state["unity"]))
        self.referee_trust = float(self._sigmoid(self.z_state["referee_trust"]))
        self.roles["President"]["media_sensitivity"] = float(self._sigmoid(self.z_state["media_sensitivity"]))
        self.tactical_controls["risk_budget"] = float(self._sigmoid(self.z_state["risk_tolerance"]))
        self.tactical_controls["pressing_intensity"] = float(self._sigmoid(self.z_state["pressing_intensity"]))

    def _emotion_scalar(self):
        # Scalar intensity from canonical emotion profile.
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
        referee_controversy = float(np.clip(e.get("referee_controversy", 0.0), 0.0, 1.0))
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
        z_novelty = 0.60 * abs(score_diff) + 1.10 * upset_factor + 0.40 * referee_controversy - 0.30
        z_control = 0.90 * (self.coach_authority - self.icon_influence) + 0.70 * self.team_cohesion - 0.50 * self.conflict_heat
        z_certainty = 1.20 * abs(score_diff) + 0.80 * stage_pressure - 0.60
        z_norm = 1.60 * referee_controversy + 0.22 * max(0.0, social_chaos) - 0.70 * self.referee_trust
        z_agency = agency_raw + 0.35 * np.tanh(self.coach_authority - self.icon_influence)

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
            "anger": -3.0 * impact - 2.0 * agency + 2.5 * norm_violation + 1.0 * novelty,
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
        risk_shift = float(np.tanh(1.2 * anger + 0.8 * determination - 1.0 * fear + 0.2 * pride))
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

    def simulate_internal_game(self, stage_pressure=0.2):
        """
        Continuous internal game:
        - Coach authority vs icon influence determines strategic discipline.
        - Conflict heat rises with pressure and low stability.
        """
        pressure = float(np.clip(stage_pressure, 0.0, 1.0))
        stability = float(np.clip((self.hidden_state[1] + 1.0) / 2.0, 0.0, 1.0))
        fatigue_term = float(np.tanh(self.fatigue))
        icon_drive = self.icon_influence * (1.0 + 0.35 * pressure)
        coach_drive = self.coach_authority * (1.0 + 0.25 * self.roles["Manager"]["rationality"])
        negotiation_gap = coach_drive - icon_drive

        coordination = self._bounded_sigmoid(2.8 * negotiation_gap + 1.2 * self.team_cohesion - 1.1 * fatigue_term)
        tension_push = pressure * (1.0 - stability) * (0.55 + 0.45 * self.icon_influence)
        cooldown = 0.18 * coordination + 0.10 * self.team_cohesion
        self.conflict_heat = float(np.clip(self.conflict_heat * np.exp(-cooldown) + tension_push, 0.0, 1.05))
        self.team_cohesion = float(np.clip(self.team_cohesion * np.exp(-0.06 * self.conflict_heat) + 0.05 * coordination, 0.05, 0.99))

        performance_bonus = 5.2 * (coordination - 0.5) + 2.2 * (self.team_cohesion - 0.5)
        volatility = 0.25 + 0.55 * self.conflict_heat

        self._register_memory_event(
            content=f"Internal game: coordination={coordination:.2f}, conflict_heat={self.conflict_heat:.2f}",
            layer="episodic",
            importance=5.8 + 2.0 * pressure,
            emotion=min(1.0, 0.25 + self.conflict_heat * 0.45),
            tags=["internal_game", "coach_player"],
            write_temperature=1.05 + 0.4 * pressure,
        )
        return {
            "coordination": coordination,
            "performance_bonus": float(performance_bonus),
            "volatility": float(volatility),
            "conflict_heat": self.conflict_heat,
        }

    def apply_referee_dynamics(self, referee_strictness, bias_signal):
        """
        Referee-game coupling:
        - bias_signal in [-1, 1], >0 means calls mildly favor this team.
        - strictness in [0, 1], higher means harsher whistle intensity.
        """
        strictness = float(np.clip(self._finite(referee_strictness, 0.5), 0.0, 1.0))
        bias = float(np.clip(self._finite(bias_signal, 0.0), -1.0, 1.0))
        grievance_shock = 0.52 * strictness * (0.45 - 0.35 * bias) * (1.0 + self.conflict_heat * 0.35)
        trust_repair = 0.10 + 0.18 * max(0.0, bias)

        self.referee_grievance = float(np.clip(self.referee_grievance * np.exp(-0.16) + grievance_shock, 0.0, 0.92))
        self.referee_trust = float(np.clip(self.referee_trust * np.exp(-0.10 * strictness) + trust_repair, 0.05, 0.99))

        whistle_drag = 4.8 * strictness * (0.55 + self.referee_grievance) * (1.0 - 0.35 * max(0.0, bias))
        decision_lift = 2.0 * bias * self.referee_trust
        status_delta = float(decision_lift - whistle_drag)
        chaos_delta = float(0.35 + 1.4 * self.referee_grievance - 0.4 * self.referee_trust)

        self._register_memory_event(
            content=f"Referee dynamics: strict={strictness:.2f}, bias={bias:.2f}, grievance={self.referee_grievance:.2f}",
            layer="episodic",
            importance=6.2 + 1.8 * strictness,
            emotion=min(1.0, 0.20 + abs(bias) * 0.25 + self.referee_grievance * 0.35),
            tags=["referee_game"],
            write_temperature=1.0 + strictness * 0.3,
        )
        return {
            "status_delta": status_delta,
            "chaos_delta": chaos_delta,
            "grievance": self.referee_grievance,
        }

    def relax_referee_grievance_post_match(self, match_result, referee_bias_signal, drama_score):
        """
        Settlement after whistle + chatter: grievance shouldn't ratchet forever.
        Wins, calm fixtures, evenly-called games, or sides that were mildly favored bleed heat faster.
        """
        bias = float(np.clip(referee_bias_signal, -1.0, 1.0))
        drama = float(np.clip(drama_score, 0.0, 1.0))
        factor = np.exp(-0.10)
        factor *= np.exp(-0.065) if match_result == "win" else np.exp(-0.035) if match_result == "draw" else 1.0
        factor *= np.exp(-0.09 * max(0.0, bias))
        factor *= np.exp(-0.05 * drama)
        factor *= np.exp(-0.05 * np.exp(-3.6 * bias**2))
        self.referee_grievance = float(np.clip(self.referee_grievance * factor, 0.0, 0.92))

    def update_governance_post_match(self, match_result, drama_score, governance_signal):
        outcome = {"win": 0.22, "draw": 0.04, "loss": -0.26}.get(match_result, 0.0)
        signal = self._finite(governance_signal, 0.0)
        drama = float(np.clip(self._finite(drama_score, 0.0), 0.0, 1.5))

        self.coach_authority = float(np.clip(self.coach_authority + 0.08 * outcome + 0.05 * signal - 0.03 * drama, 0.05, 0.98))
        self.icon_influence = float(np.clip(self.icon_influence - 0.05 * outcome + 0.04 * drama + 0.03 * self.conflict_heat, 0.05, 0.98))
        self.team_cohesion = float(np.clip(self.team_cohesion + 0.06 * outcome - 0.04 * self.conflict_heat + 0.03 * signal, 0.05, 0.99))

        self._register_memory_event(
            content=(
                f"Governance shift after {match_result}: coach={self.coach_authority:.2f}, "
                f"icon={self.icon_influence:.2f}, cohesion={self.team_cohesion:.2f}"
            ),
            layer="procedural",
            importance=6.8 + 2.0 * abs(outcome),
            emotion=min(1.0, 0.25 + drama * 0.4),
            tags=["governance_update"],
            write_temperature=1.1,
        )

    def _bounded_sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def _clip01(self, value):
        try:
            return float(np.clip(float(value), 0.0, 1.0))
        except Exception:
            return 0.5

    def set_tactical_controls(self, controls):
        controls = controls or {}
        self.tactical_controls["pressing_intensity"] = self._clip01(
            controls.get("pressing_intensity", self.tactical_controls["pressing_intensity"])
        )
        self.tactical_controls["risk_budget"] = self._clip01(
            controls.get("risk_budget", self.tactical_controls["risk_budget"])
        )
        self.tactical_controls["line_height"] = self._clip01(
            controls.get("line_height", self.tactical_controls["line_height"])
        )
        self.tactical_controls["rotation_aggressiveness"] = self._clip01(
            controls.get("rotation_aggressiveness", self.tactical_controls["rotation_aggressiveness"])
        )
        if getattr(self, "_tactical_vector", None) is None:
            self._tactical_vector = {}
        for k, v in controls.items():
            if k in self._tactical_vector or k not in self.tactical_controls:
                try:
                    self._tactical_vector[k] = self._clip01(float(v))
                except (TypeError, ValueError):
                    pass

    def refresh_tactical_vector(self):
        from src.match_engine.tactical_profile import sync_agent_controls_from_vector

        self._tactical_vector = sync_agent_controls_from_vector(self)

    def tactical_effects(self):
        p = self.tactical_controls["pressing_intensity"]
        r = self.tactical_controls["risk_budget"]
        h = self.tactical_controls["line_height"]
        rot = self.tactical_controls["rotation_aggressiveness"]

        status_delta = (
            4.0 * (p - 0.5)
            + 3.2 * (h - 0.5)
            + 2.6 * (r - 0.5)
            - 2.0 * max(0.0, h - 0.75) * (1.0 - p)
        )
        volatility = 0.30 + 0.85 * r + 0.35 * abs(h - 0.5)
        fatigue_load_multiplier = 1.0 + 0.60 * p + 0.35 * r - 0.45 * rot
        chaos_push = 0.25 + 0.70 * r + 0.20 * p
        discipline = 1.0 - 0.45 * r + 0.25 * rot
        return {
            "status_delta": float(status_delta),
            "volatility": float(max(0.15, volatility)),
            "fatigue_load_multiplier": float(max(0.55, fatigue_load_multiplier)),
            "chaos_push": float(max(0.0, chaos_push)),
            "discipline": float(np.clip(discipline, 0.4, 1.3)),
        }

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

        self.memory_clock += 1
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
            "valence": float((appraisal or self.appraisal_state).get("impact", 0.0)),
            "arousal": float((appraisal or self.appraisal_state).get("novelty", 0.5)),
            "confidence": float(self._sigmoid(self.z_state["confidence"])),
            "novelty": float(novelty),
            "speaker": speaker,
            "tags": tags or [],
            "opponent": opponent,
            "affected_states": affected_states or {},
            "summary": summary or str(content),
            "appraisal": appraisal or dict(self.appraisal_state),
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
        rec["salience"] = float(rec["memory_weight"])  # compatibility alias

        if layer == "episodic":
            self.episodic_memory.append(rec)
        elif layer == "procedural":
            self.procedural_memory.append(rec)
        else:
            # Semantic updates are merged to long-term profile.
            self.semantic_memory[str(tags[0] if tags else f"semantic_{self.memory_clock}")] = content

        self.memory_event_log.append(
            {"step": self.memory_clock, "layer": layer, "p_write": float(p_write), "id": rec["id"]}
        )
        self._decay_and_prune_memory()
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

    def record_decision_event(self, opponent, stage_name, controls, outcomes, opponent_style=None, stage_pressure=None):
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
            },
            force_write=True,
        )
        if mem_rec is not None:
            rec["step"] = mem_rec["created_step"]
            rec["evidence_id"] = mem_rec.get("id")
            self.update_beliefs_from_match_event({"memory_rec": mem_rec})

    def retrieve_similar_decision_memories(self, controls, top_k=4, current_stage=None, current_opponent_style=None):
        if not self.decision_memory:
            return []
        target = self._control_vector(controls)
        stage_target = current_stage or (self.decision_memory[-1].get("stage") if self.decision_memory else None)
        style_target = current_opponent_style or (self.decision_memory[-1].get("opponent_style") if self.decision_memory else None)
        scored = []
        for rec in self.decision_memory:
            vec = self._control_vector(rec.get("controls", {}))
            dist = float(np.linalg.norm(target - vec))
            out = rec.get("outcomes", {})
            utility = (
                0.9 * float(out.get("goal_diff", 0.0))
                + 0.25 * float(out.get("xg_for", 0.0) - out.get("xg_against", 0.0))
                - 0.7 * float(out.get("fatigue_delta", 0.0))
                - 0.15 * abs(float(out.get("chaos", 0.0)))
            )
            # Stage and style context weighting for better transfer.
            stage_bonus = 0.18 if stage_target and rec.get("stage") == stage_target else 0.0
            style_bonus = 0.20 if style_target and rec.get("opponent_style") == style_target else 0.0
            score = utility - 1.25 * dist + stage_bonus + style_bonus
            scored.append((score, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[: max(1, top_k)]]

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

    def _memory_vector(self, rec):
        app = rec.get("appraisal", {})
        emo = rec.get("emotion_profile", {})
        cop = rec.get("coping", {})
        return np.array(
            [
                float(app.get("impact", 0.0)),
                float(app.get("novelty", 0.5)),
                float(app.get("control", 0.5)),
                float(app.get("certainty", 0.5)),
                float(app.get("norm_violation", 0.0)),
                float(app.get("agency", 0.0)),
                float(emo.get("pride", 0.0)),
                float(emo.get("anger", 0.0)),
                float(emo.get("shame", 0.0)),
                float(emo.get("fear", 0.0)),
                float(emo.get("determination", 0.0)),
                float(cop.get("planning", 0.0)),
                float(cop.get("self_correction", 0.0)),
                float(cop.get("external_blame", 0.0)),
                float(cop.get("risk_shift", 0.0)),
            ],
            dtype=float,
        )

    def _cosine_similarity(self, a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na <= self.eps or nb <= self.eps:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _query_vector(self, opponent=None, stage_pressure=0.3):
        app = dict(self.appraisal_state)
        app["novelty"] = float(np.clip(0.6 * app.get("novelty", 0.5) + 0.4 * stage_pressure, 0.0, 1.0))
        if opponent is not None:
            app["agency"] = float(np.tanh(app.get("agency", 0.0) + 0.15 * self.rivalry_database.get(opponent, 0.0)))
        rec = {
            "appraisal": app,
            "emotion_profile": dict(self.emotion_profile),
            "coping": dict(self.coping_profile),
        }
        return self._memory_vector(rec)

    def retrieve_memory_context(self, top_k=None, opponent=None):
        if not self.episodic_memory:
            return []
        q = self._query_vector(opponent=opponent, stage_pressure=0.3)
        logits = []
        recs = []
        for rec in self.episodic_memory:
            sim = self._cosine_similarity(q, self._memory_vector(rec))
            mw = float(max(self.eps, rec.get("memory_weight", rec.get("salience", 0.0))))
            logit = self.gamma_retrieval * sim + self.eta_retrieval * np.log(mw + self.eps)
            if opponent and rec.get("opponent") == opponent:
                logit += 0.15
            logits.append(logit)
            recs.append(rec)
        arr = np.array(logits, dtype=float)
        arr = arr - np.max(arr)
        w = np.exp(arr)
        w = w / max(self.eps, float(np.sum(w)))
        weighted = []
        for ww, rec in zip(w.tolist(), recs):
            rc = dict(rec)
            rc["retrieval_weight"] = float(ww)
            weighted.append(rc)
        weighted.sort(key=lambda r: r.get("retrieval_weight", 0.0), reverse=True)
        selected = weighted if top_k is None else weighted[: max(1, top_k)]
        selected_ids = {record.get("id") for record in selected}
        for record in self.episodic_memory:
            if record.get("id") in selected_ids:
                record["retrieval_count"] = int(record.get("retrieval_count", 0)) + 1
        if top_k is None:
            return selected
        # top_k is only for display/export convenience, not for core weighting math.
        return selected

    def record_memory_utility(self, memory_ids, utility):
        """Attribute delayed downstream utility to retrieved evidence."""
        return attribute_delayed_utility(
            self.episodic_memory + self.procedural_memory,
            memory_ids or [], utility,
        )

    def retrieve_memory_context_display(self, top_k=10, opponent=None):
        return self.retrieve_memory_context(top_k=max(1, int(top_k)), opponent=opponent)

    def retrieve_memory_by_tags(self, tags, top_k=8):
        tags = set(tags or [])
        if not tags:
            return self.retrieve_memory_context_display(top_k=top_k)
        base = self.retrieve_memory_context(top_k=None)
        for rec in base:
            overlap = len(tags & set(rec.get("tags", [])))
            rec["retrieval_weight"] = float(rec.get("retrieval_weight", 0.0) * (1.0 + 0.35 * overlap))
        base.sort(key=lambda r: r.get("retrieval_weight", 0.0), reverse=True)
        return base[: max(1, top_k)]

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
                        "prototype": np.random.normal(0.0, 0.05, size=m_vec.shape[0]).tolist(),
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

    def recursive_update(self, match_result, score_diff, prof_score, social_chaos, opp_name, opp_status):
        """
        Final V13 Logic: 
        - Rivalry amplification persists.
        - Momentum field flips social impact for 'Cinderella' teams.
        """
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

        # Latent state update (continuous, bounded by tanh/sigmoid projection only).
        noise = np.random.normal(0.0, self.state_noise_sigma, size=8)
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

    def perform_reflection(self, llm):
        print(f"\n  [V13 SUMMIT] {self.name}: Reflecting on the momentum and power...")
        memory_context = self.retrieve_memory_context_display(top_k=8)
        stage_hint = self.decision_memory[-1].get("stage") if self.decision_memory else None
        style_hint = self.decision_memory[-1].get("opponent_style") if self.decision_memory else None
        similar_decisions = self.retrieve_similar_decision_memories(
            self.tactical_controls,
            top_k=4,
            current_stage=stage_hint,
            current_opponent_style=style_hint,
        )
        reflection_payload = {
            "high_salience_memory": memory_context,
            "similar_decision_outcomes": similar_decisions,
            "current_controls": self.tactical_controls,
        }
        verdict_str = llm.perform_agent_reflection(
            self.name,
            self.momentum,
            self.hidden_state.tolist(),
            self.roles["Icon"]["patience"],
            memory_context=json.dumps(reflection_payload, ensure_ascii=False),
        )
        try:
            data = json.loads(verdict_str)
            self.apply_llm_reflection(data)
            self._register_memory_event(
                content=f"Reflection update: {self.reflection_diary[:160]}",
                layer="procedural",
                importance=8.0,
                emotion=self._emotion_scalar(),
                event_type="reflection_update",
                tags=["reflection"],
                write_temperature=1.25,
                metadata={"latest_reflection_audit": self.llm_reflection_audit[-1] if self.llm_reflection_audit else {}},
            )
            self.consolidate_memories()
        except Exception as e:
            print(f"  [ERROR] Reflection parsing failed for {self.name}: {e}")

    def ingest_micro_cognitive_memory(self, cognitive_plans: list, team_name: str = "") -> None:
        """Absorb in-match System 2 narratives into episodic memory for cross-match continuity."""
        if not cognitive_plans:
            return
        for rec in cognitive_plans:
            if not isinstance(rec, dict):
                continue
            trig = rec.get("trigger") or {}
            if team_name and trig.get("team_id") and trig.get("team_id") != team_name:
                if trig.get("entity_tier") == "coach" and team_name not in str(trig.get("entity_id", "")):
                    continue
            plan = rec.get("plan") or {}
            narrative = str(plan.get("narrative", "") or plan.get("reasoning", "")).strip()
            if len(narrative) < 8:
                continue
            self._register_memory_event(
                content=f"In-match ({trig.get('kind', 'event')}): {narrative[:200]}",
                layer="episodic",
                importance=6.0 + 2.0 * float(trig.get("salience", 0)),
                emotion=self._emotion_scalar(),
                event_type="micro_cognitive",
                tags=["cognitive", str(trig.get("kind", ""))],
                write_temperature=1.1,
                metadata={"trigger": trig, "applied": rec.get("applied", False)},
            )

    def apply_llm_reflection(self, reflection):
        from src.simulation.meta_learning import MetaLearningController

        audit_log = MetaLearningController().apply(self, reflection)
        self.llm_reflection_audit.append(audit_log)
        return audit_log

    def locker_room_tension(self):
        """Continuous 0..1 display tension; logistic squash keeps most crews mid-band, extremes reserved."""
        stability = float(self.hidden_state[1])
        patience = float(self.roles["Icon"]["patience"])
        ch = float(self.conflict_heat)
        media = float(np.clip(self.media_exposure, 0.0, 1.2))
        latent = np.clip(
            0.36 * (1.0 - np.tanh(0.95 * stability))
            + 0.29 * np.clip(ch / 1.05, 0.0, 1.0) ** 0.88
            + 0.11 * np.clip((1.0 - np.tanh(patience)), 0.0, 1.0)
            + 0.07 * np.clip(media - 0.35, 0.0, 0.95),
            0.0,
            1.05,
        )
        # Shift baseline lower so normal-state tension has more narrative separation.
        k, mid = 7.20, 0.60
        sig = float(1.0 / (1.0 + np.exp(-k * (latent - mid))))
        smoothed = 0.22 + 0.62 * sig
        return float(np.clip(smoothed, 0.0, 1.0))

    def rare_locker_room_explosion(self, rng=None):
        """Headline-level crisis; intentionally rare so it keeps punch."""
        rng = np.random if rng is None else rng
        t = self.locker_room_tension()
        if t < 0.86:
            return False
        p = ((t - 0.86) ** 2.15) * 0.085 + 0.002
        return bool(rng.random() < p)

    def apply_match_wear(self, intensity=0.3):
        # Continuous wear model (no hard thresholds):
        # fatigue follows exponential smoothing; injury_load follows hazard-driven stochastic drift.
        effective_intensity = float(intensity) * (1.0 - 0.35 * np.tanh(self.momentum))
        self.fatigue = 0.82 * self.fatigue + effective_intensity

        fatigue_pressure = 1.0 / (1.0 + np.exp(-3.2 * (self.fatigue - 0.55)))
        hazard = fatigue_pressure * (0.35 + 0.65 * effective_intensity) * (1.0 + 0.5 * self.injury_load)
        shock = np.random.gamma(shape=1.4, scale=max(1e-6, hazard * 0.30))

        # Keep injury load in a normalized [0, 1] band for stable downstream interpretation.
        self.injury_load = float(np.clip(self.injury_load * np.exp(-0.10) + np.tanh(shock), 0.0, 1.0))
        self.readiness = float(np.exp(-0.9 * self.fatigue - 1.35 * self.injury_load))

        event_rate = max(0.0, hazard * 1.6)
        events = np.random.poisson(event_rate)
        if events > 0:
            note = f"Medical load increased: hazard={hazard:.2f}, injury_load={self.injury_load:.2f}"
            self.injury_list.append(note)
            self.add_memory(note, importance=7)
            return "Medical Load Spike"
        return None

    def recover(self, rest_units=1.0):
        rest_units = max(0.0, float(rest_units))
        self.fatigue = self.fatigue * np.exp(-0.34 * rest_units)
        self.injury_load = float(np.clip(self.injury_load * np.exp(-0.22 * rest_units), 0.0, 1.0))
        self.readiness = float(np.exp(-0.9 * self.fatigue - 1.35 * self.injury_load))

    def get_effective_status(self, matchup_bonus=0.0):
        # Smooth match-day strength mapping with logistic envelope.
        morale_bonus = 0.07 * np.tanh(self.hidden_state[0])
        stability_bonus = 0.05 * np.tanh(self.hidden_state[1])
        governance_term = 0.12 * (self.coach_authority - self.icon_influence) + 0.10 * self.team_cohesion - 0.08 * self.conflict_heat
        condition_core = (
            1.0
            - 0.42 * np.tanh(self.fatigue)
            - 0.48 * np.tanh(self.injury_load)
            + morale_bonus
            + stability_bonus
            + governance_term
        )
        multiplier = 0.52 + 0.56 * (1.0 / (1.0 + np.exp(-3.0 * (condition_core - 0.64))))
        return max(12.0, self.status_score * multiplier + matchup_bonus)

    def coach_intervention(self, verdict_str):
        try:
            parts = verdict_str.split('|')
            for p in parts:
                if 'Formation:' in p: self.formation = p.split(':')[1].strip()
        except: pass

    def get_context_for_llm(self):
        hs = self.hidden_state
        morale = "High" if hs[0] > 0.4 else "Low"
        top_beliefs = [b.get("claim", "")[:60] for b in sorted(self.beliefs, key=lambda x: x.get("confidence", 0.0), reverse=True)[:2]]
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
        media_pressure = float(2.0 * self._sigmoid(self.z_state["media_sensitivity"]) - 1.0)
        return np.array([morale, stability, media_pressure], dtype=float)

    def update_rivalry(self, opp_name, drama_score):
        current = self.rivalry_database.get(opp_name, 0.0)
        self.rivalry_database[opp_name] = current + (drama_score * 0.1)

    def decide_action(self, feed):
        if not feed.posts:
            return "POST"
        base_post_prob = 0.4 + (self.personality["arrogance"] - 0.5) * 0.3
        if self.hidden_state[2] > 0.3:
            base_post_prob += 0.1
        return "POST" if random.random() < np.clip(base_post_prob, 0.15, 0.85) else "REPLY"

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

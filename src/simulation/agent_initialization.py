"""Ordered initialization stages for SocietyAgent."""

import random

import numpy as np


class AgentInitializationMixin:
    def _initialize_identity(self, name, stats):
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

    def _initialize_roles_and_strategy(self, stats, tactical_info):
        
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

    def _initialize_runtime_state(self):
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

    def _initialize_affective_state(self):
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


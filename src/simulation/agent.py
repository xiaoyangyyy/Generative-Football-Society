import numpy as np
import json

from src.simulation.agent_memory import AgentMemoryMixin
from src.simulation.agent_dynamics import AgentMatchDynamicsMixin
from src.simulation.agent_initialization import AgentInitializationMixin
from src.simulation.agent_social import SocialAgentMixin
from src.simulation.random_control import named_py_rng, named_rng


class SocietyAgent(
    AgentInitializationMixin,
    AgentMatchDynamicsMixin,
    AgentMemoryMixin,
    SocialAgentMixin,
):
    @staticmethod
    def _finite(value, default=0.0):
        try:
            v = float(value)
        except Exception:
            return float(default)
        if not np.isfinite(v):
            return float(default)
        return v

    def __init__(self, name, stats, tactical_info=None,
                 initialization_rng=None, random_root_seed=42):
        self.random_root_seed = int(random_root_seed)
        self._initialization_rng = initialization_rng or named_py_rng(
            self.random_root_seed, "initialization", name)
        self._initialize_identity(name, stats)
        self._initialize_roles_and_strategy(stats, tactical_info)
        self._initialize_runtime_state()
        self._initialize_affective_state()
        del self._initialization_rng

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


    def _reflection_context_payload(self):
        memory_context = self.retrieve_memory_context_display(top_k=8)
        stage_hint = self.decision_memory[-1].get("stage") if self.decision_memory else None
        style_hint = self.decision_memory[-1].get("opponent_style") if self.decision_memory else None
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
            previous = next((
                record for record in self.llm_reflection_audit
                if isinstance(record, dict)
                and record.get("operation_id") == operation_id
            ), None)
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

    def apply_llm_reflection(self, reflection, *, operation_id=None):
        from src.simulation.meta_learning import MetaLearningController

        audit_log = MetaLearningController().apply(
            self, reflection, operation_id=operation_id,
        )
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
        rng = rng or named_rng(
            self.random_root_seed, "locker_room_explosion", self.memory_clock,
        )
        t = self.locker_room_tension()
        if t < 0.86:
            return False
        p = ((t - 0.86) ** 2.15) * 0.085 + 0.002
        return bool(rng.random() < p)

    def apply_match_wear(self, intensity=0.3, *, rng=None):
        # Continuous wear model (no hard thresholds):
        # fatigue follows exponential smoothing; injury_load follows hazard-driven stochastic drift.
        effective_intensity = float(intensity) * (1.0 - 0.35 * np.tanh(self.momentum))
        self.fatigue = 0.82 * self.fatigue + effective_intensity

        fatigue_pressure = 1.0 / (1.0 + np.exp(-3.2 * (self.fatigue - 0.55)))
        hazard = fatigue_pressure * (0.35 + 0.65 * effective_intensity) * (1.0 + 0.5 * self.injury_load)
        rng = rng or named_rng(
            self.random_root_seed, "match_wear", self.memory_clock,
        )
        shock = rng.gamma(shape=1.4, scale=max(1e-6, hazard * 0.30))

        # Keep injury load in a normalized [0, 1] band for stable downstream interpretation.
        self.injury_load = float(np.clip(self.injury_load * np.exp(-0.10) + np.tanh(shock), 0.0, 1.0))
        self.readiness = float(np.exp(-0.9 * self.fatigue - 1.35 * self.injury_load))

        event_rate = max(0.0, hazard * 1.6)
        events = rng.poisson(event_rate)
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

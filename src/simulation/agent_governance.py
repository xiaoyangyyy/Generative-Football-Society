"""Own internal power, referee-trust and post-match governance dynamics."""

import numpy as np


class AgentGovernanceMixin:
    """Own internal power, referee-trust and post-match governance dynamics."""

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
        coach_drive = self.coach_authority * (
            1.0 + 0.25 * self.roles["Manager"]["rationality"]
        )
        negotiation_gap = coach_drive - icon_drive

        coordination = self._bounded_sigmoid(
            2.8 * negotiation_gap + 1.2 * self.team_cohesion - 1.1 * fatigue_term
        )
        tension_push = (
            pressure * (1.0 - stability) * (0.55 + 0.45 * self.icon_influence)
        )
        cooldown = 0.18 * coordination + 0.10 * self.team_cohesion
        self.conflict_heat = float(
            np.clip(self.conflict_heat * np.exp(-cooldown) + tension_push, 0.0, 1.05)
        )
        self.team_cohesion = float(
            np.clip(
                self.team_cohesion * np.exp(-0.06 * self.conflict_heat)
                + 0.05 * coordination,
                0.05,
                0.99,
            )
        )

        performance_bonus = 5.2 * (coordination - 0.5) + 2.2 * (
            self.team_cohesion - 0.5
        )
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
        grievance_shock = (
            0.52 * strictness * (0.45 - 0.35 * bias) * (1.0 + self.conflict_heat * 0.35)
        )
        trust_repair = 0.10 + 0.18 * max(0.0, bias)

        self.referee_grievance = float(
            np.clip(self.referee_grievance * np.exp(-0.16) + grievance_shock, 0.0, 0.92)
        )
        self.referee_trust = float(
            np.clip(
                self.referee_trust * np.exp(-0.10 * strictness) + trust_repair,
                0.05,
                0.99,
            )
        )

        whistle_drag = (
            4.8
            * strictness
            * (0.55 + self.referee_grievance)
            * (1.0 - 0.35 * max(0.0, bias))
        )
        decision_lift = 2.0 * bias * self.referee_trust
        status_delta = float(decision_lift - whistle_drag)
        chaos_delta = float(
            0.35 + 1.4 * self.referee_grievance - 0.4 * self.referee_trust
        )

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

    def relax_referee_grievance_post_match(
        self, match_result, referee_bias_signal, drama_score
    ):
        """
        Settlement after whistle + chatter: grievance shouldn't ratchet forever.
        Wins, calm fixtures, evenly-called games, or sides that were mildly favored bleed heat faster.
        """
        bias = float(np.clip(referee_bias_signal, -1.0, 1.0))
        drama = float(np.clip(drama_score, 0.0, 1.0))
        factor = np.exp(-0.10)
        factor *= (
            np.exp(-0.065)
            if match_result == "win"
            else np.exp(-0.035)
            if match_result == "draw"
            else 1.0
        )
        factor *= np.exp(-0.09 * max(0.0, bias))
        factor *= np.exp(-0.05 * drama)
        factor *= np.exp(-0.05 * np.exp(-3.6 * bias**2))
        self.referee_grievance = float(
            np.clip(self.referee_grievance * factor, 0.0, 0.92)
        )

    def update_governance_post_match(
        self, match_result, drama_score, governance_signal
    ):
        outcome = {"win": 0.22, "draw": 0.04, "loss": -0.26}.get(match_result, 0.0)
        signal = self._finite(governance_signal, 0.0)
        drama = float(np.clip(self._finite(drama_score, 0.0), 0.0, 1.5))

        self.coach_authority = float(
            np.clip(
                self.coach_authority + 0.08 * outcome + 0.05 * signal - 0.03 * drama,
                0.05,
                0.98,
            )
        )
        self.icon_influence = float(
            np.clip(
                self.icon_influence
                - 0.05 * outcome
                + 0.04 * drama
                + 0.03 * self.conflict_heat,
                0.05,
                0.98,
            )
        )
        self.team_cohesion = float(
            np.clip(
                self.team_cohesion
                + 0.06 * outcome
                - 0.04 * self.conflict_heat
                + 0.03 * signal,
                0.05,
                0.99,
            )
        )

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

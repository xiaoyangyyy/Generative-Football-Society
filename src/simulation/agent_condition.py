"""Own locker-room risk, physical wear, recovery and effective status."""

import numpy as np

from src.simulation.random_control import named_rng


class AgentConditionMixin:
    """Own locker-room risk, physical wear, recovery and effective status."""

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
            self.random_root_seed,
            "locker_room_explosion",
            self.memory_clock,
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
        hazard = (
            fatigue_pressure
            * (0.35 + 0.65 * effective_intensity)
            * (1.0 + 0.5 * self.injury_load)
        )
        rng = rng or named_rng(
            self.random_root_seed,
            "match_wear",
            self.memory_clock,
        )
        shock = rng.gamma(shape=1.4, scale=max(1e-6, hazard * 0.30))

        # Keep injury load in a normalized [0, 1] band for stable downstream interpretation.
        self.injury_load = float(
            np.clip(self.injury_load * np.exp(-0.10) + np.tanh(shock), 0.0, 1.0)
        )
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
        self.injury_load = float(
            np.clip(self.injury_load * np.exp(-0.22 * rest_units), 0.0, 1.0)
        )
        self.readiness = float(np.exp(-0.9 * self.fatigue - 1.35 * self.injury_load))

    def get_effective_status(self, matchup_bonus=0.0):
        # Smooth match-day strength mapping with logistic envelope.
        morale_bonus = 0.07 * np.tanh(self.hidden_state[0])
        stability_bonus = 0.05 * np.tanh(self.hidden_state[1])
        governance_term = (
            0.12 * (self.coach_authority - self.icon_influence)
            + 0.10 * self.team_cohesion
            - 0.08 * self.conflict_heat
        )
        condition_core = (
            1.0
            - 0.42 * np.tanh(self.fatigue)
            - 0.48 * np.tanh(self.injury_load)
            + morale_bonus
            + stability_bonus
            + governance_term
        )
        multiplier = 0.52 + 0.56 * (
            1.0 / (1.0 + np.exp(-3.0 * (condition_core - 0.64)))
        )
        return max(12.0, self.status_score * multiplier + matchup_bonus)

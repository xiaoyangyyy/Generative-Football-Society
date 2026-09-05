"""Own bounded tactical controls and coach interventions."""

import numpy as np


class AgentTacticsMixin:
    """Own bounded tactical controls and coach interventions."""

    def _clip01(self, value):
        try:
            return float(np.clip(float(value), 0.0, 1.0))
        except Exception:
            return 0.5

    def set_tactical_controls(self, controls):
        controls = controls or {}
        self.tactical_controls["pressing_intensity"] = self._clip01(
            controls.get(
                "pressing_intensity", self.tactical_controls["pressing_intensity"]
            )
        )
        self.tactical_controls["risk_budget"] = self._clip01(
            controls.get("risk_budget", self.tactical_controls["risk_budget"])
        )
        self.tactical_controls["line_height"] = self._clip01(
            controls.get("line_height", self.tactical_controls["line_height"])
        )
        self.tactical_controls["rotation_aggressiveness"] = self._clip01(
            controls.get(
                "rotation_aggressiveness",
                self.tactical_controls["rotation_aggressiveness"],
            )
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

    def coach_intervention(self, verdict_str):
        if not isinstance(verdict_str, str):
            return
        marker = "Formation:"
        for part in verdict_str.split("|"):
            if marker not in part:
                continue
            formation = part.split(marker, 1)[1].strip()
            if formation:
                self.formation = formation

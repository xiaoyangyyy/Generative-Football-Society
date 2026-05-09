import numpy as np


class FusionController:
    """
    Multi-system hierarchical fusion:
    - each subsystem exports compact expert signals
    - a single fusion layer computes final match-day adjustments
    """

    def __init__(self):
        # Context gate parameters (kept small, interpretable).
        self.base_logits = {
            "phys": 0.05,
            "affect": 0.10,
            "social": 0.08,
            "tactic": 0.16,
            "governance": 0.10,
        }

    def _softmax(self, logits_dict):
        keys = list(logits_dict.keys())
        arr = np.array([float(logits_dict[k]) for k in keys], dtype=float)
        arr = arr - np.max(arr)
        exps = np.exp(arr)
        probs = exps / max(1e-9, float(np.sum(exps)))
        return {k: float(p) for k, p in zip(keys, probs)}

    def _clip01(self, x):
        return float(np.clip(float(x), 0.0, 1.0))

    def build_context(self, stage_pressure, referee_strictness, social_chaos_hint):
        return {
            "stage_pressure": float(np.clip(stage_pressure, 0.0, 1.0)),
            "referee_strictness": float(np.clip(referee_strictness, 0.0, 1.0)),
            "social_chaos_hint": float(social_chaos_hint),
        }

    def _gate_weights(self, context):
        p = context["stage_pressure"]
        strict = context["referee_strictness"]
        chaos = float(np.tanh(context["social_chaos_hint"] / 5.0))
        logits = dict(self.base_logits)
        logits["phys"] += 0.38 * p
        logits["affect"] += 0.42 * p + 0.25 * abs(chaos)
        logits["social"] += 0.35 * strict + 0.45 * abs(chaos)
        logits["tactic"] += 0.22 * (1.0 - p) + 0.10 * strict
        logits["governance"] += 0.30 * p + 0.20 * strict
        return self._softmax(logits)

    def export_expert_signals(self, agent, internal_game, referee_game, tactical):
        phys_signal = {
            "status": -2.6 * np.tanh(agent.fatigue) - 3.1 * np.tanh(agent.injury_load) + 1.2 * agent.readiness,
            "volatility": 0.30 + 0.25 * np.tanh(agent.fatigue),
            "chaos": 0.15 + 0.45 * np.tanh(agent.injury_load),
        }
        emo = getattr(agent, "emotion_profile", {})
        affect_signal = {
            "status": 2.4 * float(emo.get("pride", 0.0)) - 1.9 * float(emo.get("fear", 0.0)) - 1.4 * float(emo.get("shame", 0.0)),
            "volatility": 0.20 + 0.45 * float(emo.get("anger", 0.0)) + 0.30 * float(emo.get("fear", 0.0)),
            "chaos": 0.30 + 0.60 * float(emo.get("anger", 0.0)),
        }
        social_signal = {
            "status": 1.2 * max(0.0, agent.referee_trust - 0.5) - 1.8 * agent.referee_grievance - 1.3 * agent.conflict_heat,
            "volatility": 0.25 + 0.55 * agent.conflict_heat + 0.25 * agent.referee_grievance,
            "chaos": 0.25 + 1.2 * agent.referee_grievance + 0.60 * agent.conflict_heat,
        }
        tactic_signal = {
            "status": float(tactical.get("status_delta", 0.0)),
            "volatility": float(tactical.get("volatility", 0.25)),
            "chaos": float(tactical.get("chaos_push", 0.0)),
        }
        gov_signal = {
            "status": float(internal_game.get("performance_bonus", 0.0) + referee_game.get("status_delta", 0.0)),
            "volatility": float(internal_game.get("volatility", 0.25)),
            "chaos": float(max(0.0, referee_game.get("chaos_delta", 0.0))),
        }
        return {
            "phys": phys_signal,
            "affect": affect_signal,
            "social": social_signal,
            "tactic": tactic_signal,
            "governance": gov_signal,
        }

    def fuse(self, expert_signals, context):
        weights = self._gate_weights(context)
        status_delta = 0.0
        volatility = 0.0
        chaos_push = 0.0
        for name, sig in expert_signals.items():
            w = weights.get(name, 0.0)
            status_delta += w * float(sig.get("status", 0.0))
            volatility += w * float(sig.get("volatility", 0.0))
            chaos_push += w * float(sig.get("chaos", 0.0))

        fused = {
            "weights": weights,
            "status_delta": float(status_delta),
            "volatility": float(max(0.12, volatility)),
            "chaos_push": float(max(0.0, chaos_push)),
            "discipline_scalar": float(np.clip(1.08 - 0.22 * volatility - 0.08 * context["referee_strictness"], 0.70, 1.20)),
        }
        return fused

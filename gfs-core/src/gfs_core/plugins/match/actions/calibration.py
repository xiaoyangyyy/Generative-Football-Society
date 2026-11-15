from __future__ import annotations

import json
import math
from pathlib import Path

DEFAULT = {
    "overall_completion": 0.8224,
    "sim_targets": {
        "overall": 0.8224,
        "short": 0.853,
        "through": 0.74,
        "long": 0.6221,
        "intercept_share_max": 0.05,
    },
}


class PassCalibration:
    def __init__(self, data, *, logit_blend=None, intercept_scale=None) -> None:
        self.data = data
        self.logit_blend_override = logit_blend
        self.intercept_scale_override = intercept_scale

    @classmethod
    def load(cls, project_root=".", **overrides):
        path = Path(project_root) / "data" / "calibration" / "statsbomb_pass_rates.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            targets = data.get("sim_targets", {})
            overall = float(data.get("overall_completion", targets.get("overall", 0.8224)))
            targets.setdefault("overall", overall)
            by_length = data.get("by_length_m", {})
            targets.setdefault("short", float(by_length.get("short_lt_15", 0.853)))
            targets.setdefault("long", float(by_length.get("long_gte_30", 0.6221)))
            targets.setdefault("through", float(targets.get("through", 0.74)))
            data["sim_targets"] = targets
            data["overall_completion"] = overall
        except (OSError, json.JSONDecodeError):
            data = dict(DEFAULT)
        return cls(data, **overrides)

    def overall_logit(self):
        return _logit(self.data.get("overall_completion", 0.8224))

    def target_logit(self, kind, distance):
        targets = self.data.get("sim_targets", DEFAULT["sim_targets"])
        short = float(targets.get("short", 0.853))
        through = float(targets.get("through", 0.74))
        long = float(targets.get("long", 0.6221))
        if kind == "long" or distance >= 0.38:
            target = long
        elif kind == "through":
            target = through
        elif kind == "short" or distance <= 0.22:
            target = short
        else:
            weight = (float(distance) - 0.22) / 0.16
            weight = max(0.0, min(1.0, weight))
            target = (1.0 - weight) * short + weight * through
        return _logit(target)

    def logit_blend(self):
        if self.logit_blend_override is not None:
            return float(self.logit_blend_override)
        tuning = self.data.get("sim_tuning", {})
        legacy = tuning.get("logit_residual_uplift")
        if legacy is not None and "logit_target_blend" not in tuning:
            return min(0.95, 0.55 + float(legacy))
        return float(tuning.get("logit_target_blend", 0.88))

    def intercept_scale(self):
        if self.intercept_scale_override is not None:
            return float(self.intercept_scale_override)
        return float(self.data.get("sim_tuning", {}).get("intercept_risk_scale", 0.028))

    def mix_log_bias(self, kind):
        mix = self.data.get("sim_tuning", {}).get(
            "pass_mix_targets",
            {"short": 0.72, "through": 0.16, "long": 0.09, "wall": 0.03},
        )
        probability = float(mix.get(kind, 0.25))
        reference = 0.25 if kind == "wall" else (1.0 - float(mix.get("wall", 0.03))) / 3.0
        return math.log(max(1e-4, probability) / max(1e-4, reference))


def _logit(probability):
    probability = max(1e-4, min(1.0 - 1e-4, float(probability)))
    return math.log(probability / (1.0 - probability))

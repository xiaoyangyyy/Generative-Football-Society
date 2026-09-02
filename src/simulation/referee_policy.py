"""Referee profile configuration and sampling for tournament matches.

Keeping this policy outside ``TournamentManager`` makes the tournament
orchestrator responsible for sequencing matches rather than also owning the
probability model used to configure officials.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np


DEFAULT_REFEREE_PROFILES = {
    "lenient": {
        "strictness_mean": 0.34,
        "strictness_kappa": 16.0,
        "bias_noise": 0.22,
        "exposure_bias_weight": 0.34,
    },
    "balanced": {
        "strictness_mean": 0.52,
        "strictness_kappa": 22.0,
        "bias_noise": 0.15,
        "exposure_bias_weight": 0.42,
    },
    "strict": {
        "strictness_mean": 0.71,
        "strictness_kappa": 28.0,
        "bias_noise": 0.11,
        "exposure_bias_weight": 0.50,
    },
}

DEFAULT_REFEREE_WEIGHTS = {"lenient": 0.30, "balanced": 0.48, "strict": 0.22}


def normalize_weights(
    weights: Mapping[str, float],
    keys: Iterable[str],
) -> dict[str, float]:
    """Return a positive probability distribution for the requested keys."""
    normalized = {key: max(1e-6, float(weights.get(key, 0.0))) for key in keys}
    total = sum(normalized.values())
    if total <= 0:
        uniform = 1.0 / max(1, len(normalized))
        return {key: uniform for key in normalized}
    return {key: value / total for key, value in normalized.items()}


@dataclass
class RefereePolicy:
    """Stage-aware referee archetype distribution and sampler."""

    profiles: Mapping[str, Mapping[str, float]] = field(
        default_factory=lambda: DEFAULT_REFEREE_PROFILES
    )
    weights: Mapping[str, float] = field(
        default_factory=lambda: DEFAULT_REFEREE_WEIGHTS
    )
    stage_morph_strength: float = 1.0

    def __post_init__(self) -> None:
        self.profiles = {
            name: dict(profile) for name, profile in self.profiles.items()
        }
        self.weights = normalize_weights(self.weights, self.profiles)
        self.stage_morph_strength = float(
            np.clip(self.stage_morph_strength, 0.1, 3.0)
        )

    def stage_distribution(self, stage_pressure: float) -> dict[str, float]:
        pressure = float(np.clip(stage_pressure, 0.0, 1.0))
        morph_pressure = self.stage_morph_strength * pressure
        logits = {}
        for name, base_weight in self.weights.items():
            base_log = np.log(max(1e-9, base_weight))
            morph = {
                "lenient": -1.10,
                "balanced": -0.15,
                "strict": 1.20,
            }.get(name, 0.0)
            logits[name] = base_log + morph * morph_pressure

        max_logit = max(logits.values())
        scores = {name: np.exp(value - max_logit) for name, value in logits.items()}
        total = sum(scores.values())
        return {name: value / total for name, value in scores.items()}

    def sample(
        self, first_agent, second_agent, stage_pressure: float,
        *, rng: np.random.Generator | None = None,
    ) -> dict:
        rng = rng or np.random.default_rng()
        distribution = self.stage_distribution(stage_pressure)
        profile_names = list(distribution)
        probabilities = np.array(
            [distribution[name] for name in profile_names],
            dtype=float,
        )
        if not np.all(np.isfinite(probabilities)) or probabilities.sum() <= 0:
            raise ValueError(
                f"Invalid referee profile probabilities: {probabilities}"
            )

        chosen = rng.choice(
            profile_names,
            p=probabilities / probabilities.sum(),
        )
        profile = self.profiles[chosen]
        mean = float(np.clip(profile["strictness_mean"], 0.02, 0.98))
        concentration = float(max(4.0, profile["strictness_kappa"]))
        strictness = rng.beta(
            max(1e-6, mean * concentration),
            max(1e-6, (1.0 - mean) * concentration),
        )
        strictness = float(
            np.clip(
                strictness + 0.18 * stage_pressure * (1.0 - strictness),
                0.05,
                0.98,
            )
        )

        exposure_first = float(
            np.clip(
                np.nan_to_num(
                    first_agent.media_exposure,
                    nan=0.5,
                    posinf=1.0,
                    neginf=0.0,
                ),
                0.0,
                1.0,
            )
        )
        exposure_second = float(
            np.clip(
                np.nan_to_num(
                    second_agent.media_exposure,
                    nan=0.5,
                    posinf=1.0,
                    neginf=0.0,
                ),
                0.0,
                1.0,
            )
        )
        base_bias = profile["exposure_bias_weight"] * np.tanh(
            (exposure_first - exposure_second) * 1.8
        )
        noise = rng.normal(
            0.0,
            profile["bias_noise"] * (1.0 - 0.30 * stage_pressure),
        )
        bias_first = float(np.clip(base_bias + noise, -0.85, 0.85))
        return {
            "profile_name": chosen,
            "profile_distribution": distribution,
            "strictness": strictness,
            "bias_t1": bias_first,
            "bias_t2": -bias_first,
        }

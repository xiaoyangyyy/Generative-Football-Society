from __future__ import annotations

import numpy as np

TOURNAMENT_WEIGHTS = {
    "FIFA World Cup": 1.00,
    "UEFA Euro": 0.85,
    "Copa América": 0.85,
    "African Cup of Nations": 0.70,
    "AFC Asian Cup": 0.70,
    "Gold Cup": 0.70,
    "OFC Nations Cup": 0.60,
    "FIFA World Cup qualification": 0.60,
    "UEFA Euro qualification": 0.55,
    "African Cup of Nations qualification": 0.45,
    "AFC Asian Cup qualification": 0.45,
    "CONCACAF Nations League": 0.45,
    "UEFA Nations League": 0.50,
    "Friendly": 0.20,
}
DEFAULT_WEIGHT = 0.35
MAJOR_TOURNAMENTS = ("FIFA World Cup", "UEFA Euro", "Copa América")


def tournament_weight(name: str) -> float:
    return TOURNAMENT_WEIGHTS.get(name, DEFAULT_WEIGHT)


def time_decay(years_ago, tournament_weights, base_half_life: float = 15.0):
    half_life = np.where(
        tournament_weights >= 0.8,
        50.0,
        np.where(tournament_weights >= 0.5, 25.0, base_half_life),
    )
    decay_rate = np.log(2) / half_life
    return np.exp(-decay_rate * np.clip(years_ago, 0, None))

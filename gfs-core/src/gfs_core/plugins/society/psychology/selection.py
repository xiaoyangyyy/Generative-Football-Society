import numpy as np


def pick_random_outfield_player(team, rng: np.random.Generator, exclude_gk: bool = True):
    pool = [
        player for player in team.players
        if player.on_pitch and (not exclude_gk or player.role != "GK")
    ]
    if not pool:
        return team.players[0]
    weights = np.array([
        1.4 if player.role in {"ST", "LW", "RW"} else 1.0
        for player in pool
    ], dtype=float)
    weights /= weights.sum()
    return pool[int(rng.choice(len(pool), p=weights))]

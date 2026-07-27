"""Fixed-size observation vectors for the latent world model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.match_engine.world_model.config import WorldModelConfig

if TYPE_CHECKING:
    from src.match_engine.state import MatchAffectiveState

# Layout: 5×(gx·gy) grids + 10 global + 88 players + 8 tactics + 1 attack_dir
_GRID_CHANNELS = 5
_GLOBAL_DIM = 10
_PLAYER_DIM = 88  # 22 × 4
_TACTICS_DIM = 8
_ATTACK_FLAG = 1
OBS_DIM = _GRID_CHANNELS * 8 * 5 + _GLOBAL_DIM + _PLAYER_DIM + _TACTICS_DIM + _ATTACK_FLAG


def _downsample(grid: np.ndarray, gx: int, gy: int) -> np.ndarray:
    sh = grid.shape
    if sh[0] < 2 or sh[1] < 2:
        return np.zeros((gx, gy), dtype=np.float32)
    x_idx = np.linspace(0, sh[0] - 1, gx).astype(int)
    y_idx = np.linspace(0, sh[1] - 1, gy).astype(int)
    return grid[np.ix_(x_idx, y_idx)].astype(np.float32)


def encode_observation(
    state: "MatchAffectiveState",
    *,
    attacking_home: bool | None = None,
    cfg: WorldModelConfig | None = None,
) -> np.ndarray:
    """Encode match state into a fixed vector in [0, 1] (mostly)."""
    cfg = cfg or WorldModelConfig.from_env()
    gx, gy = cfg.grid_gx, cfg.grid_gy
    sg = state.spatial

    grids = [
        np.clip(_downsample(sg.rho_home, gx, gy) / 2.0, 0.0, 1.0),
        np.clip(_downsample(sg.rho_away, gx, gy) / 2.0, 0.0, 1.0),
        np.clip(_downsample(sg.press, gx, gy) / 2.0, 0.0, 1.0),
        np.clip(_downsample(sg.phi_home, gx, gy), 0.0, 1.0),
        np.clip(_downsample(sg.phi_away, gx, gy), 0.0, 1.0),
    ]
    grid_flat = np.concatenate([g.ravel() for g in grids])

    ball = state.ball
    poss_home = 1.0 if ball.possession_team_id == state.home.team_id else 0.0
    global_feats = np.array(
        [
            float(np.clip(ball.position[0], 0, 1)),
            float(np.clip(ball.position[1], 0, 1)),
            float(np.clip(ball.velocity[0] * 2 + 0.5, 0, 1)),
            float(np.clip(ball.velocity[1] * 2 + 0.5, 0, 1)),
            float(np.clip(state.home.score / 5.0, 0, 1)),
            float(np.clip(state.away.score / 5.0, 0, 1)),
            float(np.clip((state.home.score - state.away.score + 3) / 6.0, 0, 1)),
            float(np.clip(state.clock_seconds / (90.0 * 60.0), 0, 1)),
            float(np.clip(state.crowd.psi * 0.25 + 0.5, 0, 1)),
            poss_home,
        ],
        dtype=np.float32,
    )

    player_feats = np.zeros(_PLAYER_DIM, dtype=np.float32)
    all_players = list(state.home.players) + list(state.away.players)
    for i, p in enumerate(all_players[:22]):
        base = i * 4
        player_feats[base] = float(np.clip(p.position[0], 0, 1))
        player_feats[base + 1] = float(np.clip(p.position[1], 0, 1))
        player_feats[base + 2] = float(np.clip(p.velocity[0] * 3 + 0.5, 0, 1))
        player_feats[base + 3] = float(np.clip(p.velocity[1] * 3 + 0.5, 0, 1))

    tac_h = state.home.coach.tactical_current or {}
    tac_a = state.away.coach.tactical_current or {}
    tactics = np.array(
        [
            float(tac_h.get("pressing_intensity", 0.5)),
            float(tac_h.get("risk_budget", 0.5)),
            float(tac_h.get("line_height", 0.5)),
            float(tac_h.get("rotation_aggressiveness", 0.5)),
            float(tac_a.get("pressing_intensity", 0.5)),
            float(tac_a.get("risk_budget", 0.5)),
            float(tac_a.get("line_height", 0.5)),
            float(tac_a.get("rotation_aggressiveness", 0.5)),
        ],
        dtype=np.float32,
    )

    if attacking_home is None:
        if ball.possession_team_id == state.home.team_id:
            attacking_home = True
        elif ball.possession_team_id == state.away.team_id:
            attacking_home = False
        else:
            attacking_home = True
    attack_flag = np.array([1.0 if attacking_home else 0.0], dtype=np.float32)

    obs = np.concatenate([grid_flat, global_feats, player_feats, tactics, attack_flag])
    assert obs.shape[0] == OBS_DIM, (obs.shape[0], OBS_DIM)
    return np.clip(
        np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0),
        0.0,
        1.0,
    ).astype(np.float32)


def decode_ball_xy(obs: np.ndarray) -> tuple[float, float]:
    return float(obs[200]), float(obs[201])


def synthesize_obs_from_ball_event(
    event: dict,
    *,
    meta: dict | None = None,
    attacking_home: bool | None = None,
) -> np.ndarray:
    """
    Reconstruct a minimal observation from ball_log when full state was not snapshotted.
    Uses ball xy, clock, and score from meta.
    """
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    et = event.get("type", "pass")
    if et == "pass":
        xy = event.get("from_xy", [0.5, 0.5])
        land = event.get("land_xy", xy)
    elif et == "shot":
        xy = event.get("from_xy", [0.5, 0.5])
        land = xy
    else:
        xy = event.get("land_xy", event.get("from_xy", [0.5, 0.5]))
        land = xy

    obs[200] = float(np.clip(xy[0], 0, 1))
    obs[201] = float(np.clip(xy[1], 0, 1))
    obs[202] = float(np.clip((land[0] - xy[0]) * 2 + 0.5, 0, 1))
    obs[203] = float(np.clip((land[1] - xy[1]) * 2 + 0.5, 0, 1))
    t_sec = float(event.get("t_sec", 0.0))
    obs[207] = float(np.clip(t_sec / (90.0 * 60.0), 0, 1))
    if meta and "score" in meta:
        try:
            sh, sa = [int(x) for x in str(meta["score"]).split("-")]
            obs[204] = float(np.clip(sh / 5.0, 0, 1))
            obs[205] = float(np.clip(sa / 5.0, 0, 1))
            obs[206] = float(np.clip((sh - sa + 3) / 6.0, 0, 1))
        except Exception:
            pass
    team = str(event.get("team", ""))
    home = (meta or {}).get("home", "")
    if attacking_home is None:
        attacking_home = team == home if home else True
    obs[-1] = 1.0 if attacking_home else 0.0
    return obs.astype(np.float32)

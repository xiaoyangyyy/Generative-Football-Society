"""Load and run the latent world model at match time."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from src.match_engine.world_model.action_codec import ACTION_DIM, zero_action
from src.match_engine.world_model.config import WorldModelConfig, default_checkpoint_path
from src.match_engine.world_model.model import LatentWorldModel, WorldModelOutput, load_checkpoint
from src.match_engine.world_model.observation import OBS_DIM, encode_observation

try:
    import torch

    _TORCH = True
except ImportError:
    _TORCH = False


class WorldModelRuntime:
    """Thin wrapper for planning and trace collection."""

    def __init__(self, model: LatentWorldModel, cfg: WorldModelConfig):
        self.model = model
        self.cfg = cfg
        self._hidden: Optional[np.ndarray] = None

    @classmethod
    def load(cls, path: str) -> "WorldModelRuntime":
        if not _TORCH:
            raise RuntimeError("PyTorch required for MATCH_WORLD_MODEL=1")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"World model checkpoint not found: {path}")
        model, cfg, meta = load_checkpoint(path)
        rt = cls(model, cfg)
        kind = getattr(model, "transition_kind", cfg.transition_type)
        print(f"  [WORLD_MODEL] transition={kind} latent={cfg.latent_dim} (meta rows={meta.get('rows', '?')})")
        return rt

    @classmethod
    def load_default(cls, base_dir: str) -> Optional["WorldModelRuntime"]:
        path = default_checkpoint_path(base_dir)
        if not os.path.isfile(path):
            print(f"  [WORLD_MODEL] No checkpoint at {path} — run scripts/ensure_world_model.py")
            return None
        try:
            rt = cls.load(path)
            print(f"  [WORLD_MODEL] Loaded from {path}")
            return rt
        except Exception as exc:
            print(f"  [WORLD_MODEL] Failed to load: {exc}")
            return None

    def reset_hidden(self) -> None:
        self._hidden = None

    def imagine(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        *,
        steps: int | None = None,
        carry_hidden: bool = False,
    ) -> WorldModelOutput:
        steps = steps if steps is not None else self.cfg.imagination_steps
        h = self._hidden if carry_hidden else None
        out = self.model.imagine(obs, action, steps=steps, h=h)
        if carry_hidden and self.model.transition_kind == "gru":
            self._hidden = out.latent.reshape(1, 1, -1)
        return out

    def score_action(self, obs: np.ndarray, action: np.ndarray) -> float:
        out = self.imagine(obs, action, carry_hidden=False)
        attacking = obs[-1] > 0.5
        bx = float(out.next_obs[200])
        progress = bx if attacking else (1.0 - bx)
        progress = float(np.clip(progress, 0, 1))
        xg_term = float(np.clip(out.xg_delta_attacking, -0.5, 0.5)) * 2.0
        pass_term = float(out.pass_success) * 0.35
        turnover = float(action[0:6].dot(np.array([0, 0, 0, 0, 1, 0], dtype=np.float32)))
        intercept_pen = 0.25 * turnover * (1.0 - float(out.pass_success))
        return progress * 0.42 + xg_term * 0.33 + pass_term - intercept_pen

    def score_shot_action(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        *,
        attacking_home: bool,
    ) -> float:
        out = self.imagine(obs, action)
        bx = float(out.next_obs[200])
        progress = bx if attacking_home else (1.0 - bx)
        progress = float(np.clip(progress, 0, 1))
        xg_term = float(np.clip(out.xg_delta_attacking, 0.0, 1.0)) * 1.5
        goal_term = float(out.shot_goal_prob) * 0.55
        return progress * 0.35 + xg_term * 0.35 + goal_term

    def encode_state(self, state, *, attacking_home: bool | None = None) -> np.ndarray:
        return encode_observation(state, attacking_home=attacking_home, cfg=self.cfg)

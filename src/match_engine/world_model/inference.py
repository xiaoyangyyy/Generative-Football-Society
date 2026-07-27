"""Load and run the latent world model at match time."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from src.match_engine.math_utils import finite_float
from src.match_engine.world_model.config import WorldModelConfig, default_checkpoint_path
from src.match_engine.world_model.model import LatentWorldModel, WorldModelOutput, load_checkpoint
from src.match_engine.world_model.observation import OBS_DIM, encode_observation
from src.match_engine.world_model.schema import observation_coverage, strip_outcome_leakage
from src.match_engine.world_model.probabilistic import ProbabilisticFuture, future_from_ensemble

try:
    import torch

    _TORCH = True
except ImportError:
    _TORCH = False


class WorldModelRuntime:
    """Thin wrapper for planning and trace collection."""

    def __init__(self, model: LatentWorldModel, cfg: WorldModelConfig, meta: dict | None = None):
        self.model = model
        self.cfg = cfg
        self.meta = meta or {}
        self._hidden: Optional[np.ndarray] = None
        self.last_uncertainty = 1.0
        version = int(getattr(model, "checkpoint_version", 2))
        validation = self.meta.get("validation", {}) if isinstance(self.meta, dict) else {}
        if version < 4:
            self.base_quality = float(np.clip(cfg.legacy_quality, 0.0, 1.0))
        else:
            self.base_quality = float(np.clip(validation.get("planner_quality", 0.0), 0.0, 1.0))
        self.pass_quality = float(
            np.clip(validation.get("pass_planner_quality", self.base_quality), 0.0, 1.0)
        )
        self.shot_quality = float(
            np.clip(validation.get("shot_planner_quality", self.base_quality), 0.0, 1.0)
        )
        calibration = validation.get("pass_calibration", {})
        self.pass_calibration_scale = float(calibration.get("scale", 1.0))
        self.pass_calibration_bias = float(calibration.get("bias", 0.0))
        self.pass_threshold = float(validation.get("pass_threshold", 0.5))

    @classmethod
    def load(cls, path: str) -> "WorldModelRuntime":
        if not _TORCH:
            raise RuntimeError("PyTorch required for MATCH_WORLD_MODEL=1")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"World model checkpoint not found: {path}")
        model, cfg, meta = load_checkpoint(path)
        rt = cls(model, cfg, meta)
        kind = getattr(model, "transition_kind", cfg.transition_type)
        print(f"  [WORLD_MODEL] transition={kind} latent={cfg.latent_dim} (meta rows={meta.get('rows', '?')})")
        print(
            f"  [WORLD_MODEL] checkpoint=v{getattr(model, 'checkpoint_version', 2)} "
            f"quality pass={rt.pass_quality:.3f} shot={rt.shot_quality:.3f}"
        )
        return rt

    @classmethod
    def load_default(cls, base_dir: str) -> Optional["WorldModelRuntime"]:
        path = default_checkpoint_path(base_dir)
        if not os.path.isfile(path):
            print(f"  [WORLD_MODEL] No checkpoint at {path} 鈥?run scripts/ensure_world_model.py")
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
        clean_obs = np.clip(
            np.nan_to_num(np.asarray(obs, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0),
            0.0, 1.0,
        )
        raw_action = np.asarray(action, dtype=np.float32).copy()
        if raw_action.shape[0] > 13 and not np.isfinite(raw_action[13]):
            raw_action[13] = 0.5
        clean_action = strip_outcome_leakage(
            np.nan_to_num(raw_action, nan=0.0, posinf=1.0, neginf=0.0)
        )
        out = self.model.imagine(clean_obs, clean_action, steps=steps, h=h)
        raw = float(np.clip(out.pass_success, 1e-6, 1.0 - 1e-6))
        raw_logit = np.log(raw / (1.0 - raw))
        out.pass_success = float(
            1.0 / (1.0 + np.exp(-(self.pass_calibration_scale * raw_logit + self.pass_calibration_bias)))
        )
        quality_uncertainty = 1.0 - self.planner_confidence(clean_obs)
        out.uncertainty = float(
            np.clip(max(quality_uncertainty, 2.0 * float(out.uncertainty)), 0.0, 1.0)
        )
        self.last_uncertainty = out.uncertainty
        if carry_hidden and self.model.transition_kind == "gru":
            self._hidden = out.latent.reshape(1, 1, -1)
        return out

    def planner_confidence(self, obs: np.ndarray, kind: str = "general") -> float:
        coverage = observation_coverage(obs)
        quality = self.base_quality
        if kind == "pass":
            quality = self.pass_quality
        elif kind == "shot":
            quality = self.shot_quality
        confidence = quality * (0.25 + 0.75 * coverage)
        if confidence < float(self.cfg.min_planner_quality):
            return 0.0
        return float(np.clip(confidence, 0.0, 1.0))

    def imagine_pass(self, obs: np.ndarray, action: np.ndarray, *, steps: int | None = None) -> WorldModelOutput:
        """Backward-compatible pass imagination alias."""
        return self.imagine(obs, action, steps=steps, carry_hidden=False)

    def predict_future(self, obs: np.ndarray, action: np.ndarray, *, action_kind: str, horizon_s: float = 10.0) -> ProbabilisticFuture:
        """Expose multimodal event and progress uncertainty from ensemble heads."""
        if action_kind not in {"pass", "shot", "hold", "cross"}:
            raise ValueError("unsupported action kind")
        clean_obs = np.clip(np.nan_to_num(np.asarray(obs, dtype=np.float32)), 0.0, 1.0)
        clean_action = strip_outcome_leakage(np.nan_to_num(np.asarray(action, dtype=np.float32), nan=0.0))
        with torch.no_grad():
            obs_t = torch.from_numpy(clean_obs).unsqueeze(0)
            action_t = torch.from_numpy(clean_action).unsqueeze(0)
            z = self.model.encode(obs_t)
            z_next, _ = self.model._transition_step(z, action_t, None)
            pass_heads, progress_heads, shot_heads = self.model.outcome_ensemble(z_next, action_t)
            pass_probability = torch.sigmoid(pass_heads).reshape(-1).numpy()
            shot_probability = torch.sigmoid(shot_heads).reshape(-1).numpy()
            progress = progress_heads.reshape(-1).numpy()
        return future_from_ensemble(
            pass_probabilities=pass_probability, shot_probabilities=shot_probability,
            progress_samples=progress, action_kind=action_kind, horizon_s=horizon_s,
        )

    def score_action(self, obs: np.ndarray, action: np.ndarray) -> float:
        obs = np.nan_to_num(np.asarray(obs, dtype=float), nan=0.0, posinf=1.0, neginf=-1.0)
        action = np.nan_to_num(np.asarray(action, dtype=float), nan=0.0, posinf=1.0, neginf=-1.0)
        out = self.imagine(obs, action, carry_hidden=False)
        attacking = obs[-1] > 0.5
        bx = finite_float(float(out.next_obs[200]), 0.5)
        predicted_progress = bx if attacking else (1.0 - bx)
        current_bx = finite_float(float(obs[200]), 0.5)
        current_progress = current_bx if attacking else (1.0 - current_bx)
        state_delta = float(np.clip(predicted_progress - current_progress, -0.5, 0.5))
        learned_delta = float(np.clip(finite_float(out.progress_delta, 0.0), -0.5, 0.5))
        pass_term = finite_float(float(out.pass_success), 0.5) - 0.5
        turnover = float(action[0:6].dot(np.array([0, 0, 0, 0, 1, 0], dtype=np.float32)))
        intercept_pen = 0.25 * turnover * (1.0 - finite_float(float(out.pass_success), 0.5))
        return finite_float(
            0.45 * state_delta + 0.35 * learned_delta + 0.20 * pass_term - intercept_pen,
            0.0,
        )

    def score_shot_action(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        *,
        attacking_home: bool,
    ) -> float:
        obs = np.nan_to_num(np.asarray(obs, dtype=float), nan=0.0, posinf=1.0, neginf=-1.0)
        action = np.nan_to_num(np.asarray(action, dtype=float), nan=0.0, posinf=1.0, neginf=-1.0)
        out = self.imagine(obs, action)
        progress = float(np.clip(finite_float(out.progress_delta, 0.0), -0.5, 0.5))
        xg_prior = float(np.clip(action[13], 0.0, 1.0))
        goal_term = finite_float(float(out.shot_goal_prob), 0.1)
        return finite_float(0.10 * progress + 0.25 * xg_prior + 0.65 * goal_term, 0.0)

    def encode_state(self, state, *, attacking_home: bool | None = None) -> np.ndarray:
        return encode_observation(state, attacking_home=attacking_home, cfg=self.cfg)

"""Load and run the latent world model at match time."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

import numpy as np

from src.match_engine.math_utils import finite_float
from src.match_engine.world_model.config import WorldModelConfig, default_checkpoint_path
from src.match_engine.world_model.action_codec import decode_action_kind
from src.match_engine.world_model.model import LatentWorldModel, WorldModelOutput, load_checkpoint
from src.match_engine.world_model.observation import OBS_DIM, encode_observation
from src.match_engine.world_model.schema import (
    HORIZON_INDEX,
    HORIZON_SCALE_SECONDS,
    observation_coverage,
    strip_outcome_leakage,
)
from src.match_engine.world_model.probabilistic import ProbabilisticFuture, future_from_ensemble
from src.match_engine.world_model.online_calibration import OnlineTransitionCalibrator
from src.match_engine.world_model.uncertainty import (
    compose_uncertainty,
    compound_uncertainty,
    ensemble_uncertainty_decomposition,
)

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
        signature_payload = {
            "checkpoint_version": int(getattr(model, "checkpoint_version", 2)),
            "transition_kind": str(getattr(model, "transition_kind", "unknown")),
            "latent_dim": int(cfg.latent_dim),
            "hidden_dim": int(cfg.hidden_dim),
            "outcome_ensemble_size": int(cfg.ensemble_size),
            "transition_ensemble_size": int(cfg.transition_ensemble_size),
            "transition_ensemble_trained": bool(getattr(
                model, "transition_ensemble_trained", False,
            )),
            "validation": self.meta.get("validation", {}),
        }
        self.checkpoint_signature = "meta:" + hashlib.sha256(
            json.dumps(
                signature_payload, sort_keys=True, default=str,
            ).encode("utf-8")
        ).hexdigest()[:16]
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
        self.progress_aleatoric_scale = float(np.clip(
            validation.get(
                "progress_rmse",
                np.sqrt(max(0.0, float(validation.get("weighted_obs_mse", 0.02)))),
            ),
            0.0,
            1.0,
        ))
        self.online_calibrator = OnlineTransitionCalibrator(
            expected_weighted_mse=float(
                validation.get("weighted_obs_mse", 0.02)
            ),
        )

    @classmethod
    def load(cls, path: str) -> "WorldModelRuntime":
        if not _TORCH:
            raise RuntimeError("PyTorch required for MATCH_WORLD_MODEL=1")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"World model checkpoint not found: {path}")
        model, cfg, meta = load_checkpoint(path)
        rt = cls(model, cfg, meta)
        digest = hashlib.sha256()
        with open(path, "rb") as checkpoint_file:
            for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                digest.update(chunk)
        rt.checkpoint_signature = "sha256:" + digest.hexdigest()
        kind = getattr(model, "transition_kind", cfg.transition_type)
        print(f"  [WORLD_MODEL] transition={kind} latent={cfg.latent_dim} (meta rows={meta.get('rows', '?')})")
        print(
            f"  [WORLD_MODEL] checkpoint=v{getattr(model, 'checkpoint_version', 2)} "
            f"quality pass={rt.pass_quality:.3f} shot={rt.shot_quality:.3f} "
            f"dynamics_members={model.transition_member_count} "
            f"trained={model.transition_ensemble_trained}"
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
        quality_kind: str = "general",
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
        quality_uncertainty = 1.0 - self.planner_confidence(
            clean_obs, kind=quality_kind,
        )
        samples = out.uncertainty_samples or {}
        if samples:
            calibrated_pass_logit = np.clip(
                self.pass_calibration_scale
                * np.asarray(samples["pass_logits"], dtype=float)
                + self.pass_calibration_bias,
                -40.0,
                40.0,
            )
            pass_probability = 1.0 / (1.0 + np.exp(-calibrated_pass_logit))
            shot_logit = np.clip(np.asarray(
                samples["shot_logits"], dtype=float,
            ), -40.0, 40.0)
            shot_probability = 1.0 / (1.0 + np.exp(-shot_logit))
            decomposition = ensemble_uncertainty_decomposition(
                pass_probability,
                shot_probability,
                np.asarray(samples["progress"], dtype=float),
                progress_aleatoric=self.progress_aleatoric_scale,
                transition_state_samples=np.asarray(
                    samples["transition_states"], dtype=float,
                ),
                transition_ensemble_trained=bool(
                    samples["transition_ensemble_trained"]
                ),
            )
            model_epistemic = float(
                decomposition["epistemic_uncertainty"]
            )
            aleatoric = float(decomposition["aleatoric_uncertainty"])
            out.uncertainty_components = dict(decomposition["components"])
            base_source = str(decomposition["source"])
        else:
            model_epistemic = float(out.epistemic_uncertainty or 0.0)
            aleatoric = float(np.clip(
                float(out.aleatoric_uncertainty or 0.0)
                + 0.10 * self.progress_aleatoric_scale,
                0.0,
                0.50,
            ))
            base_source = str(out.uncertainty_source)
        epistemic = float(np.clip(
            max(quality_uncertainty, 2.0 * model_epistemic), 0.0, 1.0,
        ))
        out.epistemic_uncertainty = epistemic
        out.aleatoric_uncertainty = aleatoric
        out.uncertainty = compose_uncertainty(epistemic, aleatoric)
        out.uncertainty_source = (
            base_source + "+validation_quality_and_progress_residual"
        )
        out.uncertainty_components = {
            **(out.uncertainty_components or {}),
            "validation_quality_epistemic": quality_uncertainty,
            "progress_residual_aleatoric": self.progress_aleatoric_scale,
        }
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
        confidence *= self.online_calibrator.trust_factor(kind)
        return float(np.clip(confidence, 0.0, 1.0))

    def observe_transition(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        next_observation: np.ndarray,
    ) -> dict[str, float]:
        """Calibrate against the exact transition target used at inference."""
        action_kind = decode_action_kind(action)
        quality_kind = "shot" if action_kind == "shot" else "pass"
        output = self.imagine(
            observation, action, carry_hidden=False,
            quality_kind=quality_kind,
        )
        return self.online_calibrator.observe(
            action_kind=action_kind,
            current_observation=observation,
            predicted_observation=output.next_obs,
            actual_observation=next_observation,
            predicted_uncertainty=output.uncertainty,
        )

    def online_calibration_diagnostics(self) -> dict:
        return self.online_calibrator.diagnostics()

    def imagine_pass(self, obs: np.ndarray, action: np.ndarray, *, steps: int | None = None) -> WorldModelOutput:
        """Backward-compatible pass imagination alias."""
        return self.imagine(
            obs, action, steps=steps, carry_hidden=False,
            quality_kind="pass",
        )

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
            z_members, _ = self.model.transition_ensemble_step(
                z, action_t,
            )
            next_obs_members = torch.stack([
                self.model.decode_next(obs_t, member)
                for member in z_members
            ], dim=0)
            outcome_members = [
                self.model.outcome_ensemble(member, action_t)
                for member in z_members
            ]
            pass_heads = torch.stack([
                item[0] for item in outcome_members
            ], dim=0)
            progress_heads = torch.stack([
                item[1] for item in outcome_members
            ], dim=0)
            shot_heads = torch.stack([
                item[2] for item in outcome_members
            ], dim=0)
            pass_probability = torch.sigmoid(
                self.pass_calibration_scale * pass_heads
                + self.pass_calibration_bias
            ).numpy()
            shot_probability = torch.sigmoid(shot_heads).numpy()
            progress = progress_heads.numpy()
        return future_from_ensemble(
            pass_probabilities=pass_probability, shot_probabilities=shot_probability,
            progress_samples=progress, action_kind=action_kind, horizon_s=horizon_s,
            progress_aleatoric=self.progress_aleatoric_scale,
            transition_state_samples=next_obs_members.numpy(),
            transition_ensemble_trained=(
                self.model.transition_ensemble_trained
            ),
        )

    def evaluate_action_from_observation(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        *,
        action_kind: str,
        attacking_home: bool,
        horizon_s: float = 10.0,
    ) -> dict:
        """Score one action and its uncertainty with a single model rollout."""
        if action_kind not in {"pass", "shot", "hold", "cross"}:
            raise ValueError("unsupported action kind")
        clean_obs = np.clip(
            np.nan_to_num(np.asarray(obs, dtype=np.float32)), 0.0, 1.0,
        )
        clean_action = strip_outcome_leakage(np.nan_to_num(
            np.asarray(action, dtype=np.float32), nan=0.0,
        ))
        quality_kind = "shot" if action_kind == "shot" else "pass"
        output = self.imagine(
            clean_obs,
            clean_action,
            carry_hidden=False,
            quality_kind=quality_kind,
        )
        samples = output.uncertainty_samples or {}
        pass_logits = np.clip(
            self.pass_calibration_scale
            * np.asarray(samples.get("pass_logits", [[[0.0]]]), dtype=float)
            + self.pass_calibration_bias,
            -40.0,
            40.0,
        )
        shot_logits = np.clip(np.asarray(
            samples.get("shot_logits", [[[0.0]]]), dtype=float,
        ), -40.0, 40.0)
        progress_samples = np.asarray(
            samples.get("progress", [[[output.progress_delta]]]), dtype=float,
        )
        transition_states = samples.get("transition_states")
        future = future_from_ensemble(
            pass_probabilities=1.0 / (1.0 + np.exp(-pass_logits)),
            shot_probabilities=1.0 / (1.0 + np.exp(-shot_logits)),
            progress_samples=progress_samples,
            action_kind=action_kind,
            horizon_s=horizon_s,
            progress_aleatoric=self.progress_aleatoric_scale,
            transition_state_samples=(
                np.asarray(transition_states, dtype=float)
                if transition_states is not None else None
            ),
            transition_ensemble_trained=bool(samples.get(
                "transition_ensemble_trained", False,
            )),
        )
        future = ProbabilisticFuture(
            event_probabilities=future.event_probabilities,
            event_time_s=future.event_time_s,
            progress_quantiles=future.progress_quantiles,
            state_uncertainty=float(output.uncertainty),
            epistemic_uncertainty=float(output.epistemic_uncertainty),
            aleatoric_uncertainty=float(output.aleatoric_uncertainty),
            uncertainty_source=str(output.uncertainty_source),
            uncertainty_components=dict(output.uncertainty_components or {}),
        )
        if action_kind == "shot":
            expected_value = (
                0.10 * float(np.clip(output.progress_delta, -0.5, 0.5))
                + 0.25 * float(np.clip(clean_action[13], 0.0, 1.0))
                + 0.65 * finite_float(float(output.shot_goal_prob), 0.1)
            )
        else:
            direction = 1.0 if attacking_home else -1.0
            state_delta = float(np.clip(
                direction * (
                    finite_float(float(output.next_obs[200]), 0.5)
                    - finite_float(float(clean_obs[200]), 0.5)
                ),
                -0.5,
                0.5,
            ))
            learned_delta = float(np.clip(
                finite_float(output.progress_delta, 0.0), -0.5, 0.5,
            ))
            pass_term = finite_float(output.pass_success, 0.5) - 0.5
            intercept_penalty = (
                0.25 * float(clean_action[4])
                * (1.0 - finite_float(output.pass_success, 0.5))
            )
            expected_value = (
                0.45 * state_delta
                + 0.35 * learned_delta
                + 0.20 * pass_term
                - intercept_penalty
            )
        return {
            "expected_value": finite_float(expected_value, 0.0),
            "future": future,
            "next_observation": np.asarray(output.next_obs, dtype=np.float32),
            "model_calls": 1,
        }

    def two_step_planning_gate(self) -> dict:
        """Authorize predicted-state search only with grouped two-step evidence."""
        validation = self.meta.get("validation", {}) if isinstance(
            self.meta, dict
        ) else {}
        evidence = validation.get("two_step_rollout") or {}
        samples = int(evidence.get("samples", 0))
        groups = int(evidence.get("groups", 0))
        skill = finite_float(evidence.get("skill_vs_persistence") or 0.0, 0.0)
        model_error = finite_float(evidence.get("weighted_mse") or 0.0, 0.0)
        baseline_error = finite_float(
            evidence.get("persistence_weighted_mse") or 0.0, 0.0,
        )
        ensemble_trained = bool(getattr(
            self.model, "transition_ensemble_trained", False,
        ))
        active = bool(
            ensemble_trained
            and samples >= 32
            and groups >= 4
            and baseline_error > 1e-10
            and model_error < baseline_error
            and skill >= 0.02
        )
        sample_factor = samples / (samples + 128.0)
        group_factor = groups / (groups + 8.0)
        skill_factor = float(np.clip(skill / 0.25, 0.0, 1.0))
        authority = float(np.clip(
            0.50 * sample_factor * group_factor * skill_factor
            if active else 0.0,
            0.0,
            0.50,
        ))
        return {
            "version": 1,
            "active": active,
            "authority": authority,
            "reason": (
                "grouped_two_step_holdout_gain"
                if active else "two_step_holdout_gate_closed"
            ),
            "samples": samples,
            "groups": groups,
            "weighted_mse": model_error,
            "persistence_weighted_mse": baseline_error,
            "skill_vs_persistence": skill,
            "transition_ensemble_trained": ensemble_trained,
            "action_sequence": str(evidence.get(
                "action_sequence", "unavailable",
            )),
        }

    def predict_policy_utility(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        *,
        action_kind: str,
        attacking_home: bool,
        horizon_s: float,
    ) -> dict[str, float | int | str]:
        """Predict the same declared utility later measured by policy outcomes."""
        rollout_steps = max(1, int(np.ceil(
            max(0.1, float(horizon_s)) / HORIZON_SCALE_SECONDS
        )))
        segment_horizon_s = max(0.1, float(horizon_s)) / rollout_steps
        conditioned_action = np.asarray(action, dtype=np.float32).copy()
        conditioned_action[HORIZON_INDEX] = float(np.clip(
            segment_horizon_s / HORIZON_SCALE_SECONDS, 0.0, 1.0,
        ))
        quality_kind = "shot" if action_kind == "shot" else "pass"
        output = self.imagine(
            obs,
            conditioned_action,
            steps=rollout_steps,
            carry_hidden=False,
            quality_kind=quality_kind,
        )
        current = np.asarray(obs, dtype=float)
        future = np.asarray(output.next_obs, dtype=float)
        direction = 1.0 if attacking_home else -1.0
        progress = float(np.clip(
            direction * (future[200] - current[200]), -1.0, 1.0,
        ))
        current_goal_diff = direction * 5.0 * (current[204] - current[205])
        future_goal_diff = direction * 5.0 * (future[204] - future[205])
        goal_delta = float(np.clip(
            future_goal_diff - current_goal_diff, -2.0, 2.0,
        ))
        possession_home = float(np.clip(future[209], 0.0, 1.0))
        retention_probability = (
            possession_home if attacking_home else 1.0 - possession_home
        )
        retention_edge = 2.0 * retention_probability - 1.0
        xg_net = float(np.clip(
            output.progress_delta * rollout_steps, -1.0, 1.0,
        ))
        epistemic_uncertainty = compound_uncertainty(
            output.epistemic_uncertainty, rollout_steps,
        )
        aleatoric_uncertainty = compound_uncertainty(
            output.aleatoric_uncertainty, rollout_steps,
        )
        uncertainty = compose_uncertainty(
            epistemic_uncertainty, aleatoric_uncertainty,
        )
        utility = (
            goal_delta
            + 0.35 * xg_net
            + 0.15 * progress
            + 0.05 * retention_edge
        )
        return {
            "prediction_source": "autoregressive_action_persistence_rollout",
            "horizon_s": float(horizon_s),
            "rollout_steps": rollout_steps,
            "segment_horizon_s": segment_horizon_s,
            "policy_utility": float(utility),
            "progress": progress,
            "retention_probability": retention_probability,
            "xg_net_delta": xg_net,
            "goal_diff_delta": goal_delta,
            "uncertainty": uncertainty,
            "epistemic_uncertainty": epistemic_uncertainty,
            "aleatoric_uncertainty": aleatoric_uncertainty,
            "uncertainty_source": output.uncertainty_source,
            "uncertainty_components": output.uncertainty_components or {},
        }

    def score_action(self, obs: np.ndarray, action: np.ndarray) -> float:
        obs = np.nan_to_num(np.asarray(obs, dtype=float), nan=0.0, posinf=1.0, neginf=-1.0)
        action = np.nan_to_num(np.asarray(action, dtype=float), nan=0.0, posinf=1.0, neginf=-1.0)
        out = self.imagine(
            obs, action, carry_hidden=False, quality_kind="pass",
        )
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
        out = self.imagine(obs, action, quality_kind="shot")
        progress = float(np.clip(finite_float(out.progress_delta, 0.0), -0.5, 0.5))
        xg_prior = float(np.clip(action[13], 0.0, 1.0))
        goal_term = finite_float(float(out.shot_goal_prob), 0.1)
        return finite_float(0.10 * progress + 0.25 * xg_prior + 0.65 * goal_term, 0.0)

    def encode_state(self, state, *, attacking_home: bool | None = None) -> np.ndarray:
        return encode_observation(state, attacking_home=attacking_home, cfg=self.cfg)

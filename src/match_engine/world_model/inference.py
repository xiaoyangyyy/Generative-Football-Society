"""Load and run the latent world model at match time."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

import numpy as np

from src.match_engine.math_utils import finite_float
from src.match_engine.world_model.config import (
    WorldModelConfig,
    default_checkpoint_path,
    default_shot_head_path,
)
from src.match_engine.world_model.action_codec import ACTION_DIM, decode_action_kind
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
    transition_ensemble_uncertainty,
)
from src.match_engine.world_model.state_scales import (
    FALSIFIABLE_SEMANTIC_EVENTS,
    PREDICTIVE_MECHANISM_PATHS,
    multiscale_state_forecast,
    semantic_path_statistics,
)
from src.match_engine.world_model.policy_utility import (
    POLICY_UTILITY_VERSION,
    policy_utility_sequence_validation_gate,
    policy_utility_validation_gate,
    transition_policy_utility_numpy,
)
from src.match_engine.world_model.cross_validation import (
    replay_cross_action_validation,
)

try:
    import torch

    _TORCH = True
except ImportError:
    _TORCH = False


def _quality_kind_for_action(action_kind: str) -> str:
    if action_kind == "shot":
        return "shot"
    if action_kind == "cross":
        return "cross"
    return "pass"


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
        # Decision uncertainty excludes validation quality.  Validation is a
        # hard authorization gate; folding it into every per-action confidence
        # term made an authorized head effectively inert at sampling time.
        self.last_decision_uncertainty = 1.0
        version = int(getattr(model, "checkpoint_version", 2))
        validation = self.meta.get("validation", {}) if isinstance(self.meta, dict) else {}
        if version < 4:
            self.base_quality = float(np.clip(cfg.legacy_quality, 0.0, 1.0))
        else:
            self.base_quality = float(np.clip(validation.get("planner_quality", 0.0), 0.0, 1.0))
        self.pass_quality = float(
            np.clip(validation.get("pass_planner_quality", self.base_quality), 0.0, 1.0)
        )
        self.joint_shot_quality = float(
            np.clip(validation.get("shot_planner_quality", self.base_quality), 0.0, 1.0)
        )
        # The joint outcome head is trained and validated inside the backbone
        # development split.  It is diagnostic only: direct shot authority
        # requires an independently sealed FrozenShotHead that beats physics xG.
        self.shot_quality = 0.0
        self.cross_validation = replay_cross_action_validation(
            validation.get("cross_action_validation")
        )
        self.cross_quality = float(self.cross_validation["quality"])
        self.frozen_shot_head = None
        self.shot_probability_source = "physics_xg_prior"
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
            expected_weighted_mse_by_branch={
                "cross": self.cross_validation.get(
                    "model_weighted_mse",
                    validation.get("weighted_obs_mse", 0.02),
                ),
            },
        )

    @classmethod
    def load(
        cls, path: str, *, shot_head_path: str | None = None,
    ) -> "WorldModelRuntime":
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
        if shot_head_path:
            rt.attach_shot_head(shot_head_path)
        kind = getattr(model, "transition_kind", cfg.transition_type)
        print(f"  [WORLD_MODEL] transition={kind} latent={cfg.latent_dim} (meta rows={meta.get('rows', '?')})")
        print(
            f"  [WORLD_MODEL] checkpoint=v{getattr(model, 'checkpoint_version', 2)} "
            f"quality pass={rt.pass_quality:.3f} cross={rt.cross_quality:.3f} "
            f"shot={rt.shot_quality:.3f} "
            f"joint_shot_diagnostic={rt.joint_shot_quality:.3f} "
            f"shot_source={rt.shot_probability_source} "
            f"dynamics_members={model.transition_member_count} "
            f"trained={model.transition_ensemble_trained}"
        )
        return rt

    @classmethod
    def load_default(
        cls, base_dir: str, *, required: bool = False,
    ) -> Optional["WorldModelRuntime"]:
        path = default_checkpoint_path(base_dir)
        if not os.path.isfile(path):
            message = f"world-model checkpoint not found: {path}"
            if required:
                raise FileNotFoundError(message)
            print(f"  [WORLD_MODEL] {message}")
            return None
        try:
            rt = cls.load(path, shot_head_path=default_shot_head_path(base_dir))
            print(f"  [WORLD_MODEL] Loaded from {path}")
            return rt
        except Exception as exc:
            if required:
                raise RuntimeError(
                    f"required world model failed to load: {path}"
                ) from exc
            print(f"  [WORLD_MODEL] Failed to load: {exc}")
            return None

    def reset_hidden(self) -> None:
        self._hidden = None

    def attach_shot_head(self, path: str) -> None:
        from src.match_engine.world_model.shot_head import FrozenShotHead

        head = FrozenShotHead.load(path)
        head.assert_compatible(self.checkpoint_signature)
        self.frozen_shot_head = head
        self.shot_quality = head.quality
        self.shot_probability_source = "frozen_backbone_shot_head"

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
        self.last_decision_uncertainty = compose_uncertainty(
            float(np.clip(2.0 * model_epistemic, 0.0, 1.0)),
            aleatoric,
        )
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
        authority = self.planner_authority(obs, kind=kind)
        if not authority["authorized"]:
            return 0.0
        return float(authority["validation_quality"] * authority["decision_confidence"])

    def planner_authority(self, obs: np.ndarray, kind: str = "general") -> dict:
        """Separate checkpoint authorization from state-specific authority.

        Validation quality is used once, as a hard release gate.  Observation
        coverage and online calibration then determine how much of the bounded
        action-policy budget may be used for this decision.
        """
        coverage = observation_coverage(obs)
        quality = self.base_quality
        if kind == "pass":
            quality = self.pass_quality
        elif kind == "cross":
            quality = self.cross_quality
        elif kind == "shot":
            quality = self.shot_quality
        threshold = float(self.cfg.min_planner_quality)
        authorized = bool(quality >= threshold)
        trust = float(np.clip(self.online_calibrator.trust_factor(kind), 0.0, 1.0))
        decision_confidence = float(np.clip(
            (0.25 + 0.75 * coverage) * trust, 0.0, 1.0,
        )) if authorized else 0.0
        return {
            "authorized": authorized,
            "validation_quality": float(quality),
            "minimum_validation_quality": threshold,
            "observation_coverage": float(coverage),
            "online_trust": trust,
            "decision_confidence": decision_confidence,
        }

    def policy_utility_authority(self, action_kind: str) -> dict:
        """Authorize M2 value use only from action-specific grouped holdout."""
        validation = self.meta.get("validation", {}) if isinstance(
            self.meta, dict
        ) else {}
        return policy_utility_validation_gate(
            validation.get("policy_utility"),
            action_kind=action_kind,
        )

    def policy_utility_sequence_authority(
        self,
        action_kind: str,
        *,
        rollout_steps: int = 2,
    ) -> dict:
        """Authorize changing-action utility only at an explicitly proven depth."""
        validation = self.meta.get("validation", {}) if isinstance(
            self.meta, dict
        ) else {}
        return policy_utility_sequence_validation_gate(
            validation.get("policy_utility_two_step"),
            action_kind=action_kind,
            rollout_steps=rollout_steps,
        )

    def observe_transition(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        next_observation: np.ndarray,
    ) -> dict[str, float]:
        """Calibrate against the exact transition target used at inference."""
        action_kind = decode_action_kind(action)
        quality_kind = _quality_kind_for_action(action_kind)
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
        quality_kind = _quality_kind_for_action(action_kind)
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
        """Authorize search only after trained and calibrated two-step evidence."""
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
        training_pairs = int(evidence.get("training_pairs", 0))
        training_groups = int(evidence.get("training_groups", 0))
        training_weight = finite_float(
            evidence.get("max_curriculum_weight_applied") or 0.0, 0.0,
        )
        optimization_steps = int(evidence.get("optimization_steps", 0))
        trained_multi_step = bool(evidence.get(
            "trained_with_autoregressive_multi_step_objective", False,
        ))
        ensemble_gain = finite_float(
            evidence.get("ensemble_gain_vs_member_mean") or 0.0, 0.0,
        )
        disagreement_correlation = finite_float(
            evidence.get("disagreement_error_correlation") or 0.0, 0.0,
        )
        calibration_reported = bool(
            "ensemble_gain_vs_member_mean" in evidence
            and "disagreement_error_correlation" in evidence
        )
        training_contract = bool(
            trained_multi_step
            and evidence.get("autoregressive_predicted_state") is True
            and evidence.get("action_sequence") == "observed_changing_actions"
            and training_pairs >= 32
            and training_groups >= 4
            and training_weight > 0.0
            and optimization_steps > 0
        )
        ensemble_trained = bool(getattr(
            self.model, "transition_ensemble_trained", False,
        ))
        active = bool(
            ensemble_trained
            and training_contract
            and calibration_reported
            and samples >= 32
            and groups >= 4
            and baseline_error > 1e-10
            and model_error < baseline_error
            and skill >= 0.02
            and ensemble_gain >= -1e-8
            and disagreement_correlation >= 0.0
        )
        sample_factor = samples / (samples + 128.0)
        group_factor = groups / (groups + 8.0)
        skill_factor = float(np.clip(skill / 0.25, 0.0, 1.0))
        calibration_factor = float(np.clip(
            0.50 + disagreement_correlation, 0.25, 1.0,
        ))
        authority = float(np.clip(
            0.50 * sample_factor * group_factor * skill_factor
            * calibration_factor
            if active else 0.0,
            0.0,
            0.50,
        ))
        return {
            "version": 2,
            "active": active,
            "authority": authority,
            "reason": (
                "trained_calibrated_grouped_two_step_gain"
                if active
                else (
                    "two_step_training_contract_missing"
                    if not training_contract
                    else "two_step_evidence_gate_closed"
                )
            ),
            "samples": samples,
            "groups": groups,
            "weighted_mse": model_error,
            "persistence_weighted_mse": baseline_error,
            "skill_vs_persistence": skill,
            "transition_ensemble_trained": ensemble_trained,
            "training_pairs": training_pairs,
            "training_groups": training_groups,
            "max_curriculum_weight_applied": training_weight,
            "optimization_steps": optimization_steps,
            "trained_multi_step": trained_multi_step,
            "ensemble_gain_vs_member_mean": ensemble_gain,
            "disagreement_error_correlation": disagreement_correlation,
            "calibration_reported": calibration_reported,
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
    ) -> dict[str, object]:
        """Predict the same declared utility later measured by policy outcomes."""
        rollout_steps = max(1, int(np.ceil(
            max(0.1, float(horizon_s)) / HORIZON_SCALE_SECONDS
        )))
        segment_horizon_s = max(0.1, float(horizon_s)) / rollout_steps
        conditioned_action = np.asarray(action, dtype=np.float32).copy()
        conditioned_action[HORIZON_INDEX] = float(np.clip(
            segment_horizon_s / HORIZON_SCALE_SECONDS, 0.0, 1.0,
        ))
        quality_kind = _quality_kind_for_action(action_kind)
        output = self.imagine(
            obs,
            conditioned_action,
            steps=rollout_steps,
            carry_hidden=False,
            quality_kind=quality_kind,
        )
        current = np.asarray(obs, dtype=float)
        future = np.asarray(output.next_obs, dtype=float)
        components = transition_policy_utility_numpy(
            current,
            future,
            attacking_home=attacking_home,
        )
        progress = float(np.asarray(components["territorial_delta"]))
        goal_delta = float(np.asarray(components["goal_diff_delta"]))
        possession_delta = float(np.asarray(components["possession_delta"]))
        possession_home = float(np.clip(future[209], 0.0, 1.0))
        retention_probability = (
            possession_home if attacking_home else 1.0 - possession_home
        )
        epistemic_uncertainty = compound_uncertainty(
            output.epistemic_uncertainty, rollout_steps,
        )
        aleatoric_uncertainty = compound_uncertainty(
            output.aleatoric_uncertainty, rollout_steps,
        )
        uncertainty = compose_uncertainty(
            epistemic_uncertainty, aleatoric_uncertainty,
        )
        utility = float(np.asarray(components["policy_utility"]))
        transition_samples = (output.uncertainty_samples or {}).get(
            "transition_states",
            np.asarray(future, dtype=np.float32).reshape(1, 1, -1),
        )
        ensemble_trained = bool((output.uncertainty_samples or {}).get(
            "transition_ensemble_trained", False,
        ))
        state_scales = multiscale_state_forecast(
            current,
            transition_samples,
            attacking_home=attacking_home,
            horizon_s=horizon_s,
            ensemble_trained=ensemble_trained,
            transition_trajectory_samples=(
                (output.uncertainty_samples or {}).get(
                    "transition_state_trajectory"
                )
            ),
        )
        future_members = np.asarray(transition_samples, dtype=np.float32)
        if future_members.ndim == 2:
            future_members = future_members[:, None, :]
        with torch.no_grad():
            learned_event_probabilities = torch.sigmoid(
                self.model.semantic_event_logits(
                    torch.from_numpy(current.astype(np.float32)).unsqueeze(0),
                    torch.from_numpy(future_members),
                    torch.from_numpy(conditioned_action).unsqueeze(0),
                )
            ).mean(dim=0).squeeze(0).numpy()
        projected_events = dict(
            state_scales["semantic_event_probabilities"]
        )
        event_fusion = {}
        for index, event in enumerate(FALSIFIABLE_SEMANTIC_EVENTS):
            gate = self.semantic_event_head_gate(
                event, rollout_steps=rollout_steps,
            )
            projected = float(projected_events[event])
            learned = float(np.clip(
                learned_event_probabilities[index], 0.0, 1.0,
            ))
            authority = float(gate["authority"])
            fused = float(np.clip(
                projected + authority * (learned - projected), 0.0, 1.0,
            ))
            projected_events[event] = fused
            event_fusion[event] = {
                "projection_probability": projected,
                "learned_probability": learned,
                "fused_probability": fused,
                "gate": gate,
            }
        state_scales["semantic_event_probabilities"] = projected_events
        state_scales["semantic_event_fusion"] = event_fusion
        active_learned_events = [
            event for event, audit in event_fusion.items()
            if audit["gate"]["active"]
        ]
        state_scales["claims"]["learned_event_heads_active"] = (
            active_learned_events
        )
        if active_learned_events:
            state_scales["claims"]["probability_source"] = (
                "per_event_grouped_validation_gated_blend"
            )
        with torch.no_grad():
            learned_path_probabilities = (
                self.model.semantic_path_joint_probabilities(
                    self.model.semantic_path_logits(
                        torch.from_numpy(
                            current.astype(np.float32)
                        ).unsqueeze(0),
                        torch.from_numpy(future_members),
                        torch.from_numpy(conditioned_action).unsqueeze(0),
                    )
                ).mean(dim=0).squeeze(0).numpy()
            )
        path_fusion = {}
        projected_paths = state_scales.get("predictive_mechanism_paths") or {}
        for path_index, path in enumerate(PREDICTIVE_MECHANISM_PATHS):
            path_key = "->".join(path)
            projection = projected_paths.get(path_key) or {}
            raw_joint = projection.get("joint_probabilities") or {}
            if len(raw_joint) != 8:
                continue
            projected_joint = np.asarray(
                list(raw_joint.values()), dtype=np.float64,
            )
            learned_joint = np.asarray(
                learned_path_probabilities[path_index], dtype=np.float64,
            )
            gate = self.semantic_path_head_gate(
                path_key, rollout_steps=rollout_steps,
            )
            authority = float(gate["authority"])
            fused_joint = projected_joint + authority * (
                learned_joint - projected_joint
            )
            fused_joint = np.clip(fused_joint, 1e-9, None)
            fused_joint /= fused_joint.sum()
            statistics = semantic_path_statistics(fused_joint)
            projection.update(statistics)
            projection["probability_source"] = (
                "grouped_validated_autoregressive_path_blend"
                if gate["active"] else "member_aligned_three_event_projection"
            )
            path_fusion[path_key] = {
                "projection_joint_probabilities": dict(raw_joint),
                "learned_joint_probabilities": dict(zip(
                    raw_joint, map(float, learned_joint),
                )),
                "fused_joint_probabilities": dict(
                    statistics["joint_probabilities"]
                ),
                "gate": gate,
            }
        state_scales["semantic_path_fusion"] = path_fusion
        state_scales["claims"]["learned_path_heads_active"] = [
            key for key, audit in path_fusion.items()
            if audit["gate"]["active"]
        ]
        from src.match_engine.world_model.distributional_utility import (
            member_policy_utility_distribution,
        )

        distributional_utility = member_policy_utility_distribution(
            current,
            transition_samples,
            (output.uncertainty_samples or {}).get(
                "progress", np.zeros((len(future_members), 1)),
            ),
            attacking_home=attacking_home,
            rollout_steps=rollout_steps,
            ensemble_trained=ensemble_trained,
        )
        if distributional_utility.get("available"):
            distributional_utility["aggregate_mean_identity_error"] = abs(
                float(distributional_utility["mean_utility"])
                - float(utility)
            )
        return {
            "prediction_source": "autoregressive_action_persistence_rollout",
            "horizon_s": float(horizon_s),
            "rollout_steps": rollout_steps,
            "segment_horizon_s": segment_horizon_s,
            "policy_utility": float(utility),
            "policy_utility_version": POLICY_UTILITY_VERSION,
            "progress": progress,
            "retention_probability": retention_probability,
            "possession_delta": possession_delta,
            "territorial_value_delta": progress,
            "xg_net_delta": progress,
            "goal_diff_delta": goal_delta,
            "uncertainty": uncertainty,
            "epistemic_uncertainty": epistemic_uncertainty,
            "aleatoric_uncertainty": aleatoric_uncertainty,
            "uncertainty_source": output.uncertainty_source,
            "uncertainty_components": output.uncertainty_components or {},
            "state_scales": state_scales,
            "semantic_event_probabilities": dict(
                state_scales["semantic_event_probabilities"]
            ),
            "distributional_policy_utility": distributional_utility,
        }

    def predict_policy_utility_sequence(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        *,
        action_kind: str,
        attacking_home: bool,
    ) -> dict[str, object]:
        """Evaluate an explicit two-action plan with member-consistent dynamics.

        This is deliberately separate from ``predict_policy_utility``. The
        latter repeats one action when a long horizon spans several model
        steps, while M2 sequence evidence is trained on observed action
        sequences. Runtime action control must therefore supply both actions
        explicitly and expose the exact sequence in its audit record.
        """
        clean_obs = np.clip(
            np.nan_to_num(
                np.asarray(obs, dtype=np.float32),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ),
            0.0,
            1.0,
        )
        if clean_obs.shape != (OBS_DIM,):
            raise ValueError(
                f"M2 sequence observation must have shape [{OBS_DIM}]"
            )
        raw_actions = np.asarray(actions, dtype=np.float32)
        if raw_actions.ndim != 2 or raw_actions.shape != (2, ACTION_DIM):
            raise ValueError(
                f"M2 sequence actions must have shape [2, {ACTION_DIM}]"
            )
        if not np.isfinite(raw_actions).all():
            raise ValueError("M2 sequence actions must be finite")
        clean_actions = strip_outcome_leakage(raw_actions)
        clean_actions[:, HORIZON_INDEX] = np.clip(
            clean_actions[:, HORIZON_INDEX], 0.0, 1.0,
        )
        action_sequence = [
            decode_action_kind(action) for action in clean_actions
        ]
        if any(kind == "other" for kind in action_sequence):
            raise ValueError("M2 sequence actions must have explicit action kinds")
        with torch.no_grad():
            member_predictions = self.model.transition_rollout_predictions(
                torch.from_numpy(clean_obs).unsqueeze(0),
                torch.from_numpy(clean_actions).unsqueeze(0),
            )
        future_members = member_predictions.detach().cpu().numpy()
        future = np.asarray(
            future_members.mean(axis=0).squeeze(0), dtype=np.float64,
        )
        components = transition_policy_utility_numpy(
            np.asarray(clean_obs, dtype=np.float64),
            future,
            attacking_home=attacking_home,
        )
        ensemble = transition_ensemble_uncertainty(
            future_members,
            trained=bool(getattr(
                self.model, "transition_ensemble_trained", False,
            )),
        )
        epistemic_uncertainty = float(ensemble["epistemic_uncertainty"])
        aleatoric_uncertainty = compound_uncertainty(
            self.progress_aleatoric_scale, 2,
        )
        uncertainty = compose_uncertainty(
            epistemic_uncertainty, aleatoric_uncertainty,
        )
        self.last_decision_uncertainty = uncertainty
        self.last_uncertainty = uncertainty
        possession_home = float(np.clip(future[209], 0.0, 1.0))
        segment_horizons_s = [
            float(action[HORIZON_INDEX] * HORIZON_SCALE_SECONDS)
            for action in clean_actions
        ]
        return {
            "prediction_source": "explicit_changing_action_sequence_rollout",
            "planning_mode": "open_loop_shared_continuation_policy",
            "rollout_steps": 2,
            "action_sequence": action_sequence,
            "segment_horizons_s": segment_horizons_s,
            "horizon_s": float(sum(segment_horizons_s)),
            "policy_utility": float(np.asarray(
                components["policy_utility"],
            )),
            "policy_utility_version": POLICY_UTILITY_VERSION,
            "progress": float(np.asarray(
                components["territorial_delta"],
            )),
            "retention_probability": (
                possession_home if attacking_home else 1.0 - possession_home
            ),
            "possession_delta": float(np.asarray(
                components["possession_delta"],
            )),
            "goal_diff_delta": float(np.asarray(
                components["goal_diff_delta"],
            )),
            "uncertainty": uncertainty,
            "epistemic_uncertainty": epistemic_uncertainty,
            "aleatoric_uncertainty": aleatoric_uncertainty,
            "uncertainty_source": (
                "transition_ensemble_plus_compounded_progress_residual"
            ),
            "uncertainty_components": ensemble,
            "sequence_gate": self.policy_utility_sequence_authority(
                action_kind, rollout_steps=2,
            ),
        }

    def semantic_event_head_gate(
        self,
        event: str,
        *,
        rollout_steps: int,
    ) -> dict[str, object]:
        """Authorize learned event blending only for its exact validated depth."""
        validation = self.meta.get("validation", {}) if isinstance(
            self.meta, dict
        ) else {}
        contract = validation.get("semantic_event_heads") or {}
        steps = max(1, int(rollout_steps))
        scope = "one_step" if steps == 1 else (
            "two_step" if steps == 2 else "unsupported_depth"
        )
        scope_report = contract.get(scope) or {}
        profile = (scope_report.get("events") or {}).get(str(event)) or {}
        def safe_int(name: str, default: int = 0) -> int:
            try:
                return int(profile.get(name, default))
            except (TypeError, ValueError):
                return int(default)

        def safe_float(name: str, default: float) -> float:
            try:
                return finite_float(float(profile.get(name, default)), default)
            except (TypeError, ValueError):
                return float(default)

        samples = safe_int("samples")
        groups = safe_int("groups")
        positives = safe_int("positives")
        negatives = safe_int("negatives")
        learned_brier = safe_float("learned_brier", 1.0)
        baseline_brier = safe_float("best_baseline_brier", 0.0)
        skill = safe_float("skill_vs_best_baseline", -1.0)
        ensemble_gain = safe_float("ensemble_gain_vs_member_mean", -1.0)
        calibration_error = safe_float("calibration_error", 1.0)
        disagreement_correlation = safe_float(
            "disagreement_error_correlation", -1.0,
        )
        trained = bool(
            getattr(self.model, "semantic_event_heads_trained", False)
            and contract.get("trained")
        )
        optimization_key = (
            "one_step_optimization_steps"
            if steps == 1 else "two_step_optimization_steps"
        )
        try:
            optimization_steps = int(contract.get(optimization_key, 0))
        except (TypeError, ValueError):
            optimization_steps = 0
        active = bool(
            event in FALSIFIABLE_SEMANTIC_EVENTS
            and scope != "unsupported_depth"
            and trained
            and optimization_steps > 0
            and samples >= 32
            and groups >= 4
            and positives >= 4
            and negatives >= 4
            and baseline_brier > 1e-10
            and learned_brier < baseline_brier
            and skill >= 0.01
            and ensemble_gain >= -1e-8
            and disagreement_correlation >= 0.0
            and calibration_error <= 0.20
        )
        sample_factor = samples / (samples + 128.0)
        group_factor = groups / (groups + 8.0)
        support_factor = min(positives, negatives) / (
            min(positives, negatives) + 16.0
        )
        skill_factor = float(np.clip(skill / 0.20, 0.0, 1.0))
        calibration_factor = float(np.clip(
            1.0 - calibration_error / 0.20, 0.0, 1.0,
        ))
        authority = float(np.clip(
            0.50 * sample_factor * group_factor * support_factor
            * skill_factor * calibration_factor if active else 0.0,
            0.0,
            0.50,
        ))
        return {
            "version": 1,
            "active": active,
            "authority": authority,
            "reason": (
                "grouped_event_head_gain"
                if active else (
                    "rollout_depth_not_validated"
                    if scope == "unsupported_depth"
                    else "semantic_event_head_gate_closed"
                )
            ),
            "event": str(event),
            "rollout_steps": steps,
            "validation_scope": scope,
            "samples": samples,
            "groups": groups,
            "positives": positives,
            "negatives": negatives,
            "learned_brier": learned_brier,
            "best_baseline_brier": baseline_brier,
            "skill_vs_best_baseline": skill,
            "calibration_error": calibration_error,
            "disagreement_error_correlation": disagreement_correlation,
            "optimization_steps": optimization_steps,
            "heads_trained": trained,
        }

    def semantic_path_head_gate(
        self, path: str, *, rollout_steps: int,
    ) -> dict[str, object]:
        """Authorize learned joint blending only for exact path and depth."""
        validation = self.meta.get("validation", {}) if isinstance(
            self.meta, dict
        ) else {}
        contract = validation.get("semantic_path_heads") or {}
        steps = max(1, int(rollout_steps))
        scope = "one_step" if steps == 1 else (
            "two_step" if steps == 2 else "unsupported_depth"
        )
        profile = ((contract.get(scope) or {}).get("paths") or {}).get(
            str(path),
        ) or {}
        def safe_int(name: str) -> int:
            try:
                return int(profile.get(name, 0))
            except (TypeError, ValueError):
                return 0

        def safe_float(name: str, default: float) -> float:
            try:
                return finite_float(float(profile.get(name, default)), default)
            except (TypeError, ValueError):
                return float(default)

        samples = safe_int("samples")
        groups = safe_int("groups")
        occupied = safe_int("occupied_cells")
        positives = safe_int("completion_positives")
        negatives = safe_int("completion_negatives")
        learned_brier = safe_float("learned_brier", 2.0)
        baseline_brier = safe_float("best_baseline_brier", 0.0)
        skill = safe_float("skill_vs_best_baseline", -1.0)
        ensemble_gain = safe_float("ensemble_gain_vs_member_mean", -1.0)
        optimization_key = (
            "one_step_optimization_steps" if steps == 1
            else "two_step_optimization_steps"
        )
        try:
            optimization_steps = int(contract.get(optimization_key, 0))
        except (TypeError, ValueError):
            optimization_steps = 0
        trained = bool(
            getattr(self.model, "semantic_path_heads_trained", False)
            and contract.get("trained")
        )
        known_paths = {"->".join(value) for value in PREDICTIVE_MECHANISM_PATHS}
        active = bool(
            path in known_paths and scope != "unsupported_depth" and trained
            and optimization_steps > 0 and samples >= 48 and groups >= 4
            and occupied >= 4 and positives >= 4 and negatives >= 4
            and baseline_brier > 1e-10 and learned_brier < baseline_brier
            and skill >= 0.01 and ensemble_gain >= -1e-8
        )
        sample_factor = samples / (samples + 192.0)
        group_factor = groups / (groups + 8.0)
        support_factor = min(positives, negatives) / (
            min(positives, negatives) + 16.0
        )
        cell_factor = occupied / 8.0
        skill_factor = float(np.clip(skill / 0.20, 0.0, 1.0))
        authority = float(np.clip(
            0.50 * sample_factor * group_factor * support_factor
            * cell_factor * skill_factor if active else 0.0,
            0.0, 0.50,
        ))
        return {
            "version": 1, "active": active, "authority": authority,
            "reason": (
                "grouped_autoregressive_path_head_gain" if active
                else "rollout_depth_not_validated"
                if scope == "unsupported_depth" else "semantic_path_head_gate_closed"
            ),
            "path": str(path), "rollout_steps": steps,
            "validation_scope": scope, "samples": samples,
            "groups": groups, "occupied_cells": occupied,
            "completion_positives": positives,
            "completion_negatives": negatives,
            "learned_brier": learned_brier,
            "best_baseline_brier": baseline_brier,
            "skill_vs_best_baseline": skill,
            "ensemble_gain_vs_member_mean": ensemble_gain,
            "optimization_steps": optimization_steps,
            "heads_trained": trained,
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

    def score_cross_action(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        *,
        attacking_home: bool,
    ) -> float:
        """Score cross and continuation through cross-validated transitions."""
        obs = np.nan_to_num(
            np.asarray(obs, dtype=float), nan=0.0, posinf=1.0, neginf=-1.0,
        )
        action = np.nan_to_num(
            np.asarray(action, dtype=float), nan=0.0, posinf=1.0, neginf=-1.0,
        )
        out = self.imagine(
            obs, action, carry_hidden=False, quality_kind="cross",
        )
        direction = 1.0 if attacking_home else -1.0
        next_ball_x = finite_float(float(out.next_obs[200]), 0.5)
        current_ball_x = finite_float(float(obs[200]), 0.5)
        state_delta = float(np.clip(
            direction * (next_ball_x - current_ball_x), -0.5, 0.5,
        ))
        learned_delta = float(np.clip(
            finite_float(out.progress_delta, 0.0), -0.5, 0.5,
        ))
        possession_home = float(np.clip(
            finite_float(float(out.next_obs[209]), 0.5), 0.0, 1.0,
        ))
        retention = possession_home if attacking_home else 1.0 - possession_home
        retention_edge = 2.0 * retention - 1.0
        return finite_float(
            0.50 * state_delta + 0.35 * learned_delta + 0.15 * retention_edge,
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
        xg_prior = float(np.clip(action[13], 0.0, 1.0))
        if self.frozen_shot_head is not None:
            with torch.no_grad():
                latent = self.model.encode(
                    torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
                ).squeeze(0).numpy()
            goal_term = self.frozen_shot_head.predict(latent, action)
        else:
            goal_term = xg_prior
        authority = self.planner_authority(obs, kind="shot")
        self.last_decision_uncertainty = float(np.clip(
            1.0 - float(authority["decision_confidence"]), 0.0, 1.0,
        ))
        self.last_uncertainty = self.last_decision_uncertainty
        return finite_float(float(np.clip(goal_term, 0.0, 1.0)), xg_prior)

    def encode_state(self, state, *, attacking_home: bool | None = None) -> np.ndarray:
        return encode_observation(state, attacking_home=attacking_home, cfg=self.cfg)

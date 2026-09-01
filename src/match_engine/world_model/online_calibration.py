"""Online same-target transition calibration for world-model planning trust."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from src.match_engine.world_model.observation import OBS_DIM
from src.match_engine.world_model.schema import observation_loss_weights


CALIBRATION_BRANCHES = ("general", "pass", "cross", "shot")


def action_calibration_branch(action_kind: str) -> str:
    if action_kind == "shot":
        return "shot"
    if action_kind == "cross":
        return "cross"
    return "pass"


@dataclass
class _BranchState:
    samples: int = 0
    model_error_sum: float = 0.0
    persistence_error_sum: float = 0.0
    ewma_error: float = 0.0
    ewma_uncertainty: float = 0.0
    error_samples: list[float] = field(default_factory=list)
    uncertainty_samples: list[float] = field(default_factory=list)


class OnlineTransitionCalibrator:
    """Reduce planning trust when live transition error exceeds validation."""

    def __init__(
        self,
        *,
        expected_weighted_mse: float,
        expected_weighted_mse_by_branch: dict[str, float] | None = None,
        min_samples: int = 8,
        ewma_alpha: float = 0.08,
    ) -> None:
        expected = float(expected_weighted_mse)
        self.expected_weighted_mse = (
            max(1e-6, expected) if math.isfinite(expected) else 0.02
        )
        self.expected_weighted_mse_by_branch = {
            branch: self.expected_weighted_mse
            for branch in CALIBRATION_BRANCHES
        }
        for branch, value in (expected_weighted_mse_by_branch or {}).items():
            if branch not in self.expected_weighted_mse_by_branch:
                continue
            try:
                branch_value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(branch_value) and branch_value > 0.0:
                self.expected_weighted_mse_by_branch[branch] = max(
                    1e-6, branch_value,
                )
        self.min_samples = max(2, int(min_samples))
        self.ewma_alpha = float(np.clip(ewma_alpha, 0.01, 1.0))
        self._branches = {
            name: _BranchState() for name in CALIBRATION_BRANCHES
        }

    @staticmethod
    def _validate_vector(value: Any) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        if vector.shape != (OBS_DIM,):
            raise ValueError(f"online calibration vector must have shape ({OBS_DIM},)")
        if not np.isfinite(vector).all():
            raise ValueError("online calibration vectors must be finite")
        return vector

    def _update_branch(
        self,
        branch: str,
        *,
        model_error: float,
        persistence_error: float,
        uncertainty: float,
    ) -> None:
        state = self._branches[branch]
        state.samples += 1
        state.model_error_sum += model_error
        state.persistence_error_sum += persistence_error
        alpha = self.ewma_alpha
        if state.samples == 1:
            state.ewma_error = model_error
            state.ewma_uncertainty = uncertainty
        else:
            state.ewma_error = (
                (1.0 - alpha) * state.ewma_error + alpha * model_error
            )
            state.ewma_uncertainty = (
                (1.0 - alpha) * state.ewma_uncertainty + alpha * uncertainty
            )
        state.error_samples.append(model_error)
        state.uncertainty_samples.append(uncertainty)

    def observe(
        self,
        *,
        action_kind: str,
        current_observation: Any,
        predicted_observation: Any,
        actual_observation: Any,
        predicted_uncertainty: float,
    ) -> dict[str, float]:
        current = self._validate_vector(current_observation)
        prediction = self._validate_vector(predicted_observation)
        actual = self._validate_vector(actual_observation)
        weights = observation_loss_weights().astype(np.float64)
        model_error = float(np.mean(
            np.square(prediction - actual, dtype=np.float64) * weights
        ))
        persistence_error = float(np.mean(
            np.square(current - actual, dtype=np.float64) * weights
        ))
        try:
            uncertainty_value = float(predicted_uncertainty)
        except (TypeError, ValueError):
            uncertainty_value = 1.0
        uncertainty = float(np.clip(
            uncertainty_value if math.isfinite(uncertainty_value) else 1.0,
            0.0, 1.0,
        ))
        branch = action_calibration_branch(action_kind)
        self._update_branch(
            branch,
            model_error=model_error,
            persistence_error=persistence_error,
            uncertainty=uncertainty,
        )
        self._update_branch(
            "general",
            model_error=model_error,
            persistence_error=persistence_error,
            uncertainty=uncertainty,
        )
        return {
            "weighted_mse": model_error,
            "persistence_weighted_mse": persistence_error,
            "trust_factor": self.trust_factor(branch),
        }

    def trust_factor(self, branch: str = "general") -> float:
        state = self._branches.get(branch, self._branches["general"])
        if state.samples < self.min_samples:
            return 1.0
        expected = self.expected_weighted_mse_by_branch.get(
            branch, self.expected_weighted_mse,
        )
        error_ratio = state.ewma_error / expected
        error_factor = float(np.clip(
            1.0 / math.sqrt(max(1.0, error_ratio)), 0.25, 1.0,
        ))
        persistence = state.persistence_error_sum / max(1, state.samples)
        model = state.model_error_sum / max(1, state.samples)
        skill = 1.0 - model / max(persistence, 1e-8)
        skill_factor = 1.0 if skill >= 0.0 else float(np.clip(
            math.exp(2.0 * skill), 0.25, 1.0,
        ))
        return min(error_factor, skill_factor)

    def _branch_diagnostics(self, branch: str) -> dict[str, Any]:
        state = self._branches[branch]
        model = state.model_error_sum / max(1, state.samples)
        persistence = state.persistence_error_sum / max(1, state.samples)
        if len(state.error_samples) >= 3:
            errors = np.asarray(state.error_samples, dtype=float)
            uncertainties = np.asarray(state.uncertainty_samples, dtype=float)
            if np.std(errors) > 1e-12 and np.std(uncertainties) > 1e-12:
                correlation = float(np.corrcoef(errors, uncertainties)[0, 1])
            else:
                correlation = 0.0
        else:
            correlation = 0.0
        expected = self.expected_weighted_mse_by_branch[branch]
        return {
            "samples": state.samples,
            "mean_weighted_mse": model,
            "ewma_weighted_mse": state.ewma_error,
            "expected_weighted_mse": expected,
            "error_ratio": (
                state.ewma_error / expected
                if state.samples else 0.0
            ),
            "persistence_weighted_mse": persistence,
            "skill_vs_persistence": (
                1.0 - model / persistence if persistence > 1e-8 else 0.0
            ),
            "mean_uncertainty_ewma": state.ewma_uncertainty,
            "uncertainty_error_correlation": correlation,
            "trust_factor": self.trust_factor(branch),
            "active": state.samples >= self.min_samples,
        }

    def diagnostics(self) -> dict[str, Any]:
        return {
            "version": 1,
            "calibration_target": "same_tick_next_observation",
            "min_samples": self.min_samples,
            "branches": {
                branch: self._branch_diagnostics(branch)
                for branch in CALIBRATION_BRANCHES
            },
            "policy": (
                "Online evidence may only reduce planner trust and cannot "
                "reopen a checkpoint quality gate."
            ),
        }

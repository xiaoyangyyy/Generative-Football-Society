"""Joint receiver-choice and completion residual model with physics anchoring."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


PASS_FEATURES = (
    "distance",
    "forward_progress",
    "lateral_change",
    "receiver_defender_distance",
    "lane_clearance",
    "receiver_arrival_margin",
    "lane_intercept_margin",
    "body_alignment",
    "pressure_change",
    "preferred_foot_alignment",
)


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0))))


@dataclass(frozen=True)
class JointPassPrediction:
    receiver_utility: float
    completion_probability: float
    epistemic_uncertainty: float


@dataclass(frozen=True)
class JointPassModel:
    mean: np.ndarray
    scale: np.ndarray
    receiver_weights: np.ndarray
    completion_weights: np.ndarray
    receiver_bias: float = 0.0
    completion_bias: float = 0.0
    residual_limit: float = 0.65
    calibration_scale: float = 1.0
    calibration_bias: float = 0.0
    calibration_knots: tuple[float, ...] = ()
    calibration_values: tuple[float, ...] = ()
    receiver_rank_based: bool = False

    @classmethod
    def load(cls, path: str | Path) -> "JointPassModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") not in {2, 3} or tuple(payload.get("features", ())) != PASS_FEATURES:
            raise ValueError("unsupported joint pass model schema")
        return cls(
            *(np.asarray(payload[name], dtype=float) for name in ("mean", "scale", "receiver_weights", "completion_weights")),
            receiver_bias=float(payload.get("receiver_bias", 0.0)),
            completion_bias=float(payload.get("completion_bias", 0.0)),
            residual_limit=float(payload.get("residual_limit", 0.65)),
            calibration_scale=float(payload.get("calibration_scale", 1.0)),
            calibration_bias=float(payload.get("calibration_bias", 0.0)),
            calibration_knots=tuple(payload.get("calibration_knots", ())),
            calibration_values=tuple(payload.get("calibration_values", ())),
            receiver_rank_based=bool(payload.get("receiver_rank_based", False)),
        )

    def to_dict(self) -> dict:
        return {
            "version": 3, "features": list(PASS_FEATURES),
            "mean": self.mean.tolist(), "scale": self.scale.tolist(),
            "receiver_weights": self.receiver_weights.tolist(),
            "completion_weights": self.completion_weights.tolist(),
            "receiver_bias": self.receiver_bias, "completion_bias": self.completion_bias,
            "residual_limit": self.residual_limit,
            "calibration_scale": self.calibration_scale,
            "calibration_bias": self.calibration_bias,
            "calibration_knots": list(self.calibration_knots),
            "calibration_values": list(self.calibration_values),
            "receiver_rank_based": self.receiver_rank_based,
        }

    def __post_init__(self) -> None:
        expected = (len(PASS_FEATURES),)
        for vector in (self.mean, self.scale, self.receiver_weights, self.completion_weights):
            if np.asarray(vector).shape != expected or not np.isfinite(vector).all():
                raise ValueError(f"joint pass vectors must have shape {expected}")

    def predict(
        self,
        features: np.ndarray,
        *,
        physics_prior: float,
        missing_mask: np.ndarray | None = None,
    ) -> JointPassPrediction:
        values = np.asarray(features, dtype=float)
        if values.shape != (len(PASS_FEATURES),):
            raise ValueError("invalid joint pass feature shape")
        missing = ~np.isfinite(values)
        if missing_mask is not None:
            missing |= np.asarray(missing_mask, dtype=bool)
        filled = np.where(missing, self.mean, values)
        z = (filled - self.mean) / np.maximum(self.scale, 1e-8)
        receiver = float(z @ self.receiver_weights + self.receiver_bias) if not self.receiver_rank_based else self.receiver_bias
        residual = self.residual_limit * float(np.tanh(z @ self.completion_weights + self.completion_bias))
        prior = float(np.clip(physics_prior, 0.01, 0.99))
        raw_logit = np.log(prior / (1.0 - prior)) + residual
        completion = _sigmoid(self.calibration_scale * raw_logit + self.calibration_bias)
        if self.calibration_knots:
            completion = float(np.interp(completion, self.calibration_knots, self.calibration_values))
        uncertainty = float(np.clip(missing.mean() + 0.2 * abs(residual), 0.0, 1.0))
        return JointPassPrediction(receiver, completion, uncertainty)

    def score_receiver_candidates(self, features: np.ndarray) -> np.ndarray:
        """Score one event's candidate set using provider-invariant within-event ranks."""
        values = np.asarray(features, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(PASS_FEATURES):
            raise ValueError("invalid receiver candidate feature shape")
        if not self.receiver_rank_based:
            filled = np.where(np.isfinite(values), values, self.mean)
            return ((filled - self.mean) / np.maximum(self.scale, 1e-8)) @ self.receiver_weights + self.receiver_bias
        ranked = _rank_features(values)
        return ((ranked - 0.5) / 0.3) @ self.receiver_weights + self.receiver_bias


def _rank_features(values: np.ndarray) -> np.ndarray:
    ranked = np.full(values.shape, 0.5, dtype=float)
    for column in range(values.shape[1]):
        finite = np.flatnonzero(np.isfinite(values[:, column]))
        if not len(finite):
            continue
        series = values[finite, column]
        order = np.argsort(series, kind="stable")
        ordinal = np.empty(len(finite), dtype=float)
        ordinal[order] = (np.arange(len(finite)) + 0.5) / len(finite)
        ranked[finite, column] = ordinal
    return ranked


def pass_model_gate(candidate: dict, prior: dict) -> dict[str, bool]:
    """Require meaningful, calibrated, cross-match gains over the physics prior."""
    return {
        "auc_gain": float(candidate["auc"]) >= float(prior["auc"]) + 0.01,
        "brier_gain": float(candidate["brier"]) <= float(prior["brier"]) - 0.002,
        "ece_bounded": float(candidate["ece"]) <= 0.04,
        "cross_match_stable": float(candidate.get("worst_match_auc", 0.0)) >= 0.54,
        "enough_matches": int(candidate.get("matches", 0)) >= 8,
    }


def fit_joint_pass_model(
    features: np.ndarray,
    event_ids: np.ndarray,
    is_receiver: np.ndarray,
    completion: np.ndarray,
    physics_prior: np.ndarray,
    *,
    steps: int = 800,
    learning_rate: float = 0.04,
) -> JointPassModel:
    """Fit listwise receiver utility and physics-residual completion jointly."""
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(PASS_FEATURES):
        raise ValueError("invalid training feature matrix")
    event_ids = np.asarray(event_ids)
    receiver = np.asarray(is_receiver, dtype=float)
    completion = np.asarray(completion, dtype=float)
    prior = np.clip(np.asarray(physics_prior, dtype=float), 0.01, 0.99)
    finite = np.isfinite(x)
    counts = finite.sum(axis=0)
    mean = np.divide(np.where(finite, x, 0.0).sum(axis=0), counts, out=np.zeros(x.shape[1]), where=counts > 0)
    centered = np.where(finite, x - mean, 0.0)
    variance = np.divide((centered * centered).sum(axis=0), counts, out=np.ones(x.shape[1]), where=counts > 0)
    scale = np.sqrt(variance)
    z = (np.where(np.isfinite(x), x, mean) - mean) / np.maximum(scale, 1e-8)
    receiver_w = np.zeros(z.shape[1])
    completion_w = np.zeros(z.shape[1])
    receiver_b = completion_b = 0.0
    order = np.argsort(event_ids, kind="stable")
    z_rank = np.full_like(z[order], 0.0)
    receiver_rank = receiver[order]
    event_rank = event_ids[order]
    starts = np.r_[0, np.flatnonzero(event_rank[1:] != event_rank[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(event_rank)])
    valid_group = (np.add.reduceat(receiver_rank, starts) == 1.0) & (counts >= 2)
    for start, count, valid in zip(starts, counts, valid_group):
        if valid:
            z_rank[start:start + count] = (_rank_features(x[order][start:start + count]) - 0.5) / 0.3
    valid_rows = np.repeat(valid_group, counts)
    z_rank = z_rank[valid_rows]
    receiver_rank = receiver_rank[valid_rows]
    counts_rank = counts[valid_group]
    starts_rank = np.r_[0, np.cumsum(counts_rank)[:-1]]
    completion_mask = np.isfinite(completion)
    z_completion = z[completion_mask]
    completion_truth = completion[completion_mask]
    prior_completion = prior[completion_mask]
    for _ in range(max(1, steps)):
        grad_rank = np.zeros_like(receiver_w)
        grad_rank_b = 0.0
        if len(z_rank):
            rank_logits = np.clip(z_rank @ receiver_w + receiver_b, -20, 20)
            group_max = np.maximum.reduceat(rank_logits, starts_rank)
            exponent = np.exp(rank_logits - np.repeat(group_max, counts_rank))
            denominator = np.add.reduceat(exponent, starts_rank)
            rank_probability = exponent / np.repeat(denominator, counts_rank)
            rank_error = rank_probability - receiver_rank
            grad_rank = z_rank.T @ rank_error / len(counts_rank)
            grad_rank_b = float(rank_error.sum()) / len(counts_rank)
        receiver_w -= learning_rate * (grad_rank + 0.002 * receiver_w)
        receiver_b -= learning_rate * grad_rank_b
        if len(z_completion):
            prior_logit = np.log(prior_completion / (1.0 - prior_completion))
            logits = prior_logit + 0.65 * np.tanh(z_completion @ completion_w + completion_b)
            probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
            chain = probability - completion_truth
            chain *= 0.65 * (1.0 - np.tanh(z_completion @ completion_w + completion_b) ** 2)
            completion_w -= learning_rate * (z_completion.T @ chain / len(z_completion) + 0.002 * completion_w)
            completion_b -= learning_rate * float(chain.mean())
    return JointPassModel(mean, np.maximum(scale, 1e-8), receiver_w, completion_w, receiver_b, completion_b, receiver_rank_based=True)

"""A frozen-backbone shot head with an independent sealed-evidence contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.match_engine.world_model.action_codec import ACTION_DIM
from src.match_engine.world_model.schema import strip_outcome_leakage


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FrozenShotHead:
    base_checkpoint_sha256: str
    latent_dim: int
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float
    calibration_scale: float
    calibration_bias: float
    evidence: Mapping[str, Any]
    accepted: bool

    def __post_init__(self) -> None:
        feature_dim = int(self.latent_dim) + ACTION_DIM
        for value in (self.mean, self.scale, self.weights):
            if np.asarray(value).shape != (feature_dim,) or not np.isfinite(value).all():
                raise ValueError(f"shot-head vectors must have shape ({feature_dim},)")
        if not self.base_checkpoint_sha256.startswith("sha256:"):
            raise ValueError("shot head must bind to a full sha256 checkpoint signature")

    @property
    def quality(self) -> float:
        sealed = self.evidence.get("sealed_test") or {}
        samples = max(0, int(sealed.get("samples", 0)))
        goals = max(0, int(sealed.get("goals", 0)))
        model_brier = float(sealed.get("model_brier", 1.0))
        prior_brier = float(sealed.get("physics_xg_prior_brier", 1.0))
        skill = max(0.0, 1.0 - model_brier / max(prior_brier, 1e-12))
        confidence = np.sqrt(min(1.0, samples / 200.0)) * np.sqrt(
            min(1.0, goals / 20.0)
        )
        return float(np.clip(confidence * (0.50 + 0.50 * skill), 0.0, 1.0))

    def gate(self) -> dict[str, bool]:
        sealed = self.evidence.get("sealed_test") or {}
        model_brier = float(sealed.get("model_brier", 1.0))
        prior_brier = float(sealed.get("physics_xg_prior_brier", 1.0))
        return {
            "accepted_flag": bool(self.accepted),
            "samples": int(sealed.get("samples", 0)) >= 40,
            "goals": int(sealed.get("goals", 0)) >= 5,
            "proper_score_gain": model_brier < prior_brier,
            "sealed_only": bool(sealed.get("sealed_only", False)),
        }

    @property
    def enabled(self) -> bool:
        return all(self.gate().values())

    def assert_compatible(self, checkpoint_signature: str) -> None:
        if checkpoint_signature != self.base_checkpoint_sha256:
            raise ValueError("shot head was trained against a different world-model checkpoint")
        if not self.enabled:
            failed = [name for name, passed in self.gate().items() if not passed]
            raise ValueError("shot head lacks sealed authority: " + ", ".join(failed))

    def predict(self, latent: np.ndarray, action: np.ndarray) -> float:
        latent_value = np.asarray(latent, dtype=float).reshape(-1)
        if latent_value.shape != (self.latent_dim,):
            raise ValueError("invalid latent shot-head feature shape")
        action_value = strip_outcome_leakage(
            np.asarray(action, dtype=np.float32).reshape(-1)
        ).astype(float)
        if action_value.shape != (ACTION_DIM,):
            raise ValueError("invalid action shot-head feature shape")
        features = np.concatenate([latent_value, action_value])
        standardized = (features - self.mean) / np.maximum(self.scale, 1e-8)
        logit = float(np.clip(standardized @ self.weights + self.bias, -30.0, 30.0))
        calibrated = float(np.clip(
            self.calibration_scale * logit + self.calibration_bias,
            -30.0,
            30.0,
        ))
        return float(1.0 / (1.0 + np.exp(-calibrated)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "frozen_world_model_shot_head",
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "latent_dim": self.latent_dim,
            "feature_schema": "frozen_latent_plus_leakage_stripped_action",
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "calibration": {
                "scale": self.calibration_scale,
                "bias": self.calibration_bias,
            },
            "evidence": dict(self.evidence),
            "accepted": self.accepted,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FrozenShotHead":
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported frozen shot-head schema")
        if payload.get("kind") != "frozen_world_model_shot_head":
            raise ValueError("unsupported shot-head kind")
        calibration = payload.get("calibration") or {}
        return cls(
            base_checkpoint_sha256=str(payload["base_checkpoint_sha256"]),
            latent_dim=int(payload["latent_dim"]),
            mean=np.asarray(payload["mean"], dtype=float),
            scale=np.asarray(payload["scale"], dtype=float),
            weights=np.asarray(payload["weights"], dtype=float),
            bias=float(payload["bias"]),
            calibration_scale=float(calibration.get("scale", 1.0)),
            calibration_bias=float(calibration.get("bias", 0.0)),
            evidence=dict(payload.get("evidence") or {}),
            accepted=bool(payload.get("accepted", False)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "FrozenShotHead":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

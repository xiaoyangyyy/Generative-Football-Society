from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .action_codec import decode_action_kind
from .model import load_checkpoint
from .schema import observation_coverage, strip_outcome_leakage


class WorldModelRuntime:
    def __init__(self, model, config, meta, checkpoint_signature: str) -> None:
        self.model = model
        self.config = config
        self.meta = meta
        self.checkpoint_signature = checkpoint_signature
        validation = meta.get("validation", {})
        self.base_quality = float(np.clip(
            validation.get("planner_quality", 0.0), 0.0, 1.0,
        ))
        self.pass_quality = float(np.clip(
            validation.get("pass_planner_quality", self.base_quality), 0.0, 1.0,
        ))
        self.shot_quality = 0.0
        self.cross_quality = 0.0
        calibration = validation.get("pass_calibration", {})
        self.pass_calibration_scale = float(calibration.get("scale", 1.0))
        self.pass_calibration_bias = float(calibration.get("bias", 0.0))

    @classmethod
    def load(cls, path: str | Path) -> WorldModelRuntime:
        checkpoint = Path(path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"world model checkpoint not found: {checkpoint}")
        model, config, meta = load_checkpoint(str(checkpoint))
        return cls(model, config, meta, _file_hash(checkpoint))

    def imagine(self, observation, action, *, steps: int | None = None):
        clean_observation = np.clip(
            np.nan_to_num(
                np.asarray(observation, dtype=np.float32),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ),
            0.0,
            1.0,
        )
        clean_action = strip_outcome_leakage(
            np.nan_to_num(
                np.asarray(action, dtype=np.float32),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            )
        )
        output = self.model.imagine(
            clean_observation,
            clean_action,
            steps=steps or self.config.imagination_steps,
        )
        raw = float(np.clip(output.pass_success, 1e-6, 1.0 - 1e-6))
        logit = np.log(raw / (1.0 - raw))
        output.pass_success = float(
            1.0 / (1.0 + np.exp(-(
                self.pass_calibration_scale * logit + self.pass_calibration_bias
            )))
        )
        return output

    @staticmethod
    def action_kind(action) -> str:
        return decode_action_kind(action)

    def planner_authority(self, observation, *, kind: str = "general") -> dict:
        quality = {
            "pass": self.pass_quality,
            "shot": self.shot_quality,
            "cross": self.cross_quality,
        }.get(kind, self.base_quality)
        coverage = observation_coverage(observation)
        authorized = quality >= self.config.min_planner_quality
        return {
            "authorized": authorized,
            "validation_quality": quality,
            "minimum_validation_quality": self.config.min_planner_quality,
            "observation_coverage": coverage,
            "online_trust": 1.0,
            "decision_confidence": float(
                np.clip(0.25 + 0.75 * coverage, 0.0, 1.0)
            ) if authorized else 0.0,
        }

    def status(self) -> dict:
        return {
            "checkpoint_signature": self.checkpoint_signature,
            "checkpoint_version": int(getattr(self.model, "checkpoint_version", 0)),
            "transition_kind": getattr(self.model, "transition_kind", "unknown"),
            "latent_dim": self.config.latent_dim,
            "hidden_dim": self.config.hidden_dim,
            "transition_members": self.model.transition_member_count,
            "transition_ensemble_trained": bool(
                getattr(self.model, "transition_ensemble_trained", False)
            ),
            "pass_quality": self.pass_quality,
            "shot_quality": self.shot_quality,
            "cross_quality": self.cross_quality,
            "deployed": False,
        }


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

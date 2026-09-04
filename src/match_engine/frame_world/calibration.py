"""Verified loading for calibrated continuous-time routing artifacts."""

from __future__ import annotations
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from .continuous import load_continuous_time
from .router import ActionTransitionRouter


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=8)
def _load_calibrated_temporal_assets(artifact: str | Path):
    artifact = Path(artifact)
    root = (
        artifact.resolve().parents[2]
        if artifact.parent.name == "frame_world"
        else Path.cwd()
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    models = {}
    thresholds = {}
    intervals = {}
    if payload.get("deployment") not in {"evaluated_only", "operational_candidate"}:
        raise ValueError("unaccepted temporal calibration artifact")
    for provider, entry in payload.get("providers", {}).items():
        model_path = Path(entry["model"])
        model_path = model_path if model_path.is_absolute() else root / model_path
        if not model_path.is_file() or _sha(model_path) != entry["model_sha256"]:
            raise RuntimeError(f"temporal model integrity failure: {provider}")
        model, _, meta = load_continuous_time(model_path)
        model.eval()
        models[provider] = model
        thresholds[provider] = (
            float(meta["continue_threshold"]),
            float(meta["subtype_threshold"]),
        )
        intervals[provider] = entry["radii_s"]
    if not models:
        raise ValueError("calibration artifact has no providers")
    return models, thresholds, intervals


def load_calibrated_temporal_router(
    artifact: str | Path, **router_options
) -> ActionTransitionRouter:
    models, thresholds, intervals = _load_calibrated_temporal_assets(
        str(Path(artifact).resolve())
    )
    return ActionTransitionRouter(
        temporal_models=models,
        temporal_thresholds=thresholds,
        temporal_intervals=intervals,
        **router_options,
    )

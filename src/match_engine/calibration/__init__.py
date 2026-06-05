"""Calibration infrastructure: observable contract, parameter registry, ablations."""

from src.match_engine.calibration.ablation import (
    ADDITIVE_LAYERS,
    ABLATION_PRESETS,
    PIPELINE_PRESETS,
    SUBTRACTIVE_ABLATIONS,
    AblationSpec,
    apply_ablation,
    calibration_env,
)
from src.match_engine.calibration.contract import (
    evaluate_rows,
    load_contract,
    load_statsbomb_baselines,
)
from src.match_engine.calibration.apply_params import apply_params_to_config, export_profile_overrides
from src.match_engine.calibration.objective import calibration_loss, calibration_score
from src.match_engine.calibration.profile import CalibrationProfile, load_param_registry, load_profile

__all__ = [
    "ADDITIVE_LAYERS",
    "ABLATION_PRESETS",
    "PIPELINE_PRESETS",
    "SUBTRACTIVE_ABLATIONS",
    "AblationSpec",
    "CalibrationProfile",
    "apply_ablation",
    "calibration_env",
    "evaluate_rows",
    "load_contract",
    "apply_params_to_config",
    "calibration_loss",
    "calibration_score",
    "export_profile_overrides",
    "load_param_registry",
    "load_profile",
    "load_statsbomb_baselines",
]

from pathlib import Path

import numpy as np
import pytest

from gfs_core.kernel import PluginKernel
from gfs_studio.bootstrap.builtin_plugins import build_research_registry
from gfs_studio.plugins.training.datasets.manifest import TraceDataset
from gfs_studio.plugins.world_model.model import build_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def start_research():
    kernel = PluginKernel(build_research_registry(PROJECT_ROOT))
    kernel.start(["gfs.studio.research"])
    return kernel, kernel.context.require("research-bundle")


def test_probability_calibration_and_real_candidate_evaluation():
    kernel, research = start_research()
    try:
        report = research["calibration"].probability_report(
            [0.25, 0.75, 0.75, 0.25], [0, 1, 0, 1], bins=2,
        )
        assert report["brier_score"] == pytest.approx(0.3125)
        assert report["expected_calibration_error"] == pytest.approx(0.25)
        uncertainty = research["calibration"].uncertainty_report(
            [0.1, 0.2, 0.8, 0.9], [0, 0, 0, 0],
            [0.1, 0.2, 0.8, 0.9],
        )
        assert uncertainty["uncertainty_error_correlation"] == pytest.approx(1.0)
        assert uncertainty["risk_coverage"][0]["mean_absolute_error"] < (
            uncertainty["risk_coverage"][-1]["mean_absolute_error"]
        )
        assert "held_out_split_identity" in uncertainty["scope"]
        rng = np.random.default_rng(7)
        observations = rng.random((8, 307), dtype=np.float32)
        dataset = TraceDataset(
            observations,
            rng.random((8, 18), dtype=np.float32),
            observations.copy(),
            np.asarray([f"match-{index // 4}" for index in range(8)]),
        )
        result = research["calibration"].evaluate_candidate(build_model(), dataset)
        assert np.isfinite(result["transition_prediction"]["model_mse"])
        assert result["transition_prediction"]["persistence_mse"] == 0.0
        assert result["transition_prediction"]["scope"] == "transition_prediction_only_not_policy_or_deployment"
        assert result["new_training"] is False
        assert research["calibration"].history()["new_runs"] == 0
    finally:
        kernel.stop()


def test_frozen_ablation_and_reproduction_do_not_claim_new_runs():
    kernel, research = start_research()
    try:
        assert set(research) == {"calibration", "ablation", "experiments", "reproduction"}
        ablation = research["ablation"].summary()
        assert ablation["complete"] is True
        assert ablation["new_runs"] == 0
        assert ablation["evidence"] == "frozen_historical"
        assert research["ablation"].variant("no_spatial")["result"]["all_pass"] is False
        with pytest.raises(KeyError, match="unknown frozen"):
            research["ablation"].variant("invented")
        reproduction = research["reproduction"].status()
        assert reproduction["independent_reproduced"] is False
        assert reproduction["ready_to_start"] is False
        assert reproduction["new_runs"] == 0
        assert reproduction["protocol_identity"]["current"] is True
    finally:
        kernel.stop()

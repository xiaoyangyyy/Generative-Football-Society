from pathlib import Path

import numpy as np
import pytest

from gfs_core.kernel import PluginKernel
from gfs_studio.api import frame_manifest, plan, recommend
from gfs_studio.bootstrap.builtin_plugins import build_planning_registry
from gfs_studio.contracts.product import DecisionCase
from gfs_studio.plugins.world_model.action_codec import ACTION_DIM
from gfs_studio.plugins.world_model.observation import OBS_DIM
from gfs_studio.plugins.world_model.policies.service import PolicyService

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_real_planning_forecast_and_policy_recommendation():
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[0] = 1.0
    action[13] = 0.7
    action[17] = 0.5
    observation = np.zeros(OBS_DIM, dtype=np.float32)
    forecast = plan(PROJECT_ROOT, observation, action, kind="pass")
    assert forecast["pass_success"] == pytest.approx(0.7065701263339648)
    assert forecast["authority"]["authorized"] is True
    assert forecast["deployed"] is False
    recommendation = recommend(PROJECT_ROOT, observation, action)
    assert recommendation["action"] == "pass"
    assert recommendation["authority"] is True
    assert recommendation["outcome_improvement_claim"] is False


def test_unauthorized_policy_preserves_baseline():
    class Planning:
        @staticmethod
        def forecast(observation, action, **kwargs):
            return {"authority": {"authorized": False}}

    result = PolicyService(Planning()).recommend(
        np.zeros(OBS_DIM), np.zeros(ACTION_DIM), baseline_action="hold",
    )
    assert result["action"] == "hold"
    assert result["authority"] is False
    assert result["baseline_preserved"] is True


def test_frame_manifest_has_identity_and_no_unverified_artifact_claim():
    status = frame_manifest(PROJECT_ROOT)
    assert status["matches"] > 0
    assert "metrica" in status["providers"]
    assert "skillcorner" in status["providers"]
    assert status["artifact_verified"] is False


def test_real_model_assessment_is_bound_to_decision_case():
    observation = np.zeros(OBS_DIM, dtype=np.float32)
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[0] = 1.0
    action[13] = 0.7
    action[17] = 0.5
    kernel = PluginKernel(build_planning_registry(PROJECT_ROOT))
    try:
        kernel.start(["gfs.studio.world-model.policies"])
        case = DecisionCase(
            "case-1", "Brazil", "Argentina", 42,
            ("preserve_current_vector", "attacking_pressure"),
        )
        assessment = kernel.context.require("policies").assess_case(
            case, observation, action,
        )
        assert assessment["case_id"] == "case-1"
        assert assessment["candidate_id"] == "pass_forecast"
        assert assessment["predicted_metrics"]["pass_success"] == pytest.approx(0.7065701263339648)
        assert assessment["observed_simulation_metrics"] is None
    finally:
        kernel.stop()

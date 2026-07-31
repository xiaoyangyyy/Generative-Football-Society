"""Smoke tests for latent world model encoding and checkpoint."""

from __future__ import annotations

import os
import sys
import json

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.match_engine.macro_bridge import build_match_affective_state
from src.match_engine.world_model.action_codec import (
    ACTION_DIM,
    encode_from_ball_log_event,
    encode_pass_candidate,
    zero_action,
)
from src.match_engine.world_model.observation import OBS_DIM, encode_observation
from src.match_engine.world_model.schema import (
    PASS_OUTCOME_INDEX,
    SHOT_GOAL_INDEX,
    observation_coverage,
    observation_loss_weights,
    strip_outcome_leakage,
)


class _FakeAgent:
    team_name = "Home"
    status_score = 55.0
    emotion_profile = {"pride": 0.5, "anger": 0.1, "fear": 0.2, "determination": 0.5}
    formation = "4-3-3"
    referee_grievance = 0.0
    tactical_controls = {
        "pressing_intensity": 0.5,
        "risk_budget": 0.5,
        "line_height": 0.5,
        "rotation_aggressiveness": 0.5,
    }
    tactical_vector = {}


def test_observation_dim():
    a1, a2 = _FakeAgent(), _FakeAgent()
    a2.team_name = "Away"
    state = build_match_affective_state(a1, a2, neutral_venue=True)
    obs = encode_observation(state)
    assert obs.shape == (OBS_DIM,)
    assert np.all(np.isfinite(obs))


def test_action_dim():
    a = np.zeros(ACTION_DIM, dtype=np.float32)
    assert a.shape[0] == ACTION_DIM
    assert zero_action().shape[0] == ACTION_DIM


def test_training_outcomes_are_stripped_from_model_action():
    action = encode_from_ball_log_event(
        {
            "type": "pass",
            "land_xy": [0.7, 0.4],
            "kind": "through",
            "outcome": "COMPLETE",
            "p_success": 0.61,
        }
    )
    assert action[PASS_OUTCOME_INDEX] == pytest.approx(1.0)
    clean = strip_outcome_leakage(action)
    assert clean[PASS_OUTCOME_INDEX] == pytest.approx(0.0)
    assert clean[13] == pytest.approx(0.61)


def test_shot_goal_is_real_outcome_not_xg_label():
    action = encode_from_ball_log_event(
        {"type": "shot", "from_xy": [0.8, 0.5], "xg": 0.18, "outcome": "GOAL"}
    )
    assert action[13] == pytest.approx(0.18)
    assert action[SHOT_GOAL_INDEX] == pytest.approx(1.0)


def test_observation_weights_prioritize_ball_over_sparse_grids():
    weights = observation_loss_weights()
    assert weights.shape == (OBS_DIM,)
    assert weights[200] > weights[0]
    sparse = np.zeros(OBS_DIM, dtype=np.float32)
    sparse[200:204] = 0.5
    dense = np.full(OBS_DIM, 0.5, dtype=np.float32)
    assert observation_coverage(dense) > observation_coverage(sparse)


def test_v3_residual_model_preserves_state_with_zero_parameters():
    torch = pytest.importorskip("torch")
    from src.match_engine.world_model.config import WorldModelConfig
    from src.match_engine.world_model.model import build_model

    model = build_model(WorldModelConfig(latent_dim=16, hidden_dim=32))
    for param in model.parameters():
        torch.nn.init.zeros_(param)
    obs = np.linspace(0.0, 1.0, OBS_DIM, dtype=np.float32)
    out = model.imagine(obs, zero_action(), steps=1)
    assert np.allclose(out.next_obs, obs, atol=1e-6)


def test_world_model_config_rejects_schema_drift():
    from src.match_engine.world_model.config import WorldModelConfig

    with pytest.raises(ValueError, match="grid_gx=8"):
        WorldModelConfig(grid_gx=7)


def test_trace_loader_excludes_aggregate_backfill_and_returns_groups(tmp_path):
    from src.match_engine.world_model.recorder import load_trace_batches

    row = {
        "obs": np.zeros(OBS_DIM).tolist(),
        "action": np.zeros(ACTION_DIM).tolist(),
        "next_obs": np.ones(OBS_DIM).tolist(),
    }
    (tmp_path / "match_a.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (tmp_path / "_ball_log_backfill.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    obs, actions, next_obs, groups = load_trace_batches(
        str(tmp_path),
        return_groups=True,
    )
    assert len(obs) == len(actions) == len(next_obs) == 1
    assert groups.tolist() == ["match_a"]


def test_structured_encoder_has_multiple_outcome_members():
    torch = pytest.importorskip("torch")
    from src.match_engine.world_model.config import WorldModelConfig
    from src.match_engine.world_model.model import build_model

    model = build_model(WorldModelConfig(latent_dim=16, hidden_dim=32, ensemble_size=3))
    obs = torch.zeros((2, OBS_DIM), dtype=torch.float32)
    action = torch.zeros((2, ACTION_DIM), dtype=torch.float32)
    z, *_ = model(obs, action)
    pass_heads, progress_heads, shot_heads = model.outcome_ensemble(z, action)
    assert pass_heads.shape == progress_heads.shape == shot_heads.shape == (3, 2, 1)


def test_v6_graph_checkpoint_round_trip(tmp_path):
    pytest.importorskip("torch")
    from src.match_engine.world_model.config import WorldModelConfig
    from src.match_engine.world_model.model import build_model, load_checkpoint, save_checkpoint

    cfg = WorldModelConfig(latent_dim=16, hidden_dim=32, ensemble_size=2)
    model = build_model(cfg)
    path = tmp_path / "wm.pt"
    save_checkpoint(str(path), model, cfg, {"quality": 0.0})
    loaded, _, meta = load_checkpoint(str(path))
    assert loaded.checkpoint_version == 6
    assert meta["quality"] == 0.0


def test_v6_zero_pass_residual_preserves_physics_prior():
    torch = pytest.importorskip("torch")
    from src.match_engine.world_model.config import WorldModelConfig
    from src.match_engine.world_model.model import build_model

    model = build_model(WorldModelConfig(latent_dim=16, hidden_dim=32, ensemble_size=2))
    for head in [model.head_pass, *model.extra_pass_heads]:
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
    z = torch.zeros((3, 16), dtype=torch.float32)
    action = torch.zeros((3, ACTION_DIM), dtype=torch.float32)
    action[:, 13] = torch.tensor([0.2, 0.5, 0.8])
    pass_logits, _, _ = model.outcome_ensemble(z, action)
    probability = torch.sigmoid(pass_logits.mean(dim=0)).view(-1)
    assert torch.allclose(probability, action[:, 13], atol=1e-6)


def test_runtime_predicts_declared_policy_utility_at_requested_horizon():
    pytest.importorskip("torch")
    from src.match_engine.world_model.config import WorldModelConfig
    from src.match_engine.world_model.inference import WorldModelRuntime
    from src.match_engine.world_model.model import build_model

    cfg = WorldModelConfig(latent_dim=16, hidden_dim=32, ensemble_size=2)
    runtime = WorldModelRuntime(build_model(cfg), cfg, {
        "validation": {"planner_quality": 0.8, "weighted_obs_mse": 0.02},
    })
    observation = np.full(OBS_DIM, 0.5, dtype=np.float32)
    prediction = runtime.predict_policy_utility(
        observation,
        zero_action(),
        action_kind="hold",
        attacking_home=True,
        horizon_s=60.0,
    )
    assert prediction["prediction_source"] == (
        "autoregressive_action_persistence_rollout"
    )
    assert prediction["horizon_s"] == 60.0
    assert np.isfinite(prediction["policy_utility"])
    assert 0.0 <= prediction["retention_probability"] <= 1.0
    assert 0.0 <= prediction["uncertainty"] <= 1.0
    longer = runtime.predict_policy_utility(
        observation,
        zero_action(),
        action_kind="hold",
        attacking_home=True,
        horizon_s=180.0,
    )
    assert longer["rollout_steps"] == 3
    assert longer["segment_horizon_s"] == 60.0
    assert longer["uncertainty"] >= prediction["uncertainty"]


@pytest.mark.skipif(
    not os.path.isfile(
        os.path.join(os.path.dirname(__file__), "..", "data", "world_model", "latent_wm.pt")
    ),
    reason="checkpoint not trained",
)
def test_checkpoint_load_and_imagine():
    torch = pytest.importorskip("torch")
    base = os.path.join(os.path.dirname(__file__), "..")
    from src.match_engine.world_model.inference import WorldModelRuntime

    rt = WorldModelRuntime.load(
        os.path.join(base, "data", "world_model", "latent_wm.pt")
    )
    a1, a2 = _FakeAgent(), _FakeAgent()
    a2.team_name = "Away"
    state = build_match_affective_state(a1, a2)
    obs = rt.encode_state(state)
    out = rt.imagine_pass(obs, zero_action(), steps=2)
    assert out.next_obs.shape[0] == OBS_DIM
    assert np.isfinite(out.pass_success)

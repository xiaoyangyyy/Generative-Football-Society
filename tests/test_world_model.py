"""Smoke tests for latent world model encoding and checkpoint."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.match_engine.macro_bridge import build_match_affective_state
from src.match_engine.world_model.action_codec import ACTION_DIM, encode_pass_candidate, zero_action
from src.match_engine.world_model.observation import OBS_DIM, encode_observation


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

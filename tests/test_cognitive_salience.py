"""Cognitive bus salience and tier caps."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.match_engine.cognitive.bus import MatchCognitiveBus
from src.match_engine.cognitive.config import CognitiveMatchConfig
from src.match_engine.cognitive.salience import base_salience_from_micro, should_fire
from src.match_engine.micro_events import MicroEvent, MicroEventType
from src.match_engine.state import (
    CrowdState,
    MatchAffectiveState,
    RefereeAffectiveState,
    TeamAffectiveState,
    CoachAffectiveState,
)


def _minimal_state():
    home = TeamAffectiveState(team_id="H", coach=CoachAffectiveState(team_id="H"))
    away = TeamAffectiveState(team_id="A", coach=CoachAffectiveState(team_id="A"))
    ref = RefereeAffectiveState()
    return MatchAffectiveState(
        home=home,
        away=away,
        referee=ref,
        crowd=CrowdState(home_team_id="H"),
    )


def test_goal_salience_positive():
    state = _minimal_state()
    ev = MicroEvent(120.0, MicroEventType.GOAL_SCORED, "H", intensity=1.0, player_id="p1")
    sal = base_salience_from_micro(ev, state)
    assert sal > 0.8


def test_cooldown_blocks_repeat():
    cfg = CognitiveMatchConfig(enabled=True, cooldown_sec=300.0)
    rng = np.random.default_rng(0)
    last = {}
    assert should_fire(1.5, "coach:H", "coach", 100.0, last, cfg, rng) in (True, False)
    last["coach:H"] = 100.0
    assert should_fire(1.5, "coach:H", "coach", 150.0, last, cfg, rng) is False


def test_bus_queues_goal_trigger():
    cfg = CognitiveMatchConfig(enabled=True, cooldown_sec=0.0, salience_center=0.3)
    cfg.tier_caps = (10, 10, 10, 10, 10)
    bus = MatchCognitiveBus(cfg, np.random.default_rng(1))
    state = _minimal_state()
    ev = MicroEvent(60.0, MicroEventType.GOAL_SCORED, "H", intensity=1.0, player_id="p9")
    bus.ingest_micro_event(ev, state)
    pending = bus.drain_pending()
    assert len(pending) >= 1
    assert any(t.entity_tier == "coach" for t in pending)
def test_policy_bridge_experiment_config_is_safely_bounded():
    cfg = CognitiveMatchConfig.from_mapping({
        "MATCH_WM_LLM_ACTION_BRIDGE": "1",
        "MATCH_WM_LLM_ACTION_BIAS_MAX": "9",
        "MATCH_WM_LLM_ACTION_CONTROL_RATE": "0.8",
        "MATCH_WM_LLM_ACTION_MIN_ARM_SAMPLES": "1",
        "MATCH_WM_LLM_OUTCOME_HORIZONS": "180,60,60,-1,bad",
        "MATCH_WM_LLM_RESIDUAL_MIN_SAMPLES": "1",
        "MATCH_WM_ACTIVE_LEARNING": "1",
        "MATCH_WM_EXPLORATION_BUDGET": "0.9",
        "MATCH_WM_EXPLORATION_MAX_REGRET": "0.9",
        "MATCH_WM_EXPLORATION_MIN_INFORMATION": "-1",
        "MATCH_WM_EXPLORATION_STRENGTH_SCALE": "2",
        "MATCH_WM_TRAJECTORY_BRANCH_BUDGET": "999",
        "MATCH_WM_LLM_CONTRASTIVE_REPAIR": "1",
        "MATCH_WM_LLM_CONTRASTIVE_REPAIR_PATH_BUDGET": "999",
    })
    assert cfg.world_model_action_bridge
    assert cfg.world_model_action_bias_max == 0.5
    assert cfg.world_model_action_control_rate == 0.5
    assert cfg.world_model_action_min_arm_samples == 2
    assert cfg.world_model_outcome_horizons_s == (0.0, 60.0, 180.0)
    assert cfg.world_model_residual_min_samples == 2
    assert cfg.world_model_active_learning
    assert cfg.world_model_exploration_budget == 0.5
    assert cfg.world_model_exploration_max_regret == 0.30
    assert cfg.world_model_exploration_min_information == 0.0
    assert cfg.world_model_exploration_strength_scale == 1.0
    assert cfg.world_model_trajectory_branch_budget == 112
    assert cfg.world_model_contrastive_repair
    assert cfg.world_model_contrastive_repair_path_budget == 128

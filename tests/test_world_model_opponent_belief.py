from __future__ import annotations

import json

import numpy as np
import pytest

from src.match_engine.cognitive.config import CognitiveMatchConfig
from src.match_engine.cognitive.events import CognitiveTriggerEvent, ENTITY_TIER_COACH
from src.match_engine.cognitive.executor import CognitiveExecutor
from src.match_engine.state import (
    CoachAffectiveState,
    CrowdState,
    MatchAffectiveState,
    RefereeAffectiveState,
    TeamAffectiveState,
)
from src.match_engine.tactical_catalog import TACTICAL_PRESETS
from src.match_engine.world_model.decision_support import (
    _condition_on_opponent_hypothesis,
    build_coach_decision_packet,
)
from src.match_engine.world_model.observation import OBS_DIM
from src.match_engine.world_model.opponent_belief import (
    OPPONENT_HYPOTHESES,
    assimilate_llm_opponent_hypothesis,
    opponent_belief_diagnostics,
    reweight_counterfactual_candidates,
    update_opponent_belief,
    validate_llm_opponent_hypothesis,
)
from src.match_engine.world_model.probabilistic import ProbabilisticFuture


def _state(opponent_preset: str = "low_block") -> MatchAffectiveState:
    controls = {
        key: TACTICAL_PRESETS[opponent_preset][key]
        for key in (
            "pressing_intensity", "risk_budget", "line_height",
            "rotation_aggressiveness",
        )
    }
    return MatchAffectiveState(
        home=TeamAffectiveState(
            team_id="Home",
            coach=CoachAffectiveState(team_id="Home"),
        ),
        away=TeamAffectiveState(
            team_id="Away",
            coach=CoachAffectiveState(
                team_id="Away", tactical_current=controls,
            ),
        ),
        referee=RefereeAffectiveState(),
        crowd=CrowdState(),
    )


class _OpponentSensitiveRuntime:
    checkpoint_signature = "test-opponent-belief"

    def encode_state(self, state, *, attacking_home):
        observation = np.full(OBS_DIM, 0.5, dtype=np.float32)
        observation[302:306] = [
            state.away.coach.tactical_current.get(key, 0.5)
            for key in (
                "pressing_intensity", "risk_budget", "line_height",
                "rotation_aggressiveness",
            )
        ]
        return observation

    def planner_confidence(self, observation, kind="general"):
        return 0.85

    def predict_future(self, observation, action, *, action_kind, horizon_s):
        pressure = float(observation[302])
        turnover = 0.12 + 0.18 * pressure if action_kind == "pass" else 0.18
        retain = 0.70 - 0.15 * pressure if action_kind == "pass" else 0.55
        shot = 0.22 if action_kind == "shot" else 0.05
        raw = np.asarray([retain, turnover, shot, 0.03, 0.02], dtype=float)
        raw /= raw.sum()
        return ProbabilisticFuture(
            dict(zip(("retain", "turnover", "shot", "foul", "out"), raw)),
            (max(0.1, horizon_s * 0.5), 1.0),
            (-0.05, 0.08, 0.20),
            0.12,
        )

    def score_action(self, observation, action):
        pressure = float(observation[302])
        if action[0] > 0.5:
            return 0.85 - 0.75 * pressure
        if action[2] > 0.5:
            return 0.35 + 0.20 * pressure
        if action[3] > 0.5:
            return 0.42
        return 0.30

    def score_shot_action(self, observation, action, *, attacking_home):
        return 0.38 + 0.25 * float(observation[302])

    def predict_policy_utility(
        self, observation, action, *, action_kind, attacking_home, horizon_s,
    ):
        value = (
            self.score_shot_action(observation, action, attacking_home=attacking_home)
            if action_kind == "shot" else self.score_action(observation, action)
        )
        return {
            "policy_utility": value,
            "progress": 0.1,
            "retention_probability": 0.6,
            "xg_net_delta": 0.02,
            "goal_diff_delta": 0.0,
            "uncertainty": 0.12,
            "epistemic_uncertainty": 0.08,
            "aleatoric_uncertainty": 0.04,
        }


def test_numeric_observations_build_a_persistent_non_degenerate_posterior():
    state = _state("low_block")
    first = update_opponent_belief(state, "Home")
    state.clock_seconds = 90.0
    second = update_opponent_belief(state, "Home")

    assert first["map_hypothesis"] == "low_block"
    assert second["map_hypothesis"] == "low_block"
    assert second["evidence_count"] == 2
    assert second["posterior"]["low_block"] > first["posterior"]["low_block"]
    assert 0.0 < second["normalized_entropy"] < 1.0
    assert abs(sum(second["posterior"].values()) - 1.0) < 1e-9


def test_llm_hypothesis_is_schema_constrained_and_likelihood_bounded():
    state = _state("low_block")
    update_opponent_belief(state, "Home")
    cleaned = validate_llm_opponent_hypothesis({
        "tactical_preset": "gegenpress",
        "confidence": 5.0,
        "evidence_features": ["pressing_intensity", "invented_metric"],
        "rationale": "Observed pressure.",
    })
    assert cleaned == {
        "tactical_preset": "gegenpress",
        "confidence": 1.0,
        "evidence_features": ["pressing_intensity"],
        "rationale": "Observed pressure.",
    }
    prior = update_opponent_belief(state, "Home")["posterior"]["gegenpress"]
    audit = assimilate_llm_opponent_hypothesis(state, "Home", cleaned)

    assert 0.0 <= audit["bounded_influence"] <= 0.20
    assert audit["observation_likelihood_support"] < 1.0
    assert audit["posterior_probability"] - prior <= 0.20
    assert not assimilate_llm_opponent_hypothesis(
        state,
        "Home",
        {
            "tactical_preset": "unknown",
            "confidence": 1.0,
            "evidence_features": ["pressing_intensity", "line_height"],
        },
    )["accepted"]


def test_repeated_llm_calls_cannot_exceed_one_observations_influence_budget():
    state = _state("low_block")
    update_opponent_belief(state, "Home")
    hypothesis = {
        "tactical_preset": "low_block",
        "confidence": 1.0,
        "evidence_features": [
            "pressing_intensity", "risk_budget", "line_height",
            "rotation_aggressiveness",
        ],
    }
    audits = [
        assimilate_llm_opponent_hypothesis(state, "Home", hypothesis)
        for _ in range(5)
    ]

    assert sum(audit["bounded_influence"] for audit in audits) <= 0.20 + 1e-12
    assert audits[-1]["bounded_influence"] == 0.0
    assert audits[-1]["observation_influence_used_before"] == 0.20


def test_world_model_values_actions_under_every_opponent_hypothesis():
    packet = build_coach_decision_packet(
        _OpponentSensitiveRuntime(), _state("balanced"), "Home",
        outcome_horizons_s=(0.0,),
    )

    assert packet["available"]
    assert set(packet["opponent_belief"]["posterior"]) == set(OPPONENT_HYPOTHESES)
    for candidate in packet["candidates"]:
        assert set(candidate["opponent_hypothesis_values"]) == set(
            OPPONENT_HYPOTHESES
        )
        assert candidate["opponent_belief_value_std"] >= 0.0
        assert candidate["opponent_belief_tail_value"] <= (
            candidate["opponent_belief_expected_value"] + 1e-12
        )
        assert 0.0 <= candidate["opponent_hypothesis_discrimination"] <= 1.0
        assert candidate["active_learning"][
            "opponent_hypothesis_discrimination"
        ] == candidate["opponent_hypothesis_discrimination"]
        assert np.isfinite(candidate["risk_adjusted_value"])
    passing = next(item for item in packet["candidates"] if item["action"] == "pass")
    assert passing["opponent_belief_value_std"] > 0.0


def test_hypothesis_intervention_changes_only_the_opponents_tactical_slice():
    observation = np.linspace(0.0, 1.0, OBS_DIM, dtype=np.float32)
    home_view = _condition_on_opponent_hypothesis(
        observation, team_is_home=True, hypothesis="low_block",
    )
    away_view = _condition_on_opponent_hypothesis(
        observation, team_is_home=False, hypothesis="gegenpress",
    )

    assert np.array_equal(home_view[298:302], observation[298:302])
    assert not np.array_equal(home_view[302:306], observation[302:306])
    assert np.array_equal(away_view[302:306], observation[302:306])
    assert not np.array_equal(away_view[298:302], observation[298:302])
    unchanged = np.ones(OBS_DIM, dtype=bool)
    unchanged[298:306] = False
    assert np.array_equal(home_view[unchanged], observation[unchanged])
    assert np.array_equal(away_view[unchanged], observation[unchanged])


def test_robust_tail_ignores_an_unsupported_one_percent_extreme():
    packet = {
        "candidates": [{
            "action": "pass",
            "effective_confidence": 0.8,
            "opponent_hypothesis_values": {
                "balanced": 1.0,
                "low_block": -10.0,
            },
        }],
    }
    reweight_counterfactual_candidates(
        packet, {"balanced": 0.99, "low_block": 0.01},
    )

    candidate = packet["candidates"][0]
    assert candidate["opponent_belief_tail_value"] == 1.0
    assert candidate["opponent_belief_expected_value"] == pytest.approx(0.89)


class _HypothesisLLM:
    def __init__(self):
        self.facts = None

    def coach_in_match_plan(self, team_name, facts, kind):
        self.facts = facts
        return json.dumps({
            "reasoning": "The compact shape remains the main concern.",
            "confidence": 0.8,
            "controls_delta": {},
            "world_model_action": facts["world_model_decision_support"][
                "recommended_action"
            ],
            "opponent_hypothesis": {
                "tactical_preset": "low_block",
                "confidence": 0.8,
                "evidence_features": ["line_height", "risk_budget"],
                "rationale": "Their line and risk controls remain low.",
            },
        })


def test_executor_closes_model_llm_belief_loop_and_exports_audit():
    state = _state("low_block")
    llm = _HypothesisLLM()
    executor = CognitiveExecutor(
        CognitiveMatchConfig(enabled=True, world_model_action_control_rate=0.0),
        llm,
        world_model_runtime=_OpponentSensitiveRuntime(),
    )
    record = executor.process_trigger(CognitiveTriggerEvent(
        60.0, "shape_change", ENTITY_TIER_COACH,
        "coach:Home", team_id="Home", salience=1.0,
    ), state)

    audit = record.plan["opponent_belief_audit"]
    assert audit["accepted"]
    assert 0.0 < audit["bounded_influence"] <= 0.20
    assert record.plan["world_model_action"] in {"hold", "pass", "cross", "shot"}
    assert llm.facts["world_model_decision_support"]["opponent_belief_reweighted"]
    context = state._wm_coach_decision_adoption[-1]["opponent_belief_context"]
    assert context["map_hypothesis"] == "low_block"
    assert context["posterior"]["low_block"] == audit["posterior"]["low_block"]
    assert context["selected_action_hypothesis_values"]
    exported = {
        "cognitive_plans": executor.export_log(),
        "world_model_decision_adoption": {
            "records": state._wm_coach_decision_adoption,
        },
    }
    diagnostics = opponent_belief_diagnostics([exported])
    assert diagnostics["hypotheses_submitted"] == 1
    assert diagnostics["hypotheses_accepted"] == 1
    assert diagnostics["mean_counterfactual_action_sensitivity"] > 0.0
    assert diagnostics["decision_belief_snapshots"] == 1
    assert not diagnostics["calibrated_against_hidden_truth"]

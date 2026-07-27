from types import SimpleNamespace

import numpy as np
import pytest

from src.data_engine.tracking_contract import (
    TrackingFrame,
    grouped_leave_one_domain_out,
    perturb_player_state,
    tracking_quality,
)
from src.data_engine.skillcorner_adapter import load_skillcorner_frames
from src.match_engine.joint_pass_model import JointPassModel, PASS_FEATURES, fit_joint_pass_model, pass_model_gate
from src.match_engine.shot_decision import SHOT_FEATURES, ShotEvidence, ShotProbabilityModel, fit_shot_probability_model, shot_counterfactual_value
from src.match_engine.world_model.probabilistic import EVENTS, EventTransitionModel, fit_event_transition_model, future_from_ensemble
from src.model_registry import ModelRegistry
from src.simulation.meta_learning import MetaLearningController
from src.simulation.psychological_state import PsychologicalState
from src.simulation.regression_gate import benchmark_latency, distribution_gate


def test_tracking_contract_reports_cross_provider_evidence():
    frames = [
        TrackingFrame(provider, "league", match, 1, index, float(index), {"h": np.zeros(2)}, {"a": np.ones(2)})
        for index, (provider, match) in enumerate([("a", "m1"), ("a", "m2"), ("b", "m3"), ("b", "m4")], 1)
    ]
    rows = [{"provider": f.provider, "match_id": f.match_id} for f in frames]
    assert tracking_quality(frames)["cross_provider_ready"]
    assert len(grouped_leave_one_domain_out(rows)) == 2
    assert perturb_player_state(np.zeros(4), np.random.default_rng(1)).shape == (4,)


def test_skillcorner_adapter_streams_provider_neutral_frames(tmp_path):
    metadata = {
        "id": 7, "home_team": {"id": 1},
        "competition_edition": {"competition": {"name": "League"}},
        "players": [{"trackable_object": 10, "team_id": 1}, {"trackable_object": 20, "team_id": 2}],
    }
    frame = {
        "frame": 4, "timestamp": "00:00:00.4", "period": 1,
        "player_data": [{"player_id": 10, "x": 1, "y": 2}, {"player_id": 20, "x": -1, "y": -2}],
        "ball_data": {"x": 0, "y": 0},
    }
    match, tracking = tmp_path / "match.json", tmp_path / "tracking.jsonl"
    match.write_text(__import__("json").dumps(metadata), encoding="utf-8")
    tracking.write_text(__import__("json").dumps(frame) + "\n", encoding="utf-8")
    parsed = list(load_skillcorner_frames(match, tracking))
    assert parsed[0].provider == "skillcorner"
    assert set(parsed[0].home) == {"10"} and set(parsed[0].away) == {"20"}


def test_skillcorner_adapter_skips_frames_without_match_clock(tmp_path):
    metadata = {"id": 7, "home_team": {"id": 1}, "competition_edition": {"competition": {"name": "League"}}, "players": []}
    invalid = {"frame": 0, "timestamp": None, "period": None, "player_data": [], "ball_data": None}
    valid = {"frame": 1, "timestamp": "00:00:00.1", "period": 1, "player_data": [], "ball_data": None}
    match, tracking = tmp_path / "match.json", tmp_path / "tracking.jsonl"
    match.write_text(__import__("json").dumps(metadata), encoding="utf-8")
    tracking.write_text("\n".join(map(__import__("json").dumps, (invalid, valid))) + "\n", encoding="utf-8")
    parsed = list(load_skillcorner_frames(match, tracking))
    assert len(parsed) == 1 and parsed[0].period == 1


def test_joint_pass_model_is_physics_anchored_and_uncertainty_aware():
    size = len(PASS_FEATURES)
    model = JointPassModel(np.zeros(size), np.ones(size), np.ones(size), np.ones(size))
    complete = model.predict(np.zeros(size), physics_prior=0.7)
    missing = model.predict(np.full(size, np.nan), physics_prior=0.7)
    assert complete.completion_probability == pytest.approx(0.7)
    assert missing.epistemic_uncertainty > complete.epistemic_uncertainty
    assert all(pass_model_gate(
        {"auc": 0.62, "brier": 0.16, "ece": 0.02, "worst_match_auc": 0.55, "matches": 9},
        {"auc": 0.60, "brier": 0.17},
    ).values())


def test_joint_pass_training_learns_receiver_and_completion_residuals():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(60, len(PASS_FEATURES)))
    events = np.repeat(np.arange(20), 3)
    receivers = np.zeros(60)
    receivers[np.arange(20) * 3 + np.argmax(x.reshape(20, 3, -1)[:, :, 0], axis=1)] = 1
    completion = (x[:, 1] > 0).astype(float)
    model = fit_joint_pass_model(x, events, receivers, completion, np.full(60, 0.5), steps=30)
    assert model.receiver_weights[0] > 0
    assert model.completion_weights[1] > 0


def test_joint_pass_training_treats_fully_missing_feature_as_neutral():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(30, len(PASS_FEATURES)))
    x[:, -1] = np.nan
    model = fit_joint_pass_model(x, np.repeat(np.arange(10), 3), np.tile([1, 0, 0], 10),
                                 np.tile([1, 0, 1], 10), np.full(30, .5), steps=2)
    assert model.mean[-1] == 0 and model.scale[-1] == 1
    assert np.isfinite(model.predict(x[0], physics_prior=.5).completion_probability)


def test_receiver_group_scoring_is_invariant_to_feature_scale():
    size = len(PASS_FEATURES)
    model = JointPassModel(np.zeros(size), np.ones(size), np.arange(1, size + 1), np.zeros(size), receiver_rank_based=True)
    values = np.arange(3 * size, dtype=float).reshape(3, size)
    assert np.allclose(model.score_receiver_candidates(values), model.score_receiver_candidates(values * 100 + 7))


def test_joint_pass_fit_accepts_completion_only_singletons():
    x = np.zeros((4, len(PASS_FEATURES)))
    model = fit_joint_pass_model(x, np.arange(4), np.ones(4), np.array([1, 0, 1, 0]), np.full(4, .5), steps=2)
    assert np.allclose(model.receiver_weights, 0)


def test_probabilistic_future_is_multimodal_and_normalized():
    future = future_from_ensemble(
        pass_probabilities=np.array([0.6, 0.8, 0.7]),
        shot_probabilities=np.array([0.1, 0.2, 0.15]),
        progress_samples=np.array([-0.1, 0.2, 0.4]),
        action_kind="pass",
        horizon_s=5.0,
    )
    assert set(future.event_probabilities) == set(EVENTS)
    assert sum(future.event_probabilities.values()) == pytest.approx(1.0)
    assert future.progress_quantiles[0] < future.progress_quantiles[2]


def test_psychological_state_supports_causal_ablation():
    state = PsychologicalState()
    signals = {"unity_signal": 0.8, "player_anxiety": 0.2, "media_amplification": 0.6}
    treated = state.updated(signals, 0.8)
    assert treated != state
    assert state.updated(signals, 0.8, enabled=False) == state
    assert set(treated.decision_modifiers()) == {"risk", "execution", "coordination"}


def test_meta_adaptation_expires_audits_and_rejects_sealed_data(tmp_path):
    agent = SimpleNamespace(
        name="A", tactical_controls={"risk_budget": 0.5},
        roles={"Icon": {"patience": 0.8}}, W_h=0.6, W_x=0.4,
        memory_event_log=[{"id": "e1"}],
    )
    controller = MetaLearningController(tmp_path / "audit.jsonl")
    proposal = controller.propose(agent, {
        "suggested_adjustments": {"risk_budget": 0.1},
        "evidence_memory_ids": ["e1"], "created_step": 2, "expires_after": 2,
    })
    controller.commit(agent, proposal)
    assert not controller.expire(agent, current_step=3)
    assert controller.expire(agent, current_step=4) == [proposal.proposal_id]
    assert (tmp_path / "audit.jsonl").read_text().count("\n") == 3
    with pytest.raises(ValueError, match="sealed"):
        controller.propose(agent, {"uses_sealed_data": True})


def test_shot_planner_requires_real_evidence_and_prices_uncertainty():
    weak = ShotEvidence(45, 5, 9, 1, 0.09, 0.10)
    strong = ShotEvidence(1200, 100, 24, 2, 0.08, 0.09)
    assert not weak.enabled and strong.enabled
    certain = shot_counterfactual_value(
        goal_probability=0.3, rebound_value=0.1, turnover_cost=0.2,
        best_continuation_value=0.1, uncertainty=0.0,
    )
    uncertain = shot_counterfactual_value(
        goal_probability=0.3, rebound_value=0.1, turnover_cost=0.2,
        best_continuation_value=0.1, uncertainty=0.8,
    )
    assert certain > uncertain > 0.0


def test_shot_probability_model_round_trips(tmp_path):
    rng = np.random.default_rng(12)
    features = rng.normal(size=(80, len(SHOT_FEATURES)))
    goals = (features[:, 0] < -0.2).astype(float)
    model = fit_shot_probability_model(features, goals, steps=40)
    path = tmp_path / "shot.json"
    path.write_text(__import__("json").dumps(model.to_dict()), encoding="utf-8")
    restored = ShotProbabilityModel.load(path)
    assert restored.predict(features[0]) == pytest.approx(model.predict(features[0]))


def test_event_transition_model_is_smoothed_and_serializable(tmp_path):
    model = fit_event_transition_model(np.array(["pass", "pass", "shot"]), np.array(["retain", "turnover", "shot"]))
    assert set(model.predict("pass")) == set(EVENTS)
    assert all(value > 0 for value in model.predict("pass").values())
    path = tmp_path / "transition.json"
    path.write_text(__import__("json").dumps(model.to_dict()), encoding="utf-8")
    assert EventTransitionModel.load(path).predict("pass") == model.predict("pass")


def test_model_registry_enforces_linear_promotion_and_hashes(tmp_path):
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"model")
    registry = ModelRegistry(tmp_path / "registry.json")
    registry.register("pass", "7.0.0-candidate", artifact)
    registry.promote("pass", "7.0.0-candidate", "evaluated")
    registry.promote("pass", "7.0.0-candidate", "sealed", {"gate": True})
    registry.attest("pass", "7.0.0-candidate", {"release": "v7"})
    assert registry.data["models"]["pass:7.0.0-candidate"]["evidence"] == {"release": "v7"}
    with pytest.raises(ValueError):
        registry.promote("pass", "7.0.0-candidate", "evaluated")


def test_distribution_and_latency_regression_gates():
    gate = distribution_gate({"goals": 2.6}, {"goals": 2.5}, {"goals": 0.2})
    assert gate["goals"]
    latency = benchmark_latency(lambda: sum(range(20)), warmup=1, repeats=3)
    assert latency["repeats"] == 3 and latency["p95_ms"] >= 0

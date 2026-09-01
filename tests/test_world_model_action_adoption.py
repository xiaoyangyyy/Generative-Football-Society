"""Direct world-model guidance reaches action sampling with bounded auditability."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.match_engine.action_engine import ActionEngine
from src.match_engine.aerial_duel import AerialDuelEngine
from src.match_engine.calibration.benchmark_core import row_from_summary
from src.match_engine.macro_bridge import build_match_affective_state
from src.match_engine.match_micro_runner import _init_micro_state
from src.match_engine.micro_config import MicroMatchConfig
from src.match_engine.passing_engine import PassingEngine
from src.match_engine.shot_engine import ShotEngine
from src.match_engine.spatial_intelligence import SpatialIntelligenceEngine
from src.match_engine.state import PlayerModulators
from src.match_engine.world_model.action_adoption import (
    direct_action_adoption_diagnostics,
    mask_infeasible_action_probabilities,
    mix_direct_action_probabilities,
    record_pass_target_policy_sample,
    record_action_policy_sample,
    register_action_policy_opportunity,
    sample_action_from_uniform,
)
from src.match_engine.world_model.action_codec import decode_action_kind
from src.match_engine.world_model.config import WorldModelConfig
from src.match_engine.world_model.inference import WorldModelRuntime
from src.match_engine.world_model.observation import OBS_DIM
from src.match_engine.world_model.planner import (
    action_imagination_adjustments,
    pass_candidate_policy_probabilities,
)
from tests.test_affective_phase1b import _FakeAgent


class _ValidatedPassRuntime:
    def __init__(self, *, pass_confidence: float = 0.8):
        self.cfg = SimpleNamespace(planner_blend=0.30, shot_planner_blend=0.25)
        self.pass_confidence = pass_confidence
        self.last_uncertainty = 0.1

    def encode_state(self, state, *, attacking_home):
        return np.ones(8, dtype=np.float32)

    def planner_confidence(self, observation, kind="general"):
        return self.pass_confidence if kind == "pass" else 0.0

    def score_action(self, observation, action):
        self.last_uncertainty = 0.1
        return 0.30 if decode_action_kind(action) == "pass" else 0.0


class _RankedPassRuntime(_ValidatedPassRuntime):
    def score_action(self, observation, action):
        self.last_uncertainty = 0.1
        return float(action[6])


class _ValidatedShotRuntime:
    def __init__(self):
        self.cfg = SimpleNamespace(planner_blend=0.30, shot_planner_blend=0.25)
        self.last_uncertainty = 0.1

    def encode_state(self, state, *, attacking_home):
        return np.ones(8, dtype=np.float32)

    def planner_confidence(self, observation, kind="general"):
        return 0.8 if kind == "shot" else 0.0

    def score_action(self, observation, action):
        self.last_uncertainty = 0.1
        return 0.0

    def score_shot_action(self, observation, action, *, attacking_home):
        self.last_uncertainty = 0.1
        return 0.9


def test_validation_quality_gate_precedes_coverage_discount():
    runtime = object.__new__(WorldModelRuntime)
    runtime.cfg = WorldModelConfig(min_planner_quality=0.15)
    runtime.base_quality = 0.01
    runtime.pass_quality = 0.18
    runtime.shot_quality = 0.01
    runtime.online_calibrator = SimpleNamespace(trust_factor=lambda kind: 1.0)
    sparse_observation = np.zeros(OBS_DIM, dtype=np.float32)

    assert 0.0 < runtime.planner_confidence(sparse_observation, kind="pass") < 0.18
    assert runtime.planner_confidence(sparse_observation, kind="shot") == 0.0


def test_authorized_quality_is_not_reused_as_action_policy_attenuation():
    runtime = object.__new__(WorldModelRuntime)
    runtime.cfg = WorldModelConfig(min_planner_quality=0.15)
    runtime.base_quality = 0.01
    runtime.pass_quality = 0.151
    runtime.shot_quality = 0.01
    runtime.online_calibrator = SimpleNamespace(trust_factor=lambda kind: 0.8)
    sparse_observation = np.zeros(OBS_DIM, dtype=np.float32)

    authority = runtime.planner_authority(sparse_observation, kind="pass")

    assert authority["authorized"] is True
    assert authority["validation_quality"] == 0.151
    assert np.isclose(authority["decision_confidence"], 0.26)
    assert runtime.planner_confidence(sparse_observation, kind="pass") < 0.04


def test_infeasible_actions_have_zero_sampling_mass():
    masked = mask_infeasible_action_probabilities(
        ["pass", "shot", "cross", "hold"],
        [0.05, 0.70, 0.20, 0.05],
        {"pass", "hold"},
    )

    assert np.allclose(masked, [0.5, 0.0, 0.0, 0.5])


def test_bounded_controller_has_material_effect_after_quality_authorization():
    state = SimpleNamespace()
    labels = ["pass", "hold"]
    baseline = np.array([0.5, 0.5], dtype=float)
    register_action_policy_opportunity(
        state, team_id="home", t_sec=10.0,
        feasible_actions=set(labels), base_utilities=[0.0, 0.2],
        adjusted_utilities=[0.1, 0.2], labels=labels,
        model_adjustments={"pass": 0.1, "hold": 0.0},
        quality_gates={
            "pass": {
                "open": True,
                "validation_quality": 0.151,
                "minimum_validation_quality": 0.15,
                "decision_confidence": 0.8,
                "decision_certainty": 0.75,
                "policy_blend": 0.30,
                "model_advantage": 0.35,
            },
            "hold": {"open": False},
        },
    )

    adjusted = mix_direct_action_probabilities(
        state, labels=labels, probabilities=baseline,
    )

    assert adjusted[0] - baseline[0] > 0.08
    assert state._wm_pending_direct_action_adoption["recommended_action"] == "pass"


def test_controller_competes_across_validated_actions_on_one_simplex():
    state = SimpleNamespace()
    labels = ["pass", "shot", "cross", "hold"]
    baseline = np.array([0.45, 0.10, 0.0, 0.45], dtype=float)
    register_action_policy_opportunity(
        state, team_id="home", t_sec=12.0,
        feasible_actions={"pass", "shot", "hold"},
        base_utilities=[0.4, 0.1, 0.0, 0.4],
        adjusted_utilities=[0.3, 0.3, 0.0, 0.4], labels=labels,
        model_adjustments={"pass": -0.1, "shot": 0.2, "cross": 0.0, "hold": 0.0},
        quality_gates={
            "pass": {
                "open": True, "decision_confidence": 0.8,
                "decision_certainty": 0.8, "policy_blend": 0.3,
                "model_advantage": -0.2,
            },
            "shot": {
                "open": True, "decision_confidence": 0.9,
                "decision_certainty": 0.8, "policy_blend": 0.25,
                "model_advantage": 0.3,
            },
            "cross": {
                "open": True, "decision_confidence": 1.0,
                "decision_certainty": 1.0, "policy_blend": 0.35,
                "model_advantage": 0.35,
            },
            "hold": {"open": False},
        },
    )

    adjusted = mix_direct_action_probabilities(
        state, labels=labels, probabilities=baseline,
    )

    assert np.isclose(adjusted.sum(), 1.0)
    assert adjusted[1] > baseline[1]
    assert adjusted[0] < baseline[0]
    assert adjusted[2] == 0.0
    gates = state._wm_pending_direct_action_adoption["quality_gates"]
    assert gates["shot"]["model_target_action_probability"] > baseline[1]
    assert "model_target_action_probability" not in gates["cross"]
    record_action_policy_sample(
        state, actual_action="shot", labels=labels,
        base_probabilities=baseline, adjusted_probabilities=adjusted,
        sampling_uniform=0.5, counterfactual_baseline_action="pass",
    )
    audit = direct_action_adoption_diagnostics(state)
    assert audit["probability_policy_version"] == "validated_action_simplex_v1"
    assert set(audit["action_signal_breakdown"]) == {"pass", "shot"}
    assert audit["action_signal_breakdown"]["pass"]["negative_guidance"] == 1
    assert audit["action_signal_breakdown"]["shot"]["positive_guidance"] == 1
    assert audit["action_signal_breakdown"]["shot"]["locally_changed_to_action"] == 1


def test_world_model_ranks_and_audits_the_executed_pass_target(monkeypatch):
    monkeypatch.setenv("MATCH_WORLD_MODEL", "1")
    monkeypatch.setenv("MATCH_WM_PLAN", "1")
    rng = np.random.default_rng(7)
    state = build_match_affective_state(
        _FakeAgent("Home"), _FakeAgent("Away"), rng=rng,
    )
    carrier, receiver_a, receiver_b = state.home.players[:3]
    carrier.on_pitch = receiver_a.on_pitch = receiver_b.on_pitch = True
    meta = [
        (receiver_a, "short", np.array([0.35, 0.45]), 0.7, 0.2, 0.0, 0.8),
        (receiver_b, "through", np.array([0.80, 0.55]), 0.7, 0.2, 0.0, 0.8),
    ]
    base = np.array([0.5, 0.5], dtype=float)

    adjusted, evidence = pass_candidate_policy_probabilities(
        _RankedPassRuntime(), state, carrier, meta, True, base,
    )

    assert evidence["applied"] is True
    assert adjusted[1] > base[1]
    register_action_policy_opportunity(
        state, team_id=carrier.team_id, t_sec=10.0,
        feasible_actions={"pass", "hold"},
        base_utilities=[0.0, 0.0], adjusted_utilities=[0.1, 0.0],
        labels=["pass", "hold"],
        model_adjustments={"pass": 0.1, "hold": 0.0},
        quality_gates={"pass": {
            "open": True, "confidence": 0.8, "certainty": 0.9,
            "policy_blend": 0.3, "model_advantage": 0.2,
        }},
    )
    record_pass_target_policy_sample(
        state,
        candidate_ids=["a:short:0", "b:through:1"],
        base_probabilities=base,
        adjusted_probabilities=adjusted,
        selected_index=1,
        counterfactual_index=0,
        sampling_uniform=0.51,
        evidence=evidence,
    )
    audit = direct_action_adoption_diagnostics(state)
    target = audit["records"][0]["pass_target_policy"]
    assert audit["pass_target_opportunities"] == 1
    assert audit["pass_target_influenced_opportunities"] == 1
    assert audit["pass_target_changes"] == 1
    assert target["selected_candidate_id"] == "b:through:1"
    assert target["counterfactual_baseline_candidate_id"] == "a:short:0"


def _planner_state():
    return SimpleNamespace(
        ball=SimpleNamespace(position=np.array([0.5, 0.5], dtype=float)),
        clock_seconds=10.0,
        _wm_horizon_s=10.0,
    )


def test_validated_pass_evidence_changes_high_level_utility_and_is_audited(
    monkeypatch,
):
    monkeypatch.setenv("MATCH_WORLD_MODEL", "1")
    monkeypatch.setenv("MATCH_WM_PLAN", "1")
    state = _planner_state()
    carrier = SimpleNamespace(team_id="home", position=state.ball.position)
    base = np.array([0.1, 0.2, -0.1, 0.0], dtype=float)

    adjusted = action_imagination_adjustments(
        _ValidatedPassRuntime(), state, carrier, True, base,
        ["pass", "shot", "cross", "hold"], dist_goal=0.5,
        feasible_actions={"pass", "hold"},
    )

    assert adjusted[0] > base[0]
    assert np.array_equal(adjusted[1:], base[1:])
    audit = direct_action_adoption_diagnostics(state)
    assert audit["opportunities"] == 1
    assert audit["influenced_opportunities"] == 1
    record = audit["records"][0]
    assert record["quality_gates"]["pass"]["open"] is True
    assert record["quality_gates"]["pass"]["model_advantage"] == 0.30
    assert record["quality_gates"]["shot"]["reason"] == "action_infeasible"
    assert record["quality_gates"]["cross"]["reason"] == "no_action_specific_validation"


def test_validated_shot_advantage_reaches_probability_controller(monkeypatch):
    monkeypatch.setenv("MATCH_WORLD_MODEL", "1")
    monkeypatch.setenv("MATCH_WM_PLAN", "1")
    state = _planner_state()
    carrier = SimpleNamespace(team_id="home", position=state.ball.position)
    labels = ["pass", "shot", "cross", "hold"]
    utilities = np.array([0.2, 0.0, -0.1, 0.2], dtype=float)

    adjusted_utilities = action_imagination_adjustments(
        _ValidatedShotRuntime(), state, carrier, True, utilities, labels,
        dist_goal=0.2, feasible_actions={"pass", "shot", "hold"},
    )
    baseline = np.array([0.45, 0.10, 0.0, 0.45], dtype=float)
    adjusted = mix_direct_action_probabilities(
        state, labels=labels, probabilities=baseline,
    )

    gate = state._wm_pending_direct_action_adoption["quality_gates"]["shot"]
    assert adjusted_utilities[1] > utilities[1]
    assert gate["open"] is True
    assert gate["reason"] == "validated_shot_vs_continuation_advantage"
    assert gate["model_advantage"] > 0.0
    assert adjusted[1] > baseline[1]
    assert gate["model_target_action_probability"] > baseline[1]


def test_closed_quality_gate_records_opportunity_without_changing_policy(
    monkeypatch,
):
    monkeypatch.setenv("MATCH_WORLD_MODEL", "1")
    monkeypatch.setenv("MATCH_WM_PLAN", "1")
    state = _planner_state()
    carrier = SimpleNamespace(team_id="home", position=state.ball.position)
    base = np.array([0.1, 0.2, -0.1, 0.0], dtype=float)

    adjusted = action_imagination_adjustments(
        _ValidatedPassRuntime(pass_confidence=0.0),
        state, carrier, True, base,
        ["pass", "shot", "cross", "hold"], dist_goal=0.5,
        feasible_actions={"pass", "hold"},
    )

    assert np.array_equal(adjusted, base)
    audit = direct_action_adoption_diagnostics(state)
    assert audit["influenced_opportunities"] == 0
    assert audit["records"][0]["quality_gates"]["pass"]["reason"] == (
        "pass_quality_gate_closed"
    )


def test_shared_uniform_identifies_world_model_changed_action():
    state = SimpleNamespace()
    labels = ["pass", "hold"]
    baseline = np.array([0.9, 0.1], dtype=float)
    register_action_policy_opportunity(
        state, team_id="home", t_sec=10.0,
        feasible_actions=set(labels), base_utilities=[1.0, 0.0],
        adjusted_utilities=[1.0, 0.0], labels=labels,
        model_adjustments={"pass": 0.0, "hold": 0.0},
        quality_gates={
            "pass": {
                "open": True, "confidence": 1.0, "certainty": 1.0,
                "policy_blend": 0.3, "model_advantage": -0.35,
            },
            "hold": {"open": False},
        },
    )
    adjusted = mix_direct_action_probabilities(
        state, labels=labels, probabilities=baseline,
    )
    draw = 0.75
    baseline_action = sample_action_from_uniform(labels, baseline, draw)
    adjusted_action = sample_action_from_uniform(labels, adjusted, draw)
    record_action_policy_sample(
        state, actual_action=adjusted_action, labels=labels,
        base_probabilities=baseline, adjusted_probabilities=adjusted,
        sampling_uniform=draw,
        counterfactual_baseline_action=baseline_action,
    )

    assert baseline_action == "pass"
    assert adjusted_action == "hold"
    audit = direct_action_adoption_diagnostics(state)
    assert audit["counterfactual_action_changes"] == 1
    assert audit["expected_counterfactual_action_changes"] > 0.0
    assert audit["attribution_eligible_opportunities"] == 1
    assert audit["counterfactual_change_rate"] == 1.0
    assert audit["records"][0]["policy_changed_action"] is True
    assert audit["records"][0]["total_variation_distance"] > 0.0


def test_action_engine_resolves_direct_policy_against_actual_sample(monkeypatch):
    monkeypatch.setenv("MATCH_WORLD_MODEL", "1")
    monkeypatch.setenv("MATCH_WM_PLAN", "1")
    cfg = MicroMatchConfig.fast_demo()
    cfg.enable_wall_pass = False
    rng = np.random.default_rng(17)
    state = build_match_affective_state(
        _FakeAgent("Home"), _FakeAgent("Away"), rng=rng,
    )
    _init_micro_state(state, cfg, rng)
    state.clock_seconds = 11.0
    spatial = SpatialIntelligenceEngine(cfg)
    passing = PassingEngine(cfg, spatial)
    engine = ActionEngine(
        cfg, passing, ShotEngine(cfg, spatial), AerialDuelEngine(cfg),
        wm_runtime=_ValidatedPassRuntime(),
    )
    mod_home = [PlayerModulators(player_id=p.player_id) for p in state.home.players]
    mod_away = [PlayerModulators(player_id=p.player_id) for p in state.away.players]

    actual_action, _ = engine.step(state, mod_home, mod_away, rng)

    audit = direct_action_adoption_diagnostics(state)
    assert audit["resolved"] == 1
    assert audit["influenced_opportunities"] == 1
    record = audit["records"][0]
    assert record["actual_action"] == actual_action
    assert record["sampling_uniform"] is not None
    assert record["counterfactual_baseline_action"] in {
        "pass", "shot", "cross", "hold",
    }
    assert record["attribution_eligible"] is True
    assert record["adjusted_probability"]["pass"] > record["base_probability"]["pass"]
    assert record["adopted"] is (actual_action == record["recommended_action"])
    assert record["resolution"] == "sampled_after_world_model_adjustment"


def test_benchmark_row_exports_action_adoption_mechanism_metrics():
    summary = SimpleNamespace(
        passes_home=1, passes_away=1,
        pass_completion_home=1.0, pass_completion_away=0.0,
        pass_intercepts_home=0, pass_intercepts_away=0,
        shots_home=0, shots_away=0, goals_micro_home=0, goals_micro_away=0,
        micro_xg_home=0.0, micro_xg_away=0.0,
        through_balls_home=0, through_balls_away=0,
        long_passes_home=0, long_passes_away=0,
        shots_on_target_home=0, shots_on_target_away=0,
        fouls_committed_home=0, fouls_committed_away=0,
        yellow_cards_home=0, yellow_cards_away=0,
        red_cards_home=0, red_cards_away=0, possession_home=0.5,
        crosses_attempted=0, headers_attempted=0,
        tackles_home=0, tackles_away=0,
        world_model_action_adoption={
            "opportunities": 8, "influenced_opportunities": 6,
            "adopted": 4, "attributable_adoptions": 3,
            "attribution_eligible_opportunities": 6,
            "adoption_rate": 0.5,
            "counterfactual_action_changes": 2,
            "expected_counterfactual_action_changes": 2.5,
            "counterfactual_change_rate": 0.25,
            "expected_counterfactual_change_rate": 0.3125,
            "mean_recommended_probability_shift": 0.02,
            "pass_target_opportunities": 5,
            "pass_target_influenced_opportunities": 4,
            "pass_target_changes": 1,
            "pass_target_expected_changes": 1.5,
        },
    )

    row = row_from_summary(summary)

    assert row["wm_action_opportunities"] == 8.0
    assert row["wm_action_influenced_opportunities"] == 6.0
    assert row["wm_action_attributable_adoptions"] == 3.0
    assert row["wm_action_attribution_eligible_opportunities"] == 6.0
    assert row["wm_action_counterfactual_changes"] == 2.0
    assert row["wm_action_expected_counterfactual_changes"] == 2.5
    assert row["wm_action_counterfactual_change_rate"] == 0.25
    assert row["wm_action_expected_counterfactual_change_rate"] == 0.3125
    assert row["wm_action_mean_probability_shift"] == 0.02
    assert row["wm_pass_target_opportunities"] == 5.0
    assert row["wm_pass_target_influenced_opportunities"] == 4.0
    assert row["wm_pass_target_changes"] == 1.0
    assert row["wm_pass_target_expected_changes"] == 1.5

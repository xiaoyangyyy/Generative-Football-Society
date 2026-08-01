"""Cross-match residual memory is contextual, conformal, and checkpoint-scoped."""

import json
from dataclasses import replace

import pytest

from src.match_engine.cognitive.config import CognitiveMatchConfig
from src.match_engine.micro_config import MicroMatchConfig
from src.match_engine.world_model.residual_memory import (
    compile_contextual_residual_memory,
    detect_residual_drift,
    load_contextual_residual_memory,
    policy_environment_signature,
)


def _payload(signature, residuals, *, action="pass", horizon="60s"):
    records = []
    for residual in residuals:
        records.append({
            "checkpoint_signature": signature,
            "environment_signature": "environment_unspecified",
            "policy_utility_version": 1,
            "team_id": "Home",
            "intervention_actual_action": action,
            "outcome_baseline": {
                "opponent_team_id": "Away",
                "zone": "middle",
                "score_state": "level",
                "match_phase": "early",
            },
            "multi_horizon_regime_outcomes": {
                horizon: {
                    "policy_utility": residual,
                    "world_model_prediction": {
                        "raw_policy_utility": 0.0,
                        "policy_utility": 0.0,
                        "uncertainty": 0.2,
                    },
                },
            },
        })
    return {
        "home": "Home",
        "away": "Away",
        "world_model_decision_adoption": {"records": records},
    }


def _multi_horizon_payload(signature, residual_paths):
    records = []
    for near, far in residual_paths:
        records.append({
            "checkpoint_signature": signature,
            "environment_signature": "environment_unspecified",
            "policy_utility_version": 1,
            "team_id": "Home",
            "intervention_actual_action": "pass",
            "outcome_baseline": {
                "opponent_team_id": "Away",
                "zone": "middle",
                "score_state": "level",
                "match_phase": "early",
            },
            "multi_horizon_regime_outcomes": {
                "60s": {
                    "policy_utility": near,
                    "world_model_prediction": {
                        "raw_policy_utility": 0.0,
                        "policy_utility": 0.0,
                        "uncertainty": 0.2,
                    },
                },
                "180s": {
                    "policy_utility": far,
                    "world_model_prediction": {
                        "raw_policy_utility": 0.0,
                        "policy_utility": 0.0,
                        "uncertainty": 0.2,
                    },
                },
            },
        })
    return {
        "home": "Home", "away": "Away",
        "world_model_decision_adoption": {"records": records},
    }


def test_split_conformal_memory_corrects_repeated_contextual_bias():
    memory = compile_contextual_residual_memory(
        [_payload("sig-a", [0.2] * 8)],
        checkpoint_signature="sig-a",
        min_samples=8,
    )
    corrected = memory.calibrate(
        {"policy_utility": 0.0, "uncertainty": 0.2},
        context={
            "team_id": "Home",
            "opponent_team_id": "Away",
            "zone": "middle",
            "score_state": "level",
            "match_phase": "early",
        },
        action="pass",
        horizon_key="60s",
    )
    assert corrected["raw_policy_utility"] == 0.0
    assert corrected["policy_utility"] == pytest.approx(0.2)
    assert corrected["residual_memory"]["scope"] == "team_opponent_context"
    assert corrected["residual_memory"]["interval_90"] == pytest.approx(
        [0.2, 0.2]
    )
    assert corrected["residual_memory_trust_factor"] == 1.0
    assert corrected["residual_memory"]["residual_quantile_levels"] == [
        0.1, 0.25, 0.5, 0.75, 0.9,
    ]
    assert corrected["residual_memory"]["residual_quantiles"] == pytest.approx(
        [0.0] * 5
    )
    assert corrected["residual_memory"]["residual_quantiles_split"] == (
        "held_out_calibration"
    )


def test_memory_rejects_other_checkpoint_and_rejects_harmful_bias_fit():
    mismatch = compile_contextual_residual_memory(
        [_payload("old", [0.2] * 8)],
        checkpoint_signature="new",
        min_samples=8,
    )
    assert mismatch.rows == 0
    assert mismatch.groups == {}

    changing = compile_contextual_residual_memory(
        [_payload("sig", [1.0] * 4 + [-1.0] * 4)],
        checkpoint_signature="sig",
        min_samples=8,
    )
    corrected = changing.calibrate(
        {"policy_utility": 0.0, "uncertainty": 0.2},
        context={
            "team_id": "Home",
            "opponent_team_id": "Away",
            "zone": "middle",
            "score_state": "level",
            "match_phase": "early",
        },
        action="pass",
        horizon_key="60s",
    )
    assert corrected["residual_memory"]["bias_correction"] == 0.0
    assert corrected["policy_utility"] == 0.0


def test_context_backoff_uses_action_horizon_when_exact_context_is_sparse():
    logs = [
        _payload("sig", [0.1, 0.1], action="cross")
        for _ in range(4)
    ]
    for index, payload in enumerate(logs):
        payload["home"] = f"Team-{index}"
        for record in payload["world_model_decision_adoption"]["records"]:
            record["team_id"] = f"Team-{index}"
            record["outcome_baseline"]["opponent_team_id"] = (
                f"Opponent-{index}"
            )
    memory = compile_contextual_residual_memory(
        logs, checkpoint_signature="sig", min_samples=8,
    )
    corrected = memory.calibrate(
        {"policy_utility": 0.0, "uncertainty": 0.2},
        context={
            "team_id": "Different",
            "opponent_team_id": "Opponent",
            "zone": "final_third",
            "score_state": "trailing",
            "match_phase": "late",
        },
        action="cross",
        horizon_key="60s",
    )
    assert corrected["residual_memory"]["scope"] == "action_horizon"
    assert corrected["policy_utility"] == pytest.approx(0.1)


def test_loader_reads_persisted_logs_and_ignores_malformed_files(tmp_path):
    log_dir = tmp_path / "data" / "persistence" / "cognitive_log"
    log_dir.mkdir(parents=True)
    (log_dir / "valid.json").write_text(
        json.dumps(_payload("sig", [0.2] * 8)), encoding="utf-8",
    )
    (log_dir / "broken.json").write_text("{not-json", encoding="utf-8")
    memory = load_contextual_residual_memory(
        tmp_path, checkpoint_signature="sig", min_samples=8,
    )
    assert memory.source_logs == 1
    assert memory.rows == 8
    assert memory.groups


def test_policy_environment_signature_tracks_decision_relevant_settings():
    micro = MicroMatchConfig()
    cognitive = CognitiveMatchConfig()
    signature = policy_environment_signature(micro, cognitive)

    assert signature.startswith("policy-env:")
    assert signature == policy_environment_signature(
        MicroMatchConfig(), CognitiveMatchConfig(),
    )
    assert signature != policy_environment_signature(
        replace(micro, action_tau=micro.action_tau + 0.01), cognitive,
    )
    assert signature != policy_environment_signature(
        micro,
        replace(
            cognitive,
            world_model_action_control_rate=(
                cognitive.world_model_action_control_rate + 0.01
            ),
        ),
    )
    assert signature != policy_environment_signature(
        micro,
        replace(cognitive, world_model_contrastive_repair=True),
    )


def test_memory_never_pools_matching_checkpoint_across_environments():
    payload = _payload("sig", [0.2] * 8)
    for record in payload["world_model_decision_adoption"]["records"]:
        record["environment_signature"] = "env-a"

    mismatch = compile_contextual_residual_memory(
        [payload],
        checkpoint_signature="sig",
        environment_signature="env-b",
        min_samples=8,
    )
    assert mismatch.rows == 0
    assert mismatch.groups == {}

    for record in payload["world_model_decision_adoption"]["records"]:
        record.pop("environment_signature")
    legacy = compile_contextual_residual_memory(
        [payload], checkpoint_signature="sig", min_samples=8,
    )
    current = compile_contextual_residual_memory(
        [payload],
        checkpoint_signature="sig",
        environment_signature="policy-env:current",
        min_samples=8,
    )
    assert legacy.rows == 8
    assert current.rows == 0


def test_residual_drift_quarantines_point_correction():
    payload = _payload("sig", [0.0] * 16 + [2.0] * 16)
    memory = compile_contextual_residual_memory(
        [payload], checkpoint_signature="sig", min_samples=8,
    )

    corrected = memory.calibrate(
        {"policy_utility": 0.3, "uncertainty": 0.2},
        context={
            "team_id": "Home",
            "opponent_team_id": "Away",
            "zone": "middle",
            "score_state": "level",
            "match_phase": "early",
        },
        action="pass",
        horizon_key="60s",
    )
    assert memory.drift["status"] == "quarantined"
    assert corrected["policy_utility"] == 0.3
    assert corrected["residual_memory"]["scope"] == (
        "quarantined_distribution_drift"
    )
    assert corrected["residual_memory_trust_factor"] == 0.5


def test_context_drift_enters_watch_and_shrinks_memory_influence():
    payload = _payload("sig", [0.2] * 32)
    records = payload["world_model_decision_adoption"]["records"]
    for record in records[16:]:
        record["outcome_baseline"].update({
            "zone": "final_third",
            "score_state": "trailing",
            "match_phase": "late",
        })
    memory = compile_contextual_residual_memory(
        [payload], checkpoint_signature="sig", min_samples=8,
    )
    corrected = memory.calibrate(
        {"policy_utility": 0.0, "uncertainty": 0.2},
        context={
            "team_id": "Home",
            "opponent_team_id": "Away",
            "zone": "final_third",
            "score_state": "trailing",
            "match_phase": "late",
        },
        action="pass",
        horizon_key="60s",
    )

    assert memory.drift["status"] == "watch"
    assert memory.drift["context_total_variation"] == 1.0
    assert corrected["policy_utility"] == pytest.approx(0.1)
    assert corrected["residual_memory_trust_factor"] <= 0.75


def test_adjacent_windows_recover_after_new_regime_stabilizes():
    rows = []
    for residual in [0.0] * 16 + [2.0] * 32:
        rows.append({
            "actual": residual,
            "predicted": 0.0,
            "context": {
                "zone": "middle",
                "score_state": "level",
                "match_phase": "early",
            },
        })

    drift = detect_residual_drift(rows)
    assert drift["status"] == "stable"
    assert drift["window_samples"] == 16
    assert drift["standardized_mean_shift"] == 0.0


def test_temporal_memory_learns_held_out_cross_action_rank_templates():
    anti_correlated = [
        (-0.3, 0.3), (-0.1, 0.1), (0.1, -0.1), (0.3, -0.3),
    ] * 4
    memory = compile_contextual_residual_memory(
        [_multi_horizon_payload("sig", anti_correlated)],
        checkpoint_signature="sig", min_samples=8,
    )
    coupling = memory.temporal_rank_coupling(
        context={
            "team_id": "Home", "opponent_team_id": "Away",
            "zone": "middle", "score_state": "level",
            "match_phase": "early",
        },
        horizon_keys=("180s", "60s"),
    )

    assert coupling["available"]
    assert coupling["scope"] == "team_opponent_context"
    assert coupling["horizon_keys"] == ["60s", "180s"]
    assert coupling["calibration_samples"] == 8
    assert coupling["rank_templates"][0][0] < coupling[
        "rank_templates"
    ][0][1]
    assert coupling["rank_templates"][2][0] > coupling[
        "rank_templates"
    ][2][1]
    assert coupling["mean_absolute_rank_correlation"] == pytest.approx(1.0)
    assert coupling["shared_across_candidate_actions"]
    assert not coupling["temporal_joint_calibrated"]
    assert memory.summary()["temporal_residual_rank_memory"][
        "active_groups"
    ] > 0
    diagnostics = memory.diagnostics()["temporal_residual_rank_memory"]
    assert diagnostics["groups"][0]["rank_templates"]

    mismatch = compile_contextual_residual_memory(
        [_multi_horizon_payload("sig", anti_correlated)],
        checkpoint_signature="other", min_samples=8,
    )
    assert not mismatch.temporal_rank_coupling(
        context={}, horizon_keys=("60s", "180s"),
    )["available"]

    other_environment = json.loads(json.dumps(
        _multi_horizon_payload("sig", anti_correlated)
    ))
    for record in other_environment["world_model_decision_adoption"][
        "records"
    ]:
        record["environment_signature"] = "other-environment"
    isolated = compile_contextual_residual_memory(
        [other_environment], checkpoint_signature="sig", min_samples=8,
    )
    assert not isolated.temporal_rank_coupling(
        context={}, horizon_keys=("60s", "180s"),
    )["available"]

    memory.temporal_rank_memory.drift = {"status": "watch"}
    gated = memory.temporal_rank_coupling(
        context={}, horizon_keys=("60s", "180s"),
    )
    assert not gated["available"]
    assert gated["reason"] == "temporal_rank_memory_drift_gate_closed"

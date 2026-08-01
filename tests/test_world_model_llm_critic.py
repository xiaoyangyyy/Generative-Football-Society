from __future__ import annotations

import json

import pytest

from src.match_engine.world_model.llm_critic import (
    apply_llm_world_model_critique,
    llm_critic_signature,
    validate_llm_world_model_critique,
)
from src.match_engine.world_model.llm_critic_memory import (
    compile_llm_critic_memory,
    llm_critic_diagnostics,
    load_llm_critic_memory,
)


def _record(
    residual=0.08,
    *,
    proposal=0.08,
    checkpoint="checkpoint-a",
    environment="env-a",
):
    return {
        "team_id": "Home",
        "checkpoint_signature": checkpoint,
        "environment_signature": environment,
        "intervention_actual_action": "pass",
        "llm_world_model_critique_context": {
            "version": 1,
            "authority_active": False,
            "critique": {
                "action": "pass",
                "horizon": "60s",
                "direction": "underestimate",
            },
            "proposed_policy_utility_correction": proposal,
            "applied_ranking_correction": 0.0,
            "world_model_prediction_mutated": False,
            "can_update_world_model": False,
            "can_update_critic_memory": False,
        },
        "multi_horizon_regime_outcomes": {
            "60s": {
                "policy_utility": residual,
                "world_model_prediction": {
                    "policy_utility": 0.0,
                    "raw_policy_utility": 0.0,
                },
            },
        },
    }


def _match(residual=0.08, *, repeats=1, **kwargs):
    return {
        "world_model_decision_adoption": {
            "records": [_record(residual, **kwargs) for _ in range(repeats)],
        },
    }


def _packet():
    return {
        "recommended_action": "hold",
        "checkpoint_signature": "checkpoint-a",
        "environment_signature": "env-a",
        "decision_context": {
            "zone": "middle",
            "score_state": "level",
            "match_phase": "early",
        },
        "candidates": [
            {
                "action": "hold",
                "effective_confidence": 0.8,
                "risk_adjusted_value": 0.50,
                "multi_horizon_predictions": {"60s": {"policy_utility": 0.1}},
            },
            {
                "action": "pass",
                "effective_confidence": 0.8,
                "risk_adjusted_value": 0.49,
                "multi_horizon_predictions": {"60s": {"policy_utility": 0.1}},
            },
        ],
    }


def _critique():
    return {
        "action": "pass",
        "horizon": "60s",
        "direction": "underestimate",
        "confidence": 1.0,
        "evidence_features": ["zone", "epistemic_uncertainty"],
        "rationale": "The semantic regime may be underrepresented.",
    }


def test_critic_memory_activates_only_after_match_held_out_gain():
    memory = compile_llm_critic_memory(
        [_match() for _ in range(8)],
        checkpoint_signature="checkpoint-a",
        environment_signature="env-a",
    )
    profile = memory.profile("pass", "60s")

    assert profile is not None
    assert profile.training_matches == 4
    assert profile.validation_matches == 4
    assert profile.fitted_scale == pytest.approx(1.0)
    assert profile.validation_skill > 0.02
    assert profile.active
    assert 0.0 < profile.trust <= 0.35


def test_critic_signature_is_stable_and_model_specific():
    assert llm_critic_signature("model-a") == llm_critic_signature("model-a")
    assert llm_critic_signature("model-a") != llm_critic_signature("model-b")


def test_critic_memory_is_match_clustered_and_rejects_scope_mismatch():
    memory = compile_llm_critic_memory(
        [
            _match(repeats=8),
            _match(),
            _match(checkpoint="checkpoint-b"),
            _match(environment="env-b"),
        ],
        checkpoint_signature="checkpoint-a",
        environment_signature="env-a",
    )
    profile = memory.profile("pass", "60s")

    assert memory.compatible_matches == 2
    assert profile is not None
    assert profile.matches == 2
    assert profile.rows == 9


def test_harmful_validation_keeps_critic_at_zero_authority():
    memory = compile_llm_critic_memory(
        [
            *[_match(0.08) for _ in range(4)],
            *[_match(-0.08) for _ in range(4)],
        ],
        checkpoint_signature="checkpoint-a",
        environment_signature="env-a",
    )
    profile = memory.profile("pass", "60s")

    assert profile is not None
    assert not profile.active
    assert profile.trust == 0.0
    assert not memory.authority("pass", "60s")["active"]


def test_validated_critique_is_bounded_and_does_not_mutate_forecast():
    memory = compile_llm_critic_memory(
        [_match() for _ in range(8)],
        checkpoint_signature="checkpoint-a",
        environment_signature="env-a",
    )
    packet = _packet()
    original_prediction = dict(
        packet["candidates"][1]["multi_horizon_predictions"]["60s"]
    )
    audit = apply_llm_world_model_critique(
        packet, _critique(), memory, selected_action="pass",
    )

    assert audit["accepted"]
    assert audit["authority_active"]
    assert 0.0 < audit["applied_ranking_correction"] <= 0.04
    assert not audit["world_model_prediction_mutated"]
    assert not audit["can_update_world_model"]
    assert not audit["can_update_critic_memory"]
    assert packet["candidates"][1]["multi_horizon_predictions"][
        "60s"
    ] == original_prediction

    incompatible_packet = _packet()
    incompatible_packet["environment_signature"] = "env-b"
    incompatible = apply_llm_world_model_critique(
        incompatible_packet, _critique(), memory, selected_action="pass",
    )
    assert not incompatible["authority_active"]
    assert not incompatible["memory_compatible"]
    assert incompatible["applied_ranking_correction"] == 0.0

    different_llm = apply_llm_world_model_critique(
        _packet(),
        _critique(),
        memory,
        selected_action="pass",
        critic_signature="different-llm-contract",
    )
    assert not different_llm["authority_active"]
    assert not different_llm["memory_compatible"]

    caution_packet = _packet()
    caution = apply_llm_world_model_critique(
        caution_packet,
        {**_critique(), "direction": "overestimate"},
        memory,
        selected_action="pass",
    )
    assert caution["applied_ranking_correction"] < 0.0
    assert 0.5 <= caution_packet["candidates"][1][
        "llm_semantic_critic_bridge_factor"
    ] < 1.0


def test_unvalidated_critique_runs_in_shadow_and_schema_is_grounded():
    assert validate_llm_world_model_critique({
        **_critique(), "evidence_features": ["invented_fact"],
    }) is None
    packet = _packet()
    audit = apply_llm_world_model_critique(
        packet, _critique(), None, selected_action="pass",
    )
    rejected = apply_llm_world_model_critique(
        _packet(), _critique(), None, selected_action="hold",
    )

    assert audit["accepted"]
    assert not audit["authority_active"]
    assert audit["applied_ranking_correction"] == 0.0
    assert audit["reason"] == "shadow_critique_for_validation"
    assert not rejected["accepted"]


def test_loader_ignores_corrupt_logs_and_diagnostics_score_realizations(tmp_path):
    directory = tmp_path / "data" / "persistence" / "cognitive_log"
    directory.mkdir(parents=True)
    payload = _match()
    context = payload["world_model_decision_adoption"]["records"][0][
        "llm_world_model_critique_context"
    ]
    context["authority_active"] = True
    context["correction_applied"] = True
    context["applied_ranking_correction"] = 0.02
    (directory / "valid.json").write_text(
        json.dumps(payload), encoding="utf-8",
    )
    (directory / "broken.json").write_text("{bad", encoding="utf-8")

    memory = load_llm_critic_memory(
        tmp_path,
        checkpoint_signature="checkpoint-a",
        environment_signature="env-a",
    )
    report = llm_critic_diagnostics([payload])

    assert memory.source_logs == 1
    assert report["realized_critiques"] == 1
    assert report["realized_authority_active_critiques"] == 1
    assert report["realized_validated_corrections"] == 1
    assert report["match_clustered_corrected_mse"] < (
        report["match_clustered_baseline_mse"]
    )
    assert report["all_non_persistent"]
    assert report["all_predictions_immutable"]
    assert report["all_bridge_authority_non_increasing"]

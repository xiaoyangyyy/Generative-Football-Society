import pytest

from src.product.m2_research_control import build_m2_research_control


def _control(**overrides):
    values = {
        "protocol_available": True,
        "protocol_state": (
            "preregistered_code_ready_awaiting_sealed_checkpoint"
        ),
        "preflight_available": False,
        "preflight_current": False,
        "preflight_ready": False,
        "preflight_status": None,
        "frozen_training_command": "python train.py --frozen",
        "candidate_evidence_available": False,
        "candidate_evidence_current": False,
        "candidate_eligible": False,
        "checkpoint_path": None,
        "execution_state": (
            "preregistered_code_ready_awaiting_sealed_checkpoint"
        ),
        "runs_executed": 0,
        "fixed_run_budget": 360,
        "result_current": False,
        "result_status": None,
        "promotion_supported": False,
        "promotion_gates": {},
    }
    values.update(overrides)
    return build_m2_research_control(**values)


def test_m2_control_exposes_one_zero_training_next_action():
    control = _control()

    assert control["status"] == "awaiting_current_preflight"
    assert control["zero_execution_projection"] is True
    assert control["next_action"]["id"] == "record_training_preflight"
    assert control["next_action"]["training_required"] is False
    assert control["next_action"]["formal_matches_required"] is False
    assert [row["stage_id"] for row in control["stages"]] == [
        "freeze_protocol",
        "verify_training_readiness",
        "qualify_sealed_candidate",
        "execute_mirrored_study",
        "decide_promotion",
    ]


def test_m2_control_separates_training_from_formal_execution():
    awaiting = _control(
        preflight_available=True,
        preflight_current=True,
        preflight_ready=True,
        preflight_status="ready_for_m2_training",
    )
    assert awaiting["status"] == "awaiting_candidate"
    assert awaiting["next_action"]["training_required"] is True
    assert awaiting["next_action"]["formal_matches_required"] is False
    assert awaiting["next_action"]["commands"][0] == "python train.py --frozen"

    ready = _control(
        preflight_available=True,
        preflight_current=True,
        preflight_ready=True,
        preflight_status="ready_for_m2_training",
        candidate_evidence_available=True,
        candidate_evidence_current=True,
        candidate_eligible=True,
        checkpoint_path="data/world_model/m2.pt",
        execution_state="candidate_qualified_awaiting_execution",
    )
    assert ready["status"] == "ready_for_authorized_execution"
    assert ready["next_action"]["id"] == "authorize_fixed_m2_study"
    assert ready["next_action"]["training_required"] is False
    assert ready["next_action"]["formal_matches_required"] is True
    assert ready["next_action"]["requires_explicit_authorization"] is True


def test_m2_control_preserves_rejection_and_identity_drift():
    rejected = _control(
        preflight_available=True,
        preflight_current=True,
        preflight_ready=True,
        preflight_status="ready_for_m2_training",
        candidate_evidence_available=True,
        candidate_evidence_current=True,
        candidate_eligible=False,
        checkpoint_path="data/world_model/rejected.pt",
        execution_state="candidate_rejected",
    )
    assert rejected["status"] == "candidate_rejected"
    assert rejected["next_action"]["id"] == (
        "preregister_new_candidate_after_rejection"
    )
    assert "never tune on the opened sealed result" in (
        rejected["next_action"]["label"]
    )

    stale = _control(
        preflight_available=True,
        preflight_current=True,
        preflight_ready=True,
        preflight_status="ready_for_m2_training",
        candidate_evidence_available=True,
        candidate_evidence_current=False,
        execution_state="stale_current_code_or_input_identity",
    )
    assert stale["status"] == "candidate_identity_drift"
    assert stale["next_action"]["id"] == "requalify_candidate_identity"


def test_m2_control_requires_resume_and_never_restarts_partial_budget():
    control = _control(
        candidate_evidence_available=True,
        candidate_evidence_current=True,
        candidate_eligible=True,
        checkpoint_path="data/world_model/m2.pt",
        execution_state="running",
        runs_executed=81,
    )

    assert control["status"] == "formal_execution_interrupted"
    assert control["progress_fraction"] == 0.225
    assert control["next_action"]["id"] == "resume_identity_matched_m2_study"
    assert "Resume the same immutable" in control["next_action"]["label"]


def test_m2_control_only_surfaces_promotion_after_complete_current_result():
    control = _control(
        candidate_evidence_available=True,
        candidate_evidence_current=True,
        candidate_eligible=True,
        checkpoint_path="data/world_model/m2.pt",
        execution_state="completed",
        runs_executed=360,
        result_current=True,
        result_status="promotion_supported",
        promotion_supported=True,
        promotion_gates={"all_preregistered_gates": True},
    )

    assert control["status"] == "completed_promotion_supported"
    assert control["next_action"]["id"] == "request_independent_reproduction"
    assert control["next_action"]["commands"] == []
    assert control["diagnosis_artifact"] is None
    assert control["promotion_gate_results"] == {
        "all_preregistered_gates": True,
    }

def test_m2_control_points_no_promotion_at_frozen_diagnosis():
    control = _control(
        candidate_evidence_available=True,
        candidate_evidence_current=True,
        candidate_eligible=True,
        checkpoint_path="data/world_model/m2.pt",
        execution_state="completed",
        runs_executed=360,
        result_current=True,
        result_status="research_only_default_off",
        promotion_supported=False,
        promotion_gates={
            "controlled_micro_xg_effect_is_meaningful": False,
        },
    )

    assert control["status"] == "completed_no_promotion"
    assert control["next_action"]["id"] == "retain_research_only_and_diagnose"
    assert control["next_action"]["training_required"] is False
    assert control["next_action"]["formal_matches_required"] is False
    assert control["next_action"]["commands"] == [
        "data/evaluation/m2_mirrored_policy_diagnosis_v1.json",
        "docs/M2_MIRRORED_POLICY_DIAGNOSIS_V1.md",
    ]
    assert control["diagnosis_artifact"] == (
        "data/evaluation/m2_mirrored_policy_diagnosis_v1.json"
    )
    assert control["zero_execution_projection"] is True


def test_m2_control_requires_resume_rejects_partial_budget_with_result():
    with pytest.raises(ValueError, match="complete fixed run budget"):
        _control(
            candidate_evidence_available=True,
            candidate_evidence_current=True,
            candidate_eligible=True,
            runs_executed=359,
            result_current=True,
            result_status="promotion_supported",
            promotion_supported=True,
            promotion_gates={"all_preregistered_gates": True},
        )

"""Fail-closed product workflow for the preregistered M2 research pipeline."""

from __future__ import annotations

from typing import Any, Mapping


_STAGE_IDS = (
    "freeze_protocol",
    "verify_training_readiness",
    "qualify_sealed_candidate",
    "execute_mirrored_study",
    "decide_promotion",
)

_PREFLIGHT_COMMAND = (
    "python scripts/preflight_m2_training.py --out "
    "data/evaluation/m2_training_preflight_v1.json"
)
_DIAGNOSIS_JSON = "data/evaluation/m2_mirrored_policy_diagnosis_v1.json"
_DIAGNOSIS_DOC = "docs/M2_MIRRORED_POLICY_DIAGNOSIS_V1.md"


def _stage(stage_id: str, status: str, evidence: str) -> dict[str, str]:
    return {"stage_id": stage_id, "status": status, "evidence": evidence}


def build_m2_research_control(
    *,
    protocol_available: bool,
    protocol_state: str | None,
    preflight_available: bool,
    preflight_current: bool,
    preflight_ready: bool,
    preflight_status: str | None,
    frozen_training_command: str | None,
    candidate_evidence_available: bool,
    candidate_evidence_current: bool,
    candidate_eligible: bool,
    checkpoint_path: str | None,
    execution_state: str | None,
    runs_executed: int,
    fixed_run_budget: int,
    result_current: bool,
    result_status: str | None,
    promotion_supported: bool,
    promotion_gates: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Project evidence into one honest, non-executing M2 operator workflow."""
    runs = int(runs_executed)
    budget = int(fixed_run_budget)
    if runs < 0 or budget < 0 or runs > budget:
        raise ValueError("M2 run progress must remain inside the fixed budget")
    if candidate_eligible and not candidate_evidence_current:
        raise ValueError("M2 candidate eligibility requires current identity evidence")
    if preflight_ready and not preflight_current:
        raise ValueError("M2 preflight readiness requires current identity evidence")
    if result_current and (budget <= 0 or runs != budget):
        raise ValueError("a current M2 result requires the complete fixed run budget")
    if promotion_supported and not result_current:
        raise ValueError("M2 promotion cannot precede a current complete result")

    protocol_frozen = bool(
        protocol_available
        and protocol_state == "preregistered_code_ready_awaiting_sealed_checkpoint"
    )
    readiness_satisfied = bool(
        preflight_current and preflight_ready
        or candidate_eligible
        or result_current
    )
    stages = [
        _stage(
            "freeze_protocol",
            "completed" if protocol_frozen else "blocked",
            "current preregistered protocol" if protocol_frozen
            else "valid frozen protocol unavailable",
        ),
        _stage(
            "verify_training_readiness",
            (
                "completed" if readiness_satisfied
                else "failed" if preflight_current
                else "stale" if preflight_available
                else "pending"
            ),
            (
                "superseded by stronger current candidate qualification"
                if not preflight_current and (candidate_eligible or result_current)
                else
                str(preflight_status or "identity-verified zero-training report")
                if preflight_current else
                "recorded preflight does not match current protocol, manifest and code"
                if preflight_available else "identity-bound preflight not recorded"
            ),
        ),
        _stage(
            "qualify_sealed_candidate",
            (
                "completed" if candidate_evidence_current and candidate_eligible
                else "failed" if candidate_evidence_current
                else "stale" if candidate_evidence_available
                else "blocked" if not preflight_ready
                else "pending"
            ),
            (
                "identity-bound candidate passed development and sealed gates"
                if candidate_evidence_current and candidate_eligible
                else "identity-bound candidate failed at least one frozen gate"
                if candidate_evidence_current
                else "candidate evidence does not match current identities"
                if candidate_evidence_available
                else "no candidate qualification has been recorded"
            ),
        ),
        _stage(
            "execute_mirrored_study",
            (
                "completed" if budget > 0 and runs == budget
                else "running" if runs > 0
                else "ready" if candidate_eligible
                else "blocked"
            ),
            f"{runs}/{budget} frozen full-match runs",
        ),
        _stage(
            "decide_promotion",
            (
                "completed_supported" if result_current and promotion_supported
                else "completed_not_supported" if result_current
                else "stale" if result_status
                else "blocked"
            ),
            (
                str(result_status)
                if result_current else
                "historical analysis is not applicable to current identities"
                if result_status else "complete identity-matched analysis unavailable"
            ),
        ),
    ]

    if not protocol_frozen:
        status = "blocked_protocol"
        next_action = {
            "id": "repair_or_restore_frozen_protocol",
            "label": "Restore and validate the preregistered M2 protocol.",
            "commands": ["python scripts/run_m2_mirrored_policy_study.py"],
            "requires_explicit_authorization": False,
            "training_required": False,
            "formal_matches_required": False,
        }
    elif not preflight_current and not candidate_evidence_current:
        status = "awaiting_current_preflight"
        next_action = {
            "id": "record_training_preflight",
            "label": (
                "Record a current identity-bound zero-training readiness report."
            ),
            "commands": [_PREFLIGHT_COMMAND],
            "requires_explicit_authorization": False,
            "training_required": False,
            "formal_matches_required": False,
        }
    elif not preflight_ready and not candidate_eligible:
        status = "blocked_before_training"
        next_action = {
            "id": "resolve_preflight_failures",
            "label": (
                "Resolve every failed readiness check before any optimization."
            ),
            "commands": [_PREFLIGHT_COMMAND],
            "requires_explicit_authorization": False,
            "training_required": False,
            "formal_matches_required": False,
        }
    elif candidate_evidence_available and not candidate_evidence_current:
        status = "candidate_identity_drift"
        next_action = {
            "id": "requalify_candidate_identity",
            "label": (
                "Re-run qualification against the current protocol, code and inputs."
            ),
            "commands": [
                "python scripts/run_m2_mirrored_policy_study.py "
                "--checkpoint {checkpoint} --eligibility-out "
                "data/evaluation/m2_candidate_eligibility_v1.json"
            ],
            "requires_explicit_authorization": False,
            "training_required": False,
            "formal_matches_required": False,
            "required_inputs": ["checkpoint"],
        }
    elif candidate_evidence_current and not candidate_eligible:
        status = "candidate_rejected"
        next_action = {
            "id": "preregister_new_candidate_after_rejection",
            "label": (
                "Keep this rejection immutable; preregister a new candidate "
                "before further tuning and never tune on the opened sealed result."
            ),
            "commands": [],
            "requires_explicit_authorization": False,
            "training_required": False,
            "formal_matches_required": False,
        }
    elif not candidate_eligible:
        status = "awaiting_candidate"
        commands = []
        if frozen_training_command:
            commands.append(str(frozen_training_command))
        commands.append(
            "python scripts/run_m2_mirrored_policy_study.py "
            "--checkpoint {checkpoint} --eligibility-out "
            "data/evaluation/m2_candidate_eligibility_v1.json"
        )
        next_action = {
            "id": "train_or_supply_and_qualify_candidate",
            "label": (
                "Train or supply one frozen-contract checkpoint, then record "
                "its development and sealed qualification."
            ),
            "commands": commands,
            "requires_explicit_authorization": True,
            "training_required": True,
            "formal_matches_required": False,
            "required_inputs": ["checkpoint"],
        }
    elif result_current:
        status = (
            "completed_promotion_supported"
            if promotion_supported else "completed_no_promotion"
        )
        next_action = {
            "id": (
                "request_independent_reproduction"
                if promotion_supported else "retain_research_only_and_diagnose"
            ),
            "label": (
                "Request identity-matched independent reproduction before release."
                if promotion_supported else
                "Retain research-only status and diagnose without post-hoc promotion."
            ),
            "commands": (
                [] if promotion_supported else [_DIAGNOSIS_JSON, _DIAGNOSIS_DOC]
            ),
            "requires_explicit_authorization": False,
            "training_required": False,
            "formal_matches_required": False,
        }
    elif runs > 0:
        status = (
            "awaiting_final_analysis" if runs == budget
            else "formal_execution_interrupted"
        )
        next_action = {
            "id": "resume_identity_matched_m2_study",
            "label": (
                "Resume the same immutable M2 execution; do not restart or "
                "change the checkpoint."
            ),
            "commands": [
                "python scripts/run_m2_mirrored_policy_study.py "
                "--checkpoint {checkpoint} --execute"
            ],
            "requires_explicit_authorization": True,
            "training_required": False,
            "formal_matches_required": True,
            "required_inputs": ["checkpoint"],
        }
    else:
        status = "ready_for_authorized_execution"
        next_action = {
            "id": "authorize_fixed_m2_study",
            "label": "Authorize and execute the fixed 360-match mirrored study.",
            "commands": [
                "python scripts/run_m2_mirrored_policy_study.py "
                "--checkpoint {checkpoint} --execute"
            ],
            "requires_explicit_authorization": True,
            "training_required": False,
            "formal_matches_required": True,
            "required_inputs": ["checkpoint"],
        }

    payload = {
        "schema_version": 1,
        "status": status,
        "zero_execution_projection": True,
        "stages": stages,
        "completed_stage_count": sum(
            str(row["status"]).startswith("completed") for row in stages
        ),
        "stage_count": len(stages),
        "runs_executed": runs,
        "fixed_run_budget": budget,
        "progress_fraction": round(runs / budget, 6) if budget else 0.0,
        "checkpoint_path": checkpoint_path if candidate_evidence_current else None,
        "next_action": next_action,
        "diagnosis_artifact": (
            _DIAGNOSIS_JSON
            if status == "completed_no_promotion" else None
        ),
        "promotion_gate_results": (
            dict(promotion_gates or {}) if result_current else {}
        ),
        "claim_boundary": (
            "Workflow readiness is not model quality, policy effect, match-outcome "
            "causality or real-football causality. Promotion requires a current "
            "complete result and all preregistered gates."
        ),
    }
    validate_m2_research_control(payload)
    return payload


def validate_m2_research_control(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported M2 research control schema")
    stages = payload.get("stages")
    if not isinstance(stages, list) or tuple(
        row.get("stage_id") for row in stages if isinstance(row, Mapping)
    ) != _STAGE_IDS:
        raise ValueError("M2 research control stages are incomplete or reordered")
    if payload.get("zero_execution_projection") is not True:
        raise ValueError("M2 research control must remain non-executing")
    action = payload.get("next_action")
    if not isinstance(action, Mapping) or not action.get("id"):
        raise ValueError("M2 research control requires exactly one next action")
    runs = payload.get("runs_executed")
    budget = payload.get("fixed_run_budget")
    if (
        isinstance(runs, bool) or not isinstance(runs, int)
        or isinstance(budget, bool) or not isinstance(budget, int)
        or runs < 0 or budget < 0 or runs > budget
    ):
        raise ValueError("invalid M2 research control progress")
    if payload.get("status") == "completed_promotion_supported" and not (
        payload.get("promotion_gate_results")
    ):
        raise ValueError("M2 promotion requires visible current gate results")


__all__ = ["build_m2_research_control", "validate_m2_research_control"]

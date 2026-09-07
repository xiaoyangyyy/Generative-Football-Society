"""Read-only evidence and readiness projections for a product workspace."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from src.infrastructure import (
    code_identity_manifest,
    file_sha256,
    portable_text_hash_matches,
    verify_artifact_manifest,
)
from src.match_engine.world_model.mirrored_policy_evaluation import (
    study_execution_identity,
    study_preflight_identity,
    validate_m2_candidate_receipt,
    validate_m2_preflight_receipt,
)
from src.product.m2_research_control import build_m2_research_control
from src.simulation.runtime import environment_snapshot


ArtifactResolver = Callable[[Path, Any], Path | None]


def _formal_evidence_identity(
    root: Path,
    protocol_relative: str,
    protocol: Mapping[str, Any],
    progress: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed when a sealed result does not describe current code."""
    if not protocol or not decision:
        return {"verified": False, "reason": "formal_result_unavailable"}
    candidate = protocol.get("candidate") or {}
    integrity = protocol.get("integrity") or {}
    code_files = integrity.get("code_identity_files") or []
    checkpoint_relative = candidate.get("checkpoint")
    checkpoint_expected = candidate.get("checkpoint_sha256")
    if (
        not isinstance(code_files, list) or not code_files
        or not checkpoint_relative or not checkpoint_expected
    ):
        return {"verified": False, "reason": "identity_contract_missing"}
    try:
        resolved_root = root.resolve()
        protocol_path = (root / protocol_relative).resolve()
        checkpoint_path = (root / str(checkpoint_relative)).resolve()
        for path in (protocol_path, checkpoint_path):
            if not path.is_file() or resolved_root not in path.parents:
                raise ValueError("identity artifact unavailable or outside root")
        checkpoint_sha = file_sha256(checkpoint_path)
        if checkpoint_sha != str(checkpoint_expected):
            return {"verified": False, "reason": "checkpoint_identity_mismatch"}
        code_sha = code_identity_manifest(
            resolved_root,
            code_files,
            mode=str(integrity.get("code_identity_mode") or "explicit_files_v1"),
        )
        expected = {
            "protocol_sha256": file_sha256(protocol_path),
            "checkpoint_sha256": checkpoint_sha,
            "code_sha256": code_sha,
        }
    except (OSError, TypeError, ValueError):
        return {"verified": False, "reason": "identity_replay_unavailable"}
    if progress.get("execution_identity") != expected:
        return {"verified": False, "reason": "progress_code_identity_stale"}
    if decision.get("execution_identity") != expected:
        return {"verified": False, "reason": "decision_code_identity_stale"}
    return {"verified": True, "reason": "current_code_identity_verified"}


def build_workspace_evidence(
    *, root: Path, artifact_path: ArtifactResolver,
) -> dict[str, Any]:
    def read(relative: str) -> dict:
        path = root / relative
        return (
            json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        )

    release = read("data/releases/current.json")
    staged = read("data/evaluation/staged_completion_v1.json")
    phase5 = read("data/evaluation/phase5_research_layer_validation_v1.json")
    action_protocol = read("data/evaluation/action_adoption_protocol_v1.json")
    cross_protocol = read(
        "data/evaluation/cross_action_validation_protocol_v1.json"
    )
    cross_verification = read(
        "data/evaluation/cross_action_validation_verification_v1.json"
    )
    cross_identity_verified = False
    if cross_protocol and cross_verification:
        try:
            cross_protocol_path = (
                root
                / "data/evaluation/cross_action_validation_protocol_v1.json"
            ).resolve()
            cross_candidate = cross_protocol.get("candidate") or {}
            cross_checkpoint = (
                root / str(cross_candidate.get("checkpoint") or "")
            ).resolve()
            cross_code_sha = {}
            for relative in cross_protocol.get("required_code_paths") or []:
                path = (root / str(relative)).resolve()
                path.relative_to(root.resolve())
                cross_code_sha[str(relative)] = file_sha256(path)
            expected_cross_identity = {
                "protocol_sha256": file_sha256(cross_protocol_path),
                "checkpoint_sha256": file_sha256(cross_checkpoint),
                "code_sha256": cross_code_sha,
            }
            cross_identity_verified = (
                cross_verification.get("execution_identity")
                == expected_cross_identity
            )
        except (OSError, TypeError, ValueError):
            cross_identity_verified = False
    manager_advisor_protocol = read(
        "data/evaluation/manager_advisor_protocol_v1.json"
    )
    action_progress = read("data/evaluation/action_adoption_v1/progress.json")
    action_decision = read("data/evaluation/action_adoption_v1/decision.json")
    outcome_protocol = read("data/evaluation/action_outcome_protocol_v1.json")
    outcome_progress = read("data/evaluation/action_outcome_v1/progress.json")
    outcome_decision = read("data/evaluation/action_outcome_v1/decision.json")
    m2_protocol_relative = (
        "data/evaluation/m2_mirrored_policy_protocol_v1.json"
    )
    m2_protocol = read(m2_protocol_relative)
    m2_preflight = read(
        "data/evaluation/m2_training_preflight_v1.json"
    )
    m2_candidate_report = read(
        "data/evaluation/m2_candidate_eligibility_v1.json"
    )
    m2_progress = read(
        "data/evaluation/m2_mirrored_policy_progress_v1.json"
    )
    m2_preflight_identity_verified = False
    m2_preflight_identity_reason = (
        "not_recorded" if not m2_preflight else "missing_evidence_identity"
    )
    if m2_protocol and m2_preflight.get("evidence_identity"):
        try:
            validate_m2_preflight_receipt(m2_preflight, m2_protocol)
            manifest_relative = str(
                (m2_protocol.get("candidate") or {})["dataset_manifest"]
            )
            expected_preflight_identity = study_preflight_identity(
                root,
                root / m2_protocol_relative,
                m2_protocol,
                root / manifest_relative,
            )
            m2_preflight_identity_verified = bool(
                m2_preflight.get("schema_version") == 1
                and m2_preflight.get("protocol_id")
                == m2_protocol.get("protocol_id")
                and m2_preflight.get("evidence_identity")
                == expected_preflight_identity
            )
            m2_preflight_identity_reason = (
                "current_protocol_manifest_and_training_code_verified"
                if m2_preflight_identity_verified else
                "protocol_manifest_or_training_code_identity_stale"
            )
        except (KeyError, OSError, TypeError, ValueError):
            m2_preflight_identity_reason = (
                "invalid_or_unavailable_preflight_identity"
            )
    m2_candidate_source = (
        m2_progress if m2_progress.get("execution_identity")
        else m2_candidate_report
    )
    m2_stored_identity = (
        m2_candidate_source.get("execution_identity") or {}
    )
    m2_candidate_contract_valid = bool(
        m2_progress.get("execution_identity")
    )
    if m2_candidate_report and not m2_progress.get("execution_identity"):
        try:
            validate_m2_candidate_receipt(
                m2_candidate_report, m2_protocol,
            )
            m2_candidate_contract_valid = True
        except (KeyError, TypeError, ValueError):
            m2_candidate_contract_valid = False
    m2_identity_verified = False
    m2_identity_reason = (
        "not_started" if m2_protocol and not m2_candidate_source
        else "missing_execution_identity"
    )
    if m2_protocol and m2_stored_identity:
        try:
            checkpoint_relative = str(
                m2_stored_identity["checkpoint_path"]
            )
            recomputed_m2_identity = study_execution_identity(
                root,
                root / m2_protocol_relative,
                m2_protocol,
                root / checkpoint_relative,
            )
            m2_identity_match = (
                recomputed_m2_identity == m2_stored_identity
            )
            m2_identity_verified = bool(
                m2_candidate_contract_valid and m2_identity_match
            )
            if m2_identity_verified:
                m2_identity_reason = (
                    "current_protocol_checkpoint_code_and_input_identity_"
                    "verified"
                )
            elif not m2_candidate_contract_valid:
                m2_identity_reason = (
                    "invalid_candidate_qualification_receipt"
                )
            else:
                m2_identity_reason = (
                    "protocol_checkpoint_code_or_input_identity_stale"
                )
        except (KeyError, OSError, TypeError, ValueError):
            m2_identity_reason = "invalid_or_unavailable_execution_identity"
    m2_analysis = (
        m2_progress.get("analysis")
        if isinstance(m2_progress.get("analysis"), Mapping)
        else {}
    )
    m2_result_current = bool(
        m2_identity_verified
        and m2_progress.get("state") == "completed"
        and m2_analysis.get("state") == "completed"
    )
    if m2_analysis and not m2_result_current:
        m2_execution_state = "stale_current_code_or_input_identity"
    elif m2_progress:
        m2_execution_state = m2_progress.get("state", "invalid_progress")
    elif m2_candidate_report:
        m2_execution_state = (
            "candidate_qualified_awaiting_execution"
            if m2_identity_verified
            and m2_candidate_report.get("candidate_eligible") is True
            else "candidate_rejected"
            if m2_identity_verified else
            m2_identity_reason
        )
    else:
        m2_execution_state = m2_protocol.get("state", "absent")
    m2_runs_executed = sum(
        len((row or {}).get("rows") or [])
        for row in (m2_progress.get("arms") or {}).values()
    )
    m2_fixed_run_budget = int(
        (m2_protocol.get("design") or {}).get("runs_total") or 0
    )
    m2_candidate_eligibility = (
        m2_candidate_source.get("candidate_eligibility") or {}
    )
    m2_candidate_eligible = bool(
        m2_identity_verified
        and m2_candidate_eligibility.get("eligible", False)
    )
    m2_preflight_current = bool(
        m2_preflight_identity_verified
        and m2_preflight.get("status")
        in {"ready_for_m2_training", "blocked_before_training"}
    )
    m2_preflight_ready = bool(
        m2_preflight_current
        and m2_preflight.get("ready") is True
        and m2_preflight.get("status") == "ready_for_m2_training"
    )
    m2_research_control = build_m2_research_control(
        protocol_available=bool(m2_protocol),
        protocol_state=m2_protocol.get("state"),
        preflight_available=bool(m2_preflight),
        preflight_current=m2_preflight_current,
        preflight_ready=m2_preflight_ready,
        preflight_status=m2_preflight.get("status"),
        frozen_training_command=m2_preflight.get(
            "frozen_training_command"
        ),
        candidate_evidence_available=bool(m2_candidate_source),
        candidate_evidence_current=m2_identity_verified,
        candidate_eligible=m2_candidate_eligible,
        checkpoint_path=m2_stored_identity.get("checkpoint_path"),
        execution_state=m2_execution_state,
        runs_executed=m2_runs_executed,
        fixed_run_budget=m2_fixed_run_budget,
        result_current=m2_result_current,
        result_status=m2_analysis.get("decision"),
        promotion_supported=bool(
            m2_result_current
            and m2_analysis.get("promotion_supported", False)
        ),
        promotion_gates=m2_analysis.get("promotion_gates"),
    )
    mechanism_identity = _formal_evidence_identity(
        root,
        "data/evaluation/action_adoption_protocol_v1.json",
        action_protocol, action_progress, action_decision,
    )
    outcome_identity = _formal_evidence_identity(
        root,
        "data/evaluation/action_outcome_protocol_v1.json",
        outcome_protocol, outcome_progress, outcome_decision,
    )
    mechanism_current = bool(mechanism_identity["verified"])
    outcome_current = bool(outcome_identity["verified"])
    if mechanism_current:
        mechanism_execution_state = action_decision.get("status")
    elif action_decision:
        mechanism_execution_state = "stale_current_code_identity"
    else:
        mechanism_execution_state = action_progress.get(
            "state", "ready_not_started" if action_protocol else "absent",
        )
    if outcome_current:
        outcome_execution_state = outcome_decision.get("decision")
    elif outcome_decision:
        outcome_execution_state = "stale_current_code_identity"
    else:
        outcome_execution_state = outcome_progress.get(
            "state", "not_started" if outcome_protocol else "absent",
        )
    candidate = phase5.get("world_model_candidate") or {}
    live_llm = phase5.get("live_llm_evidence") or {}
    frozen_shot_decision = read(
        "data/evaluation/frozen_shot_head_decision.json"
    )
    sealed_shot = (candidate.get("sealed_test") or {}).get("shot") or {}
    frozen_shot_artifact = artifact_path(
        root, frozen_shot_decision.get("promotion_artifact")
    )
    frozen_shot_identity_verified = bool(
        frozen_shot_decision.get("accepted")
        and frozen_shot_artifact is not None
        and frozen_shot_artifact.is_file()
        and frozen_shot_decision.get("promotion_artifact_sha256")
        and file_sha256(frozen_shot_artifact)
        == frozen_shot_decision.get("promotion_artifact_sha256")
    )
    return {
        "stable_release": release.get("active"),
        "stable_release_manifest": release.get("manifest"),
        "stable_release_manifest_sha256": release.get("manifest_sha256"),
        "rollback_release": (release.get("rollback") or {}).get("release"),
        "all_research_stages_complete": staged.get("all_stages_complete", False),
        "world_model_candidate_accepted": candidate.get("accepted", False),
        "world_model_checkpoint": candidate.get("checkpoint"),
        "world_model_checkpoint_sha256": candidate.get("checkpoint_sha256"),
        "shot_planner_active": candidate.get("shot_planner_active", False),
        "shot_fallback": candidate.get("shot_fallback"),
        "frozen_shot_head": frozen_shot_decision,
        "shot_action_validation": {
            "available": bool(sealed_shot or frozen_shot_decision),
            "status": (
                "sealed_frozen_head_authorized"
                if frozen_shot_identity_verified
                else "physics_xg_fallback_joint_head_not_authorized"
            ),
            "probability_source": (
                "frozen_backbone_shot_head"
                if frozen_shot_identity_verified else "physics_xg_prior"
            ),
            "joint_head_authorized": False,
            "frozen_head_authorized": frozen_shot_identity_verified,
            "sealed_samples": int(sealed_shot.get("samples", 0) or 0),
            "sealed_goals": int(sealed_shot.get("goals", 0) or 0),
            "model_brier": sealed_shot.get("model_brier"),
            "physics_xg_prior_brier": sealed_shot.get(
                "physics_xg_prior_brier"
            ),
            "skill_vs_physics_xg_prior": sealed_shot.get(
                "skill_vs_physics_xg_prior"
            ),
            "claim_scope": (
                "sealed_simulator_shot_probability_proper_score_only_"
                "no_match_outcome_or_real_football_claim"
            ),
        },
        "live_llm_evaluable": live_llm.get("evaluable", False),
        "action_adoption_mechanism": {
            "available": bool(action_protocol),
            "historical_result_available": bool(action_decision),
            "result_identity_verified": mechanism_current,
            "result_applicable_to_current_code": mechanism_current,
            "identity_reason": mechanism_identity["reason"],
            "protocol_id": action_protocol.get("protocol_id"),
            "claim_scope": action_protocol.get("claim_scope"),
            "protocol_state": action_protocol.get("state"),
            "execution_state": mechanism_execution_state,
            "runs_executed": sum(
                len((row or {}).get("rows") or [])
                for row in (action_progress.get("arms") or {}).values()
            ),
            "fixed_run_budget": (
                (action_protocol.get("design") or {}).get("runs_total")
            ),
            "historical_result_status": action_decision.get("status"),
            "result_status": (
                action_decision.get("status") if mechanism_current else None
            ),
            "passed": action_decision.get("passed") if mechanism_current else None,
            "mechanism": (
                dict(action_decision.get("mechanism") or {})
                if mechanism_current else {}
            ),
            "promotion_authorized": bool(
                mechanism_current
                and action_decision.get("promotion_authorized", False)
            ),
        },
        "cross_action_validation": {
            "available": bool(cross_protocol and cross_verification),
            "result_identity_verified": cross_identity_verified,
            "protocol_id": cross_protocol.get("protocol_id"),
            "protocol_state": cross_protocol.get("state"),
            "status": (
                cross_verification.get("status")
                if cross_identity_verified else "stale_current_code_identity"
            ),
            "code_ready": bool(
                cross_identity_verified
                and cross_verification.get("code_ready", False)
            ),
            "checkpoint_identity_verified": bool(
                cross_verification.get("checkpoint_identity_verified", False)
            ),
            "cross_planning_authorized": bool(
                cross_identity_verified
                and cross_verification.get("cross_planning_authorized", False)
            ),
            "runtime_cross_quality": float(
                cross_verification.get("runtime_cross_quality", 0.0) or 0.0
                if cross_identity_verified else 0.0
            ),
            "checkpoint_reason": (
                cross_verification.get("checkpoint_cross_validation") or {}
            ).get("reason"),
            "minimum_samples": (
                cross_protocol.get("validation") or {}
            ).get("minimum_samples"),
            "minimum_groups": (
                cross_protocol.get("validation") or {}
            ).get("minimum_groups"),
            "minimum_skill_vs_persistence": (
                cross_protocol.get("validation") or {}
            ).get("minimum_skill_vs_persistence"),
            "claim_scope": cross_protocol.get("claim_scope"),
            "training_executed": bool(
                cross_verification.get("training_executed", False)
            ),
            "matches_executed": int(
                cross_verification.get("matches_executed", 0) or 0
            ),
            "provider_calls_made": bool(
                cross_verification.get("provider_calls_made", False)
            ),
        },
        "action_outcome_study": {
            "available": bool(outcome_protocol),
            "historical_result_available": bool(outcome_decision),
            "result_identity_verified": outcome_current,
            "result_applicable_to_current_code": outcome_current,
            "identity_reason": outcome_identity["reason"],
            "protocol_id": outcome_protocol.get("protocol_id"),
            "claim_scope": (
                "full_match_simulator_outcome_not_real_football_causality"
            ),
            "execution_state": outcome_execution_state,
            "runs_executed": sum(
                len((row or {}).get("rows") or [])
                for row in (outcome_progress.get("arms") or {}).values()
            ),
            "fixed_run_budget": (
                (outcome_protocol.get("design") or {}).get("runs_total")
            ),
            "historical_result_status": outcome_decision.get("decision"),
            "promotion_supported": bool(
                outcome_current
                and outcome_decision.get("promotion_supported", False)
            ),
            "result_status": (
                outcome_decision.get("decision") if outcome_current else None
            ),
            "pairs_total": (
                outcome_decision.get("pairs_total") if outcome_current else None
            ),
            "primary": (
                dict(outcome_decision.get("primary") or {})
                if outcome_current else {}
            ),
            "minimum_meaningful_delta_loss": (
                outcome_decision.get("minimum_meaningful_delta_loss")
                if outcome_current else None
            ),
            "behavior": (
                dict(outcome_decision.get("behavior") or {})
                if outcome_current else {}
            ),
            "promotion_gates": (
                dict(outcome_decision.get("promotion_gates") or {})
                if outcome_current else {}
            ),
        },
        "outcome_aligned_m2_study": {
            "available": bool(m2_protocol),
            "protocol_id": m2_protocol.get("protocol_id"),
            "claim_scope": m2_protocol.get("claim_scope"),
            "protocol_state": m2_protocol.get("state"),
            "training_contract": dict(m2_protocol.get("training") or {}),
            "required_sealed_validation": list(
                (m2_protocol.get("candidate") or {}).get(
                    "required_sealed_validation"
                ) or []
            ),
            "execution_state": m2_execution_state,
            "checkpoint_bound": bool(m2_stored_identity.get("checkpoint_path")),
            "candidate_eligible": m2_candidate_eligible,
            "training_preflight": {
                "available": bool(m2_preflight),
                "result_identity_verified": (
                    m2_preflight_identity_verified
                ),
                "result_applicable_to_current_code": m2_preflight_current,
                "identity_reason": m2_preflight_identity_reason,
                "status": (
                    m2_preflight.get("status")
                    if m2_preflight_current else None
                ),
                "ready": m2_preflight_ready,
                "checks": (
                    dict(m2_preflight.get("checks") or {})
                    if m2_preflight_current else {}
                ),
                "training_executed": False,
                "claim_scope": (
                    "zero_training_readiness_only_no_model_or_outcome_evidence"
                ),
            },
            "candidate_qualification": {
                "available": bool(m2_candidate_source),
                "source": (
                    "formal_progress"
                    if m2_progress.get("execution_identity")
                    else "qualification_report"
                    if m2_candidate_report else None
                ),
                "result_identity_verified": m2_identity_verified,
                "eligible": m2_candidate_eligible,
                "checkpoint_path": (
                    m2_stored_identity.get("checkpoint_path")
                    if m2_identity_verified else None
                ),
                "details": (
                    dict(m2_candidate_eligibility)
                    if m2_identity_verified else {}
                ),
                "claim_scope": (
                    "checkpoint_qualification_only_no_policy_effect_or_"
                    "outcome_claim"
                ),
            },
            "historical_result_available": bool(m2_analysis),
            "result_identity_verified": m2_identity_verified,
            "result_applicable_to_current_code": m2_result_current,
            "identity_reason": m2_identity_reason,
            "runs_executed": m2_runs_executed,
            "fixed_run_budget": m2_fixed_run_budget,
            "result_status": (
                m2_analysis.get("decision") if m2_result_current else None
            ),
            "promotion_supported": bool(
                m2_result_current
                and m2_analysis.get("promotion_supported", False)
            ),
            "primary": (
                dict(m2_analysis.get("primary_controlled_micro_xg_effect") or {})
                if m2_result_current else {}
            ),
            "mechanism": (
                dict(m2_analysis.get("mechanism_realized_policy_utility") or {})
                if m2_result_current else {}
            ),
            "mechanism_coverage": (
                dict(m2_analysis.get("mechanism_attributable_coverage") or {})
                if m2_result_current else {}
            ),
            "expected_counterfactual_changes": (
                dict(
                    m2_analysis.get(
                        "mechanism_expected_counterfactual_changes"
                    ) or {}
                ) if m2_result_current else {}
            ),
            "external_validity": (
                dict(
                    m2_analysis.get(
                        "external_continuous_calibration_noninferiority"
                    ) or {}
                ) if m2_result_current else {}
            ),
            "secondary_goal_difference": (
                dict(m2_analysis.get("secondary_goal_difference_effect") or {})
                if m2_result_current else {}
            ),
            "promotion_gates": (
                dict(m2_analysis.get("promotion_gates") or {})
                if m2_result_current else {}
            ),
            "runtime_row_identity": (
                dict(m2_analysis.get("runtime_row_identity") or {})
                if m2_result_current else {}
            ),
            "experimental_unit_identity": (
                dict(m2_analysis.get("experimental_unit_identity") or {})
                if m2_result_current else {}
            ),
            "research_control": m2_research_control,
        },
        "manager_advisor_adoption": {
            "available": bool(manager_advisor_protocol),
            "protocol_id": manager_advisor_protocol.get("protocol_id"),
            "claim_scope": manager_advisor_protocol.get("claim_scope"),
            "protocol_state": manager_advisor_protocol.get("state"),
            "fixed_information_windows": list(
                (manager_advisor_protocol.get("analysis") or {}).get(
                    "fixed_information_windows"
                )
                or []
            ),
            "results_available": (
                manager_advisor_protocol.get("execution") or {}
            ).get("results_available", False),
            "causal_effect_authorized": (
                manager_advisor_protocol.get("analysis") or {}
            ).get("causal_effect_authorized", False),
            "promotion_authorized": (
                manager_advisor_protocol.get("decision_rules") or {}
            ).get("product_or_academic_promotion_authorized", False),
        },
        "production_promotion_ready": staged.get(
            "production_promotion_ready", False
        ),
    }
def build_workspace_readiness(
    *, root: Path, mode: str, evidence: Mapping[str, Any],
    artifact_path: ArtifactResolver,
) -> dict[str, Any]:
    checkpoint = artifact_path(
        root,
        (
            evidence.get("world_model_checkpoint")
            or "data/world_model/latent_wm_rollout_calibrated_candidate.pt"
        ),
    )
    release_manifest = artifact_path(
        root,
        evidence.get("stable_release_manifest"),
    )
    expected_release_sha256 = str(
        evidence.get("stable_release_manifest_sha256") or ""
    )
    release_identity_verified = bool(
        release_manifest is not None
        and release_manifest.is_file()
        and expected_release_sha256
        and portable_text_hash_matches(
            release_manifest,
            expected_release_sha256,
        )
    )
    release_artifact_verification: dict[str, Any] = {
        "ok": False,
        "artifacts": 0,
        "failures": [{"reason": "release_manifest_identity_unverified"}],
    }
    if release_identity_verified and release_manifest is not None:
        try:
            release_payload = json.loads(
                release_manifest.read_text(encoding="utf-8-sig")
            )
            release_artifact_verification = verify_artifact_manifest(
                root,
                release_payload,
            )
        except (OSError, ValueError, TypeError) as exc:
            release_artifact_verification = {
                "ok": False,
                "artifacts": 0,
                "failures": [
                    {
                        "reason": "invalid_release_manifest",
                        "error_type": type(exc).__name__,
                    }
                ],
            }
    expected_checkpoint_sha256 = str(
        evidence.get("world_model_checkpoint_sha256") or ""
    )
    actual_checkpoint_sha256 = (
        file_sha256(checkpoint)
        if checkpoint is not None and checkpoint.is_file()
        else None
    )
    shot_decision = evidence.get("frozen_shot_head") or {}
    shot_artifact_value = shot_decision.get("promotion_artifact")
    shot_artifact = artifact_path(root, shot_artifact_value)
    shot_identity_verified = not shot_decision.get("accepted") or bool(
        shot_artifact is not None
        and shot_artifact.is_file()
        and shot_decision.get("promotion_artifact_sha256")
        and file_sha256(shot_artifact)
        == shot_decision.get("promotion_artifact_sha256")
    )
    from src.simulation.llm_gateway import (
        LLMGatewayConfig,
        llm_credentials_available,
    )

    llm_values = environment_snapshot()
    llm_config_error = None
    try:
        llm_config = LLMGatewayConfig.from_env()
        llm_provider = llm_config.public_summary()
    except (TypeError, ValueError) as exc:
        llm_config_error = str(exc)
        llm_provider = None
    checks = {
        "stable_release_available": bool(evidence.get("stable_release")),
        "stable_release_identity_verified": release_identity_verified,
        "stable_release_artifacts_verified": bool(
            release_artifact_verification.get("ok")
        ),
        "research_checkpoint_available": bool(
            checkpoint is not None and checkpoint.is_file()
        ),
        "research_checkpoint_accepted": bool(
            evidence.get("world_model_candidate_accepted")
        ),
        "research_checkpoint_identity_verified": bool(
            expected_checkpoint_sha256
            and actual_checkpoint_sha256 == expected_checkpoint_sha256
        ),
        "shot_head_identity_verified": shot_identity_verified,
        "llm_credentials_available": llm_credentials_available(llm_values),
        "llm_config_valid": llm_config_error is None,
    }
    if mode == "stable":
        ready = all(
            checks[name]
            for name in (
                "stable_release_available",
                "stable_release_identity_verified",
                "stable_release_artifacts_verified",
            )
        )
        blockers: list[str] = (
            []
            if ready
            else [
                name
                for name in (
                    "stable_release_available",
                    "stable_release_identity_verified",
                    "stable_release_artifacts_verified",
                )
                if not checks[name]
            ]
        )
    elif mode == "research":
        ready = all(
            checks[name]
            for name in (
                "research_checkpoint_available",
                "research_checkpoint_accepted",
                "research_checkpoint_identity_verified",
                "shot_head_identity_verified",
            )
        )
        blockers = (
            []
            if ready
            else [
                name
                for name in (
                    "research_checkpoint_available",
                    "research_checkpoint_accepted",
                    "research_checkpoint_identity_verified",
                    "shot_head_identity_verified",
                )
                if not checks[name]
            ]
        )
    else:
        ready = all(
            checks[name]
            for name in (
                "research_checkpoint_available",
                "research_checkpoint_accepted",
                "research_checkpoint_identity_verified",
                "shot_head_identity_verified",
                "llm_credentials_available",
                "llm_config_valid",
            )
        )
        blockers = [
            name
            for name in (
                "research_checkpoint_available",
                "research_checkpoint_accepted",
                "research_checkpoint_identity_verified",
                "shot_head_identity_verified",
                "llm_credentials_available",
                "llm_config_valid",
            )
            if not checks[name]
        ]
    return {
        "mode": mode,
        "ready": ready,
        "checks": checks,
        "blockers": blockers,
        "release_artifact_verification": release_artifact_verification,
        "llm_provider": llm_provider,
        "llm_config_error": llm_config_error,
    }

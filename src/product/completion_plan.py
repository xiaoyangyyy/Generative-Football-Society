"""Dependency-aware, zero-execution plan for reaching evidence maturity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CompletionStep:
    gate_id: str
    stage: str
    priority: int
    execution_class: str
    operator: str
    dependencies: tuple[str, ...]
    commands: tuple[str, ...]
    required_inputs: tuple[str, ...] = ()
    requires_explicit_authorization: bool = False


STEPS = (
    CompletionStep(
        "credential_security_closure", "security", 10, "external_manual",
        "credential_owner", (),
        ("python scripts/verify_security_closure.py --attestation "
         "data/evaluation/security_closure_v1/attestation.json --out "
         "data/evaluation/security_closure_verification_v1.json",),
        required_inputs=(
            "provider-console revocation receipt with all credential values redacted",
            "data/evaluation/security_closure_v1/attestation.json",
        ),
    ),
    CompletionStep(
        "container_build", "deployment", 20, "local_tooling",
        "release_engineer", (),
        ("python scripts/verify_container_runtime.py --execute --out "
         "data/evaluation/container_runtime_verification_v1.json",),
        required_inputs=("Docker-compatible engine on the release host",),
        requires_explicit_authorization=True,
    ),
    CompletionStep(
        "target_user_validation", "product_validation", 30, "human_study",
        "study_lead", (),
        ("python scripts/product_validation_study.py --analyze --records "
         "data/evaluation/product_validation_v1/participant_records.jsonl "
         "--external-review data/evaluation/product_validation_v1/"
         "external_review.json --out "
         "data/evaluation/product_validation_v1/decision.json",),
        required_inputs=(
            "data/evaluation/product_validation_v1/participant_records.jsonl",
            "at least ten valid target-user participants with frozen role quotas",
        ),
    ),
    CompletionStep(
        "external_accessibility_review", "product_validation", 40,
        "independent_review", "accessibility_reviewer", (),
        ("python scripts/product_validation_study.py --analyze --records "
         "data/evaluation/product_validation_v1/participant_records.jsonl "
         "--external-review data/evaluation/product_validation_v1/"
         "external_review.json --out "
         "data/evaluation/product_validation_v1/decision.json",),
        required_inputs=(
            "data/evaluation/product_validation_v1/external_review.json",
            "independent WCAG 2.2 AA reviewer attestation",
        ),
    ),
    CompletionStep(
        "external_security_review", "security", 50, "independent_review",
        "security_reviewer", ("credential_security_closure",),
        ("python scripts/product_validation_study.py --analyze --records "
         "data/evaluation/product_validation_v1/participant_records.jsonl "
         "--external-review data/evaluation/product_validation_v1/"
         "external_review.json --out "
         "data/evaluation/product_validation_v1/decision.json",),
        required_inputs=(
            "data/evaluation/product_validation_v1/external_review.json",
            "independent application-security reviewer attestation",
        ),
    ),
    CompletionStep(
        "production_operations_validation", "production", 60,
        "authorized_compute", "release_engineer", ("container_build",),
        ("python scripts/run_production_validation.py --execute --workspace . "
         "--authorization I_AUTHORIZE_GFS_100_MATCH_PRODUCTION_VALIDATION "
         "--deployment-instance-id INSTANCE_ID",),
        required_inputs=(
            "verified deployment container",
            "stable deployment instance identifier and isolated validation workspace",
        ),
        requires_explicit_authorization=True,
    ),
    CompletionStep(
        "user_value_validation", "product_validation", 70, "human_study",
        "study_lead", ("target_user_validation",),
        ("python scripts/product_value_study.py --analyze --records "
         "data/evaluation/product_value_validation_v1/participant_records.jsonl "
         "--out data/evaluation/product_value_validation_v1/decision.json",),
        required_inputs=(
            "data/evaluation/product_value_validation_v1/participant_records.jsonl",
            "valid randomized comparative target-user records",
        ),
    ),
    CompletionStep(
        "confirmatory_results", "academic_execution", 80,
        "authorized_compute", "research_lead", (),
        ("python scripts/run_formal_experiment.py --execute",
         "python scripts/academic_replication_study.py --execute "
         "--authorization I_AUTHORIZE_GFS_ACADEMIC_REPLICATION_V1"),
        required_inputs=(
            "frozen confirmatory protocol and candidate checkpoint",
            "explicit authorization for both fixed execution budgets",
        ),
        requires_explicit_authorization=True,
    ),
    CompletionStep(
        "independent_reproduction", "academic_review", 90,
        "independent_review", "independent_reproducer",
        ("confirmatory_results",),
        ("python scripts/verify_independent_reproduction.py --review "
         "data/evaluation/independent_reproduction_v1/review.json "
         "--out data/evaluation/independent_reproduction_verification_v1.json",),
        required_inputs=(
            "data/evaluation/independent_reproduction_v1/review.json",
            "independently reproduced confirmatory and replication decisions",
            "independent execution log and reviewer signature",
        ),
    ),
    CompletionStep(
        "completed_manuscript", "paper_finalization", 100,
        "document_finalization", "paper_author",
        ("confirmatory_results", "independent_reproduction"),
        ("python scripts/finalize_paper.py --build-ledger --authorization "
         "I_AUTHORIZE_GFS_PAPER_RESULT_LEDGER_V1",
         "python scripts/finalize_paper.py --verify-final --out "
         "data/evaluation/paper_finalization_v1/verification.json"),
        required_inputs=(
            "complete identity-matched confirmatory and replication results",
            "verified independent reproduction",
            "docs/PAPER_FINAL.md with the exact generated result ledger embedded",
        ),
        requires_explicit_authorization=True,
    ),
)


def build_completion_plan(gates: list[dict[str, Any]]) -> dict[str, Any]:
    """Build an honest action graph without executing or accepting evidence."""
    gate_map = {str(gate.get("id")): gate for gate in gates}
    if len(gate_map) != len(gates):
        raise ValueError("release gates must have unique ids")
    missing = [step.gate_id for step in STEPS if step.gate_id not in gate_map]
    if missing:
        raise ValueError(f"completion plan references missing gates: {missing}")

    rows: list[dict[str, Any]] = []
    for step in sorted(STEPS, key=lambda item: item.priority):
        gate = gate_map[step.gate_id]
        blockers = [
            dependency
            for dependency in step.dependencies
            if gate_map.get(dependency, {}).get("passed") is not True
        ]
        if gate.get("passed") is True:
            state = "completed"
        elif blockers:
            state = "blocked_by_dependency"
        elif step.requires_explicit_authorization:
            state = "explicit_authorization_required"
        elif step.execution_class in {
            "external_manual", "human_study", "independent_review",
            "document_finalization",
        }:
            state = "external_evidence_required"
        else:
            state = "ready_for_operator"
        rows.append({
            "gate_id": step.gate_id,
            "label": gate.get("label"),
            "stage": step.stage,
            "priority": step.priority,
            "state": state,
            "execution_class": step.execution_class,
            "operator": step.operator,
            "dependencies": list(step.dependencies),
            "blocking_gate_ids": blockers,
            "evidence": gate.get("evidence"),
            "commands": list(step.commands),
            "required_inputs": list(step.required_inputs),
            "requires_explicit_authorization": step.requires_explicit_authorization,
            "passed": gate.get("passed") is True,
        })

    open_rows = [row for row in rows if not row["passed"]]
    actionable = [
        row for row in open_rows if row["state"] != "blocked_by_dependency"
    ]
    return {
        "schema_version": 1,
        "status": "complete" if not open_rows else "actions_remaining",
        "zero_execution_plan": True,
        "steps": rows,
        "open_step_count": len(open_rows),
        "blocked_step_count": sum(
            row["state"] == "blocked_by_dependency" for row in open_rows
        ),
        "actionable_step_count": len(actionable),
        "recommended_gate_id": actionable[0]["gate_id"] if actionable else None,
        "parallel_action_gate_ids": [row["gate_id"] for row in actionable],
        "authorization_required_gate_ids": [
            row["gate_id"] for row in open_rows
            if row["requires_explicit_authorization"]
        ],
        "evidence_kit": {
            "template_only": True,
            "audit_command": "python gfs.py studio evidence-kit",
            "materialize_command": (
                "python gfs.py studio evidence-kit --materialize "
                "build/evidence-kits/gfs-excellence-evidence-kit-v1.zip"
            ),
            "materialization_is_explicit": True,
            "changes_gate_scores": False,
        },
    }

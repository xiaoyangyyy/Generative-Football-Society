import json
from pathlib import Path

import pytest

from src.cli import build_parser
from src.product.completion_plan import STEPS, build_completion_plan
from src.product.control_plane import ProductControlPlane


ROOT = Path(__file__).resolve().parents[1]


def _gates(passed=()):
    passed = set(passed)
    return [
        {"id": step.gate_id, "label": step.gate_id, "passed": step.gate_id in passed,
         "evidence": f"evidence/{step.gate_id}.json"}
        for step in STEPS
    ]


def test_plan_prioritizes_credential_closure_and_exposes_parallel_work():
    plan = build_completion_plan(_gates())
    assert plan["zero_execution_plan"] is True
    assert plan["recommended_gate_id"] == "credential_security_closure"
    assert plan["open_step_count"] == 10
    assert plan["blocked_step_count"] == 5
    assert set(plan["parallel_action_gate_ids"]) == {
        "credential_security_closure", "container_build",
        "target_user_validation", "external_accessibility_review",
        "confirmatory_results",
    }
    assert set(plan["authorization_required_gate_ids"]) == {
        "container_build", "production_operations_validation",
        "confirmatory_results", "completed_manuscript",
    }
    assert all(row["required_inputs"] for row in plan["steps"])
    assert plan["evidence_kit"]["template_only"] is True
    assert plan["evidence_kit"]["changes_gate_scores"] is False


def test_dependency_completion_unblocks_downstream_without_marking_it_passed():
    plan = build_completion_plan(_gates({
        "credential_security_closure", "container_build",
        "target_user_validation", "confirmatory_results",
    }))
    rows = {row["gate_id"]: row for row in plan["steps"]}
    assert rows["external_security_review"]["state"] == "external_evidence_required"
    assert rows["production_operations_validation"]["state"] == (
        "explicit_authorization_required"
    )
    assert rows["user_value_validation"]["state"] == "external_evidence_required"
    assert rows["independent_reproduction"]["state"] == "external_evidence_required"
    assert rows["completed_manuscript"]["state"] == "blocked_by_dependency"
    assert not rows["completed_manuscript"]["passed"]
    assert rows["independent_reproduction"]["commands"][0].endswith(
        "data/evaluation/independent_reproduction_verification_v1.json"
    )
    assert rows["completed_manuscript"]["commands"][0].endswith(
        "I_AUTHORIZE_GFS_PAPER_RESULT_LEDGER_V1"
    )


def test_plan_rejects_duplicate_or_missing_gate_identity():
    gates = _gates()
    with pytest.raises(ValueError, match="unique"):
        build_completion_plan(gates + [dict(gates[0])])
    with pytest.raises(ValueError, match="missing gates"):
        build_completion_plan(gates[:-1])


def test_live_plan_is_integrated_and_never_executes_external_work(capsys):
    release = ProductControlPlane(ROOT).release_readiness()
    assert release["next_action"] == "credential_security_closure"
    assert release["completion_plan"]["open_step_count"] == 10
    assert release["completion_plan"]["evidence_kit"]["ready"] is True
    assert release["completion_plan"]["zero_execution_plan"] is True
    parser = build_parser()
    args = parser.parse_args(["--base-dir", str(ROOT), "studio", "excellence"])
    assert args.func(args) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["scores"] == {"product": 83, "academic": 70}
    assert output["external_calls_made"] is False
    assert output["completion_plan"]["recommended_gate_id"] == (
        "credential_security_closure"
    )

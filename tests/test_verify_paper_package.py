import copy
import json
from scripts.verify_paper_package import (
    CLAIMS,
    MANUSCRIPT,
    _claim_checks,
    verify_paper_package,
)


def test_preexecution_package_fails_closed_after_confirmatory_execution():
    report = verify_paper_package()
    assert report["passed"] is False
    assert report["status"] == "failed"
    assert report["confirmatory_result_available"] is False
    assert report["independent_reproduction_available"] is False
    assert report["external_calls_made"] is False
    assert report["matches_executed"] == 0
    assert report["training_executed"] is False
    assert report["formal_experiment_executed"] is False
    assert report["claim_count"] >= 8
    assert report["checks"]["confirmatory_outputs_are_absent"] is False
    failed = {key for key, value in report["checks"].items() if not value}
    assert failed == {
        "formal_identity_preflight_passes",
        "confirmatory_outputs_are_absent",
    }
    assert any("not a completed results manuscript" in row for row in report["limitations"])
    assert any("actual Docker build" in row for row in report["limitations"])


def test_claim_audit_rejects_duplicate_or_unmarked_claims():
    registry = json.loads(CLAIMS.read_text(encoding="utf-8"))
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    assert all(_claim_checks(registry, manuscript).values())

    duplicated = copy.deepcopy(registry)
    duplicated["claims"].append(copy.deepcopy(duplicated["claims"][0]))
    assert not _claim_checks(duplicated, manuscript)[
        "claim_ids_unique_and_well_formed"
    ]

    missing_marker = manuscript.replace("<!-- claim:CLM-005 -->", "")
    assert not _claim_checks(registry, missing_marker)[
        "every_claim_marked_exactly_once"
    ]


def test_claim_audit_rejects_prohibited_confirmatory_language():
    registry = json.loads(CLAIMS.read_text(encoding="utf-8"))
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    tampered = manuscript + "\nM1 significantly outperforms M0.\n"
    assert not _claim_checks(registry, tampered)[
        "prohibited_claim_phrases_absent"
    ]

import hashlib
import json
from pathlib import Path

from scripts.verify_independent_reproduction import (
    CONTAINER_CHECKS,
    REVIEWER_ATTESTATION,
    compare_decisions,
    protocol_report,
    verify_review,
)

ROOT_PROTOCOL = Path("data/evaluation/independent_reproduction_protocol_v1.json")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _decision(point_delta=0.0):
    return {
        "schema_version": 2,
        "protocol_id": "gfs-m0-vs-sealed-m1-confirmatory-v2",
        "decision": "no_meaningful_difference_keep_research_only",
        "promotion_supported": False,
        "pairs_total": 30,
        "primary": {
            "method": "fixture_stratified_paired_bootstrap",
            "draws": 10000,
            "seed": 260806,
            "point_delta": point_delta,
            "ci95_low": -0.01,
            "ci95_high": 0.01,
        },
        "minimum_meaningful_delta_loss": 0.1,
        "behavior": {"changed_pairs": 0, "changed_pair_fraction": 0.0},
        "promotion_gates": {
            "candidate_passes_all_external_validity_gates": True,
            "upper_interval_below_negative_minimum_effect": False,
            "minimum_behavior_change_met": False,
            "execution_identity_verified": True,
        },
        "secondary_metrics": {},
        "execution_identity": {"protocol_sha256": "a" * 64},
        "analyzed_at": "2026-08-10T00:00:00Z",
    }


def _replication_decision(point_delta=0.0):
    def comparison(seed):
        return {
            "method": "fixture_stratified_paired_bootstrap",
            "draws": 10000,
            "seed": seed,
            "point_delta": point_delta,
            "ci95_low": -0.01,
            "ci95_high": 0.01,
            "candidate_all_external_gates": True,
        }
    return {
        "schema_version": 1,
        "protocol_id": "gfs-result-contingent-mechanism-replication-v1",
        "status": "passed_academic_replication",
        "passed": True,
        "branch": "adoption_path_diagnosis",
        "confirmatory_decision": "no_meaningful_difference_keep_research_only",
        "conclusion": "replicated_equivalence_and_planning_nonadoption",
        "runs_executed": 72,
        "pairs_per_arm": 24,
        "comparisons": {
            "M1_minus_M0": comparison(361000),
            "M1_minus_M1_predict_only": comparison(361001),
        },
        "mechanism": {
            "changed_pairs": 0,
            "changed_pair_fraction": 0.0,
            "replicated_promotion": False,
            "replicated_equivalence": True,
            "replicated_harm": False,
            "planning_contribution": False,
            "planning_nonadoption": True,
            "bounded_inconclusive": False,
        },
        "external_validity": {
            "source": "sportec_idsse",
            "full_candidate_mean_passes_per_team_match": 400.0,
            "full_candidate_mean_shots_per_team_match": 12.0,
            "checks": {
                "passes_inside_idsse_observed_range": True,
                "shots_inside_idsse_observed_range": True,
            },
        },
        "material_deviations": [],
        "gates": {
            "fixed_complete_paired_budget": True,
            "execution_identity_verified": True,
            "no_material_deviations": True,
            "branch_conclusion_matches_confirmatory_decision": True,
            "full_candidate_passes_internal_external_gates": True,
            "licensed_idsse_external_scale_passes": True,
        },
        "execution_identity": {"protocol_sha256": "b" * 64},
        "training_executed": False,
        "provider_calls_made": False,
        "generated_at": "2026-08-11T00:00:00Z",
    }


def _review_fixture(tmp_path):
    protocol = json.loads(ROOT_PROTOCOL.read_text(encoding="utf-8"))
    protocol_path = tmp_path / ROOT_PROTOCOL
    _write_json(protocol_path, protocol)
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM scratch\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"checkpoint")
    code = tmp_path / "code.py"
    code.write_text("VALUE = 1\n", encoding="utf-8")
    data = tmp_path / "data.json"
    data.write_text("{}\n", encoding="utf-8")

    formal_path = tmp_path / protocol["prerequisites"]["confirmatory_protocol"]
    formal = {
        "schema_version": 2,
        "state": "preregistered_not_executed",
        "protocol_id": "gfs-m0-vs-sealed-m1-confirmatory-v2",
        "design": {"runs_total": 60},
        "candidate": {
            "checkpoint": "checkpoint.bin",
            "checkpoint_sha256": _sha(checkpoint),
        },
        "integrity": {"code_identity_files": ["code.py"]},
    }
    _write_json(formal_path, formal)
    replication_protocol_path = (
        tmp_path / protocol["prerequisites"]["academic_replication_protocol"]
    )
    replication_protocol = {
        "schema_version": 1,
        "protocol_id": "gfs-result-contingent-mechanism-replication-v1",
        "state": "preregistered_waiting_for_confirmatory_decision",
        "design": {"runs_total": 72},
        "integrity": {
            "code_identity_files": ["code.py"],
            "data_identity_files": ["data.json"],
        },
    }
    _write_json(replication_protocol_path, replication_protocol)

    reference_paths = {
        "confirmatory": tmp_path
        / protocol["prerequisites"]["confirmatory_decision"],
        "academic_replication": tmp_path
        / protocol["prerequisites"]["academic_replication_decision"],
    }
    reproduced_paths = {
        "confirmatory": tmp_path
        / protocol["outputs"]["reproduced_confirmatory_decision"],
        "academic_replication": tmp_path
        / protocol["outputs"]["reproduced_academic_replication_decision"],
    }
    _write_json(reference_paths["confirmatory"], _decision())
    _write_json(reproduced_paths["confirmatory"], _decision())
    _write_json(reference_paths["academic_replication"], _replication_decision())
    _write_json(reproduced_paths["academic_replication"], _replication_decision())

    container_path = tmp_path / protocol["prerequisites"]["container_runtime_evidence"]
    container = {
        "schema_version": 2,
        "status": "passed",
        "passed": True,
        "image_built": True,
        "deployment_started": True,
        "source_revision": "1" * 40,
        "image_id": "sha256:" + "2" * 64,
        "registry_access_attempted": True,
        "checks": {key: True for key in CONTAINER_CHECKS},
        "artifact_sha256": {"Dockerfile": _sha(dockerfile)},
    }
    _write_json(container_path, container)
    release_path = tmp_path / protocol["prerequisites"][
        "reproduction_release_evidence"
    ]
    _write_json(
        release_path,
        {
            "status": "passed_code_contract",
            "passed": True,
            "artifact_sha256": {"Dockerfile": _sha(dockerfile)},
        },
    )
    review_log = tmp_path / "review-log.txt"
    review_log.write_text("independent execution log\n", encoding="utf-8")
    review = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "reviewer": {
            "public_name": "Independent Reviewer",
            "orcid": "0000-0002-1825-0097",
            "affiliation": "Independent Review Lab",
            "independent_of_implementation": True,
            "contributed_to_frozen_commit": False,
            "conflict_disclosure": "No conflicts declared.",
        },
        "source": {
            "commit_sha": "1" * 40,
            "clean_checkout": True,
            "platform": "Linux x86_64",
            "python_version": "3.12.11",
        },
        "execution": {
            "runs_total": 132,
            "stages": {
                "confirmatory": {
                    "pairs_completed": 30,
                    "runs_total": 60,
                    "runs_by_arm": {"M0": 30, "M1": 30},
                },
                "academic_replication": {
                    "pairs_per_arm_completed": 24,
                    "runs_total": 72,
                    "runs_by_arm": {
                        "M0": 24,
                        "M1_predict_only": 24,
                        "M1": 24,
                    },
                },
            },
            "interim_analysis_performed": False,
            "optional_stopping_performed": False,
            "training_executed": False,
            "provider_calls_made": False,
        },
        "identities": {
            "formal_protocol_sha256": _sha(formal_path),
            "academic_replication_protocol_sha256": _sha(
                replication_protocol_path
            ),
            "checkpoint_sha256": _sha(checkpoint),
            "critical_code_sha256": {"code.py": _sha(code)},
            "replication_data_sha256": {"data.json": _sha(data)},
            "container_runtime_report_sha256": _sha(container_path),
            "reproduction_release_report_sha256": _sha(release_path),
        },
        "comparison": {
            "reference_decision_sha256": {
                stage: _sha(path) for stage, path in reference_paths.items()
            },
            "reproduced_decision_sha256": {
                stage: _sha(path) for stage, path in reproduced_paths.items()
            },
            "stage_conclusions": {
                "confirmatory": "reproduced",
                "academic_replication": "reproduced",
            },
            "reviewer_conclusion": "reproduced",
        },
        "deviations": [
            {"classification": "none", "description": "No deviations observed."}
        ],
        "artifact_sha256": {"review-log.txt": _sha(review_log)},
        "signed_at": "2026-08-12T10:00:00Z",
        "attestation": REVIEWER_ATTESTATION,
    }
    review_path = tmp_path / protocol["outputs"]["review"]
    _write_json(review_path, review)
    return {
        "protocol_path": protocol_path,
        "reference_paths": reference_paths,
        "reproduced_paths": reproduced_paths,
        "review": review,
        "review_path": review_path,
    }


def test_independent_reproduction_protocol_is_valid_but_not_ready():
    report = protocol_report()
    assert report["passed"] is True
    assert report["ready_to_start"] is False
    assert report["review_available"] is False
    assert report["prerequisite_checks"]["confirmatory_decision_is_complete"] is False
    assert report["prerequisite_checks"][
        "academic_replication_decision_is_complete"
    ] is False
    assert report["prerequisite_checks"]["container_runtime_gate_passed"] is False
    assert report["runs_executed"] == 0
    assert report["training_executed"] is False


def test_both_decision_comparisons_ignore_timestamps_but_enforce_numbers():
    protocol = json.loads(ROOT_PROTOCOL.read_text(encoding="utf-8"))
    reference = _decision()
    reproduced = _decision()
    reproduced["analyzed_at"] = "2027-01-01T00:00:00Z"
    assert compare_decisions(reference, reproduced, protocol)["reproduction_matches"]
    reproduced["primary"]["point_delta"] = 1e-6
    assert not compare_decisions(reference, reproduced, protocol)[
        "reproduction_matches"
    ]

    replication = _replication_decision()
    repeated = _replication_decision()
    repeated["generated_at"] = "2027-01-01T00:00:00Z"
    assert compare_decisions(
        replication, repeated, protocol, stage="academic_replication"
    )["reproduction_matches"]
    repeated["mechanism"]["changed_pair_fraction"] = 0.1
    assert not compare_decisions(
        replication, repeated, protocol, stage="academic_replication"
    )["reproduction_matches"]


def test_signed_review_verifies_both_stages(tmp_path):
    fixture = _review_fixture(tmp_path)
    report = verify_review(
        fixture["review_path"],
        root=tmp_path,
        protocol_path=fixture["protocol_path"],
    )
    assert report["passed"] is True
    assert report["status"] == "verified_reproduced"
    assert report["independent_review_available"] is True
    assert report["reproduction_matches"] is True
    assert report["scope"] == "full_study_132_runs"
    assert report["stages_verified"] == ["confirmatory", "academic_replication"]
    assert len(report["artifact_sha256"]) == 5
    assert all(report["artifact_sha256"].values())
    assert set(report["computed_comparison"]) == {
        "confirmatory",
        "academic_replication",
    }
    assert all(report["checks"].values())
    assert report["runs_executed_by_verifier"] == 0


def test_mechanism_mismatch_is_preserved_not_reframed(tmp_path):
    fixture = _review_fixture(tmp_path)
    reproduced_path = fixture["reproduced_paths"]["academic_replication"]
    _write_json(reproduced_path, _replication_decision(point_delta=0.5))
    review = fixture["review"]
    review["comparison"]["reproduced_decision_sha256"][
        "academic_replication"
    ] = _sha(reproduced_path)
    review["comparison"]["stage_conclusions"][
        "academic_replication"
    ] = "not_reproduced"
    review["comparison"]["reviewer_conclusion"] = "not_reproduced"
    _write_json(fixture["review_path"], review)
    report = verify_review(
        fixture["review_path"],
        root=tmp_path,
        protocol_path=fixture["protocol_path"],
    )
    assert report["passed"] is True
    assert report["status"] == "verified_not_reproduced"
    assert report["independent_review_available"] is True
    assert report["reproduction_matches"] is False


def test_material_deviation_forces_overall_inconclusive(tmp_path):
    fixture = _review_fixture(tmp_path)
    review = fixture["review"]
    review["deviations"] = [
        {
            "classification": "material",
            "description": "Frozen checkpoint could not be used.",
        }
    ]
    review["comparison"]["reviewer_conclusion"] = "inconclusive"
    _write_json(fixture["review_path"], review)
    report = verify_review(
        fixture["review_path"],
        root=tmp_path,
        protocol_path=fixture["protocol_path"],
    )
    assert report["passed"] is True
    assert report["status"] == "verified_inconclusive"
    assert report["material_deviation"] is True
    assert report["reproduction_matches"] is True

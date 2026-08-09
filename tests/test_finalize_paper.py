import hashlib
import json

from scripts.finalize_paper import (
    PROTOCOL_PATH,
    _canonical,
    build_ledger,
    protocol_report,
    validate_protocol,
    verify_final,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _confirmatory():
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
            "point_delta": 0.0,
            "ci95_low": -0.01,
            "ci95_high": 0.01,
        },
        "minimum_meaningful_delta_loss": 0.1,
        "behavior": {"changed_pairs": 0, "changed_pair_fraction": 0.0},
        "promotion_gates": {"promotion_supported": False},
        "secondary_metrics": {},
        "execution_identity": {"protocol_sha256": "a" * 64},
    }


def _replication():
    comparison = {
        "method": "fixture_stratified_paired_bootstrap",
        "draws": 10000,
        "point_delta": 0.0,
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
        "confirmatory_decision": "no_meaning_difference_keep_research_only",
        "conclusion": "replicated_equivalence_and_planning_nonadoption",
        "runs_executed": 72,
        "pairs_per_arm": 24,
        "comparisons": {
            "M1_minus_M0": {**comparison, "seed": 361000},
            "M1_minus_M1_predict_only": {**comparison, "seed": 361001},
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
        "external_validity": {"source": "sportec_idsse", "checks": {}},
        "material_deviations": [],
        "gates": {"branch_conclusion_matches_confirmatory_decision": True},
        "execution_identity": {"protocol_sha256": "b" * 64},
        "training_executed": False,
        "provider_calls_made": False,
    }


def _fixture(tmp_path):
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol_path = tmp_path / "protocol.json"
    _write_json(protocol_path, protocol)
    (tmp_path / "docs").mkdir(parents=True)
    (tmp_path / "docs" / "PAPER_FINALIZATION.md").write_text(
        "protocol guide\n", encoding="utf-8"
    )
    confirmatory = _confirmatory()
    replication = _replication()
    replication["confirmatory_decision"] = confirmatory["decision"]
    values = {
        "confirmatory_decision": confirmatory,
        "academic_replication_decision": replication,
        "independent_reproduction_verification": {
            "passed": True,
            "independent_review_available": True,
            "scope": "full_study_132_runs",
            "stages_verified": ["confirmatory", "academic_replication"],
            "computed_comparison": {
                "confirmatory": {"reproduction_matches": True},
                "academic_replication": {"reproduction_matches": True},
            },
            "reviewer_conclusion": "reproduced",
            "reproduction_matches": True,
            "material_deviation": False,
        },
        "preexecution_paper_package": {
            "passed": True,
            "status": "passed_preexecution_package",
        },
    }
    for name, value in values.items():
        _write_json(tmp_path / protocol["inputs"][name], value)
    ledger = build_ledger(tmp_path, protocol)
    ledger_path = tmp_path / protocol["outputs"]["result_ledger"]
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(_canonical(ledger) + "\n", encoding="utf-8")
    contract = protocol["manuscript_contract"]
    digest = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    sections = "\n\n".join(contract["required_sections"])
    manuscript = (
        "# Final paper\n\n"
        + "\n".join(
            contract[key]
            for key in (
                "stage_marker",
                "confirmatory_marker",
                "replication_marker",
                "independent_marker",
            )
        )
        + f"\n{contract['ledger_sha_marker_prefix']}{digest}\n\n"
        + sections
        + f"\n\n{contract['ledger_begin_marker']}\n```json\n"
        + _canonical(ledger)
        + f"\n```\n{contract['ledger_end_marker']}\n"
    )
    manuscript_path = tmp_path / protocol["outputs"]["final_manuscript"]
    manuscript_path.write_text(manuscript, encoding="utf-8")
    return protocol, protocol_path, ledger_path, manuscript_path


def test_finalization_protocol_is_valid_but_waiting_for_results():
    report = protocol_report()
    assert report["passed"] is True
    assert report["ready_to_finalize"] is False
    assert report["result_ledger_available"] is False
    assert report["final_manuscript_available"] is False
    assert report["matches_executed"] == 0


def test_ledger_build_preserves_negative_and_independent_results(tmp_path):
    protocol, _, _, _ = _fixture(tmp_path)
    ledger = build_ledger(tmp_path, protocol)
    assert ledger["confirmatory"]["promotion_supported"] is False
    assert ledger["academic_replication"]["passed"] is True
    assert ledger["independent_reproduction"]["reviewer_conclusion"] == "reproduced"
    assert len(ledger["source_sha256"]) == 4


def test_completed_manuscript_verifies_against_exact_embedded_ledger(tmp_path):
    _, protocol_path, _, _ = _fixture(tmp_path)
    report = verify_final(tmp_path, protocol_path)
    assert report["passed"] is True
    assert report["status"] == "verified_completed_manuscript"
    assert all(report["checks"].values())
    assert report["matches_executed_by_verifier"] == 0


def test_stale_source_or_pending_language_fails_closed(tmp_path):
    protocol, protocol_path, _, manuscript_path = _fixture(tmp_path)
    confirmatory_path = tmp_path / protocol["inputs"]["confirmatory_decision"]
    changed = _confirmatory()
    changed["primary"]["point_delta"] = 0.2
    _write_json(confirmatory_path, changed)
    stale = verify_final(tmp_path, protocol_path)
    assert stale["passed"] is False
    assert stale["checks"]["ledger_matches_current_sources"] is False

    protocol, protocol_path, _, manuscript_path = _fixture(tmp_path / "pending")
    manuscript_path.write_text(
        manuscript_path.read_text(encoding="utf-8") + "\n**Status: NOT EXECUTED.**\n",
        encoding="utf-8",
    )
    pending = verify_final(tmp_path / "pending", protocol_path)
    assert pending["passed"] is False
    assert pending["checks"]["preexecution_claims_are_absent"] is False


def test_branch_drift_prevents_ledger_build(tmp_path):
    protocol, _, _, _ = _fixture(tmp_path)
    path = tmp_path / protocol["inputs"]["academic_replication_decision"]
    replication = _replication()
    replication["branch"] = "mechanism_confirmation"
    _write_json(path, replication)
    checks = validate_protocol(protocol, tmp_path)
    assert all(checks.values())
    try:
        build_ledger(tmp_path, protocol)
    except ValueError as exc:
        assert "academic_replication_is_complete_and_branch_locked" in str(exc)
    else:
        raise AssertionError("branch drift must prevent ledger construction")

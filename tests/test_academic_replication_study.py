import copy

import pytest

from scripts import academic_replication_study as study
from scripts.merge_formal_ablation_results import METRICS


def _decision(value: str) -> dict:
    return {
        "schema_version": 2,
        "protocol_id": "gfs-action-policy-full-match-outcome-v1",
        "decision": value,
    }


def _rows(protocol, *, passes=400.0, shots=12.0, shift=0.0):
    rows = []
    for home, away in protocol["design"]["fixtures"]:
        for sample in protocol["design"]["sample_indices"]:
            row = {metric: 1.0 + shift for metric in METRICS}
            row.update({
                "fixture": f"{home}_vs_{away}",
                "sample_index": sample,
                "passes_per_team_match": passes + shift,
                "shots_per_team_match": shots + shift,
            })
            rows.append(row)
    return rows


def _state(protocol, decision, *, passes=400.0, shots=12.0, changed=False):
    baseline = _rows(protocol, passes=passes, shots=shots)
    predict_only = _rows(protocol, passes=passes, shots=shots)
    candidate = _rows(
        protocol,
        passes=passes,
        shots=shots,
        shift=0.1 if changed else 0.0,
    )
    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "state": "completed",
        "branch": study.resolve_branch(decision),
        "execution_identity": {"identity": "frozen"},
        "material_deviations": [],
        "arms": {
            "M0": {"rows": baseline},
            "M1_predict_only": {"rows": predict_only},
            "M1": {"rows": candidate},
        },
    }


def _comparison_sequence(*results):
    queue = list(results)

    def compare(_baseline, _candidate, *, draws, seed):
        assert draws == 10000
        assert seed in {361000, 361001}
        return queue.pop(0)

    return compare


def test_action_replication_v2_is_registered_and_ready_without_execution():
    report = study.protocol_report()
    current = study.status()
    assert report["passed"] is True
    assert report["ready_to_start"] is True
    assert report["runs_executed"] == 0
    assert all(report["checks"].values())
    assert current["state"] == "ready_not_started"
    assert current["branch"] == "variance_diagnosis"
    assert current["remaining_runs"] == 72
    assert current["matches_executed"] == 0
    assert current["identity_matches_progress"] is True
    assert current["next_action"] == (
        "explicitly_authorize_fixed_replication_execution"
    )
    assert current["training_executed"] is False
    assert current["provider_calls_made"] is False


def test_confirmatory_decision_selects_exactly_one_frozen_branch():
    assert study.resolve_branch(
        _decision("promotion_candidate_pending_release_review")
    ) == "mechanism_confirmation"
    assert study.resolve_branch(
        _decision("no_meaningful_difference_keep_research_only")
    ) == "adoption_path_diagnosis"
    assert study.resolve_branch(
        _decision("inconclusive_keep_research_only")
    ) == "variance_diagnosis"
    with pytest.raises(ValueError, match="accepted frozen branch"):
        study.resolve_branch(_decision("post_hoc_better_branch"))


def test_equivalence_branch_passes_only_with_nonadoption_and_idsse_scale():
    protocol = study._read_json(study.PROTOCOL_PATH)
    decision = _decision("no_meaningful_difference_keep_research_only")
    state = _state(protocol, decision)
    comparison = _comparison_sequence(
        {"ci95_low": -0.01, "ci95_high": 0.01,
         "candidate_all_external_gates": True},
        {"ci95_low": -0.005, "ci95_high": 0.005,
         "candidate_all_external_gates": True},
    )
    result = study.analyze(
        protocol, state, decision,
        expected_identity={"identity": "frozen"},
        comparison_fn=comparison,
    )
    assert result["passed"] is True
    assert result["conclusion"] == (
        "replicated_equivalence_and_planning_nonadoption"
    )
    assert result["mechanism"]["planning_nonadoption"] is True
    assert result["external_validity"]["checks"] == {
        "passes_inside_idsse_observed_range": True,
        "shots_inside_idsse_observed_range": True,
    }


def test_promotion_branch_requires_changed_behavior_and_planning_contribution():
    protocol = study._read_json(study.PROTOCOL_PATH)
    decision = _decision("promotion_candidate_pending_release_review")
    state = _state(protocol, decision, changed=True)
    comparison = _comparison_sequence(
        {"ci95_low": -0.25, "ci95_high": -0.15,
         "candidate_all_external_gates": True},
        {"ci95_low": -0.10, "ci95_high": -0.06,
         "candidate_all_external_gates": True},
    )
    result = study.analyze(
        protocol, state, decision,
        expected_identity={"identity": "frozen"},
        comparison_fn=comparison,
    )
    assert result["passed"] is True
    assert result["mechanism"]["replicated_promotion"] is True
    assert result["mechanism"]["planning_contribution"] is True
    assert result["mechanism"]["changed_pair_fraction"] == 1.0


def test_external_scale_failure_is_preserved_as_negative_result():
    protocol = study._read_json(study.PROTOCOL_PATH)
    decision = _decision("no_meaningful_difference_keep_research_only")
    state = _state(protocol, decision, passes=400.0, shots=3.0)
    comparison = _comparison_sequence(
        {"ci95_low": -0.01, "ci95_high": 0.01,
         "candidate_all_external_gates": True},
        {"ci95_low": -0.005, "ci95_high": 0.005,
         "candidate_all_external_gates": True},
    )
    result = study.analyze(
        protocol, state, decision,
        expected_identity={"identity": "frozen"},
        comparison_fn=comparison,
    )
    assert result["passed"] is False
    assert result["gates"]["licensed_idsse_external_scale_passes"] is False
    assert result["external_validity"]["checks"][
        "shots_inside_idsse_observed_range"
    ] is False


def test_clear_harm_cannot_pass_as_bounded_inconclusive():
    protocol = study._read_json(study.PROTOCOL_PATH)
    decision = _decision("inconclusive_keep_research_only")
    state = _state(protocol, decision)
    comparison = _comparison_sequence(
        {"ci95_low": 0.15, "ci95_high": 0.19,
         "candidate_all_external_gates": True},
        {"ci95_low": -0.005, "ci95_high": 0.005,
         "candidate_all_external_gates": True},
    )
    result = study.analyze(
        protocol, state, decision,
        expected_identity={"identity": "frozen"},
        comparison_fn=comparison,
    )
    assert result["passed"] is False
    assert result["mechanism"]["replicated_harm"] is True
    assert result["mechanism"]["bounded_inconclusive"] is False


def test_material_deviation_and_branch_drift_fail_closed():
    protocol = study._read_json(study.PROTOCOL_PATH)
    decision = _decision("no_meaningful_difference_keep_research_only")
    state = _state(protocol, decision)
    deviated = copy.deepcopy(state)
    deviated["material_deviations"] = ["fixture_changed"]
    comparison = _comparison_sequence(
        {"ci95_low": -0.01, "ci95_high": 0.01,
         "candidate_all_external_gates": True},
        {"ci95_low": -0.005, "ci95_high": 0.005,
         "candidate_all_external_gates": True},
    )
    result = study.analyze(
        protocol, deviated, decision,
        expected_identity={"identity": "frozen"},
        comparison_fn=comparison,
    )
    assert result["passed"] is False
    assert result["gates"]["no_material_deviations"] is False

    drifted = copy.deepcopy(state)
    drifted["branch"] = "mechanism_confirmation"
    with pytest.raises(ValueError, match="selected branch"):
        study.analyze(
            protocol, drifted, decision,
            expected_identity={"identity": "frozen"},
            comparison_fn=lambda *_args, **_kwargs: {},
        )


def test_prediction_only_arm_uses_same_m1_model_with_planning_disabled():
    prediction = study._arm_spec("M1_predict_only")
    full = study._arm_spec("M1")
    assert prediction.name == full.name == "M1"
    assert prediction.env["MATCH_WORLD_MODEL"] == "1"
    assert full.env["MATCH_WORLD_MODEL"] == "1"
    assert prediction.env["MATCH_WM_PLAN"] == "0"
    assert full.env["MATCH_WM_PLAN"] == "1"
    assert "action policy" in full.description.casefold()

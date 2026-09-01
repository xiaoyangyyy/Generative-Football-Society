import copy
import hashlib
import json

import pytest

from src.product.manager_intelligence import build_postmatch_debrief
from src.product.decision_ledger import (
    build_manager_decision_ledger,
    validate_manager_decision_ledger,
)
from src.product.season import (
    ManagerDecision,
    SeasonPlan,
    new_season_state,
)
from src.product.world_model_action_execution import (
    project_world_model_action_execution,
    validate_world_model_action_execution,
)


def _identity(payload):
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _record(
    identity, team, t_sec, baseline, actual, *, changed, influenced=True,
):
    return {
        "opportunity_id": identity,
        "team_id": team,
        "t_sec": t_sec,
        "recommended_action": actual,
        "actual_action": actual,
        "counterfactual_baseline_action": baseline,
        "influenced": influenced,
        "attribution_eligible": True,
        "policy_changed_action": changed,
        "recommended_probability_delta": 0.08 if influenced else 0.0,
        "total_variation_distance": 0.12 if influenced else 0.0,
    }


def _report():
    records = [
        _record("a-pass", "A", 12.0, "hold", "pass", changed=True),
        _record("a-hold", "A", 18.0, "hold", "hold", changed=False),
        _record("b-shot", "B", 24.0, "hold", "shot", changed=True),
    ]
    links = [
        {
            "opportunity_id": "a-pass", "team": "A",
            "t_sec": 12.0,
            "recommended_action": "pass",
            "counterfactual_baseline_action": "hold",
            "actual_action": "pass",
            "status": "direct_runtime_identity_match",
            "policy_changed_action": True,
            "attribution_eligible": True,
            "event_type": "pass", "event_clock": "0:12",
        },
        {
            "opportunity_id": "a-hold", "team": "A",
            "t_sec": 18.0,
            "recommended_action": "hold",
            "counterfactual_baseline_action": "hold",
            "actual_action": "hold",
            "status": "no_ball_trajectory_by_design",
            "policy_changed_action": False,
            "attribution_eligible": True,
            "event_type": None, "event_clock": None,
        },
        {
            "opportunity_id": "b-shot", "team": "B",
            "t_sec": 24.0,
            "recommended_action": "shot",
            "counterfactual_baseline_action": "hold",
            "actual_action": "shot",
            "status": "direct_runtime_identity_match",
            "policy_changed_action": True,
            "attribution_eligible": True,
            "event_type": "shot", "event_clock": "0:24",
        },
    ]
    return {
        "match_id": "official-1",
        "fixture": {"home": "A", "away": "B"},
        "result": {"score": {"home": 1, "away": 0}},
        "layers": {
            "management": {
                "decision": {
                    "team": "A", "tactic": "balanced",
                    "rotation": "balanced",
                },
                "effects": {}, "in_match": {},
            },
            "world_model": {
                "configured": True,
                "enabled": True,
                "action_adoption": {
                    "available": True,
                    "opportunities": 3,
                    "influenced_opportunities": 3,
                    "attribution_eligible_opportunities": 3,
                    "counterfactual_action_changes": 2,
                    "expected_counterfactual_action_changes": 0.36,
                    "mean_recommended_probability_shift": 0.08,
                    "records_retained": 3,
                    "records_truncated": False,
                    "records": records,
                },
            },
        },
        "replay": {
            "world_model_action_links": {
                "schema_version": 1,
                "available": True,
                "records": 3,
                "directly_observed": 2,
                "counterfactual_changed_and_observed": 2,
                "no_trajectory_by_design": 1,
                "unresolved_or_missing": 0,
                "causal_claim_authorized": False,
                "links": links,
            },
        },
    }


def test_official_action_execution_projects_only_manager_team_and_local_claim():
    evidence = project_world_model_action_execution(
        _report(), manager_team="A", expected_match_id="official-1",
    )

    assert evidence["available"] is True
    assert evidence["schema_version"] == 2
    assert evidence["team"] == "A"
    assert evidence["evidence_state"] == (
        "locally_attributable_action_changes_observed"
    )
    counts = evidence["retained_record_evidence"]
    assert counts == {
        "records": 2,
        "resolved_action_decisions": 2,
        "influenced_decisions": 2,
        "attribution_eligible_decisions": 2,
        "locally_attributable_action_changes": 1,
        "direct_ball_event_links": 1,
        "changed_direct_ball_event_links": 1,
        "decision_only_no_trajectory": 1,
        "unresolved_or_missing_ball_event_links": 0,
        "source_match_opportunities": 3,
        "source_records_truncated": False,
        "manager_record_coverage_complete": True,
    }
    assert evidence["retained_record_semantics"] == {
        "schema_version": 1,
        "records": 2,
        "actual_action_counts": {
            "hold": 1, "pass": 1, "cross": 0, "shot": 0, "none": 0,
        },
        "primary_signal_action_counts": {
            "hold": 1, "pass": 1, "cross": 0, "shot": 0, "none": 0,
        },
        "signal_mode_counts": {
            "direct_preference": 0, "suppression_only": 0,
            "none": 0, "legacy_unclassified": 2,
        },
        "hold_reference_redistribution_records": 0,
        "direct_cross_ball_event_links": 0,
        "locally_attributable_cross_changes": 0,
        "retained_record_coverage_complete": True,
        "source_manager_record_coverage_complete": True,
        "full_source_distribution_authorized": True,
        "outcome_attribution_authorized": False,
    }
    assert len(evidence["examples"]) == 2
    changed = evidence["examples"][0]
    assert changed["counterfactual_baseline_action"] == "hold"
    assert changed["actual_action"] == "pass"
    assert changed["simulator_local_action_attribution"] is True
    assert changed["runtime_link"]["direct_ball_event_identity"] is True
    assert changed["match_outcome_causality"] is False
    assert "a-pass" not in json.dumps(evidence)
    assert evidence["outcome_comparison_performed"] is False
    assert evidence["outcome_effect_estimate"] is None
    assert evidence["causal_effect_authorized"] is False
    validate_world_model_action_execution(evidence)


def test_official_action_execution_accepts_identity_bound_cross_trajectory():
    report = _report()
    record = report["layers"]["world_model"]["action_adoption"]["records"][0]
    record.update({
        "recommended_action": "cross",
        "actual_action": "cross",
    })
    link = report["replay"]["world_model_action_links"]["links"][0]
    link.update({
        "recommended_action": "cross",
        "actual_action": "cross",
        "event_type": "cross",
    })

    evidence = project_world_model_action_execution(
        report, manager_team="A", expected_match_id="official-1",
    )

    changed = evidence["examples"][0]
    assert changed["actual_action"] == "cross"
    assert changed["runtime_link"] == {
        "status": "direct_runtime_identity_match",
        "direct_ball_event_identity": True,
        "event_type": "cross",
        "event_clock": "0:12",
    }
    assert evidence["retained_record_semantics"][
        "actual_action_counts"
    ]["cross"] == 1
    assert evidence["retained_record_semantics"][
        "direct_cross_ball_event_links"
    ] == 1
    validate_world_model_action_execution(evidence)


def test_official_action_execution_rejects_cross_as_trajectory_free():
    report = _report()
    record = report["layers"]["world_model"]["action_adoption"]["records"][0]
    record.update({
        "recommended_action": "cross",
        "actual_action": "cross",
    })
    link = report["replay"]["world_model_action_links"]["links"][0]
    link.update({
        "recommended_action": "cross",
        "actual_action": "cross",
        "status": "no_ball_trajectory_by_design",
        "event_type": None,
        "event_clock": None,
    })
    linkage = report["replay"]["world_model_action_links"]
    linkage.update({
        "directly_observed": 1,
        "counterfactual_changed_and_observed": 1,
        "no_trajectory_by_design": 2,
    })

    with pytest.raises(ValueError, match="runtime link semantics"):
        project_world_model_action_execution(
            report, manager_team="A", expected_match_id="official-1",
        )


def test_official_execution_preserves_suppression_and_hold_reference_semantics():
    report = _report()
    adoption = report["layers"]["world_model"]["action_adoption"]
    adoption.update({
        "probability_policy_version": "validated_action_simplex_v2",
        "counterfactual_action_changes": 3,
        "mean_primary_signal_probability_shift": 0.09,
    })
    record = adoption["records"][1]
    record.update({
        "recommended_action": "none",
        "primary_signal_action": "pass",
        "signal_mode": "suppression_only",
        "counterfactual_baseline_action": "pass",
        "policy_changed_action": True,
        "recommended_probability_delta": 0.0,
        "primary_signal_probability_delta": -0.09,
        "reference_action_effect": {
            "action": "hold",
            "role": "counterfactual_baseline_only",
            "directly_authorized": False,
            "probability_delta": 0.09,
            "received_redistributed_probability": True,
            "realized_as_actual_action": True,
            "policy_changed_to_reference": True,
        },
    })
    link = report["replay"]["world_model_action_links"]["links"][1]
    link.update({
        "recommended_action": "none",
        "counterfactual_baseline_action": "pass",
        "policy_changed_action": True,
    })

    evidence = project_world_model_action_execution(
        report, manager_team="A", expected_match_id="official-1",
    )
    example = next(
        row for row in evidence["examples"]
        if row["actual_action"] == "hold"
    )

    assert example["recommended_action"] == "none"
    assert example["policy_signal"] == {
        "probability_policy_version": "validated_action_simplex_v2",
        "mode": "suppression_only",
        "primary_action": "pass",
        "primary_probability_delta": -0.09,
        "direct_recommendation": False,
    }
    assert example["reference_action_effect"] == {
        "available": True,
        "action": "hold",
        "role": "counterfactual_baseline_only",
        "directly_authorized": False,
        "probability_delta": 0.09,
        "received_redistributed_probability": True,
        "realized_as_actual_action": True,
        "policy_changed_to_reference": True,
    }
    semantics = evidence["retained_record_semantics"]
    assert semantics["signal_mode_counts"] == {
        "direct_preference": 0,
        "suppression_only": 1,
        "none": 0,
        "legacy_unclassified": 1,
    }
    assert semantics["hold_reference_redistribution_records"] == 1
    validate_world_model_action_execution(evidence)

    tampered = copy.deepcopy(evidence)
    example = next(
        row for row in tampered["examples"]
        if row["actual_action"] == "hold"
    )
    example["reference_action_effect"]["directly_authorized"] = True
    with pytest.raises(ValueError, match="reference action projection"):
        validate_world_model_action_execution(tampered)


def test_official_action_execution_fails_closed_on_source_and_projection_tamper():
    malformed = _report()
    malformed["layers"]["world_model"]["action_adoption"][
        "counterfactual_action_changes"
    ] = 4
    with pytest.raises(ValueError, match="aggregate hierarchy"):
        project_world_model_action_execution(
            malformed, manager_team="A", expected_match_id="official-1",
        )

    evidence = project_world_model_action_execution(
        _report(), manager_team="A", expected_match_id="official-1",
    )
    tampered = copy.deepcopy(evidence)
    tampered["retained_record_evidence"][
        "locally_attributable_action_changes"
    ] = 3
    frozen = copy.deepcopy(tampered)
    frozen.pop("evidence_identity")
    tampered["evidence_identity"] = _identity(frozen)
    with pytest.raises(ValueError, match="counts are invalid"):
        validate_world_model_action_execution(tampered)

    semantic_tamper = copy.deepcopy(evidence)
    semantic_tamper["retained_record_semantics"]["actual_action_counts"][
        "cross"
    ] = 1
    semantic_tamper["retained_record_semantics"]["actual_action_counts"][
        "pass"
    ] = 0
    frozen = copy.deepcopy(semantic_tamper)
    frozen.pop("evidence_identity")
    semantic_tamper["evidence_identity"] = _identity(frozen)
    with pytest.raises(ValueError, match="retained action semantics"):
        validate_world_model_action_execution(semantic_tamper)

    legacy = copy.deepcopy(evidence)
    legacy["schema_version"] = 1
    legacy.pop("retained_record_semantics")
    legacy.pop("evidence_identity")
    legacy["evidence_identity"] = _identity(legacy)
    validate_world_model_action_execution(legacy)

    false_legacy = copy.deepcopy(evidence)
    false_legacy["schema_version"] = 1
    false_legacy.pop("evidence_identity")
    false_legacy["evidence_identity"] = _identity(false_legacy)
    with pytest.raises(ValueError, match="cannot claim V2 semantics"):
        validate_world_model_action_execution(false_legacy)


def test_postmatch_debrief_contains_action_execution_and_isolates_invalid_layer():
    report = _report()
    debrief = build_postmatch_debrief(
        report, manager_team="A", expected_match_id="official-1",
    )

    assert debrief["available"] is True
    assert debrief["world_model_action_execution"]["available"] is True
    assert debrief["world_model_action_execution"][
        "retained_record_evidence"
    ]["locally_attributable_action_changes"] == 1
    assert debrief["causal_outcome_attribution"] is False

    report["replay"]["world_model_action_links"]["directly_observed"] = 99
    isolated = build_postmatch_debrief(
        report, manager_team="A", expected_match_id="official-1",
    )
    assert isolated["available"] is True
    assert isolated["world_model_action_execution"] == {
        "schema_version": 2,
        "available": False,
        "reason": "world_model_action_evidence_invalid",
    }


def test_official_action_execution_preserves_legacy_and_stable_boundaries():
    legacy = {
        "match_id": "legacy", "fixture": {"home": "A", "away": "B"},
        "layers": {},
    }
    assert project_world_model_action_execution(
        legacy, manager_team="A", expected_match_id="legacy",
    ) == {
        "schema_version": 2,
        "available": False,
        "reason": "legacy_report_without_world_model_layer",
    }
    stable = copy.deepcopy(_report())
    stable["layers"]["world_model"]["configured"] = False
    assert project_world_model_action_execution(
        stable, manager_team="A", expected_match_id="official-1",
    )["reason"] == "world_model_not_configured_for_match"


def test_official_action_execution_exposes_truncation_without_false_coverage():
    report = _report()
    adoption = report["layers"]["world_model"]["action_adoption"]
    adoption["records"] = adoption["records"][:2]
    adoption["records_retained"] = 2
    adoption["records_truncated"] = True
    adoption["counterfactual_action_changes"] = 1
    linkage = report["replay"]["world_model_action_links"]
    linkage["links"] = linkage["links"][:2]
    linkage.update({
        "records": 2,
        "directly_observed": 1,
        "counterfactual_changed_and_observed": 1,
        "no_trajectory_by_design": 1,
    })

    evidence = project_world_model_action_execution(
        report, manager_team="A", expected_match_id="official-1",
    )

    assert evidence["available"] is True
    counts = evidence["retained_record_evidence"]
    assert counts["source_records_truncated"] is True
    assert counts["manager_record_coverage_complete"] is False
    assert counts["records"] == 2
    semantics = evidence["retained_record_semantics"]
    assert semantics["records"] == 2
    assert semantics["retained_record_coverage_complete"] is True
    assert semantics["source_manager_record_coverage_complete"] is False
    assert semantics["full_source_distribution_authorized"] is False

    opponent_only = _report()
    adoption = opponent_only["layers"]["world_model"]["action_adoption"]
    adoption["records"] = adoption["records"][2:]
    adoption["records_retained"] = 1
    adoption["records_truncated"] = True
    linkage = opponent_only["replay"]["world_model_action_links"]
    linkage["links"] = linkage["links"][2:]
    linkage.update({
        "records": 1,
        "directly_observed": 1,
        "counterfactual_changed_and_observed": 1,
        "no_trajectory_by_design": 0,
    })
    unavailable = project_world_model_action_execution(
        opponent_only, manager_team="A", expected_match_id="official-1",
    )
    assert unavailable["reason"] == (
        "manager_team_action_records_not_retained_or_truncated"
    )


def test_official_action_execution_survives_ledger_and_summary_replay():
    season = new_season_state(
        SeasonPlan(("A", "B", "C", "D"), fast=True, manager_team="A"),
        season_id="season-action-execution",
        seed=17,
        created_at="2026-01-01T00:00:00Z",
    )
    fixture = next(
        row for row in season["fixtures"]
        if "A" in {row["home"], row["away"]}
    )
    fixture["manager_decision"] = ManagerDecision(
        team="A", tactic="balanced",
    ).as_dict()
    fixture.update({
        "state": "completed",
        "match_id": "official-1",
        "score": {"home": 1, "away": 0},
        "report": "matches/official-1/report.json",
    })
    debrief = build_postmatch_debrief(
        _report(), manager_team="A", expected_match_id="official-1",
    )

    ledger = build_manager_decision_ledger(
        season,
        execution_by_fixture={fixture["fixture_id"]: debrief},
    )

    entry = next(
        row for row in ledger["entries"]
        if row["fixture_id"] == fixture["fixture_id"]
    )
    evidence = entry["execution"]["world_model_action_execution"]
    assert evidence["available"] is True
    summary = ledger["summary"]["world_model_official_action_execution"]
    assert summary["fixtures_with_official_action_evidence"] == 1
    assert summary["fixtures_with_local_action_changes"] == 1
    assert summary["manager_action_records_retained"] == 2
    assert summary["locally_attributable_action_changes"] == 1
    assert summary["direct_ball_event_links"] == 1
    assert summary["retained_record_semantics"] == {
        "fixtures_with_v2_semantics": 1,
        "fixtures_without_v2_semantics": 0,
        "retained_records_with_v2_semantics": 2,
        "actual_action_counts": {
            "hold": 1, "pass": 1, "cross": 0, "shot": 0, "none": 0,
        },
        "primary_signal_action_counts": {
            "hold": 1, "pass": 1, "cross": 0, "shot": 0, "none": 0,
        },
        "signal_mode_counts": {
            "direct_preference": 0, "suppression_only": 0,
            "none": 0, "legacy_unclassified": 2,
        },
        "hold_reference_redistribution_records": 0,
        "direct_cross_ball_event_links": 0,
        "locally_attributable_cross_changes": 0,
        "fixtures_with_full_source_distribution": 1,
        "all_official_evidence_has_v2_semantics": True,
        "full_source_distribution_authorized": True,
        "outcome_attribution_authorized": False,
    }
    assert summary["outcome_effect_estimate"] is None
    assert summary["causal_effect_authorized"] is False
    thread = entry["world_evolution_thread"]
    assert thread["stages"][3]["stage_id"] == (
        "official_world_model_actions"
    )
    assert thread["stages"][3][
        "locally_attributable_action_changes"
    ] == 1
    assert thread["links"][2]["status"] == "evidence_unavailable"
    assert "tactical_runtime_binding_unavailable" in thread[
        "continuity_gaps"
    ]
    validate_manager_decision_ledger(ledger)

    thread_tamper = copy.deepcopy(ledger)
    thread_entry = next(
        row for row in thread_tamper["entries"]
        if row["fixture_id"] == fixture["fixture_id"]
    )
    thread_entry["world_evolution_thread"]["links"][2][
        "status"
    ] = "caused_world_model_actions"
    frozen_entry = copy.deepcopy(thread_entry)
    frozen_entry.pop("entry_identity")
    thread_entry["entry_identity"] = _identity(frozen_entry)
    frozen_ledger = copy.deepcopy(thread_tamper)
    frozen_ledger.pop("ledger_identity")
    thread_tamper["ledger_identity"] = _identity(frozen_ledger)
    with pytest.raises(ValueError, match="thread replay mismatch"):
        validate_manager_decision_ledger(thread_tamper)

    tampered = copy.deepcopy(ledger)
    tampered_entry = next(
        row for row in tampered["entries"]
        if row["fixture_id"] == fixture["fixture_id"]
    )
    tampered_entry["execution"]["world_model_action_execution"]["examples"][
        0
    ]["actual_action"] = "shot"
    frozen_entry = copy.deepcopy(tampered_entry)
    frozen_entry.pop("entry_identity")
    tampered_entry["entry_identity"] = _identity(frozen_entry)
    frozen_ledger = copy.deepcopy(tampered)
    frozen_ledger.pop("ledger_identity")
    tampered["ledger_identity"] = _identity(frozen_ledger)
    with pytest.raises(ValueError, match="semantics disagree"):
        validate_manager_decision_ledger(tampered)

    mismatched_debrief = copy.deepcopy(debrief)
    action = mismatched_debrief["world_model_action_execution"]
    action["team"] = "B"
    frozen_action = copy.deepcopy(action)
    frozen_action.pop("evidence_identity")
    action["evidence_identity"] = _identity(frozen_action)
    mismatched = build_manager_decision_ledger(
        season,
        execution_by_fixture={fixture["fixture_id"]: mismatched_debrief},
    )
    mismatch_entry = next(
        row for row in mismatched["entries"]
        if row["fixture_id"] == fixture["fixture_id"]
    )
    assert mismatch_entry["lifecycle_state"] == (
        "executed_evidence_unavailable"
    )
    assert mismatch_entry["execution"]["reason"] == (
        "world_model_action_execution_identity_mismatch"
    )

    invalid_debrief = copy.deepcopy(debrief)
    invalid_debrief["world_model_action_execution"]["examples"][0][
        "actual_action"
    ] = "shot"
    invalid = build_manager_decision_ledger(
        season,
        execution_by_fixture={fixture["fixture_id"]: invalid_debrief},
    )
    invalid_entry = next(
        row for row in invalid["entries"]
        if row["fixture_id"] == fixture["fixture_id"]
    )
    assert invalid_entry["lifecycle_state"] == (
        "executed_evidence_unavailable"
    )
    assert invalid_entry["execution"]["reason"] == (
        "world_model_action_execution_invalid"
    )

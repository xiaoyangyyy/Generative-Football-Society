import copy
import hashlib
import json

import pytest

from scripts.manager_advisor_study import (
    PROTOCOL_PATH,
    analyze_ledger,
    protocol_report,
    validate_protocol,
)
from src.product.decision_ledger import (
    _advisor_execution_trace,
    _future_review_execution_trace,
    world_model_advisor_summary,
    world_model_future_review_execution_summary,
    world_model_future_review_summary,
)


def _identity(payload):
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _ledger(
    count=12, *, pending=False, linked=True, diverse=True,
):
    chronological = []
    for index in range(count):
        tactic = "balanced" if not diverse or index % 2 == 0 else "gegenpress"
        lifecycle = (
            "frozen_awaiting_execution"
            if pending and index == count - 1
            else "executed_with_direct_evidence"
        )
        adoption = None if not linked else {
            "schema_version": 1,
            "intent": (
                "adopt_recommendation"
                if index % 2 == 0 else "reviewed_then_selected"
            ),
        }
        fixture_id = f'fixture-{index:02d}'
        decision_identity = _identity({
            'fixture_id': fixture_id,
            'tactic': tactic,
        })
        execution = (
            {
                'schema_version': 1,
                'available': False,
                'reason': 'fixture_not_completed',
            }
            if lifecycle == 'frozen_awaiting_execution'
            else {
                'schema_version': 1,
                'available': True,
                'match_id': f'match-{index:02d}',
                'tactical_binding': {
                    'schema_version': 1,
                    'available': True,
                    'applied_tactic': tactic,
                    'binding_identity': _identity({
                        'fixture_id': fixture_id,
                        'applied_tactic': tactic,
                    }),
                    'initial_vector_identity': _identity({
                        'fixture_id': fixture_id,
                        'vector': tactic,
                    }),
                    'changed_controls': [],
                    'final_delta_l1': 0.0,
                },
            }
        )
        entry = {
            "schema_version": 1,
            "fixture_id": f"fixture-{index:02d}",
            "lifecycle_state": lifecycle,
            "world_model_decision_support": {
                "available": True,
                "recommended_tactic": tactic,
                "selected_tactic": tactic,
                "aligned_with_recommendation": True,
                "recommendation_confidence": 0.2,
                "recommendation_margin": 0.03,
                "reliability": {"guidance": "low_authority"},
                "adoption": adoption,
            },
        }
        entry["entry_identity"] = _identity(entry)
        entry['decision_identity'] = decision_identity
        entry['execution'] = execution
        entry['advisor_execution_trace'] = _advisor_execution_trace(
            entry['world_model_decision_support'],
            decision_identity=decision_identity,
            lifecycle_state=lifecycle,
            execution=execution,
        )
        entry['world_model_future_reviews'] = []
        entry['future_review_execution_trace'] = (
            _future_review_execution_trace(
                [],
                decision_identity=decision_identity,
                selected_tactic=tactic,
                lifecycle_state=lifecycle,
                execution=execution,
            )
        )
        entry['entry_identity'] = _identity({
            key: value for key, value in entry.items()
            if key != 'entry_identity'
        })
        chronological.append(entry)
    entries = list(reversed(chronological))
    lifecycle = {
        name: sum(row["lifecycle_state"] == name for row in entries)
        for name in (
            "frozen_awaiting_execution", "executed_with_direct_evidence",
            "executed_evidence_unavailable",
        )
    }
    executed = lifecycle["executed_with_direct_evidence"] + lifecycle[
        "executed_evidence_unavailable"
    ]
    ledger = {
        "schema_version": 1,
        "available": True,
        "season_id": "season-evidence",
        "team": "Brazil",
        "entries": entries,
        "summary": {
            "decisions": count,
            **lifecycle,
            "direct_execution_coverage": round(
                lifecycle["executed_with_direct_evidence"] / max(1, executed), 6,
            ),
            "world_model_advisor": world_model_advisor_summary(entries),
            "world_model_future_reviews": world_model_future_review_summary(
                entries,
            ),
            "world_model_future_review_execution": (
                world_model_future_review_execution_summary(entries)
            ),
        },
        "claim_boundary": "test ledger",
    }
    ledger["ledger_identity"] = _identity(ledger)
    return ledger


def _protocol():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert all(validate_protocol(protocol).values())
    return protocol


def test_manager_advisor_protocol_is_frozen_and_zero_execution():
    report = protocol_report()

    assert report["passed"] is True
    assert report["ready_for_identity_bound_collection"] is True
    assert report["matches_executed"] == 0
    assert report["training_executed"] is False
    assert report["provider_calls_made"] is False
    assert report["results_available"] is False


def test_manager_advisor_analysis_waits_for_fixed_window_and_execution_close():
    protocol = _protocol()

    insufficient = analyze_ledger(_ledger(11), protocol)
    assert insufficient["state"] == "insufficient_evidence"
    assert insufficient["next_information_window"] == 12
    assert insufficient["outcome_effect_estimate"] is None

    open_window = analyze_ledger(_ledger(12, pending=True), protocol)
    assert open_window["state"] == "window_not_closed"
    assert open_window["information_window"] == 12
    assert open_window["gates"] is None


def test_manager_advisor_analysis_confirms_only_frozen_instrumentation_gates():
    result = analyze_ledger(_ledger(12), _protocol())

    assert result["state"] == "instrumentation_confirmed"
    assert result["passed"] is True
    assert all(result["gates"].values())
    assert result["evidence"]["adopted_recommendation"] == 6
    assert result["evidence"]["reviewed_then_selected"] == 6
    assert result["causal_effect_authorized"] is False
    assert result["matches_executed_by_analyzer"] == 0


def test_manager_advisor_analysis_retains_negative_instrumentation_result():
    result = analyze_ledger(
        _ledger(12, linked=False, diverse=False), _protocol(),
    )

    assert result["state"] == "instrumentation_failed"
    assert result["passed"] is False
    assert result["gates"]["minimum_explicit_interaction_coverage"] is False
    assert result["gates"]["maximum_unlinked_advice_fraction"] is False
    assert result["gates"]["minimum_distinct_recommended_tactics"] is False


def test_manager_advisor_analysis_rejects_rehashed_false_execution_claim():
    ledger = _ledger(12)
    entry = ledger["entries"][0]
    entry["execution"] = {
        "schema_version": 1,
        "available": False,
        "reason": "evidence_removed",
    }
    entry["advisor_execution_trace"] = _advisor_execution_trace(
        entry["world_model_decision_support"],
        decision_identity=entry["decision_identity"],
        lifecycle_state=entry["lifecycle_state"],
        execution=entry["execution"],
    )
    frozen_entry = copy.deepcopy(entry)
    frozen_entry.pop("entry_identity")
    entry["entry_identity"] = _identity(frozen_entry)
    frozen_ledger = copy.deepcopy(ledger)
    frozen_ledger.pop("ledger_identity")
    ledger["ledger_identity"] = _identity(frozen_ledger)

    with pytest.raises(ValueError, match="lifecycle/execution mismatch"):
        analyze_ledger(ledger, _protocol())


def test_manager_advisor_analysis_rejects_rehashed_derived_summary_tampering():
    ledger = _ledger(12)
    ledger["summary"]["world_model_advisor"]["adopted_recommendation"] = 12
    frozen = copy.deepcopy(ledger)
    frozen.pop("ledger_identity")
    ledger["ledger_identity"] = _identity(frozen)

    with pytest.raises(ValueError, match="summary replay mismatch"):
        analyze_ledger(ledger, _protocol())

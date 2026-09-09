import copy
import hashlib
import json

import pytest

from src.product.paired_society_state import (
    MAX_COUNT_DELTAS,
    MAX_TERMINAL_STATE_CHANGES,
    build_paired_society_divergence,
    validate_paired_society_divergence,
)
from src.simulation.society_continuity import PUBLIC_BOUNDARY


def _identity(payload):
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def society_snapshot(team, marker, *, morale=0.0, memories=2):
    return {
        "available": True,
        "state_identity": marker * 64,
        "source_transaction_id": f"match-{marker}-{team}",
        "memory_records": memories,
        "cognitive_memory_records": 1,
        "beliefs": 1,
        "reflections": 1,
        "tactical_controls": {
            "pressing_intensity": 0.5,
            "risk_budget": 0.5,
            "line_height": 0.5,
            "rotation_aggressiveness": 0.5,
        },
        "emotion_profile": {
            "pride": 0.2,
            "anger": 0.1,
            "shame": 0.1,
            "fear": 0.1,
            "determination": 0.6,
        },
        "social_narrative_state": {
            "trust_index": 0.5,
            "polarization": 0.2,
            "narrative_fatigue": 0.1,
        },
        "psychological_state": {
            "morale": morale,
            "pressure": 0.0,
            "trust": 0.0,
            "risk_appetite": 0.0,
            "conflict": 0.0,
            "audience_activation": 0.0,
        },
        "referee_grievance": 0.05,
        "claim_boundary": PUBLIC_BOUNDARY,
    }


def _report(marker, *, morale=0.0, memories=2):
    return {"layers": {"society": {
        "Brazil": society_snapshot(
            "Brazil", marker, morale=morale, memories=memories,
        ),
        "Argentina": society_snapshot("Argentina", marker),
    }}}


def test_paired_society_divergence_is_exact_bounded_and_noncausal():
    baseline = _report("a")
    treatment = _report("b", morale=0.25, memories=3)
    original = copy.deepcopy(treatment)

    evidence = build_paired_society_divergence(
        baseline,
        treatment,
        teams=["Brazil", "Argentina"],
        pair_eligible=True,
    )

    assert treatment == original
    assert evidence["available"] is True
    assert evidence["status"] == "terminal_society_divergence_observed"
    assert evidence["teams_with_terminal_divergence"] == 1
    assert evidence["count_deltas"] == [{
        "team": "Brazil", "field": "memory_records", "delta": 1,
    }]
    assert evidence["state_changes"] == [{
        "team": "Brazil",
        "scope": "psychological_state",
        "field": "morale",
        "baseline": 0.0,
        "treatment": 0.25,
    }]
    assert evidence["paired_terminal_state_comparison_authorized"] is True
    assert evidence["policy_to_state_causality_authorized"] is False
    assert evidence["outcome_causality_authorized"] is False
    assert evidence["real_football_causality_authorized"] is False
    assert MAX_COUNT_DELTAS == 8
    assert MAX_TERMINAL_STATE_CHANGES == 38
    assert validate_paired_society_divergence(evidence) == evidence


def test_paired_society_divergence_preserves_zero_and_missing_distinction():
    baseline = _report("a")
    treatment = _report("b")

    zero = build_paired_society_divergence(
        baseline, treatment, teams=["Brazil", "Argentina"], pair_eligible=True,
    )
    missing = build_paired_society_divergence(
        {"layers": {}}, treatment,
        teams=["Brazil", "Argentina"],
        pair_eligible=True,
    )

    assert zero["available"] is True
    assert zero["status"] == "no_terminal_society_divergence"
    assert zero["state_change_count"] == zero["count_delta_count"] == 0
    assert missing["available"] is False
    assert missing["reason"] == "legacy_or_missing_terminal_society_state"
    assert missing["paired_terminal_state_comparison_authorized"] is False


def test_paired_society_divergence_rejects_invalid_or_rehashed_tampering():
    baseline = _report("a")
    treatment = _report("b", morale=0.25)
    treatment["layers"]["society"]["Brazil"]["psychological_state"][
        "morale"
    ] = 2.0
    with pytest.raises(ValueError, match="out of bounds"):
        build_paired_society_divergence(
            baseline,
            treatment,
            teams=["Brazil", "Argentina"],
            pair_eligible=True,
        )

    evidence = build_paired_society_divergence(
        baseline,
        _report("b", morale=0.25),
        teams=["Brazil", "Argentina"],
        pair_eligible=True,
    )
    evidence["state_changes"][0]["treatment"] = 2.0
    evidence.pop("divergence_identity")
    evidence["divergence_identity"] = _identity(evidence)
    with pytest.raises(ValueError, match="state change value"):
        validate_paired_society_divergence(evidence)

    evidence = build_paired_society_divergence(
        baseline,
        _report("b", memories=3),
        teams=["Brazil", "Argentina"],
        pair_eligible=True,
    )
    evidence["count_deltas"][0]["delta"] = 100_000
    evidence.pop("divergence_identity")
    evidence["divergence_identity"] = _identity(evidence)
    with pytest.raises(ValueError, match="count delta"):
        validate_paired_society_divergence(evidence)

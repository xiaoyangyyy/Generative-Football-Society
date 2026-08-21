from src.product.manager_intelligence import (
    build_postmatch_debrief,
    build_prematch_intelligence,
)
from src.match_engine.tactical_catalog import TACTICAL_KEYS


def _catalog(team, *, fatigue=0.0, morale=0.55, unavailable=0, settled=0):
    return {
        "available": True,
        "team": team,
        "formation": "4-3-3",
        "players": [
            {
                "player_id": f"{team}-{index}", "selectable": index < 16 - unavailable,
                "minutes_ema": 80 if index < 3 else 20, "form_ema": 0.57,
            }
            for index in range(16)
        ],
        "team_condition": {
            "fatigue": fatigue, "morale": morale, "media_pressure": 0.2,
            "injured_players": unavailable, "suspended_players": 0,
            "unavailable_players": unavailable, "matches_settled": settled,
            "source": "persisted_squad_carryover" if settled else "default_simulation_baseline",
        },
    }


def _command(decision=None):
    return {
        "enabled": True,
        "current_fixture": {
            "fixture_id": "md02-fx01", "manager_position": 3,
            "manager_points": 3, "opponent_position": 1,
            "opponent_points": 6, "decision": decision,
        },
    }


def test_prematch_intelligence_uses_direct_state_without_predicting_tactic_or_result():
    intelligence = build_prematch_intelligence(
        _command(),
        _catalog("A", fatigue=0.72, morale=0.41, unavailable=3, settled=2),
        _catalog("B", settled=2),
    )

    assert intelligence["evidence_coverage"] == "direct_persisted_state"
    assert {signal["id"] for signal in intelligence["signals"]} >= {
        "high_fatigue", "player_unavailability", "low_morale",
    }
    assert intelligence["rotation_tradeoff"]["rotate"]["status_penalty"] == 3.0
    assert "win probability" in intelligence["claim_boundary"]
    assert "suggested_tactic" not in intelligence


def test_prematch_intelligence_marks_default_baseline_and_missing_contingency():
    decision = {"in_match_plan": None}
    intelligence = build_prematch_intelligence(
        _command(decision), _catalog("A"), _catalog("B"),
    )

    assert intelligence["evidence_coverage"] == "roster_plus_default_baseline"
    assert any(
        signal["id"] == "no_contingency_plan"
        for signal in intelligence["signals"]
    )
    assert intelligence["contingency_conditions"] == []

    mixed = build_prematch_intelligence(
        _command(), _catalog("A", settled=1), _catalog("B"),
    )
    assert mixed["evidence_coverage"] == "mixed_persisted_and_baseline"


def test_postmatch_debrief_separates_direct_mechanics_from_observed_result():
    from src.product.season import ClubResourcePlan, club_resource_effects

    resource_plan = ClubResourcePlan(recovery=2, medical=1, sports_science=3)
    report = {
        "match_id": "m-1",
        "fixture": {"home": "A", "away": "B"},
        "result": {"score": {"home": 2, "away": 1}},
        "layers": {"management": {
            "decision": {
                "team": "A", "tactic": "gegenpress", "rotation": "rotate",
                "lineup": {"source": "manual"},
            },
            "club_support": {
                "team": "A",
                "plan": resource_plan.as_dict(),
                "effects": club_resource_effects(resource_plan),
            },
            "effects": {"home": {
                "base_status": 70.0, "effective_status": 67.5,
                "rotation_status_penalty": 3.0,
                "fatigue_load_multiplier": 0.88,
                "club_fatigue_load_factor": 0.88,
            }},
            "in_match": {"home": {"outcomes": [
                {"scheduled_minute": 60, "condition": "trailing", "status": "applied", "reason": "condition_met"},
                {"scheduled_minute": 75, "condition": "leading", "status": "skipped", "reason": "condition_not_met"},
            ]}},
        }},
    }

    debrief = build_postmatch_debrief(report, manager_team="A", expected_match_id="m-1")

    assert debrief["direct_engine_effects"]["combined_status_delta"] == -2.5
    assert debrief["direct_engine_effects"]["club_fatigue_load_factor"] == 0.88
    assert debrief["club_support"]["available"] is True
    assert debrief["club_support"]["plan"] == {
        "recovery": 2, "medical": 1, "sports_science": 3,
    }
    assert debrief["in_match_execution"] == {
        "evaluated": 2, "applied": 1, "skipped": 1, "failed": 0,
        "outcomes": [
            {"minute": 60, "condition": "trailing", "status": "applied", "reason": "condition_met"},
            {"minute": 75, "condition": "leading", "status": "skipped", "reason": "condition_not_met"},
        ],
    }
    assert debrief["observed_result"]["descriptive_only"] is True
    assert debrief["causal_outcome_attribution"] is False

    report["layers"]["management"]["effects"]["home"][
        "club_fatigue_load_factor"
    ] = 1.0
    rejected = build_postmatch_debrief(
        report, manager_team="A", expected_match_id="m-1",
    )
    assert rejected == {
        "schema_version": 1, "available": False,
        "reason": "club_support_runtime_mismatch",
    }


def test_postmatch_debrief_fails_closed_on_report_identity():
    debrief = build_postmatch_debrief(
        {"match_id": "other"}, manager_team="A", expected_match_id="m-1",
    )
    assert debrief == {
        "schema_version": 1, "available": False,
        "reason": "report_identity_mismatch",
    }

    malformed = build_postmatch_debrief(
        {"match_id": "m-1", "fixture": []},
        manager_team="A", expected_match_id="m-1",
    )
    assert malformed == {
        "schema_version": 1, "available": False,
        "reason": "fixture_evidence_missing",
    }


def test_postmatch_debrief_verifies_runtime_tactical_binding_and_rejects_tamper():
    initial = {key: 0.5 for key in TACTICAL_KEYS}
    final = dict(initial)
    final["pressing_intensity"] = 0.7
    report = {
        "match_id": "m-binding",
        "fixture": {"home": "A", "away": "B"},
        "result": {"score": {"home": 0, "away": 0}},
        "layers": {"management": {
            "decision": {
                "team": "A", "tactic": "gegenpress", "rotation": "balanced",
            },
            "effects": {}, "in_match": {},
            "tactical_execution": {"home": {
                "schema_version": 1, "team": "A",
                "applied_tactic": "gegenpress",
                "native_archetype": "gegenpress",
                "binding_kind": "locked_preset",
                "source": "studio_user_intervention",
                "preset_locked": True,
                "initial_vector": initial, "final_vector": final,
                "changed_controls": ["pressing_intensity"],
                "final_delta_l1": 0.2,
            }},
        }},
    }

    debrief = build_postmatch_debrief(
        report, manager_team="A", expected_match_id="m-binding",
    )

    binding = debrief["tactical_binding"]
    assert binding["available"] is True
    assert binding["applied_tactic"] == "gegenpress"
    assert binding["initial_vector"]["pressing_intensity"] == 0.5
    assert binding["final_vector"]["pressing_intensity"] == 0.7
    assert binding["changed_controls"] == ["pressing_intensity"]
    assert len(binding["initial_vector_identity"]) == 64
    assert len(binding["binding_identity"]) == 64

    report["layers"]["management"]["tactical_execution"]["home"][
        "applied_tactic"
    ] = "balanced"
    rejected = build_postmatch_debrief(
        report, manager_team="A", expected_match_id="m-binding",
    )
    assert rejected == {
        "schema_version": 1, "available": False,
        "reason": "tactical_execution_binding_mismatch",
    }

"""The coach LLM must spend its optional reasoning budget deliberately."""

import copy
import json

from src.match_engine.cognitive.config import CognitiveMatchConfig
from src.match_engine.cognitive.events import (
    CognitiveTriggerEvent,
    ENTITY_TIER_COACH,
)
from src.match_engine.cognitive.executor import CognitiveExecutor
from src.match_engine.cognitive.schemas import validate_coach_plan
from src.match_engine.world_model.decision_support import (
    build_coach_decision_packet,
)
from src.match_engine.world_model.llm_decision_brief import (
    build_llm_decision_brief,
    llm_decision_brief_metadata,
)
from src.match_engine.world_model.llm_deliberation_focus import (
    TASK_CONTRACTS,
    evaluate_llm_deliberation_focus,
    llm_deliberation_focus_audit_is_valid,
    llm_deliberation_focus_diagnostics,
    validate_llm_deliberation_focus,
)
from src.match_engine.world_model.online_evaluation import (
    aggregate_online_calibration,
)
from tests.test_world_model_llm_fusion import _Runtime, _state


def _focused_plan(brief):
    task = brief["deliberation_agenda"]["recommended_focus"][0]
    contract_key, audit_key = TASK_CONTRACTS[task]
    return task, {
        "world_model_deliberation_focus": {
            "tasks": [task],
            "confidence": 0.8,
            "rationale": "Use the highest-priority eligible reasoning task.",
        },
        contract_key: {"validated_contract_placeholder": True},
        audit_key: {"accepted": True},
    }


def test_focus_schema_is_bounded_and_preserved_by_coach_plan():
    focus = {
        "tasks": ["opponent_information_query"],
        "confidence": 0.8,
        "rationale": "Opponent uncertainty dominates.",
    }
    assert validate_llm_deliberation_focus(focus) == focus
    assert validate_llm_deliberation_focus({
        **focus, "tasks": ["rewrite_world_model"],
    }) is None
    assert validate_llm_deliberation_focus({
        **focus, "tasks": [
            "world_model_critique", "world_model_event_hypothesis",
            "world_model_event_option", "world_model_risk_constraint",
        ],
    }) is None
    plan = validate_coach_plan({
        "world_model_action": "pass",
        "world_model_deliberation_focus": focus,
    })
    assert plan["world_model_deliberation_focus"] == focus


def test_model_checks_focus_budget_alignment_and_priority_efficiency():
    packet = build_coach_decision_packet(_Runtime(), _state(), "Home")
    brief = build_llm_decision_brief(packet)
    task, plan = _focused_plan(brief)
    audit = evaluate_llm_deliberation_focus(
        brief, plan, focus_signature="llm-deliberation-focus:test",
    )

    assert audit["accepted"]
    assert audit["focus"]["tasks"] == [task]
    assert audit["focus_priority_efficiency"] == 1.0
    assert audit["model_checked_consistent"]
    assert not audit["unsupported_selected_tasks"]
    assert not audit["unfocused_emitted_contracts"]
    assert not audit["missing_selected_contracts"]
    assert not audit["rejected_selected_contracts"]
    assert llm_deliberation_focus_audit_is_valid(audit)
    assert not audit["can_change_current_action"]
    assert not audit["can_relax_downstream_validators"]

    other = next(name for name in TASK_CONTRACTS if name != task)
    extra_contract, extra_audit = TASK_CONTRACTS[other]
    unfocused_plan = copy.deepcopy(plan)
    unfocused_plan[extra_contract] = {"extra": True}
    unfocused_plan[extra_audit] = {"accepted": True}
    unfocused = evaluate_llm_deliberation_focus(
        brief, unfocused_plan,
        focus_signature="llm-deliberation-focus:test",
    )
    assert not unfocused["model_checked_consistent"]
    assert unfocused["unfocused_emitted_contracts"] == [other]

    rejected_plan = copy.deepcopy(plan)
    rejected_plan[TASK_CONTRACTS[task][1]] = {"accepted": False}
    rejected = evaluate_llm_deliberation_focus(
        brief, rejected_plan,
        focus_signature="llm-deliberation-focus:test",
    )
    assert not rejected["model_checked_consistent"]
    assert rejected["rejected_selected_contracts"] == [task]

    tampered = copy.deepcopy(audit)
    tampered["focus_priority_efficiency"] = 0.0
    assert not llm_deliberation_focus_audit_is_valid(tampered)


def test_focus_diagnostics_and_strict_online_gate_are_match_clustered():
    packet = build_coach_decision_packet(_Runtime(), _state(), "Home")
    brief = build_llm_decision_brief(packet)
    _, plan = _focused_plan(brief)
    audit = evaluate_llm_deliberation_focus(
        brief, plan, focus_signature="llm-deliberation-focus:test",
    )
    record = {
        "llm_decision_brief_context": llm_decision_brief_metadata(brief),
        "llm_deliberation_focus_context": audit,
    }
    logs = [{
        "world_model_decision_adoption": {
            "records": [copy.deepcopy(record)],
        },
    } for _ in range(4)]

    diagnostics = llm_deliberation_focus_diagnostics([
        payload["world_model_decision_adoption"]["records"]
        for payload in logs
    ])
    assert diagnostics["accepted_focus_audits"] == 4
    assert diagnostics["matches"] == 4
    assert diagnostics["match_clustered_focus_declaration_rate"] == 1.0
    assert diagnostics["match_clustered_consistency_rate"] == 1.0
    assert diagnostics["match_clustered_focus_priority_efficiency"] == 1.0
    assert diagnostics["provenance_compatible"]

    report = aggregate_online_calibration(
        logs, min_transitions=0, require_llm_deliberation_focus=True,
    )
    assert report["version"] == 31
    assert report["llm_deliberation_focus_ready"]
    assert report["gates"]["llm_deliberation_focus"]

    malformed = copy.deepcopy(logs)
    malformed[0]["world_model_decision_adoption"]["records"][0][
        "llm_deliberation_focus_context"
    ]["focus_priority_efficiency"] = 0.0
    rejected = aggregate_online_calibration(
        malformed, min_transitions=0,
        require_llm_deliberation_focus=True,
    )
    assert not rejected["llm_deliberation_focus_ready"]
    assert not rejected["gates"]["llm_deliberation_focus"]


def test_executor_persists_focus_failure_without_relaxing_any_contract():
    class _FocusedLLM:
        model = "focus-test"

        def coach_in_match_plan(self, team_name, facts, kind):
            task = facts["world_model_llm_decision_brief"][
                "deliberation_agenda"
            ]["recommended_focus"][0]
            return json.dumps({
                "reasoning": "Focus the bounded reasoning budget.",
                "confidence": 0.8,
                "controls_delta": {},
                "world_model_action": "shot",
                "world_model_deliberation_focus": {
                    "tasks": [task],
                    "confidence": 0.8,
                    "rationale": "Highest model-owned agenda priority.",
                },
            })

    state = _state()
    executor = CognitiveExecutor(
        CognitiveMatchConfig(enabled=True), _FocusedLLM(),
        world_model_runtime=_Runtime(),
    )
    result = executor.process_trigger(CognitiveTriggerEvent(
        60.0, "xg_swing", ENTITY_TIER_COACH,
        "coach:Home", team_id="Home", salience=1.0,
    ), state)
    audit = result.plan["world_model_deliberation_focus_audit"]
    assert audit["accepted"]
    assert not audit["model_checked_consistent"]
    assert audit["missing_selected_contracts"]
    assert audit["rejected_selected_contracts"]
    stored = state._wm_coach_decision_adoption[-1][
        "llm_deliberation_focus_context"
    ]
    assert stored == audit
    assert llm_deliberation_focus_audit_is_valid(stored)
    assert not stored["authority_active"]
    assert not stored["can_relax_downstream_validators"]

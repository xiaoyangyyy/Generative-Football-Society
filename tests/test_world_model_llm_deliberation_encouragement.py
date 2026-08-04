"""Task-selection experiments run only after the coach action is frozen."""

import copy
import json

from src.match_engine.cognitive.config import CognitiveMatchConfig
from src.match_engine.cognitive.events import (
    CognitiveTriggerEvent,
    ENTITY_TIER_COACH,
)
from src.match_engine.cognitive.executor import CognitiveExecutor
from src.match_engine.world_model.decision_support import (
    build_coach_decision_packet,
)
from src.match_engine.world_model.llm_decision_brief import (
    build_llm_decision_brief,
    compact_world_model_facts_for_llm,
)
from src.match_engine.world_model.llm_deliberation_compute import (
    build_deliberation_compute_allocation,
    deliberation_compute_limits,
)
from src.match_engine.world_model.llm_deliberation_encouragement import (
    build_shadow_deliberation_brief,
    build_task_selection_value_memory,
    shadow_deliberation_brief_is_valid,
    task_encouragement_diagnostics,
    task_selection_value_memory_is_valid,
)
from src.match_engine.world_model.llm_deliberation_focus import (
    TASK_CONTRACTS,
    evaluate_llm_deliberation_focus,
)
from src.match_engine.world_model.online_evaluation import (
    aggregate_online_calibration,
)
from tests.test_world_model_llm_fusion import _Runtime, _state


def _experimental_packet(nonce=0):
    packet = build_coach_decision_packet(_Runtime(), _state(), "Home")
    # Isolate the event-option/contrastive encouragement experiment from the
    # independent belief-space meta-planning task introduced later.
    packet.pop("belief_space_meta_plan", None)
    packet["second_order_game"]["trajectory_rollout"]["active"] = True
    for index, candidate in enumerate(packet["candidates"]):
        candidate["risk_adjusted_value"] = [1.0, 0.92, 0.5, 0.4][index]
    packet["task_encouragement_test_nonce"] = nonce
    return packet


def test_shadow_brief_randomizes_only_an_evidence_safe_shadow_swap():
    brief = build_llm_decision_brief(_experimental_packet())
    original = copy.deepcopy(brief)
    shadow = build_shadow_deliberation_brief(brief)
    experiment = shadow["shadow_task_encouragement"]

    assert experiment["eligible"]
    assert set(experiment["treatment_portfolio"]) ^ set(
        experiment["control_portfolio"]
    ) == {
        "world_model_event_option", "world_model_contrastive_claim",
    }
    assert experiment["assigned_arm"] in {"control", "treatment"}
    assert experiment["action_frozen_before_assignment"]
    assert not experiment["can_change_current_action"]
    assert brief == original
    assert shadow_deliberation_brief_is_valid(shadow, brief)

    projected = compact_world_model_facts_for_llm({
        "world_model_decision_support": _experimental_packet(),
        "world_model_llm_decision_brief": brief,
        "world_model_shadow_deliberation_phase": True,
        "world_model_shadow_deliberation_brief": shadow,
    })
    assert projected["world_model_decision_support"] == shadow
    assert "world_model_shadow_deliberation_brief" not in projected


def test_executor_freezes_action_and_preserves_two_stage_audit_in_cache(
    tmp_path,
):
    class _TwoStageLLM:
        model = "two-stage-test"

        def __init__(self):
            self.main_calls = 0
            self.shadow_calls = 0

        def coach_in_match_plan(self, team_name, facts, kind):
            self.main_calls += 1
            return json.dumps({
                "reasoning": "Freeze the current action first.",
                "confidence": 0.8,
                "controls_delta": {"risk_budget": 0.02},
                "world_model_action": "pass",
            })

        def coach_world_model_deliberation(
            self, team_name, facts, kind, frozen_plan,
        ):
            self.shadow_calls += 1
            assert frozen_plan["world_model_action"] == "pass"
            assert facts["world_model_shadow_deliberation_phase"]
            menu = facts["world_model_shadow_deliberation_brief"][
                "opponent_information_question_menu"
            ]
            proposal = next(
                row for row in menu["proposals"]
                if row["selected_action"] == "pass"
                and row["proposal_id"] == facts[
                    "world_model_shadow_deliberation_brief"
                ]["opponent_information_cognitive_policy"][
                    "recommendations_by_action"
                ]["pass"]["recommended_proposal_id"]
            )
            return json.dumps({
                "reasoning": "Attempted mutations must be ignored.",
                "confidence": 0.9,
                "controls_delta": {"risk_budget": 0.15},
                "world_model_action": "shot",
                "world_model_deliberation_focus": {
                    "tasks": [
                        "opponent_information_policy",
                        "belief_space_meta_planning",
                    ],
                    "confidence": 0.8,
                    "rationale": "Resolve opponent uncertainty.",
                },
                "opponent_information_policy": {
                    "decision": "ask",
                    "proposal_id": proposal["proposal_id"],
                    "confidence": 0.8,
                    "rationale": "Select the model-proposed information query.",
                },
                "world_model_belief_space_meta_plan": {
                    "option_id": facts[
                        "world_model_shadow_deliberation_brief"
                    ]["belief_space_meta_plan"]["recommended_option_id"],
                    "confidence": 0.8,
                    "rationale": "Route only the next decision's shadow compute.",
                },
            })

    llm = _TwoStageLLM()
    config = CognitiveMatchConfig(
        enabled=True, cache_dir=str(tmp_path),
    )

    def run_once():
        executor = CognitiveExecutor(
            config, llm, world_model_runtime=_Runtime(),
        )
        state = _state()
        result = executor.process_trigger(CognitiveTriggerEvent(
            60.0, "xg_swing", ENTITY_TIER_COACH,
            "coach:Home", team_id="Home", salience=1.0,
        ), state)
        return result, state

    first, first_state = run_once()
    assert first.plan["world_model_action"] == "pass"
    assert first.plan["controls_delta"] == {"risk_budget": 0.02}
    audit = first.plan["world_model_deliberation_encouragement_audit"]
    assert audit["accepted"]
    assert audit["action_mutation_ignored"]
    assert audit["tactical_mutation_ignored"]
    assert first.plan["world_model_deliberation_focus_audit"][
        "task_encouragement_context"
    ]["audit_valid"]
    query_audit = first.plan["opponent_information_query_audit"]
    assert query_audit["accepted"]
    assert query_audit["question_selection"][
        "world_model_proposed_question"
    ]
    assert query_audit["question_selection"][
        "llm_selected_after_action_freeze"
    ]
    meta_audit = first.plan["world_model_belief_space_meta_plan_audit"]
    assert meta_audit["accepted"]
    assert first_state._wm_belief_space_compute_preferences["Home"] == (
        meta_audit["compute_preference"]
    )
    first_state.clock_seconds = 70.0
    next_packet = build_coach_decision_packet(
        _Runtime(), first_state, "Home",
    )
    assert next_packet["belief_space_meta_plan"]["preference_status"] == (
        "prior_llm_compute_preference_applied"
    )
    assert llm.main_calls == 1
    assert llm.shadow_calls == 1

    cached, _cached_state = run_once()
    assert cached.cached
    assert cached.plan["world_model_action"] == "pass"
    assert cached.plan[
        "world_model_deliberation_encouragement_audit"
    ] == audit
    assert llm.main_calls == 1
    assert llm.shadow_calls == 1


def _randomized_encouragement_records():
    records = []
    counts = {"control": 0, "treatment": 0}
    nonce = 0
    while min(counts.values()) < 4:
        brief = build_llm_decision_brief(_experimental_packet(nonce))
        shadow = build_shadow_deliberation_brief(brief)
        experiment = shadow["shadow_task_encouragement"]
        arm = experiment["assigned_arm"]
        if not experiment["eligible"] or counts[arm] >= 4:
            nonce += 1
            continue
        counts[arm] += 1
        tasks = list(experiment["assigned_focus"])
        plan = {
            "world_model_action": "pass",
            "world_model_deliberation_focus": {
                "tasks": tasks,
                "confidence": 0.8,
                "rationale": "Follow the randomized shadow portfolio.",
            },
        }
        allocation = build_deliberation_compute_allocation(brief, plan)
        plan["world_model_deliberation_compute_allocation"] = allocation
        for task in tasks:
            contract, audit_key = TASK_CONTRACTS[task]
            plan[contract] = {"placeholder": True}
            task_audit = {"accepted": True}
            if task == "world_model_event_option":
                limits = deliberation_compute_limits(allocation, task)
                task_audit.update({
                    "conditional_gain_vs_best_fixed": 0.1,
                    "member_evaluation_budget": limits[
                        "member_evaluation_cap"
                    ],
                    "trajectory_member_path_budget": limits[
                        "trajectory_member_path_cap"
                    ],
                })
            elif task == "world_model_contrastive_claim":
                limits = deliberation_compute_limits(allocation, task)
                task_audit.update({
                    "directionally_faithful": False,
                    "trajectory_member_path_budget": limits[
                        "trajectory_member_path_cap"
                    ],
                })
                plan["world_model_contrastive_repair_audit"] = {
                    "total_member_trajectory_path_budget": limits[
                        "repair_total_member_trajectory_path_cap"
                    ],
                }
            plan[audit_key] = task_audit
        plan["world_model_deliberation_encouragement_audit"] = {
            "version": 1,
            "accepted": True,
            "reason": "post_action_shadow_deliberation_accepted",
            "two_stage_active": True,
            "action_frozen": "pass",
            "action_mutation_ignored": False,
            "tactical_mutation_ignored": False,
            "action_preserved": True,
            "tactical_controls_preserved": True,
            "can_change_current_action": False,
            "can_change_tactical_controls": False,
            "can_relax_downstream_validators": False,
            "encouragement": experiment,
            "shadow_brief_digest": shadow["shadow_brief_digest"],
            "declared_focus_tasks": tasks,
        }
        focus = evaluate_llm_deliberation_focus(
            brief, plan, focus_signature="llm-deliberation-focus:test",
        )
        records.append({
            "created_t_sec": float(nonce + 1),
            "checkpoint_signature": "test-runtime",
            "environment_signature": "environment-unspecified",
            "llm_deliberation_focus_context": focus,
        })
        nonce += 1
    return records


def test_randomized_task_encouragement_learns_itt_value_and_strict_gate():
    records = _randomized_encouragement_records()
    diagnostics = task_encouragement_diagnostics([
        [record] for record in records
    ])
    assert diagnostics["randomized_encouragement_trials"] == 8
    assert diagnostics["matches"] == 8
    assert diagnostics["balanced_randomized_evidence"]
    assert diagnostics["overall_compliance_rate"] == 1.0
    assert not diagnostics["can_claim_match_outcome_causality"]

    memory = build_task_selection_value_memory(
        records,
        checkpoint_signature="test-runtime",
        environment_signature="environment-unspecified",
        as_of_t_sec=100.0,
        min_per_arm=4,
    )
    assert task_selection_value_memory_is_valid(memory)
    assert memory["status"] == (
        "validated_event_option_encouragement_value"
    )
    assert memory["event_option_minus_contrastive_itt"] > 0.0
    assert memory["intention_to_treat"]

    logs = [{
        "world_model_decision_adoption": {"records": [record]},
    } for record in records]
    report = aggregate_online_calibration(
        logs, min_transitions=0, require_llm_task_encouragement=True,
    )
    assert report["version"] == 43
    assert report["llm_task_encouragement_ready"]
    assert report["gates"]["llm_task_encouragement"]

    packet = _experimental_packet(999)
    packet["task_selection_value_memory"] = memory
    brief = build_llm_decision_brief(packet)
    rows = {
        row["task"]: row for row in brief["deliberation_agenda"]["tasks"]
    }
    assert rows["world_model_event_option"][
        "learned_task_selection_adjustment"
    ] > 0.0
    assert rows["world_model_contrastive_claim"][
        "learned_task_selection_adjustment"
    ] < 0.0

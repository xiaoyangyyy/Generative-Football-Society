"""World-model compute authority must bound LLM-selected deliberation."""

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
)
from src.match_engine.world_model.llm_deliberation_compute import (
    build_deliberation_compute_allocation,
    deliberation_compute_allocation_is_valid,
    deliberation_compute_limits,
)
from src.match_engine.world_model.llm_deliberation_focus import (
    TASK_CONTRACTS,
    evaluate_llm_deliberation_focus,
)
from src.match_engine.world_model.llm_deliberation_compute_value import (
    build_deliberation_compute_value_memory,
    deliberation_compute_value_diagnostics,
    deliberation_compute_value_memory_is_valid,
)
from src.match_engine.world_model.online_evaluation import (
    aggregate_online_calibration,
)
from tests.test_world_model_llm_fusion import _Runtime, _state


def _plan(tasks):
    return {
        "world_model_deliberation_focus": {
            "tasks": list(tasks),
            "confidence": 0.8,
            "rationale": "Spend compute only where model evidence is useful.",
        },
    }


def test_compute_allocation_is_bounded_deterministic_and_tamper_evident():
    packet = build_coach_decision_packet(_Runtime(), _state(), "Home")
    brief = build_llm_decision_brief(packet)
    cheap = [
        row["task"] for row in brief["deliberation_agenda"]["tasks"]
        if row["eligible"] and row["compute_cost_class"]
        == "existing_evidence_audit"
    ][:2]
    tasks = ["world_model_contrastive_claim", *cheap]
    allocation = build_deliberation_compute_allocation(brief, _plan(tasks))

    assert allocation["accepted"]
    contrastive = allocation["allocations"][
        "world_model_contrastive_claim"
    ]
    assigned = contrastive["compute_credits"]
    assert assigned in {2, 3}
    assert contrastive["planned_compute_credits"] == 3
    assert contrastive["compute_value_experiment"]["randomized"]
    assert allocation["allocated_compute_credits"] == assigned + 2
    assert allocation["unallocated_compute_credits"] == 4 - assigned
    assert sum(
        row["compute_credits"]
        for row in allocation["allocations"].values()
    ) == assigned + 2
    assert all(
        allocation["allocations"][task]["compute_credits"] == 1
        for task in cheap
    )
    assert deliberation_compute_allocation_is_valid(allocation)
    assert allocation == build_deliberation_compute_allocation(
        brief, _plan(tasks),
    )

    tampered = copy.deepcopy(allocation)
    tampered["allocations"][tasks[0]]["compute_credits"] = 99
    assert not deliberation_compute_allocation_is_valid(tampered)


def test_compute_allocation_rejects_an_ineligible_selected_task():
    packet = build_coach_decision_packet(_Runtime(), _state(), "Home")
    brief = build_llm_decision_brief(packet)
    ineligible = next(
        row["task"] for row in brief["deliberation_agenda"]["tasks"]
        if not row["eligible"]
    )
    allocation = build_deliberation_compute_allocation(
        brief, _plan([ineligible]),
    )
    assert not allocation["accepted"]
    assert allocation["reason"] == "selected_task_not_agenda_eligible"


def test_event_option_credits_map_to_hard_member_and_path_caps():
    packet = build_coach_decision_packet(_Runtime(), _state(), "Home")
    packet["second_order_game"]["trajectory_rollout"]["active"] = True
    brief = build_llm_decision_brief(packet)
    allocation = build_deliberation_compute_allocation(
        brief, _plan(["world_model_event_option"]),
    )
    limits = deliberation_compute_limits(
        allocation, "world_model_event_option",
    )
    credits = allocation["allocations"][
        "world_model_event_option"
    ]["compute_credits"]
    assert credits in {2, 3}
    assert allocation["allocated_compute_credits"] == credits
    assert allocation["unallocated_compute_credits"] == 6 - credits
    assert limits == {
        "member_evaluation_cap": {2: 16, 3: 32}[credits],
        "trajectory_member_path_cap": {2: 96, 3: 144}[credits],
    }


def test_focus_audit_rejects_a_valid_allocation_from_another_selection():
    packet = build_coach_decision_packet(_Runtime(), _state(), "Home")
    brief = build_llm_decision_brief(packet)
    tasks = brief["deliberation_agenda"]["recommended_focus"][:2]
    plan = _plan(tasks)
    for task in tasks:
        contract, audit = TASK_CONTRACTS[task]
        plan[contract] = {"placeholder": True}
        plan[audit] = {"accepted": True}
    foreign = build_deliberation_compute_allocation(
        brief, _plan(list(reversed(tasks))),
    )
    assert deliberation_compute_allocation_is_valid(foreign)
    plan["world_model_deliberation_compute_allocation"] = foreign

    audit = evaluate_llm_deliberation_focus(
        brief, plan, focus_signature="llm-deliberation-focus:test",
    )
    assert audit["compute_allocation_valid"]
    assert not audit["compute_allocation_matches_model"]
    assert not audit["compute_budget_respected"]
    assert not audit["model_checked_consistent"]


def test_executor_passes_signed_deep_caps_to_contrastive_compute(monkeypatch):
    captured = {}

    class _BudgetedLLM:
        model = "deliberation-compute-test"

        def coach_in_match_plan(self, team_name, facts, kind):
            return json.dumps({
                "reasoning": "Use a bounded contrastive probe.",
                "confidence": 0.8,
                "controls_delta": {},
                "world_model_action": "shot",
                "world_model_deliberation_focus": {
                    "tasks": ["world_model_contrastive_claim"],
                    "confidence": 0.8,
                    "rationale": "Test whether score context explains margin.",
                },
                "world_model_contrastive_claim": {
                    "selected_action": "shot",
                    "alternative_action": "hold",
                    "horizon": "transition",
                    "factor": "score_context",
                    "effect": "supports_selected",
                    "confidence": 0.8,
                    "rationale": "Neutralize score context and recompute.",
                },
            })

    def contrastive_stub(*args, **kwargs):
        captured["trajectory_cap"] = kwargs["max_member_trajectory_paths"]
        return {
            "accepted": True,
            "trajectory_member_paths": 32,
            "trajectory_member_path_budget": kwargs[
                "max_member_trajectory_paths"
            ],
            "directionally_faithful": True,
        }

    def repair_stub(*args, **kwargs):
        captured["repair_cap"] = kwargs[
            "total_member_trajectory_path_budget"
        ]
        return {
            "repair_successful": False,
            "total_member_trajectory_path_budget": kwargs[
                "total_member_trajectory_path_budget"
            ],
        }

    monkeypatch.setattr(
        "src.match_engine.world_model.contrastive_explanation."
        "evaluate_llm_contrastive_claim",
        contrastive_stub,
    )
    monkeypatch.setattr(
        "src.match_engine.world_model.contrastive_repair."
        "attempt_contrastive_explanation_repair",
        repair_stub,
    )
    state = _state()
    executor = CognitiveExecutor(
        CognitiveMatchConfig(enabled=True), _BudgetedLLM(),
        world_model_runtime=_Runtime(),
    )
    result = executor.process_trigger(CognitiveTriggerEvent(
        60.0, "xg_swing", ENTITY_TIER_COACH,
        "coach:Home", team_id="Home", salience=1.0,
    ), state)

    allocation = result.plan[
        "world_model_deliberation_compute_allocation"
    ]
    limits = deliberation_compute_limits(
        allocation, "world_model_contrastive_claim",
    )
    credits = allocation["allocations"][
        "world_model_contrastive_claim"
    ]["compute_credits"]
    assert allocation["allocations"][
        "world_model_contrastive_claim"
    ]["compute_tier"] == {2: "standard", 3: "deep"}[credits]
    assert captured["trajectory_cap"] == limits[
        "trajectory_member_path_cap"
    ] == {2: 32, 3: 64}[credits]
    assert captured["repair_cap"] == limits[
        "repair_total_member_trajectory_path_cap"
    ] == {2: 64, 3: 128}[credits]
    focus = result.plan["world_model_deliberation_focus_audit"]
    assert focus["compute_allocation_valid"]
    assert focus["compute_budget_respected"]
    assert not focus["compute_budget_mismatches"]


def test_randomized_prior_learns_value_without_current_record_leakage():
    base_packet = build_coach_decision_packet(_Runtime(), _state(), "Home")
    records = []
    arm_counts = {"control": 0, "treatment": 0}
    nonce = 0
    while min(arm_counts.values()) < 4:
        packet = copy.deepcopy(base_packet)
        packet["compute_value_test_nonce"] = nonce
        brief = build_llm_decision_brief(packet)
        plan = _plan(["world_model_contrastive_claim"])
        plan["world_model_contrastive_claim"] = {"placeholder": True}
        allocation = build_deliberation_compute_allocation(brief, plan)
        row = allocation["allocations"]["world_model_contrastive_claim"]
        arm = row["compute_value_experiment"]["arm"]
        if arm_counts[arm] >= 4:
            nonce += 1
            continue
        arm_counts[arm] += 1
        limits = row["resource_limits"]
        useful = arm == "treatment"
        plan["world_model_deliberation_compute_allocation"] = allocation
        plan["world_model_contrastive_explanation_audit"] = {
            "accepted": useful,
            "reason": (
                "accepted" if useful
                else "contrastive_trajectory_budget_insufficient"
            ),
            "directionally_faithful": useful,
            "trajectory_member_path_budget": limits[
                "trajectory_member_path_cap"
            ],
            "trajectory_member_paths_required": row[
                "planned_resource_limits"
            ]["trajectory_member_path_cap"],
        }
        plan["world_model_contrastive_repair_audit"] = {
            "total_member_trajectory_path_budget": limits[
                "repair_total_member_trajectory_path_cap"
            ],
        }
        focus = evaluate_llm_deliberation_focus(
            brief, plan, focus_signature="llm-deliberation-focus:test",
        )
        if arm == "control":
            assert focus["experimental_budget_rejections"] == [
                "world_model_contrastive_claim"
            ]
            assert not focus[
                "nonexperimental_rejected_selected_contracts"
            ]
        records.append({
            "created_t_sec": float(nonce + 1),
            "checkpoint_signature": "test-runtime",
            "environment_signature": "environment-unspecified",
            "llm_deliberation_focus_context": focus,
        })
        nonce += 1

    records.append({
        **copy.deepcopy(records[-1]),
        "created_t_sec": 100.0,
    })
    memory = build_deliberation_compute_value_memory(
        records,
        checkpoint_signature="test-runtime",
        environment_signature="environment-unspecified",
        as_of_t_sec=100.0,
        min_per_arm=4,
    )
    value = memory["task_values"]["world_model_contrastive_claim"]
    assert deliberation_compute_value_memory_is_valid(memory)
    assert value["status"] == "validated_positive_compute_value"
    assert value["control"]["samples"] == 4
    assert value["treatment"]["samples"] == 4
    assert memory["excluded_records"]["future_or_current"] == 1
    assert not value["can_claim_match_outcome_causality"]

    diagnostics = deliberation_compute_value_diagnostics(
        [[record] for record in records[:-1]]
    )
    assert diagnostics["randomized_trials"] == 8
    assert diagnostics["conclusive_randomized_trials"] == 8
    assert diagnostics["matches_with_randomized_trials"] == 8
    assert diagnostics["tasks_with_balanced_randomized_evidence"] == 1
    assert diagnostics["all_outcomes_model_internal_and_noncausal"]
    logs = [{
        "world_model_decision_adoption": {"records": [record]},
    } for record in records[:-1]]
    report = aggregate_online_calibration(
        logs, min_transitions=0,
        require_llm_deliberation_compute_value=True,
    )
    assert report["version"] == 46
    assert report["llm_deliberation_compute_value_ready"]
    assert report["gates"]["llm_deliberation_compute_value"]

    next_packet = copy.deepcopy(base_packet)
    next_packet["deliberation_compute_value_memory"] = memory
    next_brief = build_llm_decision_brief(next_packet)
    learned_task = next(
        row for row in next_brief["deliberation_agenda"]["tasks"]
        if row["task"] == "world_model_contrastive_claim"
    )
    assert learned_task["learned_compute_value_adjustment"] > 0.0
    assert learned_task["portfolio_score"] == (
        learned_task["priority"]
        + learned_task["learned_compute_value_adjustment"] - 0.03
    )
    learned = build_deliberation_compute_allocation(
        next_brief, _plan(["world_model_contrastive_claim"]),
    )
    learned_row = learned["allocations"][
        "world_model_contrastive_claim"
    ]
    assert learned_row["compute_value_status"] == (
        "validated_positive_compute_value"
    )
    assert learned_row["compute_credits"] == 3
    assert learned_row["compute_value_experiment"] == {
        "randomized": False,
        "arm": "learned_exploit",
        "treatment_probability": 1.0,
        "withheld_compute_credits": 0,
    }

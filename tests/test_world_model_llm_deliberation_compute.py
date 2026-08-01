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
    assert allocation["allocated_compute_credits"] == 5
    assert allocation["unallocated_compute_credits"] == 1
    assert sum(
        row["compute_credits"]
        for row in allocation["allocations"].values()
    ) == 5
    assert allocation["allocations"][
        "world_model_contrastive_claim"
    ]["compute_credits"] == 3
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
    assert allocation["allocated_compute_credits"] == 3
    assert allocation["unallocated_compute_credits"] == 3
    assert limits == {
        "member_evaluation_cap": 32,
        "trajectory_member_path_cap": 144,
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
    assert allocation["allocations"][
        "world_model_contrastive_claim"
    ]["compute_tier"] == "deep"
    assert captured["trajectory_cap"] == limits[
        "trajectory_member_path_cap"
    ] == 64
    assert captured["repair_cap"] == limits[
        "repair_total_member_trajectory_path_cap"
    ] == 128
    focus = result.plan["world_model_deliberation_focus_audit"]
    assert focus["compute_allocation_valid"]
    assert focus["compute_budget_respected"]
    assert not focus["compute_budget_mismatches"]

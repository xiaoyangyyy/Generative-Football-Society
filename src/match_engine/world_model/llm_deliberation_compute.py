"""World-model-owned compute allocation for LLM-selected reasoning tasks."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from src.match_engine.world_model.llm_decision_brief import (
    llm_decision_brief_self_is_valid,
)
from src.match_engine.world_model.llm_deliberation_focus import (
    validate_llm_deliberation_focus,
)


LLM_DELIBERATION_COMPUTE_VERSION = 1
TOTAL_COMPUTE_CREDITS = 6
MAX_TASK_COMPUTE_CREDITS = 3


def _digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return "llm-deliberation-compute:" + hashlib.sha256(
        encoded
    ).hexdigest()[:24]


def _resource_limits(task: str, credits: int) -> dict[str, int]:
    if task == "world_model_event_option":
        return {
            "member_evaluation_cap": {1: 8, 2: 16, 3: 32}[credits],
            "trajectory_member_path_cap": {1: 48, 2: 96, 3: 144}[
                credits
            ],
        }
    if task == "world_model_contrastive_claim":
        return {
            "trajectory_member_path_cap": {1: 16, 2: 32, 3: 64}[
                credits
            ],
            "repair_total_member_trajectory_path_cap": {
                1: 32, 2: 64, 3: 128,
            }[credits],
        }
    return {}


def _allocate_compute_credits(
    tasks: list[str], priorities: dict[str, float],
) -> tuple[dict[str, int], int]:
    credits = {task: 1 for task in tasks}
    remaining = TOTAL_COMPUTE_CREDITS - len(tasks)
    while remaining > 0:
        candidates = [
            task for task in tasks
            if (
                credits[task] < MAX_TASK_COMPUTE_CREDITS
                and _resource_limits(task, credits[task])
            )
        ]
        if not candidates:
            break
        chosen = min(candidates, key=lambda task: (
            -priorities[task] / (credits[task] + 1), task,
        ))
        credits[chosen] += 1
        remaining -= 1
    return credits, remaining


def build_deliberation_compute_allocation(
    brief: dict[str, Any], plan: dict[str, Any],
) -> dict[str, Any]:
    """Allocate a hard bounded budget after semantic task selection."""
    focus = validate_llm_deliberation_focus(
        plan.get("world_model_deliberation_focus")
    )
    base = {
        "version": LLM_DELIBERATION_COMPUTE_VERSION,
        "accepted": False,
        "reason": "compatible_focus_and_agenda_required",
        "compute_authority_active": False,
        "policy_authority_active": False,
        "can_change_current_action": False,
        "can_relax_downstream_validators": False,
    }
    if focus is None or not llm_decision_brief_self_is_valid(brief):
        return base
    agenda_rows = {
        str(row.get("task")): row
        for row in (brief.get("deliberation_agenda") or {}).get("tasks") or []
        if isinstance(row, dict)
    }
    tasks = list(focus["tasks"])
    if any(
        task not in agenda_rows or not agenda_rows[task].get("eligible")
        for task in tasks
    ):
        return {**base, "reason": "selected_task_not_agenda_eligible"}

    priorities = {
        task: float(agenda_rows[task]["priority"]) for task in tasks
    }
    credits, remaining = _allocate_compute_credits(tasks, priorities)

    allocations = {
        task: {
            "compute_credits": credits[task],
            "compute_tier": {1: "light", 2: "standard", 3: "deep"}[
                credits[task]
            ],
            "agenda_priority": float(agenda_rows[task]["priority"]),
            "resource_limits": _resource_limits(task, credits[task]),
        }
        for task in tasks
    }
    priority_weighted_compute_sum = sum(
        float(agenda_rows[task]["priority"]) * credits[task]
        for task in tasks
    )
    allocated_credits = sum(credits.values())
    payload = {
        "version": LLM_DELIBERATION_COMPUTE_VERSION,
        "accepted": True,
        "reason": "world_model_budget_allocated",
        "brief_digest": brief["brief_digest"],
        "focus_tasks": tasks,
        "total_compute_credit_budget": TOTAL_COMPUTE_CREDITS,
        "allocated_compute_credits": allocated_credits,
        "unallocated_compute_credits": remaining,
        "allocations": allocations,
        "allocation_policy": (
            "priority_weighted_diminishing_returns_for_real_resources"
        ),
        "priority_weighted_compute_sum": priority_weighted_compute_sum,
        "mean_priority_per_allocated_credit": (
            priority_weighted_compute_sum / allocated_credits
        ),
        "compute_authority_active": True,
        "policy_authority_active": False,
        "can_change_current_action": False,
        "can_relax_downstream_validators": False,
    }
    return {**payload, "allocation_digest": _digest(payload)}


def deliberation_compute_allocation_is_valid(allocation: Any) -> bool:
    if not isinstance(allocation, dict) or not allocation.get("accepted"):
        return False
    payload = dict(allocation)
    digest = payload.pop("allocation_digest", None)
    try:
        rows = payload["allocations"]
        tasks = list(payload["focus_tasks"])
        if (
            not isinstance(rows, dict)
            or any(not isinstance(task, str) for task in tasks)
        ):
            return False
        credits = int(payload["allocated_compute_credits"])
        unallocated = int(payload["unallocated_compute_credits"])
        computed = sum(int(row["compute_credits"]) for row in rows.values())
        weighted = sum(
            int(row["compute_credits"]) * float(row["agenda_priority"])
            for row in rows.values()
        )
        reported_weighted = float(payload["priority_weighted_compute_sum"])
        reported_mean = float(payload["mean_priority_per_allocated_credit"])
        total_budget = int(payload["total_compute_credit_budget"])
        expected_credits, expected_unallocated = _allocate_compute_credits(
            tasks, {
                task: float(rows[task]["agenda_priority"])
                for task in tasks
            },
        )
        expected = _digest(payload)
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return False

    def row_is_valid(task: str, row: Any) -> bool:
        if not isinstance(row, dict):
            return False
        try:
            row_credits = int(row["compute_credits"])
            priority = float(row["agenda_priority"])
            tier = {1: "light", 2: "standard", 3: "deep"}[row_credits]
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        return bool(
            math.isfinite(priority)
            and 0.0 <= priority <= 1.0
            and row.get("compute_tier") == tier
            and row.get("resource_limits") == _resource_limits(
                task, row_credits,
            )
        )

    return bool(
        digest == expected
        and payload.get("version") == LLM_DELIBERATION_COMPUTE_VERSION
        and 1 <= credits <= TOTAL_COMPUTE_CREDITS
        and 0 <= unallocated <= TOTAL_COMPUTE_CREDITS
        and credits + unallocated == TOTAL_COMPUTE_CREDITS
        and computed == credits
        and {
            task: int(row["compute_credits"])
            for task, row in rows.items()
        } == expected_credits
        and unallocated == expected_unallocated
        and total_budget == TOTAL_COMPUTE_CREDITS
        and payload.get("allocation_policy")
        == "priority_weighted_diminishing_returns_for_real_resources"
        and abs(reported_weighted - weighted) <= 1e-12
        and abs(reported_mean - weighted / credits) <= 1e-12
        and 1 <= len(tasks) <= 3
        and len(tasks) == len(set(tasks))
        and set(tasks) == set(rows)
        and all(row_is_valid(task, row) for task, row in rows.items())
        and payload.get("compute_authority_active") is True
        and payload.get("policy_authority_active") is False
        and payload.get("can_change_current_action") is False
        and payload.get("can_relax_downstream_validators") is False
    )


def deliberation_compute_limits(
    allocation: dict[str, Any], task: str,
) -> dict[str, int]:
    if not deliberation_compute_allocation_is_valid(allocation):
        return {}
    row = (allocation.get("allocations") or {}).get(task) or {}
    return {
        str(key): int(value)
        for key, value in (row.get("resource_limits") or {}).items()
    }

"""Recoverable fixed-budget world-model branch-time sensitivity workbench."""

from __future__ import annotations

import html
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from src.product.manager_future import validate_manager_future_context_shape
from src.product.match_plan import WorldModelForkSetPlan


FUTURE_SET_SCENARIO_STATUSES = {
    "descriptive_only_ineligible",
    "no_realized_action_divergence",
    "action_divergence_without_local_attribution",
    "local_action_divergence_with_descriptive_future_difference",
    "local_action_divergence_without_measured_future_difference",
}
MAX_SCENARIO_MECHANISM_EXAMPLES = 3
MECHANISM_WINDOW_SECONDS = (30, 120)
MECHANISM_WINDOW_METRICS_V1 = (
    "actions", "passes", "shots", "goals", "turnovers",
)
MECHANISM_WINDOW_METRICS = (
    "actions", "passes", "crosses", "shots", "goals", "turnovers",
)


def _identity(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _paths(workspace: Any, set_id: str) -> dict[str, Path]:
    root = Path(workspace.output_root) / "fork_sets" / set_id
    return {
        "root": root,
        "lease": root / "fork_set.lock",
        "protocol": root / "protocol.json",
        "progress": root / "progress.json",
        "result": root / "result.json",
        "dashboard": root / "index.html",
    }


def _resolve_comparison(workspace: Any, value: Any) -> Path:
    workspace_root = Path(workspace.root).resolve()
    matches_root = (Path(workspace.output_root) / "matches").resolve()
    candidate = Path(str(value or ""))
    resolved = (candidate if candidate.is_absolute() else workspace_root / candidate).resolve()
    try:
        resolved.relative_to(matches_root)
    except ValueError as exc:
        raise ValueError("fork-set comparison must stay inside Studio matches") from exc
    if not resolved.name.endswith(".comparison.json") or not resolved.is_file():
        raise ValueError("fork-set comparison artifact is unavailable")
    return resolved


def _bounded_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    number = float(value)
    return max(0, min(100_000, int(number))) if math.isfinite(number) else 0


def _safe_text(value: Any, maximum: int) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > maximum
        or any(ord(char) < 32 for char in normalized)
    ):
        return None
    return normalized


def _bounded_signed_count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        return None
    return max(-100_000, min(100_000, int(number)))


def _validate_mechanism_examples(
    examples: Any, *, changed_actions: int,
) -> list[dict[str, Any]]:
    if (
        not isinstance(examples, list)
        or len(examples) > MAX_SCENARIO_MECHANISM_EXAMPLES
        or len(examples) > changed_actions
    ):
        raise ValueError("future-set mechanism example budget is invalid")
    required_v1 = {
        "schema_version", "opportunity_identity", "team", "t_sec", "clock",
        "baseline_action", "treatment_action", "recommended_action",
        "directly_observed", "local_policy_attribution_eligible",
        "downstream_windows", "downstream_causal_attribution_authorized",
        "example_identity",
    }
    required_v2 = required_v1 | {
        "probability_policy_version", "signal_mode",
        "primary_signal_action", "hold_reference_redistributed",
        "hold_reference_probability_delta",
    }
    window_required = {
        "schema_version", "window_sec", "delta",
        "additional_policy_changes", "causal_effect_authorized",
        "window_identity",
    }
    validated = []
    prior_t_sec: float | None = None
    seen_opportunities: set[str] = set()
    for raw in examples:
        if not isinstance(raw, Mapping):
            raise ValueError("future-set mechanism example fields are invalid")
        example_schema = raw.get("schema_version")
        expected_fields = required_v2 if example_schema == 2 else required_v1
        if set(raw) != expected_fields:
            raise ValueError("future-set mechanism example fields are invalid")
        t_sec = raw.get("t_sec")
        team = _safe_text(raw.get("team"), 80)
        baseline = _safe_text(raw.get("baseline_action"), 40)
        treatment = _safe_text(raw.get("treatment_action"), 40)
        recommended = raw.get("recommended_action")
        if recommended is not None:
            recommended = _safe_text(recommended, 40)
        opportunity_identity = raw.get("opportunity_identity")
        clock = raw.get("clock")
        signal_mode = raw.get("signal_mode")
        primary_signal = raw.get("primary_signal_action")
        policy_version = raw.get("probability_policy_version")
        hold_redistributed = raw.get("hold_reference_redistributed")
        hold_delta = raw.get("hold_reference_probability_delta")
        if (
            example_schema not in {1, 2}
            or not isinstance(opportunity_identity, str)
            or re.fullmatch(r"[0-9a-f]{64}", opportunity_identity) is None
            or team is None
            or isinstance(t_sec, bool)
            or not isinstance(t_sec, (int, float))
            or not math.isfinite(float(t_sec))
            or not 0 <= float(t_sec) <= 8000
            or clock != (
                f"{int(float(t_sec) // 60)}:"
                f"{int(float(t_sec) % 60):02d}"
            )
            or baseline is None
            or treatment is None
            or baseline == treatment
            or (
                raw.get("recommended_action") is not None
                and recommended is None
            )
            or not isinstance(raw.get("directly_observed"), bool)
            or not isinstance(
                raw.get("local_policy_attribution_eligible"), bool,
            )
            or (
                raw.get("local_policy_attribution_eligible")
                and not raw.get("directly_observed")
            )
            or raw.get("downstream_causal_attribution_authorized") is not False
            or (
                example_schema == 2
                and (
                    _safe_text(policy_version, 80) is None
                    or signal_mode not in {
                        "direct_preference", "suppression_only", "none",
                        "legacy_unclassified",
                    }
                    or primary_signal not in {
                        "hold", "pass", "cross", "shot", "none",
                    }
                    or (
                        signal_mode == "direct_preference"
                        and (
                            recommended not in {"pass", "cross", "shot"}
                            or primary_signal not in {"pass", "cross", "shot"}
                        )
                    )
                    or (
                        signal_mode == "suppression_only"
                        and (
                            recommended not in {None, "none"}
                            or primary_signal not in {"pass", "cross", "shot"}
                        )
                    )
                    or (
                        signal_mode == "none"
                        and (
                            recommended not in {None, "none"}
                            or primary_signal != "none"
                        )
                    )
                    or not isinstance(hold_redistributed, bool)
                    or (
                        hold_delta is not None
                        and (
                            isinstance(hold_delta, bool)
                            or not isinstance(hold_delta, (int, float))
                            or not math.isfinite(float(hold_delta))
                            or abs(float(hold_delta)) > 1.0
                        )
                    )
                    or (
                        hold_redistributed
                        and (hold_delta is None or float(hold_delta) <= 0.0)
                    )
                )
            )
        ):
            raise ValueError("future-set mechanism example values are invalid")
        normalized_t_sec = float(t_sec)
        if (
            opportunity_identity in seen_opportunities
            or (
                prior_t_sec is not None
                and normalized_t_sec < prior_t_sec
            )
        ):
            raise ValueError("future-set mechanism examples are not ordered")
        seen_opportunities.add(opportunity_identity)
        prior_t_sec = normalized_t_sec
        windows = raw.get("downstream_windows")
        if not isinstance(windows, list) or len(windows) not in {0, 2}:
            raise ValueError("future-set mechanism windows are invalid")
        expected_windows = (
            list(MECHANISM_WINDOW_SECONDS) if windows else []
        )
        for index, window in enumerate(windows):
            if not isinstance(window, Mapping) or set(window) != window_required:
                raise ValueError("future-set mechanism window fields are invalid")
            delta = window.get("delta")
            expected_metrics = (
                MECHANISM_WINDOW_METRICS
                if example_schema == 2 else MECHANISM_WINDOW_METRICS_V1
            )
            if (
                window.get("schema_version") != example_schema
                or window.get("window_sec") != expected_windows[index]
                or not isinstance(delta, Mapping)
                or set(delta) != set(expected_metrics)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or not -100_000 <= value <= 100_000
                    for value in delta.values()
                )
                or isinstance(window.get("additional_policy_changes"), bool)
                or not isinstance(window.get("additional_policy_changes"), int)
                or not 0 <= window["additional_policy_changes"] <= 100_000
                or window.get("causal_effect_authorized") is not False
            ):
                raise ValueError("future-set mechanism window values are invalid")
            frozen_window = dict(window)
            observed_window = frozen_window.pop("window_identity", None)
            if observed_window != _identity(frozen_window):
                raise ValueError("future-set mechanism window identity mismatch")
        frozen = dict(raw)
        observed = frozen.pop("example_identity", None)
        if observed != _identity(frozen):
            raise ValueError("future-set mechanism example identity mismatch")
        validated.append(dict(raw))
    return validated


def _project_mechanism_examples(
    propagation: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    decisions = propagation.get("decisions")
    summary = propagation.get("summary")
    if not isinstance(decisions, list) or not isinstance(summary, Mapping):
        raise ValueError("future-set mechanism source is invalid")
    examples = []
    for raw in decisions[:MAX_SCENARIO_MECHANISM_EXAMPLES]:
        if not isinstance(raw, Mapping):
            raise ValueError("future-set mechanism decision is invalid")
        opportunity = _safe_text(raw.get("opportunity_id"), 160)
        team = _safe_text(raw.get("team"), 80)
        baseline = _safe_text(raw.get("baseline_action"), 40)
        treatment = _safe_text(raw.get("treatment_action"), 40)
        recommended = raw.get("recommended_action")
        recommended = (
            _safe_text(recommended, 40) if recommended is not None else None
        )
        t_sec = raw.get("t_sec")
        policy_version = (
            _safe_text(raw.get("probability_policy_version"), 80)
            or "legacy_unversioned"
        )
        signal_mode = (
            _safe_text(raw.get("signal_mode"), 40)
            or "legacy_unclassified"
        )
        primary_signal = (
            _safe_text(raw.get("primary_signal_action"), 40)
            or recommended or "none"
        )
        hold_redistributed = (
            raw.get("hold_reference_redistributed") is True
        )
        hold_delta = raw.get("hold_reference_probability_delta")
        hold_delta = (
            float(hold_delta)
            if (
                not isinstance(hold_delta, bool)
                and isinstance(hold_delta, (int, float))
                and math.isfinite(float(hold_delta))
                and abs(float(hold_delta)) <= 1.0
            )
            else None
        )
        if (
            opportunity is None
            or team is None
            or baseline is None
            or treatment is None
            or baseline == treatment
            or (
                raw.get("recommended_action") is not None
                and recommended is None
            )
            or isinstance(t_sec, bool)
            or not isinstance(t_sec, (int, float))
            or not math.isfinite(float(t_sec))
            or not 0 <= float(t_sec) <= 8000
            or signal_mode not in {
                "direct_preference", "suppression_only", "none",
                "legacy_unclassified",
            }
            or primary_signal not in {
                "hold", "pass", "cross", "shot", "none",
            }
            or (
                signal_mode == "direct_preference"
                and (
                    recommended not in {"pass", "cross", "shot"}
                    or primary_signal not in {"pass", "cross", "shot"}
                )
            )
            or (
                signal_mode == "suppression_only"
                and (
                    recommended not in {None, "none"}
                    or primary_signal not in {"pass", "cross", "shot"}
                )
            )
            or (
                signal_mode == "none"
                and (
                    recommended not in {None, "none"}
                    or primary_signal != "none"
                )
            )
            or (
                hold_redistributed
                and (hold_delta is None or hold_delta <= 0.0)
            )
        ):
            raise ValueError("future-set mechanism decision values are invalid")
        windows_source = raw.get("downstream_windows")
        windows = []
        if windows_source:
            if not isinstance(windows_source, Mapping):
                raise ValueError("future-set mechanism window source is invalid")
            for duration in MECHANISM_WINDOW_SECONDS:
                source = windows_source.get(f"{duration}s")
                delta_source = (
                    source.get("delta") if isinstance(source, Mapping) else None
                )
                if not isinstance(delta_source, Mapping):
                    raise ValueError("future-set mechanism window delta is invalid")
                delta = {}
                for metric in MECHANISM_WINDOW_METRICS:
                    value = _bounded_signed_count(delta_source.get(metric))
                    if value is None:
                        raise ValueError(
                            "future-set mechanism window metric is invalid"
                        )
                    delta[metric] = value
                raw_additional = source.get("additional_policy_changes")
                if (
                    isinstance(raw_additional, bool)
                    or not isinstance(raw_additional, (int, float))
                    or not math.isfinite(float(raw_additional))
                    or not float(raw_additional).is_integer()
                    or float(raw_additional) < 0
                ):
                    raise ValueError(
                        "future-set mechanism additional changes are invalid"
                    )
                additional = min(100_000, int(raw_additional))
                window_payload = {
                    "schema_version": 2,
                    "window_sec": duration,
                    "delta": delta,
                    "additional_policy_changes": additional,
                    "causal_effect_authorized": False,
                }
                windows.append({
                    **window_payload,
                    "window_identity": _identity(window_payload),
                })
        payload = {
            "schema_version": 2,
            "opportunity_identity": hashlib.sha256(
                opportunity.encode("utf-8")
            ).hexdigest(),
            "team": team,
            "t_sec": float(t_sec),
            "clock": (
                f"{int(float(t_sec) // 60)}:"
                f"{int(float(t_sec) % 60):02d}"
            ),
            "baseline_action": baseline,
            "treatment_action": treatment,
            "recommended_action": recommended,
            "probability_policy_version": policy_version,
            "signal_mode": signal_mode,
            "primary_signal_action": primary_signal,
            "hold_reference_redistributed": hold_redistributed,
            "hold_reference_probability_delta": hold_delta,
            "directly_observed": raw.get("directly_observed") is True,
            "local_policy_attribution_eligible": (
                raw.get("local_policy_attribution_eligible") is True
            ),
            "downstream_windows": windows,
            "downstream_causal_attribution_authorized": False,
        }
        examples.append({
            **payload, "example_identity": _identity(payload),
        })
    changed = _bounded_count(summary.get("valid_changed_decisions"))
    _validate_mechanism_examples(examples, changed_actions=changed)
    return examples, bool(
        summary.get("decisions_truncated") is True
        or changed > len(examples)
    )


def validate_fork_set_scenario_evidence(
    scenarios: Any, branch_times_sec: Any,
) -> list[dict[str, Any]]:
    """Replay the bounded per-timepoint evidence exposed to product surfaces."""
    if (
        not isinstance(branch_times_sec, (list, tuple))
        or not 2 <= len(branch_times_sec) <= 4
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0 <= float(value) <= 5400
            or (
                index > 0
                and float(value) <= float(branch_times_sec[index - 1])
            )
            for index, value in enumerate(branch_times_sec)
        )
    ):
        raise ValueError("future-set scenario branch contract is invalid")
    expected_times = [float(value) for value in branch_times_sec]
    if (
        not isinstance(scenarios, list)
        or len(scenarios) != len(expected_times)
    ):
        raise ValueError("future-set scenario evidence budget is invalid")
    base_required = {
        "schema_version", "branch_at_sec", "branch_minute",
        "future_status", "eligible", "anchor_verified",
        "branch_state_identity",
        "changed_actions", "locally_attributable_changes",
        "descriptive_future_difference_count",
        "simulator_local_action_attribution",
        "outcome_causality_authorized", "real_football_causality_authorized",
        "scenario_identity",
    }
    normalized = []
    for index, raw in enumerate(scenarios):
        schema_version = raw.get("schema_version") if isinstance(
            raw, Mapping,
        ) else None
        required = (
            base_required | {
                "mechanism_examples", "mechanism_examples_truncated",
            }
            if schema_version == 2 else base_required
        )
        if (
            not isinstance(raw, Mapping)
            or schema_version not in {1, 2}
            or set(raw) != required
        ):
            raise ValueError("future-set scenario evidence fields are invalid")
        branch = raw.get("branch_at_sec")
        minute = raw.get("branch_minute")
        state_identity = raw.get("branch_state_identity")
        if (
            isinstance(branch, bool)
            or not isinstance(branch, (int, float))
            or not math.isfinite(float(branch))
            or float(branch) != expected_times[index]
            or isinstance(minute, bool)
            or not isinstance(minute, (int, float))
            or not math.isclose(
                float(minute), float(branch) / 60.0,
                rel_tol=0.0, abs_tol=1e-9,
            )
            or raw.get("future_status") not in FUTURE_SET_SCENARIO_STATUSES
            or not isinstance(raw.get("eligible"), bool)
            or not isinstance(raw.get("anchor_verified"), bool)
            or (
                raw.get("eligible") is True
                and raw.get("anchor_verified") is not True
            )
            or (
                state_identity is not None
                and re.fullmatch(r"[0-9a-f]{64}", str(state_identity)) is None
            )
            or (
                raw.get("anchor_verified") is True
                and state_identity is None
            )
            or not isinstance(
                raw.get("simulator_local_action_attribution"), bool,
            )
            or raw.get("outcome_causality_authorized") is not False
            or raw.get("real_football_causality_authorized") is not False
        ):
            raise ValueError("future-set scenario evidence values are invalid")
        for field in (
            "changed_actions", "locally_attributable_changes",
            "descriptive_future_difference_count",
        ):
            value = raw.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= 100_000
            ):
                raise ValueError("future-set scenario evidence counts are invalid")
        changed = int(raw["changed_actions"])
        local = int(raw["locally_attributable_changes"])
        differences = int(raw["descriptive_future_difference_count"])
        local_authorized = raw["simulator_local_action_attribution"]
        status = raw["future_status"]
        semantic_status = (
            "descriptive_only_ineligible"
            if not raw["eligible"] else
            "no_realized_action_divergence"
            if changed == 0 else
            "action_divergence_without_local_attribution"
            if not local_authorized else
            "local_action_divergence_with_descriptive_future_difference"
            if differences > 0 else
            "local_action_divergence_without_measured_future_difference"
        )
        if (
            local > changed
            or (local_authorized and local == 0)
            or (not raw["eligible"] and local_authorized)
            or status != semantic_status
        ):
            raise ValueError("future-set scenario evidence semantics are invalid")
        if schema_version == 2:
            examples = _validate_mechanism_examples(
                raw.get("mechanism_examples"), changed_actions=changed,
            )
            truncated = raw.get("mechanism_examples_truncated")
            if (
                not isinstance(truncated, bool)
                or truncated is not (changed > len(examples))
                or sum(
                    row["local_policy_attribution_eligible"]
                    for row in examples
                ) > local
                or (
                    local_authorized
                    and not truncated
                    and local > 0
                    and not any(
                        row["local_policy_attribution_eligible"]
                        for row in examples
                    )
                )
            ):
                raise ValueError(
                    "future-set mechanism example summary is invalid"
                )
        frozen = dict(raw)
        observed = frozen.pop("scenario_identity")
        if (
            not isinstance(observed, str)
            or observed != _identity(frozen)
        ):
            raise ValueError("future-set scenario evidence identity mismatch")
        normalized.append(dict(raw))
    return normalized


def summarize_fork_set_scenario_evidence(
    scenarios: Any, branch_times_sec: Any,
) -> dict[str, Any]:
    """Rebuild aggregate facts from the bounded scenario evidence."""
    validated = validate_fork_set_scenario_evidence(
        scenarios, branch_times_sec,
    )
    signatures = {
        (
            row["future_status"], row["changed_actions"],
            row["locally_attributable_changes"],
            row["descriptive_future_difference_count"],
        )
        for row in validated
    }
    return {
        "eligible_scenarios": sum(
            row["eligible"] for row in validated
        ),
        "verified_anchor_scenarios": sum(
            row["anchor_verified"] for row in validated
        ),
        "action_divergence_scenarios": sum(
            row["changed_actions"] > 0 for row in validated
        ),
        "local_attribution_scenarios": sum(
            row["simulator_local_action_attribution"]
            for row in validated
        ),
        "descriptive_future_difference_scenarios": sum(
            row["descriptive_future_difference_count"] > 0
            for row in validated
        ),
        "timing_sensitivity_observed": len(signatures) > 1,
        "status_counts": dict(sorted(Counter(
            row["future_status"] for row in validated
        ).items())),
    }


def project_fork_set_scenario_evidence(
    result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build a privacy-safe, non-ranked product projection from official rows."""
    plan = result.get("plan")
    rows = result.get("rows")
    branch_times = (
        plan.get("branch_times_sec")
        if isinstance(plan, Mapping) else None
    )
    if (
        not isinstance(rows, list)
        or not isinstance(branch_times, list)
        or len(rows) != len(branch_times)
    ):
        raise ValueError("future-set scenario source is invalid")
    scenarios = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("future-set scenario source row is invalid")
        raw_branch = row.get("branch_at_sec")
        raw_minute = row.get("branch_minute")
        if (
            isinstance(raw_branch, bool)
            or not isinstance(raw_branch, (int, float))
            or not math.isfinite(float(raw_branch))
            or isinstance(raw_minute, bool)
            or not isinstance(raw_minute, (int, float))
            or not math.isfinite(float(raw_minute))
        ):
            raise ValueError("future-set scenario source time is invalid")
        raw_state_identity = str(row.get("branch_state_identity") or "")
        state_identity = (
            raw_state_identity
            if re.fullmatch(r"[0-9a-f]{64}", raw_state_identity)
            else None
        )
        payload = {
            "schema_version": 1,
            "branch_at_sec": float(raw_branch),
            "branch_minute": float(raw_minute),
            "future_status": str(row.get("future_status") or ""),
            "eligible": row.get("eligible") is True,
            "anchor_verified": row.get("branch_anchor_verified") is True,
            "branch_state_identity": state_identity,
            "changed_actions": _bounded_count(row.get("changed_actions")),
            "locally_attributable_changes": _bounded_count(
                row.get("locally_attributable_changes")
            ),
            "descriptive_future_difference_count": _bounded_count(
                row.get("descriptive_future_difference_count")
            ),
            "simulator_local_action_attribution": (
                row.get("simulator_local_action_attribution") is True
            ),
            "outcome_causality_authorized": False,
            "real_football_causality_authorized": False,
        }
        has_examples = "mechanism_examples" in row
        has_truncation = "mechanism_examples_truncated" in row
        if has_examples is not has_truncation:
            raise ValueError(
                "future-set scenario mechanism source is incomplete"
            )
        if has_examples:
            examples = _validate_mechanism_examples(
                row.get("mechanism_examples"),
                changed_actions=payload["changed_actions"],
            )
            truncated = row.get("mechanism_examples_truncated")
            if not isinstance(truncated, bool):
                raise ValueError(
                    "future-set scenario mechanism truncation is invalid"
                )
            payload.update({
                "schema_version": 2,
                "mechanism_examples": examples,
                "mechanism_examples_truncated": truncated,
            })
        scenarios.append({
            **payload, "scenario_identity": _identity(payload),
        })
    return validate_fork_set_scenario_evidence(scenarios, branch_times)


def _row(plan: WorldModelForkSetPlan, comparison: Mapping[str, Any]) -> dict[str, Any]:
    fixture = comparison.get("fixture") or {}
    intervention = comparison.get("intervention") or {}
    anchor = intervention.get("branch_anchor") or {}
    eligibility = comparison.get("eligibility") or {}
    propagation = comparison.get("policy_propagation") or {}
    summary = propagation.get("summary") or {}
    future = comparison.get("counterfactual_future_summary") or {}
    future_summary = future.get("summary") or {}
    authority = future.get("claim_authority") or {}
    branch = intervention.get("branch_at_sec")
    if isinstance(branch, bool) or not isinstance(branch, (int, float)):
        raise ValueError("comparison branch time is invalid")
    if branch not in plan.branch_times_sec:
        raise ValueError("comparison branch time is outside frozen fork set")
    if (
        fixture.get("seed") != plan.seed
        or intervention.get("scope") != "world_model_action_policy"
        or intervention.get("changed_sides") != []
        or intervention.get("baseline_policy") != "predict_only"
        or intervention.get("treatment_policy") != "action_policy"
        or intervention.get("baseline_tactics") != {
            "home": plan.home_tactic, "away": plan.away_tactic,
        }
        or intervention.get("treatment_tactics") != {
            "home": plan.home_tactic, "away": plan.away_tactic,
        }
    ):
        raise ValueError("comparison is not the frozen world-model policy contrast")
    changed = _bounded_count(summary.get("valid_changed_decisions"))
    local = _bounded_count(summary.get("locally_attributable_changes"))
    differences = _bounded_count(
        future_summary.get("descriptive_outcome_difference_count")
    )
    eligible = bool(
        propagation.get("available") is True
        and future.get("available") is True
        and anchor.get("verified") is True
        and eligibility.get(
            "eligible_for_world_model_policy_attribution"
        ) is True
        and re.fullmatch(
            r"[0-9a-f]{64}", str(anchor.get("state_identity") or "")
        ) is not None
    )
    local_authorized = bool(
        eligible and local > 0
        and authority.get("simulator_local_action_attribution") is True
    )
    status = (
        "descriptive_only_ineligible" if not eligible else
        "no_realized_action_divergence" if changed == 0 else
        "action_divergence_without_local_attribution"
        if not local_authorized else
        "local_action_divergence_with_descriptive_future_difference"
        if differences > 0 else
        "local_action_divergence_without_measured_future_difference"
    )
    row = {
        "branch_at_sec": float(branch),
        "branch_minute": float(branch) / 60.0,
        "eligible": eligible,
        "branch_anchor_verified": anchor.get("verified") is True,
        "branch_state_identity": str(anchor.get("state_identity") or "")[:64],
        "future_status": status,
        "changed_actions": changed,
        "locally_attributable_changes": local,
        "descriptive_future_difference_count": differences,
        "simulator_local_action_attribution": local_authorized,
        "outcome_causality": False,
        "real_football_causality": False,
    }
    if (
        isinstance(propagation.get("decisions"), list)
        and isinstance(propagation.get("summary"), Mapping)
    ):
        examples, truncated = _project_mechanism_examples(propagation)
        row.update({
            "mechanism_examples": examples,
            "mechanism_examples_truncated": truncated,
        })
    return row


def aggregate_fork_set(
    plan: WorldModelForkSetPlan,
    comparisons: list[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(comparisons) != len(plan.branch_times_sec):
        raise ValueError("fork-set aggregation requires the complete fixed budget")
    rows = [_row(plan, comparison) for comparison in comparisons]
    if [row["branch_at_sec"] for row in rows] != list(plan.branch_times_sec):
        raise ValueError("fork-set comparisons do not match frozen branch order")
    status_counts = Counter(row["future_status"] for row in rows)
    signatures = {
        (
            row["future_status"], row["changed_actions"],
            row["locally_attributable_changes"],
            row["descriptive_future_difference_count"],
        )
        for row in rows
    }
    return {
        "rows": rows,
        "aggregate": {
            "eligible_scenarios": sum(row["eligible"] for row in rows),
            "verified_anchor_scenarios": sum(
                row["branch_anchor_verified"] for row in rows
            ),
            "action_divergence_scenarios": sum(
                row["changed_actions"] > 0 for row in rows
            ),
            "local_attribution_scenarios": sum(
                row["simulator_local_action_attribution"] for row in rows
            ),
            "descriptive_future_difference_scenarios": sum(
                row["descriptive_future_difference_count"] > 0 for row in rows
            ),
            "timing_sensitivity_observed": len(signatures) > 1,
            "status_counts": dict(sorted(status_counts.items())),
            "ranking_performed": False,
            "best_branch_time": None,
        },
        "claim_authority": {
            "descriptive_simulator_timing_sensitivity": True,
            "best_time_recommendation": False,
            "match_outcome_causality": False,
            "population_inference": False,
            "real_football_causality": False,
            "promotion_authorized": False,
        },
    }


def render_fork_set_html(result: Mapping[str, Any]) -> str:
    rows = "".join(
        "<tr><td>{:.1f}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>"
        "<td><a href='{}'>查看双世界证据</a></td></tr>".format(
            float(row.get("branch_minute", 0.0)),
            "已验证" if row.get("branch_anchor_verified") else "未验证",
            html.escape(str(row.get("future_status") or "unknown")),
            int(row.get("changed_actions", 0)),
            int(row.get("descriptive_future_difference_count", 0)),
            html.escape(str((row.get("evidence") or {}).get("comparison_dashboard") or "#"), quote=True),
        )
        for row in result.get("rows") or []
    )
    aggregate = result.get("aggregate") or {}
    fixture = result.get("fixture") or {}
    boundary = html.escape(str(result.get("claim_boundary") or ""))
    source = result.get("source_context") or {}
    manager_binding = (
        "<section><h2>经理决策绑定</h2><p>赛季 {} · 轮次 {} · 决策上下文 {}</p>"
        "<p>本报告复用该冻结决策的正式对阵、seed 与双方实际战术；"
        "若赛季 revision 改变，任务会失败关闭。</p></section>"
    ).format(
        html.escape(str(source.get("season_id") or "")),
        html.escape(str((source.get("fixture") or {}).get("matchday") or "")),
        html.escape(str(source.get("context_identity") or "")[:16]),
    ) if source.get("kind") == "manager_prematch_world_model_future_set" else ""
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GFS 多时点未来分叉</title><style>:root{{color-scheme:dark}}*{{box-sizing:border-box}}body{{max-width:1100px;margin:auto;padding:32px 20px;background:#07111e;color:#edf4ff;font:15px/1.6 system-ui}}section{{background:#111d2e;border:1px solid #2a3a51;border-radius:14px;padding:18px;margin:16px 0}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:9px;border-bottom:1px solid #2a3a51}}a{{color:#65e6b4}}.muted{{color:#a8b6c9}}a:focus-visible{{outline:3px solid #ffc36a}}</style></head><body><p class="muted">Football Causal World Lab</p><h1>{html.escape(str(fixture.get('home') or ''))} vs {html.escape(str(fixture.get('away') or ''))} · 多时点未来分叉</h1>{manager_binding}<section><h2>固定预算已完成</h2><p>{int(result.get('scenarios_completed', 0))} / {int(result.get('fixed_scenario_budget', 0))} 个预注册时点；动作分叉 {int(aggregate.get('action_divergence_scenarios', 0))} 个，局部归因 {int(aggregate.get('local_attribution_scenarios', 0))} 个，未来描述差异 {int(aggregate.get('descriptive_future_difference_scenarios', 0))} 个。</p><p>时间敏感性：{'已观察到' if aggregate.get('timing_sensitivity_observed') else '在已测指标中未观察到'}。系统没有挑选最佳时点。</p></section><section><h2>同一控制条件下的未来集</h2><table><thead><tr><th>分叉分钟</th><th>前缀锚点</th><th>证据状态</th><th>动作变化</th><th>后续描述差异</th><th>证据</th></tr></thead><tbody>{rows}</tbody></table></section><section><h2>结论边界</h2><p>{boundary}</p><p>这是模拟器内时间敏感性描述，不授权赛果因果、总体推断、现实足球因果或产品/论文晋级。</p></section></body></html>"""


def _validate_completed_result(
    plan: WorldModelForkSetPlan, set_id: str, result: Mapping[str, Any],
    *, home: str, away: str, source_context: Mapping[str, Any] | None,
) -> None:
    rows = result.get("rows") or []
    aggregate = result.get("aggregate") or {}
    authority = result.get("claim_authority") or {}
    if (
        result.get("schema_version") != 1
        or result.get("status") != "complete"
        or result.get("set_id") != set_id
        or result.get("fixture") != {"home": home, "away": away}
        or result.get("source_context") != (
            dict(source_context) if source_context is not None else None
        )
        or result.get("plan") != plan.as_dict()
        or result.get("fixed_scenario_budget") != len(plan.branch_times_sec)
        or result.get("scenarios_completed") != len(plan.branch_times_sec)
        or result.get("interim_ranking_disclosed") is not False
        or not isinstance(rows, list)
        or len(rows) != len(plan.branch_times_sec)
        or not isinstance(aggregate, Mapping)
        or aggregate.get("ranking_performed") is not False
        or aggregate.get("best_branch_time") is not None
        or not isinstance(authority, Mapping)
        or authority.get("best_time_recommendation") is not False
        or authority.get("match_outcome_causality") is not False
        or authority.get("population_inference") is not False
        or authority.get("real_football_causality") is not False
        or authority.get("promotion_authorized") is not False
    ):
        raise ValueError("completed fork-set result identity is invalid")
    if [row.get("branch_at_sec") for row in rows if isinstance(row, Mapping)] != list(
        plan.branch_times_sec
    ):
        raise ValueError("completed fork-set branch identity is invalid")
    if any(
        not isinstance(row, Mapping)
        or row.get("outcome_causality") is not False
        or row.get("real_football_causality") is not False
        for row in rows
    ):
        raise ValueError("completed fork-set claim boundary is invalid")
    scenarios = project_fork_set_scenario_evidence(result)
    rebuilt = summarize_fork_set_scenario_evidence(
        scenarios, plan.branch_times_sec,
    )
    if any(aggregate.get(key) != value for key, value in rebuilt.items()):
        raise ValueError("completed fork-set scenario aggregate mismatch")


def execute_world_model_fork_set(
    workspace: Any, plan: WorldModelForkSetPlan, *, set_id: str,
    home: str, away: str, fast: bool,
    source_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute or resume all preregistered branch times without interim ranking."""
    if getattr(getattr(workspace, "config", None), "mode", None) != "research":
        raise ValueError("world-model fork sets require research mode")
    if not set_id.isalnum() or len(set_id) > 64:
        raise ValueError("invalid fork-set identity")
    if source_context is not None:
        source_context = validate_manager_future_context_shape(source_context)
        if (
            source_context["fixture"]["home"] != home
            or source_context["fixture"]["away"] != away
            or source_context["match_seed"] != plan.seed
            or source_context["fast"] is not fast
            or source_context["home_tactic"] != plan.home_tactic
            or source_context["away_tactic"] != plan.away_tactic
        ):
            raise ValueError("future-set source context does not match controls")
    from src.infrastructure import FileLease
    from src.product.workspace import _atomic_json

    paths = _paths(workspace, set_id)
    paths["root"].mkdir(parents=True, exist_ok=True)
    frozen = {
        "schema_version": 1,
        "set_id": set_id,
        "fixture": {"home": home, "away": away},
        "fast": fast,
        "plan": plan.as_dict(),
        "source_context": (
            dict(source_context) if source_context is not None else None
        ),
    }
    with FileLease(paths["lease"], timeout=5.0):
        if paths["protocol"].is_file():
            if json.loads(paths["protocol"].read_text(encoding="utf-8")) != frozen:
                raise ValueError("fork-set identity belongs to a different frozen plan")
        else:
            _atomic_json(paths["protocol"], frozen)
        if paths["result"].is_file():
            result = json.loads(paths["result"].read_text(encoding="utf-8"))
            if not isinstance(result, Mapping):
                raise ValueError("completed fork-set result must be an object")
            _validate_completed_result(
                plan, set_id, result, home=home, away=away,
                source_context=source_context,
            )
            if not paths["dashboard"].is_file():
                paths["dashboard"].write_text(render_fork_set_html(result), encoding="utf-8")
            return {**result, "result_path": str(paths["result"]), "dashboard_path": str(paths["dashboard"])}
        progress = {
            "schema_version": 1,
            "set_id": set_id,
            "state": "running",
            "fixed_scenario_budget": len(plan.branch_times_sec),
            "scenarios_completed": 0,
            "completed": [],
            "interim_ranking_disclosed": False,
            "analysis": None,
        }
        if paths["progress"].is_file():
            loaded = json.loads(paths["progress"].read_text(encoding="utf-8"))
            if loaded.get("set_id") != set_id:
                raise ValueError("invalid fork-set progress identity")
            progress["completed"] = list(loaded.get("completed") or [])
        observed = [
            item.get("branch_at_sec") if isinstance(item, Mapping) else None
            for item in progress["completed"]
        ]
        if observed != list(plan.branch_times_sec[:len(observed)]):
            raise ValueError("fork-set progress is not a valid frozen prefix")
        progress["scenarios_completed"] = len(observed)
        _atomic_json(paths["progress"], progress)
        try:
            for index, branch in enumerate(plan.branch_times_sec):
                if index < len(observed):
                    continue
                fork = plan.fork_plan(branch)
                baseline, treatment = workspace.run_paired_matches(
                    home, away, fast=fast,
                    baseline_plan=fork.baseline_plan(),
                    treatment_plan=fork.treatment_plan(), seed=plan.seed,
                    transaction_id=f"{set_id}-{index}",
                )
                resolved = _resolve_comparison(workspace, treatment.get("comparison_path"))
                comparison = json.loads(resolved.read_text(encoding="utf-8"))
                if (comparison.get("fixture") or {}).get("home") != home or (
                    comparison.get("fixture") or {}
                ).get("away") != away:
                    raise ValueError("fork-set comparison fixture identity mismatch")
                _row(plan, comparison)
                entry = {
                    "branch_at_sec": branch,
                    "baseline_match_id": baseline["match_id"],
                    "treatment_match_id": treatment["match_id"],
                    "comparison": resolved.relative_to(Path(workspace.root).resolve()).as_posix(),
                }
                progress["completed"].append(entry)
                progress["scenarios_completed"] = len(progress["completed"])
                _atomic_json(paths["progress"], progress)
            comparisons = []
            evidence = {}
            for entry in progress["completed"]:
                comparison_path = _resolve_comparison(workspace, entry["comparison"])
                comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
                if (comparison.get("fixture") or {}).get("home") != home or (
                    comparison.get("fixture") or {}
                ).get("away") != away:
                    raise ValueError("fork-set comparison fixture identity mismatch")
                comparisons.append(comparison)
                dashboard = comparison_path.with_suffix(".html")
                if not dashboard.is_file():
                    raise ValueError("fork-set comparison dashboard is unavailable")
                evidence[float(entry["branch_at_sec"])] = {
                    "comparison_dashboard": Path(os.path.relpath(
                        dashboard, paths["root"],
                    )).as_posix()
                }
            analysis = aggregate_fork_set(plan, comparisons)
            for row in analysis["rows"]:
                row["evidence"] = evidence[row["branch_at_sec"]]
            result = {
                "schema_version": 1,
                "set_id": set_id,
                "status": "complete",
                "fixture": {"home": home, "away": away},
                "fast": fast,
                "plan": plan.as_dict(),
                "source_context": (
                    dict(source_context) if source_context is not None else None
                ),
                "fixed_scenario_budget": len(plan.branch_times_sec),
                "scenarios_completed": len(plan.branch_times_sec),
                "interim_ranking_disclosed": False,
                **analysis,
                "claim_boundary": plan.as_dict()["claim_boundary"],
            }
            _atomic_json(paths["result"], result)
            paths["dashboard"].write_text(render_fork_set_html(result), encoding="utf-8")
            progress.update({"state": "complete", "analysis": None})
            _atomic_json(paths["progress"], progress)
            return {**result, "result_path": str(paths["result"]), "dashboard_path": str(paths["dashboard"])}
        except BaseException as exc:
            progress.update({
                "state": "failed", "error_type": type(exc).__name__,
                "analysis": None, "interim_ranking_disclosed": False,
            })
            _atomic_json(paths["progress"], progress)
            raise

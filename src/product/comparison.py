"""Auditable shared-seed tactical contrasts for GFS Studio."""

from __future__ import annotations

import html
import math
from bisect import bisect_right
from pathlib import Path
from typing import Any, Mapping

from src.product.paired_society_state import (
    build_paired_society_divergence,
    validate_paired_society_divergence,
)

from src.product.match_plan import PLAYABLE_TACTICS


class PairingError(ValueError):
    """Two reports cannot form the requested structural pair."""


METRICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("score_home", ("result", "score", "home")),
    ("score_away", ("result", "score", "away")),
    ("xg_home", ("result", "xg", "home")),
    ("xg_away", ("result", "xg", "away")),
    ("possession_home", ("result", "possession", "home")),
    ("possession_away", ("result", "possession", "away")),
    ("passes_home", ("result", "passes", "home")),
    ("passes_away", ("result", "passes", "away")),
    ("shots_home", ("result", "shots", "home")),
    ("shots_away", ("result", "shots", "away")),
    ("crowd_field", ("layers", "psychology", "crowd_field")),
    ("coach_stress_home", ("layers", "psychology", "coach_stress", "home")),
    ("coach_stress_away", ("layers", "psychology", "coach_stress", "away")),
    ("tactical_drift_home", ("layers", "psychology", "tactical_drift", "home")),
    ("tactical_drift_away", ("layers", "psychology", "tactical_drift", "away")),
    ("wm_action_opportunities", ("layers", "world_model", "action_adoption", "opportunities")),
    ("wm_influenced_opportunities", ("layers", "world_model", "action_adoption", "influenced_opportunities")),
    ("wm_counterfactual_changes", ("layers", "world_model", "action_adoption", "counterfactual_action_changes")),
    ("wm_expected_changes", ("layers", "world_model", "action_adoption", "expected_counterfactual_action_changes")),
    ("wm_mean_probability_shift", ("layers", "world_model", "action_adoption", "mean_recommended_probability_shift")),
)
MAX_PAIRED_REPLAY_EVENTS_PER_SIDE = 240
MAX_PAIRED_REPLAY_FRAMES = 240
MAX_POLICY_PROPAGATION_DECISIONS = 40
POLICY_PROPAGATION_WINDOWS_SECONDS = (30, 120)
COUNTERFACTUAL_FUTURE_LAYERS: tuple[
    tuple[str, tuple[str, ...]], ...
] = (
    ("score", ("score_home", "score_away")),
    ("chance_creation", ("xg_home", "xg_away", "shots_home", "shots_away")),
    (
        "possession_and_progression",
        (
            "possession_home", "possession_away",
            "passes_home", "passes_away",
        ),
    ),
    (
        "complex_system_state",
        (
            "crowd_field", "coach_stress_home", "coach_stress_away",
            "tactical_drift_home", "tactical_drift_away",
        ),
    ),
    (
        "world_model_mechanism",
        (
            "wm_action_opportunities", "wm_influenced_opportunities",
            "wm_counterfactual_changes", "wm_expected_changes",
            "wm_mean_probability_shift",
        ),
    ),
)


def _nested(report: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = report
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _branch_anchor(report: Mapping[str, Any]) -> Mapping[str, Any]:
    anchor = _nested(report, ("layers", "world_model", "branch_anchor"))
    return anchor if isinstance(anchor, Mapping) else {}


def _safe_text(value: Any, maximum: int = 100) -> str:
    text = str(value or "").strip()
    return "".join(char for char in text if ord(char) >= 32)[:maximum]


def _replay_point(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    x, y = _finite(value[0]), _finite(value[1])
    if x is None or y is None:
        return None
    return [min(1.0, max(0.0, x)), min(1.0, max(0.0, y))]


def _bounded_replay_side(
    report: Mapping[str, Any], *, home: str, away: str,
) -> dict[str, Any]:
    replay = report.get("replay") or {}
    if not isinstance(replay, Mapping) or not replay.get("available"):
        return {"available": False, "reason": "replay_unavailable", "events": []}
    raw_events = replay.get("events") or []
    if not isinstance(raw_events, (list, tuple)):
        return {
            "available": False, "reason": "invalid_replay_event_collection",
            "events": [],
        }
    events = []
    invalid_rows = 0
    for raw in raw_events[:MAX_PAIRED_REPLAY_EVENTS_PER_SIDE]:
        if not isinstance(raw, Mapping):
            invalid_rows += 1
            continue
        event_type = _safe_text(raw.get("type"), 20).lower()
        team = _safe_text(raw.get("team"))
        start, end = _replay_point(raw.get("start")), _replay_point(raw.get("end"))
        t_sec = _finite(raw.get("t_sec"))
        if (
            event_type not in {"pass", "shot", "cross"}
            or team not in {home, away}
            or start is None or end is None or t_sec is None
        ):
            invalid_rows += 1
            continue
        t_sec = min(8_000.0, max(0.0, t_sec))
        wm_link = raw.get("world_model_link") or {}
        if not isinstance(wm_link, Mapping):
            wm_link = {}
        wm_linked = bool(wm_link) or bool(raw.get("wm_linked"))
        wm_changed = bool(wm_link.get("policy_changed_action")) or bool(
            raw.get("wm_changed")
        )
        opportunity_id = _safe_text(
            wm_link.get("opportunity_id") or raw.get("opportunity_id"), 160,
        )
        events.append({
            "source_index": len(events), "t_sec": t_sec,
            "clock": f"{int(t_sec // 60)}:{int(t_sec % 60):02d}",
            "type": event_type, "team": team,
            "actor": _safe_text(raw.get("actor")),
            "target": _safe_text(raw.get("target")),
            "kind": _safe_text(raw.get("kind"), 40),
            "outcome": _safe_text(raw.get("outcome"), 40).upper(),
            "start": start, "end": end,
            "wm_linked": wm_linked, "wm_changed": wm_changed,
            "opportunity_id": opportunity_id,
        })
    events.sort(key=lambda event: (event["t_sec"], event["source_index"]))
    for index, event in enumerate(events):
        event["event_index"] = index
    return {
        "available": bool(events),
        "reason": "available" if events else "no_valid_replay_events",
        "events": events, "invalid_rows": invalid_rows,
        "input_truncated": len(raw_events) > (
            MAX_PAIRED_REPLAY_EVENTS_PER_SIDE
        ),
    }


def _evenly_select(values: list[float], count: int) -> list[float]:
    if count <= 0 or not values:
        return []
    if len(values) <= count:
        return list(values)
    if count == 1:
        return [values[len(values) // 2]]
    positions = {
        round(index * (len(values) - 1) / (count - 1))
        for index in range(count)
    }
    return [values[position] for position in sorted(positions)]


def _select_shared_timepoints(
    baseline_events: list[dict[str, Any]],
    treatment_events: list[dict[str, Any]],
) -> tuple[list[float], bool]:
    all_events = baseline_events + treatment_events
    all_times = sorted({float(event["t_sec"]) for event in all_events})
    if len(all_times) <= MAX_PAIRED_REPLAY_FRAMES:
        return all_times, False
    priority_times = sorted({
        float(event["t_sec"]) for event in all_events
        if event["type"] == "shot"
        or event["outcome"] in {"GOAL", "SAVED", "INTERCEPTED"}
        or event["wm_changed"]
    })
    if len(priority_times) >= MAX_PAIRED_REPLAY_FRAMES:
        return _evenly_select(priority_times, MAX_PAIRED_REPLAY_FRAMES), True
    priority_set = set(priority_times)
    remaining = [time for time in all_times if time not in priority_set]
    selected = priority_times + _evenly_select(
        remaining, MAX_PAIRED_REPLAY_FRAMES - len(priority_times),
    )
    return sorted(set(selected)), True


def _event_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "actions": len(events),
        "passes": sum(event["type"] == "pass" for event in events),
        "crosses": sum(event["type"] == "cross" for event in events),
        "shots": sum(event["type"] == "shot" for event in events),
        "goals": sum(event["outcome"] == "GOAL" for event in events),
        "wm_changed": sum(bool(event["wm_changed"]) for event in events),
    }


def build_synchronized_paired_replay(
    baseline: Mapping[str, Any], treatment: Mapping[str, Any],
    *, home: str, away: str,
) -> dict[str, Any]:
    """Build a bounded clock-aligned view without claiming event correspondence."""
    baseline_side = _bounded_replay_side(baseline, home=home, away=away)
    treatment_side = _bounded_replay_side(treatment, home=home, away=away)
    if not baseline_side["available"] or not treatment_side["available"]:
        missing = []
        if not baseline_side["available"]:
            missing.append("baseline")
        if not treatment_side["available"]:
            missing.append("treatment")
        return {
            "schema_version": 1, "available": False,
            "reason": "replay_unavailable:" + ",".join(missing),
            "baseline": baseline_side, "treatment": treatment_side,
            "frames": [], "event_correspondence_authorized": False,
        }
    baseline_events = baseline_side["events"]
    treatment_events = treatment_side["events"]
    timepoints, frames_truncated = _select_shared_timepoints(
        baseline_events, treatment_events,
    )
    baseline_times = [event["t_sec"] for event in baseline_events]
    treatment_times = [event["t_sec"] for event in treatment_events]
    frames = []
    for index, t_sec in enumerate(timepoints):
        frames.append({
            "frame_index": index, "t_sec": t_sec,
            "clock": f"{int(t_sec // 60)}:{int(t_sec % 60):02d}",
            "baseline_visible_count": bisect_right(baseline_times, t_sec),
            "treatment_visible_count": bisect_right(treatment_times, t_sec),
            "baseline_current_indices": [
                event["event_index"] for event in baseline_events
                if event["t_sec"] == t_sec
            ],
            "treatment_current_indices": [
                event["event_index"] for event in treatment_events
                if event["t_sec"] == t_sec
            ],
        })
    baseline_counts = _event_counts(baseline_events)
    treatment_counts = _event_counts(treatment_events)
    return {
        "schema_version": 1, "available": True,
        "reason": "shared_match_clock_alignment",
        "alignment_rule": "shared match clock only; no cross-match event matching",
        "event_correspondence_authorized": False,
        "causal_claim_authorized": False,
        "baseline": baseline_side, "treatment": treatment_side,
        "frames": frames, "frames_truncated": frames_truncated,
        "summary": {
            metric: {
                "baseline": baseline_counts[metric],
                "treatment": treatment_counts[metric],
                "delta": treatment_counts[metric] - baseline_counts[metric],
            }
            for metric in baseline_counts
        },
    }


def _checkpoint_identity(report: Mapping[str, Any]) -> str:
    runtime = _nested(report, ("layers", "world_model", "runtime")) or {}
    if isinstance(runtime, Mapping) and runtime.get("checkpoint_signature"):
        return str(runtime["checkpoint_signature"])
    evidence = report.get("evidence_snapshot") or {}
    return str(
        evidence.get("world_model_checkpoint_sha256") or "not_applicable"
    )


def _events_in_window(
    events: list[dict[str, Any]], *, start: float, end: float,
) -> dict[str, int]:
    selected = [
        event for event in events
        if start <= float(event["t_sec"]) <= end
    ]
    return {
        "actions": len(selected),
        "passes": sum(event["type"] == "pass" for event in selected),
        "crosses": sum(event["type"] == "cross" for event in selected),
        "shots": sum(event["type"] == "shot" for event in selected),
        "goals": sum(event["outcome"] == "GOAL" for event in selected),
        "turnovers": sum(
            event["outcome"] in {"INCOMPLETE", "INTERCEPTED", "LOST"}
            for event in selected
        ),
    }


def _window_contrast(
    baseline_events: list[dict[str, Any]],
    treatment_events: list[dict[str, Any]],
    *, start: float, duration: int,
) -> dict[str, Any]:
    end = min(8_000.0, start + duration)
    baseline = _events_in_window(baseline_events, start=start, end=end)
    treatment = _events_in_window(treatment_events, start=start, end=end)
    return {
        "start_t_sec": start,
        "end_t_sec": end,
        "baseline": baseline,
        "treatment": treatment,
        "delta": {
            metric: treatment[metric] - baseline[metric]
            for metric in baseline
        },
    }


def _runtime_event_matches_action(event: Mapping[str, Any], action: str) -> bool:
    normalized = action.strip().lower()
    expected_event_type = {
        "pass": "pass",
        "cross": "cross",
        "shot": "shot",
        "shoot": "shot",
    }.get(normalized)
    return bool(
        expected_event_type
        and event.get("type") == expected_event_type
    )


def build_world_model_policy_propagation(
    baseline: Mapping[str, Any],
    treatment: Mapping[str, Any],
    *, home: str,
    away: str,
    pair_eligible: bool,
) -> dict[str, Any]:
    """Trace local policy changes and bounded downstream descriptions.

    Direct runtime identity may support attribution of the local sampled action.
    Later event windows are deliberately descriptive because trajectories have
    diverged and more than one policy intervention may occur.
    """
    adoption = _nested(treatment, ("layers", "world_model", "action_adoption"))
    raw_records = adoption.get("records") if isinstance(adoption, Mapping) else []
    if not isinstance(raw_records, (list, tuple)):
        raw_records = []
    changed_records = []
    invalid_records = 0
    seen_opportunities: set[str] = set()
    duplicate_opportunities: set[str] = set()
    for raw in raw_records:
        if not isinstance(raw, Mapping) or not raw.get("policy_changed_action"):
            continue
        opportunity_id = _safe_text(raw.get("opportunity_id"), 160)
        team = _safe_text(raw.get("team_id"))
        t_sec = _finite(raw.get("t_sec"))
        baseline_action = _safe_text(
            raw.get("counterfactual_baseline_action"), 40,
        )
        treatment_action = _safe_text(raw.get("actual_action"), 40)
        if (
            not opportunity_id
            or team not in {home, away}
            or t_sec is None
            or not baseline_action
            or not treatment_action
        ):
            invalid_records += 1
            continue
        if opportunity_id in seen_opportunities:
            duplicate_opportunities.add(opportunity_id)
            continue
        seen_opportunities.add(opportunity_id)
        changed_records.append({
            "opportunity_id": opportunity_id,
            "team": team,
            "t_sec": min(8_000.0, max(0.0, t_sec)),
            "baseline_action": baseline_action,
            "treatment_action": treatment_action,
            "recommended_action": _safe_text(
                raw.get("recommended_action"), 40,
            ),
            "probability_policy_version": _safe_text(
                adoption.get("probability_policy_version")
                if isinstance(adoption, Mapping) else None,
                80,
            ) or "legacy_unversioned",
            "signal_mode": (
                _safe_text(raw.get("signal_mode"), 40)
                or "legacy_unclassified"
            ),
            "primary_signal_action": (
                _safe_text(raw.get("primary_signal_action"), 40)
                or _safe_text(raw.get("recommended_action"), 40)
                or "none"
            ),
            "hold_reference_redistributed": bool(
                isinstance(raw.get("reference_action_effect"), Mapping)
                and raw["reference_action_effect"].get(
                    "received_redistributed_probability"
                ) is True
            ),
            "hold_reference_probability_delta": (
                _finite(raw["reference_action_effect"].get("probability_delta"))
                if isinstance(raw.get("reference_action_effect"), Mapping)
                else None
            ),
            "record_attribution_eligible": bool(
                raw.get("attribution_eligible")
            ),
        })
    changed_records.sort(
        key=lambda row: (row["t_sec"], row["opportunity_id"]),
    )
    total_valid_changes = len(changed_records)
    changed_records = changed_records[:MAX_POLICY_PROPAGATION_DECISIONS]

    baseline_side = _bounded_replay_side(baseline, home=home, away=away)
    treatment_side = _bounded_replay_side(treatment, home=home, away=away)
    replay_available = bool(
        baseline_side.get("available") and treatment_side.get("available")
    )
    baseline_events = baseline_side.get("events") or []
    treatment_events = treatment_side.get("events") or []
    linked_events: dict[str, list[dict[str, Any]]] = {}
    for event in treatment_events:
        opportunity_id = str(event.get("opportunity_id") or "")
        if opportunity_id:
            linked_events.setdefault(opportunity_id, []).append(event)

    decisions = []
    direct_observations = 0
    locally_attributable = 0
    for record in changed_records:
        direct_matches = linked_events.get(record["opportunity_id"], [])
        direct_event = direct_matches[0] if len(direct_matches) == 1 else None
        directly_observed = bool(
            direct_event
            and direct_event.get("wm_changed")
            and direct_event.get("team") == record["team"]
            and abs(float(direct_event["t_sec"]) - record["t_sec"]) <= 1e-6
            and _runtime_event_matches_action(
                direct_event, record["treatment_action"],
            )
        )
        local_eligible = bool(
            pair_eligible
            and record["record_attribution_eligible"]
            and directly_observed
            and record["opportunity_id"] not in duplicate_opportunities
        )
        direct_observations += int(directly_observed)
        locally_attributable += int(local_eligible)
        windows = {}
        if replay_available:
            for duration in POLICY_PROPAGATION_WINDOWS_SECONDS:
                window = _window_contrast(
                    baseline_events,
                    treatment_events,
                    start=record["t_sec"],
                    duration=duration,
                )
                window["additional_policy_changes"] = sum(
                    other["opportunity_id"] != record["opportunity_id"]
                    and record["t_sec"] <= other["t_sec"] <= window["end_t_sec"]
                    for other in changed_records
                )
                window["causal_attribution_authorized"] = False
                windows[f"{duration}s"] = window
        decisions.append({
            **record,
            "clock": f'{int(record["t_sec"] // 60)}:'
            f'{int(record["t_sec"] % 60):02d}',
            "directly_observed": directly_observed,
            "observed_event_index": (
                direct_event.get("event_index") if directly_observed else None
            ),
            "local_policy_attribution_eligible": local_eligible,
            "downstream_windows": windows,
        })
    status = (
        "no_realized_action_changes"
        if total_valid_changes == 0 else
        "direct_action_changes_observed"
        if direct_observations else
        "changed_actions_not_directly_observed"
    )
    return {
        "schema_version": 1,
        "available": True,
        "status": status,
        "summary": {
            "valid_changed_decisions": total_valid_changes,
            "retained_changed_decisions": len(decisions),
            "directly_observed_changes": direct_observations,
            "locally_attributable_changes": locally_attributable,
            "invalid_changed_records": invalid_records,
            "duplicate_opportunity_ids": len(duplicate_opportunities),
            "replay_windows_available": replay_available,
            "decisions_truncated": total_valid_changes > len(decisions),
        },
        "decisions": decisions,
        "local_action_attribution_authorized": bool(locally_attributable),
        "downstream_causal_attribution_authorized": False,
        "window_alignment_rule": (
            "same match-clock ranges only; no cross-world event correspondence"
        ),
        "claim_boundary": (
            "direct runtime identity can attribute an individual sampled action "
            "to the simulator policy switch; subsequent window deltas remain "
            "descriptive and do not establish a causal path to match outcomes"
        ),
    }


def build_counterfactual_future_summary(
    *, baseline_id: str, treatment_id: str,
    metrics: Mapping[str, Any], propagation: Mapping[str, Any],
    eligibility: Mapping[str, Any], intervention: Mapping[str, Any],
    society_divergence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Unify one policy fork into one evidence-graded two-future view."""
    society_divergence = (
        society_divergence if isinstance(society_divergence, Mapping) else {}
    )
    pair_eligible = bool(
        eligibility.get("eligible_for_world_model_policy_attribution")
    )
    raw_summary = propagation.get("summary") or {}
    if not isinstance(raw_summary, Mapping):
        raw_summary = {}

    def count(key: str) -> int:
        number = _finite(raw_summary.get(key))
        return max(0, min(100_000, int(number or 0)))

    changed = count("valid_changed_decisions")
    direct = count("directly_observed_changes")
    local = count("locally_attributable_changes")
    replay_available = bool(raw_summary.get("replay_windows_available"))
    layers = []
    outcome_differences = 0
    outcome_missing = 0
    for layer_id, names in COUNTERFACTUAL_FUTURE_LAYERS:
        observed, missing = [], []
        for name in names:
            values = metrics.get(name)
            if not isinstance(values, Mapping):
                missing.append(name)
                continue
            before = _finite(values.get("baseline"))
            after = _finite(values.get("treatment"))
            delta = _finite(values.get("delta"))
            if before is None or after is None or delta is None:
                missing.append(name)
                continue
            observed.append({
                "metric": name, "baseline": before, "treatment": after,
                "delta": delta,
                "different": not math.isclose(
                    delta, 0.0, rel_tol=0.0, abs_tol=1e-12,
                ),
            })
        difference_count = sum(item["different"] for item in observed)
        if layer_id != "world_model_mechanism":
            outcome_differences += difference_count
            outcome_missing += len(missing)
        layers.append({
            "layer": layer_id, "observed_metric_count": len(observed),
            "difference_count": difference_count,
            "missing_metrics": missing, "metrics": observed,
        })
    if not pair_eligible:
        status = "descriptive_only_ineligible"
    elif changed == 0:
        status = "no_realized_action_divergence"
    elif local == 0:
        status = "action_divergence_without_local_attribution"
    elif outcome_differences:
        status = "local_action_divergence_with_descriptive_future_difference"
    else:
        status = "local_action_divergence_without_measured_future_difference"
    anchor = intervention.get("branch_anchor") or {}
    if not isinstance(anchor, Mapping):
        anchor = {}
    branch_sec = _finite(intervention.get("branch_at_sec"))
    actual_sec = _finite(anchor.get("actual_sec"))
    failed_checks = eligibility.get("failed_checks") or []
    if not isinstance(failed_checks, (list, tuple)):
        failed_checks = []
    result = {
        "schema_version": 2,
        "available": True,
        "status": status,
        "comparison_mode": "single_seed_paired_simulator_fork",
        "layers": layers,
        "intervention_point": {
            "requested_sec": branch_sec,
            "actual_sec": actual_sec,
            "anchor_verified": bool(anchor.get("verified")),
            "clock_contract": _safe_text(
                anchor.get("clock_contract"), 80,
            ) or None,
            "state_identity": (
                _safe_text(anchor.get("state_identity"), 64) or None
            ),
        },
    }
    result["intervention_point"]["clock"] = (
        "%d:%02d" % (int(actual_sec // 60), int(actual_sec % 60))
        if actual_sec is not None else None
    )
    result["worlds"] = [
        {
            "id": "baseline", "label": "基线世界",
            "match_id": baseline_id, "policy": "predict_only",
        },
        {
            "id": "treatment", "label": "干预世界",
            "match_id": treatment_id, "policy": "action_policy",
        },
    ]
    prefix_verified = bool(
        anchor.get("verified")
        and anchor.get("clock_contract") == "authoritative_tick_v2"
    )
    match_start_controlled = bool(branch_sec is None and pair_eligible)
    result["evidence_ladder"] = [
        {
            "stage": "shared_prefix",
            "status": (
                "verified" if prefix_verified else
                "match_start_controlled" if match_start_controlled else
                "not_verified"
            ),
            "claim_authorized": bool(
                prefix_verified or match_start_controlled
            ),
        },
        {
            "stage": "policy_to_action",
            "status": (
                "locally_attributed" if local else
                "observed_not_attributed" if changed else
                "no_realized_divergence"
            ),
            "changed_actions": changed,
            "directly_observed": direct,
            "locally_attributable": local,
            "claim_authorized": bool(pair_eligible and local),
        },
    ]
    outcome_stage = (
        "descriptive_difference" if outcome_differences else
        "no_measured_difference" if outcome_missing == 0 else
        "incomplete_measurement"
    )
    society_available = society_divergence.get("available") is True
    society_differences = int(
        society_divergence.get("state_change_count") or 0
    ) + int(society_divergence.get("count_delta_count") or 0)
    result["evidence_ladder"].extend([
        {
            "stage": "action_to_trajectory",
            "status": (
                "descriptive_windows_available"
                if replay_available else "replay_unavailable"
            ),
            "claim_authorized": False,
        },
        {
            "stage": "action_to_society_state",
            "status": (
                "descriptive_terminal_state_divergence"
                if society_available and society_differences else
                "no_terminal_state_divergence"
                if society_available else
                "terminal_state_evidence_unavailable"
            ),
            "difference_count": society_differences,
            "claim_authorized": False,
        },
        {
            "stage": "trajectory_to_outcome",
            "status": outcome_stage,
            "difference_count": outcome_differences,
            "missing_metric_count": outcome_missing,
            "claim_authorized": False,
        },
    ])
    result["summary"] = {
        "changed_actions": changed,
        "directly_observed_actions": direct,
        "locally_attributable_actions": local,
        "descriptive_outcome_difference_count": outcome_differences,
        "missing_outcome_metric_count": outcome_missing,
        "descriptive_society_state_difference_count": society_differences,
        "teams_with_society_state_divergence": int(
            society_divergence.get("teams_with_terminal_divergence") or 0
        ),
    }
    result["eligibility"] = {
        "pair_eligible": pair_eligible,
        "failed_checks": [
            _safe_text(value, 100) for value in failed_checks[:30]
        ],
    }
    result["claim_authority"] = {
        "simulator_local_action_attribution": bool(pair_eligible and local),
        "downstream_trajectory_causality": False,
        "downstream_society_state_causality": False,
        "match_outcome_causality": False,
        "population_effect": False,
        "real_football_causality": False,
        "world_model_promotion": False,
    }
    result["claim_boundary"] = (
        "one identity-bound simulator fork can establish a local sampled "
        "action change when runtime evidence is complete; later trajectory "
        "and outcome differences remain descriptive"
    )
    return result


def build_paired_comparison(
    baseline: Mapping[str, Any], treatment: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a same-seed contrast without overstating a single pair."""
    baseline_id = str(baseline.get("match_id") or "")
    treatment_id = str(treatment.get("match_id") or "")
    if not baseline_id or not treatment_id or baseline_id == treatment_id:
        raise PairingError("paired reports require two distinct match identities")
    baseline_fixture = baseline.get("fixture") or {}
    treatment_fixture = treatment.get("fixture") or {}
    ordered_fixture = (
        str(baseline_fixture.get("home")),
        str(baseline_fixture.get("away")),
    )
    if ordered_fixture != (
        str(treatment_fixture.get("home")),
        str(treatment_fixture.get("away")),
    ):
        raise PairingError("paired reports require the same ordered fixture")
    baseline_seed = baseline_fixture.get("seed")
    treatment_seed = treatment_fixture.get("seed")
    if baseline_seed != treatment_seed or not isinstance(baseline_seed, int):
        raise PairingError("paired reports require the same deterministic seed")
    baseline_plan = baseline.get("match_plan") or {}
    treatment_plan = treatment.get("match_plan") or {}
    if treatment_plan.get("paired_baseline_match_id") != baseline_id:
        raise PairingError("treatment does not identify the supplied baseline")
    if not treatment_plan.get("reuse_last_seed"):
        raise PairingError("treatment did not declare shared-seed reuse")

    baseline_experience = str(baseline_plan.get("experience") or "")
    treatment_experience = str(treatment_plan.get("experience") or "")
    if baseline_experience != treatment_experience:
        raise PairingError("paired reports require the same experiment type")
    changed_sides = [
        side for side in ("home", "away")
        if baseline_plan.get(f"{side}_tactic")
        != treatment_plan.get(f"{side}_tactic")
    ]
    if baseline_experience == "tactical_lab":
        if not changed_sides:
            raise PairingError(
                "paired reports contain no tactical intervention change"
            )
        scope = (
            "single_side_tactical_intervention"
            if len(changed_sides) == 1 else "joint_tactical_intervention"
        )
        policy_contract = True
    elif baseline_experience == "world_model_lab":
        if changed_sides:
            raise PairingError("world-model fork must keep both tactics fixed")
        if (
            baseline_plan.get("world_model_policy") != "predict_only"
            or treatment_plan.get("world_model_policy") != "action_policy"
        ):
            raise PairingError(
                "world-model fork must isolate predict_only to action_policy"
            )
        baseline_branch = _finite(baseline_plan.get("world_model_branch_at_sec"))
        treatment_branch = _finite(treatment_plan.get("world_model_branch_at_sec"))
        if baseline_branch != treatment_branch:
            raise PairingError("world-model fork must use the same branch time")
        scope = "world_model_action_policy"
        policy_contract = True
    else:
        raise PairingError("unsupported paired experiment type")
    same_mode = (
        (baseline.get("studio") or {}).get("mode")
        == (treatment.get("studio") or {}).get("mode")
    )
    mode = str((treatment.get("studio") or {}).get("mode") or "unknown")
    baseline_anchor = _branch_anchor(baseline)
    treatment_anchor = _branch_anchor(treatment)
    branch_requested = (
        baseline_experience == "world_model_lab"
        and _finite(baseline_plan.get("world_model_branch_at_sec")) is not None
    )
    anchor_identity = str(baseline_anchor.get("state_identity") or "")
    anchor_actual_sec = _finite(baseline_anchor.get("actual_sec"))
    same_branch_anchor = (
        bool(baseline_anchor.get("available"))
        and bool(treatment_anchor.get("available"))
        and bool(anchor_identity)
        and anchor_identity == str(treatment_anchor.get("state_identity") or "")
        and _finite(baseline_anchor.get("requested_sec"))
        == _finite(treatment_anchor.get("requested_sec"))
        and anchor_actual_sec is not None
        and anchor_actual_sec == _finite(treatment_anchor.get("actual_sec"))
    )
    checks = {
        "same_ordered_fixture": True,
        "same_seed": True,
        "same_fast_configuration": (
            baseline_fixture.get("fast") == treatment_fixture.get("fast")
        ),
        "same_studio_mode": same_mode,
        "physics_score_path_both": (
            baseline_plan.get("score_path") == "physics_official"
            and treatment_plan.get("score_path") == "physics_official"
        ),
        "integrity_accepted_both": (
            bool((baseline.get("integrity") or {}).get("accepted"))
            and bool((treatment.get("integrity") or {}).get("accepted"))
        ),
        "same_world_model_identity": (
            _checkpoint_identity(baseline) == _checkpoint_identity(treatment)
        ),
        "provider_determinism_controlled": mode != "cognitive",
        "intervention_contract_isolated": policy_contract,
    }
    if branch_requested:
        checks["same_pre_intervention_branch_anchor"] = same_branch_anchor
        baseline_clock = baseline.get("simulation_clock") or {}
        treatment_clock = treatment.get("simulation_clock") or {}
        checks["same_authoritative_product_clock"] = (
            baseline_clock.get("contract") == "authoritative_tick_v2"
            and treatment_clock.get("contract") == "authoritative_tick_v2"
            and bool(baseline_clock.get("authoritative_tick_clock"))
            and bool(treatment_clock.get("authoritative_tick_clock"))
        )
    eligible = all(checks.values())
    metrics: dict[str, dict[str, float | None]] = {}
    for metric, path in METRICS:
        before = _finite(_nested(baseline, path))
        after = _finite(_nested(treatment, path))
        metrics[metric] = {
            "baseline": before,
            "treatment": after,
            "delta": (
                after - before if before is not None and after is not None else None
            ),
        }
    paired_replay = build_synchronized_paired_replay(
        baseline, treatment, home=ordered_fixture[0], away=ordered_fixture[1],
    )
    policy_propagation = (
        build_world_model_policy_propagation(
            baseline,
            treatment,
            home=ordered_fixture[0],
            away=ordered_fixture[1],
            pair_eligible=eligible,
        )
        if baseline_experience == "world_model_lab" else
        {
            "schema_version": 1,
            "available": False,
            "status": "not_world_model_policy_fork",
            "decisions": [],
        }
    )
    try:
        society_divergence = build_paired_society_divergence(
            baseline,
            treatment,
            teams=ordered_fixture,
            pair_eligible=eligible,
        )
    except ValueError as exc:
        raise PairingError("paired terminal society evidence is invalid") from exc
    comparison = {
        "schema_version": 1,
        "comparison_id": f"{treatment_id}-paired-vs-{baseline_id}",
        "fixture": {
            "home": ordered_fixture[0], "away": ordered_fixture[1],
            "seed": baseline_seed,
        },
        "baseline": {"match_id": baseline_id, "plan": dict(baseline_plan)},
        "treatment": {"match_id": treatment_id, "plan": dict(treatment_plan)},
        "intervention": {
            "scope": scope,
            "experiment_type": baseline_experience,
            "changed_sides": changed_sides,
            "baseline_tactics": {
                side: baseline_plan.get(f"{side}_tactic")
                for side in ("home", "away")
            },
            "treatment_tactics": {
                side: treatment_plan.get(f"{side}_tactic")
                for side in ("home", "away")
            },
            "baseline_world_model_policy": baseline_plan.get(
                "world_model_policy", "mode_default"
            ),
            "treatment_world_model_policy": treatment_plan.get(
                "world_model_policy", "mode_default"
            ),
            "branch_at_sec": baseline_plan.get("world_model_branch_at_sec"),
            "branch_anchor": {
                "required": branch_requested,
                "verified": same_branch_anchor if branch_requested else False,
                "state_identity": anchor_identity if same_branch_anchor else None,
                "requested_sec": (
                    _finite(baseline_anchor.get("requested_sec"))
                    if branch_requested else None
                ),
                "actual_sec": anchor_actual_sec if same_branch_anchor else None,
                "clock_contract": (
                    str((baseline.get("simulation_clock") or {}).get(
                        "contract"
                    ) or "")
                    if branch_requested else None
                ),
                "resume_capability": "deterministic_replay_only",
            },
        },
        "eligibility": {
            "eligible_for_tactical_attribution": (
                eligible if baseline_experience == "tactical_lab" else False
            ),
            "eligible_for_world_model_policy_attribution": (
                eligible if baseline_experience == "world_model_lab" else False
            ),
            "checks": checks,
            "failed_checks": [key for key, passed in checks.items() if not passed],
        },
        "metrics": metrics,
        "paired_replay": paired_replay,
        "policy_propagation": policy_propagation,
        "society_divergence": society_divergence,
        "claim_boundary": (
            "paired deterministic simulator-policy contrast for this fixture and "
            "seed only; not a population effect, significance test, real-football "
            "causal effect, or world-model promotion"
            if baseline_experience == "world_model_lab" else
            "paired deterministic contrast for this fixture and seed only; "
            "not a population effect, significance test, or general performance claim"
        ),
    }
    if baseline_experience == "world_model_lab":
        comparison["counterfactual_future_summary"] = (
            build_counterfactual_future_summary(
                baseline_id=baseline_id,
                treatment_id=treatment_id,
                metrics=metrics,
                propagation=policy_propagation,
                eligibility=comparison["eligibility"],
                intervention=comparison["intervention"],
                society_divergence=society_divergence,
            )
        )
    else:
        comparison["counterfactual_future_summary"] = {
            "schema_version": 1,
            "available": False,
            "status": "not_world_model_policy_fork",
        }
    return comparison


def _display_tactic(value: Any) -> str:
    key = str(value or "unknown")
    return str((PLAYABLE_TACTICS.get(key) or {}).get("label") or key)


def _paired_pitch_svg(
    events: list[dict[str, Any]], *, home: str, label: str,
) -> str:
    trajectories = []
    for event in events:
        x1, y1 = 40 + 920 * event["start"][0], 40 + 560 * event["start"][1]
        x2, y2 = 40 + 920 * event["end"][0], 40 + 560 * event["end"][1]
        team_class = "pair-home" if event["team"] == home else "pair-away"
        classes = [
            "pair-event", f'pair-{event["type"]}', team_class,
            "pair-wm-linked" if event["wm_linked"] else "",
            "pair-wm-changed" if event["wm_changed"] else "",
        ]
        title = (
            f'{event["clock"]} · {event["team"]} · {event["actor"]} · '
            f'{event["kind"]} · {event["outcome"]}'
        )
        trajectories.append(
            f'<g class="{" ".join(filter(None, classes))}">'
            f'<title>{html.escape(title)}</title>'
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" '
            f'x2="{x2:.1f}" y2="{y2:.1f}" />'
            f'<circle cx="{x2:.1f}" cy="{y2:.1f}" '
            f'r="{5 if event["type"] == "shot" else 2.5}" /></g>'
        )
    return f"""<svg viewBox="0 0 1000 640" role="img" aria-label="{html.escape(label, quote=True)}">
<rect class="pair-pitch" x="20" y="20" width="960" height="600" rx="8"/><line class="pair-marking" x1="500" y1="20" x2="500" y2="620"/><circle class="pair-marking" cx="500" cy="320" r="78"/><circle class="pair-spot" cx="500" cy="320" r="4"/><rect class="pair-marking" x="20" y="155" width="150" height="330"/><rect class="pair-marking" x="830" y="155" width="150" height="330"/><rect class="pair-marking" x="20" y="235" width="55" height="170"/><rect class="pair-marking" x="925" y="235" width="55" height="170"/>{''.join(trajectories)}</svg>"""


def _current_action_text(
    events: list[dict[str, Any]], indices: list[int],
) -> str:
    selected = [
        events[index] for index in indices[:3]
        if isinstance(index, int) and 0 <= index < len(events)
    ]
    if not selected:
        return '<span class="muted">该时点无新保留动作</span>'
    items = []
    for event in selected:
        badge = (
            ' <span class="pair-badge changed">WM changed</span>'
            if event["wm_changed"] else
            ' <span class="pair-badge">WM-linked</span>'
            if event["wm_linked"] else ""
        )
        items.append(
            f'<span><b>{html.escape(event["team"])}</b> · '
            f'{html.escape(event["actor"])} → {html.escape(event["target"])} · '
            f'{html.escape(event["kind"])} · '
            f'{html.escape(event["outcome"])}{badge}</span>'
        )
    if len(indices) > len(selected):
        items.append(
            f'<span class="muted">另有 {len(indices) - len(selected)} 个同刻动作</span>'
        )
    return "".join(items)


def _paired_replay_panel(comparison: Mapping[str, Any]) -> str:
    fixture = comparison.get("fixture") or {}
    home, away = _safe_text(fixture.get("home")), _safe_text(fixture.get("away"))
    stored = comparison.get("paired_replay") or {}
    if not isinstance(stored, Mapping):
        stored = {}
    baseline_stored = stored.get("baseline") or {}
    treatment_stored = stored.get("treatment") or {}
    replay = build_synchronized_paired_replay(
        {"replay": baseline_stored if isinstance(baseline_stored, Mapping) else {}},
        {"replay": treatment_stored if isinstance(treatment_stored, Mapping) else {}},
        home=home, away=away,
    )
    if not replay.get("available"):
        return f"""<section class="card paired-replay" data-testid="paired-replay-panel">
<h2>同步动作复盘</h2><p class="muted">双场播放器不可用：{html.escape(str(replay.get('reason') or 'unknown'))}。指标比较与资格检查仍然有效。</p></section>"""
    baseline_events = replay["baseline"]["events"]
    treatment_events = replay["treatment"]["events"]
    frames = replay["frames"][:MAX_PAIRED_REPLAY_FRAMES]
    summary = replay.get("summary") or {}
    inputs = [
        '<input type="radio" name="pair-step" id="pair-step-overview" checked>'
    ]
    markers = [
        '<label for="pair-step-overview" title="显示两场完整轨迹">概览</label>'
    ]
    navigation = [
        '<div class="pair-frame pair-frame-overview"><div><span class="eyebrow">同步时间轴</span><strong>两场完整动作地图</strong><span class="muted">按共享比赛时钟对齐，不代表跨场动作对应。</span></div><label class="pair-button" for="pair-step-0">开始复盘</label></div>'
    ]
    rules = [
        '#pair-step-overview:checked~.pair-navigation .pair-frame-overview{display:flex}',
        '#pair-step-overview:checked~.pair-timeline label[for="pair-step-overview"]{border-color:var(--ok);color:var(--ok);background:#10281f}',
    ]
    for frame_index, frame in enumerate(frames):
        step_id = f"pair-step-{frame_index}"
        inputs.append(f'<input type="radio" name="pair-step" id="{step_id}">')
        markers.append(
            f'<label for="{step_id}" title="共享比赛时钟 {html.escape(frame["clock"], quote=True)}">'
            f'{frame_index + 1}<span>{html.escape(frame["clock"])}</span></label>'
        )
        previous_id = (
            "pair-step-overview" if frame_index == 0
            else f"pair-step-{frame_index - 1}"
        )
        next_id = (
            "pair-step-overview" if frame_index + 1 == len(frames)
            else f"pair-step-{frame_index + 1}"
        )
        next_label = "返回概览" if frame_index + 1 == len(frames) else "下一时点"
        navigation.append(
            f'<div class="pair-frame pair-frame-{frame_index}">'
            f'<label class="pair-button" for="{previous_id}">上一时点</label>'
            f'<div class="pair-frame-center"><span class="eyebrow">时点 '
            f'{frame_index + 1} / {len(frames)} · {html.escape(frame["clock"])}</span>'
            f'<div class="pair-current"><div><b>基线场</b>{_current_action_text(baseline_events, frame["baseline_current_indices"])}</div>'
            f'<div><b>处理场</b>{_current_action_text(treatment_events, frame["treatment_current_indices"])}</div></div></div>'
            f'<label class="pair-button" for="{next_id}">{next_label}</label></div>'
        )
        rules.extend([
            f'#{step_id}:checked~.pair-navigation .pair-frame-{frame_index}{{display:flex}}',
            f'#{step_id}:checked~.pair-timeline label[for="{step_id}"]{{border-color:var(--ok);color:var(--ok);background:#10281f}}',
            f'#{step_id}:focus-visible~.pair-timeline label[for="{step_id}"]{{outline:3px solid var(--warn);outline-offset:2px}}',
        ])
        for side_name, events, visible_key, current_key in (
            ("baseline", baseline_events, "baseline_visible_count", "baseline_current_indices"),
            ("treatment", treatment_events, "treatment_visible_count", "treatment_current_indices"),
        ):
            visible = max(0, min(len(events), int(frame[visible_key])))
            rules.extend([
                f'#{step_id}:checked~.pair-pitches .pair-side-{side_name} .pair-event line{{opacity:.12}}',
                f'#{step_id}:checked~.pair-pitches .pair-side-{side_name} .pair-event circle{{opacity:.2}}',
            ])
            if visible < len(events):
                rules.append(
                    f'#{step_id}:checked~.pair-pitches .pair-side-{side_name} '
                    f'.pair-event:nth-of-type(n+{visible + 1}){{display:none}}'
                )
            for current_index in frame[current_key]:
                nth = int(current_index) + 1
                rules.extend([
                    f'#{step_id}:checked~.pair-pitches .pair-side-{side_name} .pair-event:nth-of-type({nth}){{display:inline;filter:drop-shadow(0 0 7px #fff)}}',
                    f'#{step_id}:checked~.pair-pitches .pair-side-{side_name} .pair-event:nth-of-type({nth}) line{{opacity:1;stroke-width:8}}',
                    f'#{step_id}:checked~.pair-pitches .pair-side-{side_name} .pair-event:nth-of-type({nth}) circle{{opacity:1}}',
                ])
    summary_cards = "".join(
        f'<div><span class="eyebrow">{html.escape(metric)}</span>'
        f'<strong>{html.escape(str((values or {}).get("baseline")))} → '
        f'{html.escape(str((values or {}).get("treatment")))}</strong>'
        f'<span>Δ {html.escape(str((values or {}).get("delta")))}</span></div>'
        for metric, values in summary.items()
    )
    truncation_note = (
        "共享时点超过上限，已优先保留射门、关键结果与 WM-changed 时点。"
        if replay.get("frames_truncated") else "全部共享动作时点均已保留。"
    )
    return f"""<section class="card paired-replay" data-testid="paired-replay-panel">
<div class="pair-panel-head"><div><span class="eyebrow">Shared-clock descriptive replay</span><h2>同步动作复盘</h2></div><span class="pair-badge">{len(frames)} 时点</span></div>
<p class="muted">两场只按比赛时钟同步，不进行动作一一配对，也不新增因果资格。{truncation_note}</p>
<div class="pair-summary">{summary_cards}</div>
<fieldset class="pair-controls"><legend>双场轨迹筛选</legend>
<input type="radio" name="pair-filter" id="pair-filter-all" checked><label for="pair-filter-all">全部</label>
<input type="radio" name="pair-filter" id="pair-filter-shots"><label for="pair-filter-shots">仅射门</label>
<input type="radio" name="pair-filter" id="pair-filter-wm"><label for="pair-filter-wm">WM changed</label>
{''.join(inputs)}
<div class="pair-timeline" role="group" aria-label="跳转到共享比赛时点">{''.join(markers)}</div>
<div class="pair-navigation" aria-live="polite">{''.join(navigation)}</div>
<div class="pair-pitches"><article class="pair-side pair-side-baseline"><h3>基线场</h3>{_paired_pitch_svg(baseline_events, home=home, label='基线场动作轨迹')}</article><article class="pair-side pair-side-treatment"><h3>处理场</h3>{_paired_pitch_svg(treatment_events, home=home, label='处理场动作轨迹')}</article></div>
</fieldset><style>{''.join(rules)}</style></section>"""


def _counterfactual_future_panel(comparison: Mapping[str, Any]) -> str:
    future = comparison.get("counterfactual_future_summary") or {}
    if not isinstance(future, Mapping) or not future.get("available"):
        return ""
    summary = future.get("summary") or {}
    point = future.get("intervention_point") or {}
    authority = future.get("claim_authority") or {}
    status_labels = {
        "descriptive_only_ineligible": "仅描述：配对资格未通过",
        "no_realized_action_divergence": "策略已介入，但动作未分叉",
        "action_divergence_without_local_attribution": "动作已分叉，局部归因不足",
        "local_action_divergence_with_descriptive_future_difference": (
            "局部动作已归因，后续世界出现描述性差异"
        ),
        "local_action_divergence_without_measured_future_difference": (
            "局部动作已归因，已测后续指标未变化"
        ),
    }
    layer_labels = {
        "score": "比分",
        "chance_creation": "机会创造",
        "possession_and_progression": "控球与推进",
        "complex_system_state": "复杂系统状态",
        "world_model_mechanism": "世界模型机制",
    }
    layer_cards = []
    for raw in future.get("layers") or []:
        if not isinstance(raw, Mapping):
            continue
        layer_id = _safe_text(raw.get("layer"), 80)
        layer_cards.append(
            '<div><span class="eyebrow">'
            + html.escape(layer_labels.get(layer_id, layer_id))
            + '</span><strong>'
            + html.escape(str(max(0, int(_finite(
                raw.get("difference_count")
            ) or 0))))
            + ' 项差异</strong><span>'
            + html.escape(str(max(0, int(_finite(
                raw.get("observed_metric_count")
            ) or 0))))
            + ' 项已测</span></div>'
        )
    status = _safe_text(future.get("status"), 100)
    status_label = status_labels.get(status, status or "unknown")
    clock = _safe_text(point.get("clock"), 20) or "—"
    changed = max(0, int(_finite(summary.get("changed_actions")) or 0))
    local = max(
        0, int(_finite(summary.get("locally_attributable_actions")) or 0),
    )
    downstream = max(0, int(_finite(
        summary.get("descriptive_outcome_difference_count")
    ) or 0))
    local_authorized = bool(
        isinstance(authority, Mapping)
        and authority.get("simulator_local_action_attribution")
    )
    return f"""<section class="card future-summary" data-testid="counterfactual-future-summary">
<div class="pair-panel-head"><div><span class="eyebrow">One intervention · two futures · graded evidence</span><h2>反事实未来总览</h2></div><span class="pair-badge {'changed' if local_authorized else ''}">{html.escape(status_label)}</span></div>
<p>在 <strong>{html.escape(clock)}</strong> 从同一已验证前缀分叉：基线世界保持 <code>predict_only</code>，干预世界启用 <code>action_policy</code>。</p>
<div class="prop-stages">
<div class="prop-stage"><span>共同过去</span><strong>{'已验证' if point.get('anchor_verified') else '未验证'}</strong><small>{html.escape(str(point.get('clock_contract') or 'clock unknown'))}</small></div>
<div class="prop-stage"><span>动作分叉</span><strong>{changed} 次</strong><small>{local} 次局部归因</small></div>
<div class="prop-stage"><span>未来差异</span><strong>{downstream} 项</strong><small>跨复杂系统层的描述性指标</small></div>
<div class="prop-stage"><span>最高权限</span><strong>{'模拟器局部动作归因' if local_authorized else '配对描述'}</strong><small>不授权下游赛果因果</small></div>
</div>
<div class="pair-summary">{''.join(layer_cards)}</div>
<p class="inference-boundary"><strong>阅读方式：</strong>共同过去和局部动作可以按证据逐级核验；轨迹、比分及复杂系统状态差异只描述这一个 fixture/seed 的两个模拟未来，不是总体效应、真实足球因果或模型晋级证据。</p>
</section>"""


def _society_divergence_panel(comparison: Mapping[str, Any]) -> str:
    raw = comparison.get("society_divergence")
    if raw is None:
        return """<section class="card" data-testid="society-divergence-panel">
<h2>复杂系统终局分叉</h2><p class="muted">旧版配对证据没有终局社会状态快照；缺失不能解释为零变化。</p></section>"""
    try:
        divergence = validate_paired_society_divergence(raw)
    except ValueError:
        return """<section class="card" data-testid="society-divergence-panel">
<h2>复杂系统终局分叉</h2><p class="warn">社会状态证据校验失败，已拒绝展示。</p></section>"""
    if not divergence["available"]:
        reason = html.escape(str(divergence["reason"]))
        return f"""<section class="card" data-testid="society-divergence-panel">
<h2>复杂系统终局分叉</h2><p class="muted">终局社会状态证据不可用：{reason}。缺失不能解释为零变化。</p></section>"""
    scope_labels = {
        "tactical_controls": "战术控制",
        "emotion_profile": "情绪",
        "social_narrative_state": "社会叙事",
        "psychological_state": "心理",
        "referee_grievance": "裁判积怨",
    }
    field_labels = {
        "pressing_intensity": "压迫强度", "risk_budget": "风险预算",
        "line_height": "防线高度",
        "rotation_aggressiveness": "轮换激进度", "pride": "自豪",
        "anger": "愤怒", "shame": "羞耻", "fear": "恐惧",
        "determination": "决心", "trust_index": "信任指数",
        "polarization": "极化", "narrative_fatigue": "叙事疲劳",
        "morale": "士气", "pressure": "压力", "trust": "信任",
        "risk_appetite": "风险偏好", "conflict": "冲突",
        "audience_activation": "观众激活", "value": "数值",
    }
    count_labels = {
        "memory_records": "记忆记录", "cognitive_memory_records": "认知记忆",
        "beliefs": "信念", "reflections": "反思",
    }
    rows = []
    for row in divergence["count_deltas"]:
        rows.append(
            "<tr><td>" + html.escape(row["team"]) + "</td><td>社会记录</td><td>"
            + html.escape(count_labels[row["field"]])
            + "</td><td colspan='2'>Δ "
            + html.escape(f"{row['delta']:+d}") + "</td></tr>"
        )
    for row in divergence["state_changes"]:
        rows.append(
            "<tr><td>" + html.escape(row["team"]) + "</td><td>"
            + html.escape(scope_labels[row["scope"]]) + "</td><td>"
            + html.escape(field_labels[row["field"]]) + "</td><td>"
            + html.escape(f"{float(row['baseline']):.3f}") + "</td><td>"
            + html.escape(f"{float(row['treatment']):.3f}") + "</td></tr>"
        )
    body = "".join(rows) or (
        '<tr><td colspan="5">双方终局社会状态在已测字段中完全一致。</td></tr>'
    )
    return f"""<section class="card" data-testid="society-divergence-panel">
<div class="pair-panel-head"><div><span class="eyebrow">Paired terminal state · descriptive only</span><h2>复杂系统终局分叉</h2></div><span class="pair-badge {'changed' if rows else ''}">{divergence['state_change_count'] + divergence['count_delta_count']} 项差异</span></div>
<p class="muted">比较同一分叉时点下基线世界与干预世界最终写出的无文本社会、心理和战术控制状态。</p>
<div class="scroll"><table><thead><tr><th>球队</th><th>状态域</th><th>字段</th><th>基线世界</th><th>干预世界</th></tr></thead><tbody>{body}</tbody></table></div>
<p class="inference-boundary">这是身份绑定的终局状态差异。局部动作归因不会自动传递到社会状态、心理状态或赛果；不构成现实球队结论。</p></section>"""


def _policy_propagation_panel(comparison: Mapping[str, Any]) -> str:
    propagation = comparison.get("policy_propagation") or {}
    if not isinstance(propagation, Mapping) or not propagation.get("available"):
        reason = _safe_text(
            propagation.get("status") if isinstance(propagation, Mapping)
            else "invalid_propagation_evidence",
        )
        return f"""<section class="card propagation" data-testid="policy-propagation-panel">
<h2>动作分歧与传播链</h2><p class="muted">传播证据不可用：{html.escape(reason or 'unknown')}。双世界指标仍可独立查看。</p></section>"""
    summary = propagation.get("summary") or {}
    if not isinstance(summary, Mapping):
        summary = {}

    def count(key: str) -> int:
        value = _finite(summary.get(key))
        return max(0, min(100_000, int(value or 0)))

    changed = count("valid_changed_decisions")
    direct = count("directly_observed_changes")
    attributable = count("locally_attributable_changes")
    replay_windows = bool(summary.get("replay_windows_available"))
    eligibility = comparison.get("eligibility") or {}
    pair_eligible = bool(
        isinstance(eligibility, Mapping)
        and eligibility.get("eligible_for_world_model_policy_attribution")
    )
    stage_cards = (
        '<div class="prop-stage"><span>1 · 策略分配</span><strong>'
        f'{"隔离检查通过" if pair_eligible else "仅描述"}</strong>'
        '<small>同队、同战术、同 seed、同检查点</small></div>'
        '<div class="prop-stage"><span>2 · 局部动作</span>'
        f'<strong>{changed} 次改变</strong><small>同一随机抽样下的实际动作分歧</small></div>'
        '<div class="prop-stage"><span>3 · 运行轨迹</span>'
        f'<strong>{direct} 次直接观察</strong><small>{attributable} 次满足局部归因资格</small></div>'
        '<div class="prop-stage"><span>4 · 后续传播</span>'
        f'<strong>{"窗口可查看" if replay_windows else "回放不足"}</strong>'
        '<small>只按共享时钟描述，不声称下游因果</small></div>'
    )
    raw_decisions = propagation.get("decisions") or []
    if not isinstance(raw_decisions, (list, tuple)):
        raw_decisions = []
    decision_cards = []
    for raw in raw_decisions[:MAX_POLICY_PROPAGATION_DECISIONS]:
        if not isinstance(raw, Mapping):
            continue
        clock = _safe_text(raw.get("clock"), 20) or "—"
        team = _safe_text(raw.get("team")) or "unknown"
        before = _safe_text(raw.get("baseline_action"), 40) or "unknown"
        after = _safe_text(raw.get("treatment_action"), 40) or "unknown"
        direct_label = (
            "直接轨迹已绑定" if raw.get("directly_observed")
            else "直接轨迹未绑定"
        )
        local_label = (
            "局部归因合格" if raw.get("local_policy_attribution_eligible")
            else "仅记录动作分歧"
        )
        window_rows = []
        windows = raw.get("downstream_windows") or {}
        if isinstance(windows, Mapping):
            for label in ("30s", "120s"):
                window = windows.get(label)
                if not isinstance(window, Mapping):
                    continue
                delta = window.get("delta")
                if not isinstance(delta, Mapping):
                    continue
                values = []
                for metric in (
                    "passes", "crosses", "shots", "goals", "turnovers",
                ):
                    number = _finite(delta.get(metric))
                    values.append(
                        "—" if number is None else f"{int(number):+d}"
                    )
                extra = _finite(window.get("additional_policy_changes"))
                window_rows.append(
                    f'<tr><td>+{html.escape(label)}</td>'
                    + "".join(f"<td>{value}</td>" for value in values)
                    + f'<td>{max(0, int(extra or 0))}</td></tr>'
                )
        if window_rows:
            metric_headers = "".join(
                f"<th>{label}</th>" for label in (
                    "Passes Δ", "Crosses Δ", "Shots Δ", "Goals Δ",
                    "Turnovers Δ",
                )
            )
            window_table = (
                '<div class="scroll"><table><thead><tr>'
                '<th>Shared-clock window</th>'
                + metric_headers
                + '<th>Other policy changes</th></tr></thead><tbody>'
                + "".join(window_rows)
                + "</tbody></table></div>"
            )
        else:
            window_table = (
                '<p class="muted">双方回放不足，无法构建后续共享时钟窗口。</p>'
            )
        decision_cards.append(
            '<details class="prop-decision"><summary>'
            f'<span>{html.escape(clock)} · {html.escape(team)}</span>'
            f'<strong>{html.escape(before)} → {html.escape(after)}</strong>'
            f'<small>{direct_label} · {local_label}</small></summary>'
            f'{window_table}<p class="muted">窗口从该决策时刻开始；其中可能包含其他策略改变，'
            '只比较两场在同一比赛时钟区间内的事件计数，不匹配跨世界事件。</p></details>'
        )
    if not decision_cards:
        decision_cards.append(
            '<p class="muted">本次共享随机条件下没有产生可保留的实际动作改变。'
            '这仍是有效结果：策略概率发生影响并不保证跨过抽样边界。</p>'
        )
    truncation = (
        '<p class="muted">动作改变记录超过页面上限，仅保留最早 40 条；汇总仍使用全部有效记录。</p>'
        if summary.get("decisions_truncated") else ""
    )
    boundary = _safe_text(propagation.get("claim_boundary"), 500)
    return f"""<section class="card propagation" data-testid="policy-propagation-panel">
<div class="pair-panel-head"><div><span class="eyebrow">Policy → action → trajectory → outcome</span><h2>动作分歧与传播链</h2></div><span class="pair-badge changed">局部归因 {attributable}</span></div>
<p class="muted">这条链把已证明的局部动作改变与后续描述性差异分开显示；越过“运行轨迹”后不自动继承因果资格。</p>
<div class="prop-stages">{stage_cards}</div><div class="prop-decisions">{''.join(decision_cards)}</div>{truncation}
<p class="inference-boundary"><strong>推断边界：</strong>{html.escape(boundary)}</p></section>"""


def render_paired_comparison_html(comparison: Mapping[str, Any]) -> str:
    fixture = comparison.get("fixture") or {}
    eligibility = comparison.get("eligibility") or {}
    intervention = comparison.get("intervention") or {}
    world_model_fork = intervention.get("scope") == "world_model_action_policy"
    eligible = bool(
        eligibility.get("eligible_for_world_model_policy_attribution")
        if world_model_fork else
        eligibility.get("eligible_for_tactical_attribution")
    )
    state = "配对资格通过" if eligible else "仅描述性比较"
    before = intervention.get("baseline_tactics") or {}
    after = intervention.get("treatment_tactics") or {}
    if world_model_fork:
        page_title = "世界模型因果分叉"
        eyebrow = "GFS Football Causal World Lab · shared-seed policy fork"
        baseline_title = "基线世界"
        treatment_title = "干预世界"
        baseline_body = (
            "世界模型保持加载并预测，但 MATCH_WM_PLAN=0，"
            "预测不得进入动作策略。"
        )
        treatment_body = (
            "使用同一检查点并设置 MATCH_WM_PLAN=1，"
            "仅经质量门控的信号可以改变动作概率。"
        )
        intervention_explanation = (
            "唯一计划干预：predict_only → action_policy；球队、战术、"
            "快速配置、检查点与随机种子保持一致。"
        )
    else:
        page_title = "战术配对比较"
        eyebrow = "GFS Tactical Lab · shared-seed contrast"
        baseline_title = "基线战术"
        treatment_title = "处理战术"
        baseline_body = (
            f"主队：{_display_tactic(before.get('home'))}<br>"
            f"客队：{_display_tactic(before.get('away'))}"
        )
        treatment_body = (
            f"主队：{_display_tactic(after.get('home'))}<br>"
            f"客队：{_display_tactic(after.get('away'))}"
        )
        intervention_explanation = str(intervention.get("scope"))
    if world_model_fork:
        anchor = intervention.get("branch_anchor") or {}
        branch_sec = _finite(intervention.get("branch_at_sec"))
        branch_minute = branch_sec / 60.0 if branch_sec is not None else None
        anchor_state = "verified" if anchor.get("verified") else "not verified"
        identity = _safe_text(anchor.get("state_identity"), 64)
        branch_label = f"minute {branch_minute:g}" if branch_minute is not None else "match start"
        intervention_explanation = (
            f"{intervention_explanation} Branch at {branch_label}; "
            f"pre-intervention anchor {anchor_state}"
            f"{f'; identity {identity[:16]}' if identity else ''}. "
            "Execution uses deterministic replay, not a serialized process resume."
        )
    metric_rows = []
    for name, values in (comparison.get("metrics") or {}).items():
        values = values or {}
        metric_rows.append(
            "<tr>"
            f"<td>{html.escape(str(name))}</td>"
            f"<td>{html.escape(str(values.get('baseline')))}</td>"
            f"<td>{html.escape(str(values.get('treatment')))}</td>"
            f"<td>{html.escape(str(values.get('delta')))}</td></tr>"
        )
    check_rows = "".join(
        f"<li class=\"{'ok' if passed else 'warn'}\">"
        f"{html.escape(str(name))}: {'通过' if passed else '未通过'}</li>"
        for name, passed in (eligibility.get("checks") or {}).items()
    )
    reports = comparison.get("reports") or {}
    report_links = []
    for label, key in (("打开基线场", "baseline"), ("打开处理场", "treatment")):
        source = reports.get(key)
        if source:
            filename = Path(str(source)).with_suffix(".html").name
            report_links.append(
                f'<a href="{html.escape(filename, quote=True)}">{label}</a>'
            )
    navigation = " · ".join(report_links)
    propagation_panel = (
        _policy_propagation_panel(comparison) if world_model_fork else ""
    )
    future_panel = (
        _counterfactual_future_panel(comparison) if world_model_fork else ""
    )
    society_panel = (
        _society_divergence_panel(comparison) if world_model_fork else ""
    )
    replay_panel = _paired_replay_panel(comparison)
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(str(fixture.get('home')))} vs {html.escape(str(fixture.get('away')))} · {page_title}</title>
<style>:root{{--bg:#07111e;--panel:#111d2e;--line:#2a3a51;--ink:#edf4ff;--muted:#a8b6c9;--ok:#65e6b4;--warn:#ffc36a}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif}}main{{max-width:1180px;margin:auto;padding:32px 20px}}h1{{font-size:clamp(28px,6vw,54px);margin:.2em 0}}.eyebrow,.muted{{color:var(--muted)}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px;margin:16px 0}}.ok{{color:var(--ok)}}.warn{{color:var(--warn)}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:9px;border-bottom:1px solid var(--line)}}.scroll{{overflow:auto}}code{{color:var(--ok)}}.pair-panel-head{{display:flex;justify-content:space-between;gap:12px;align-items:start}}.pair-panel-head h2{{margin:.2em 0}}.pair-badge{{display:inline-block;border:1px solid #42516a;border-radius:999px;padding:2px 8px;color:#cbd7e8;font-size:12px}}.pair-badge.changed{{border-color:var(--ok);color:var(--ok)}}.pair-summary{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;margin:14px 0}}.pair-summary>div{{display:grid;background:#0a1424;border:1px solid var(--line);border-radius:9px;padding:10px}}.pair-summary strong{{font-size:18px}}.prop-stages{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:16px 0}}.prop-stage{{display:grid;gap:4px;background:#0a1424;border:1px solid var(--line);border-radius:10px;padding:12px;position:relative}}.prop-stage:not(:last-child)::after{{content:'→';position:absolute;right:-12px;top:32%;z-index:2;color:var(--ok)}}.prop-stage span,.prop-stage small{{color:var(--muted)}}.prop-decision{{border:1px solid var(--line);border-radius:10px;margin:8px 0;background:#0a1424}}.prop-decision summary{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;cursor:pointer;padding:12px}}.prop-decision summary small{{color:var(--muted)}}.prop-decision .scroll,.prop-decision>p{{margin:0 12px 12px}}.inference-boundary{{border-left:3px solid var(--warn);padding-left:12px;color:var(--muted)}}.pair-controls{{display:flex;flex-wrap:wrap;gap:7px;border:0;padding:0;margin:0}}.pair-controls legend{{width:100%;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.1em}}.pair-controls>input{{position:absolute;opacity:0;pointer-events:none}}.pair-controls>label{{border:1px solid #42516a;border-radius:999px;padding:6px 10px;cursor:pointer}}.pair-controls>input:focus-visible+label{{outline:3px solid var(--warn);outline-offset:2px}}.pair-controls>input:checked+label{{border-color:var(--ok);color:var(--ok);background:#10281f}}.pair-timeline{{display:flex;gap:5px;overflow:auto;width:100%;padding:12px 2px 5px}}.pair-timeline label{{flex:0 0 auto;min-width:42px;border:1px solid #42516a;border-radius:8px;padding:4px 7px;text-align:center;cursor:pointer;font-size:12px}}.pair-timeline label span{{display:block;color:var(--muted);font-size:10px}}.pair-navigation{{width:100%;margin-top:8px}}.pair-frame{{display:none;align-items:center;justify-content:space-between;gap:12px;background:#0a1424;border:1px solid var(--line);border-radius:10px;padding:10px}}.pair-frame>div{{display:grid;gap:3px;text-align:center;min-width:0}}.pair-frame-center{{flex:1}}.pair-current{{display:grid;grid-template-columns:1fr 1fr;gap:8px;text-align:left}}.pair-current>div{{display:grid;gap:2px;background:#0d192a;border-radius:8px;padding:8px}}.pair-current>div>span{{display:block}}.pair-button{{border:1px solid #42516a;border-radius:8px;padding:7px 10px;cursor:pointer;color:var(--ok);white-space:nowrap}}.pair-pitches{{display:grid;grid-template-columns:1fr 1fr;gap:12px;width:100%;margin-top:12px}}.pair-side{{background:#071c19;border:1px solid #285448;border-radius:12px;padding:10px}}.pair-side h3{{margin:0 0 6px}}.pair-side svg{{display:block;width:100%;height:auto}}.pair-pitch{{fill:#0c392d;stroke:#b8d8cd;stroke-width:3}}.pair-marking{{fill:none;stroke:#b8d8cd;stroke-width:3}}.pair-spot{{fill:#b8d8cd}}.pair-event line{{stroke-width:4;stroke-linecap:round;opacity:.42}}.pair-event circle{{opacity:.75}}.pair-home line,.pair-home circle{{stroke:#65e6b4;fill:#65e6b4}}.pair-away line,.pair-away circle{{stroke:#ff9f7a;fill:#ff9f7a}}.pair-shot line{{stroke-width:7;opacity:.85}}.pair-wm-changed line{{filter:drop-shadow(0 0 5px #fff);opacity:1}}#pair-filter-shots:checked~.pair-pitches .pair-event:not(.pair-shot),#pair-filter-wm:checked~.pair-pitches .pair-event:not(.pair-wm-changed){{display:none}}@media(max-width:760px){{th,td{{padding:7px;font-size:12px}}.pair-pitches,.pair-current,.prop-stages,.prop-decision summary{{grid-template-columns:1fr}}.prop-stage:not(:last-child)::after{{content:'↓';right:50%;top:auto;bottom:-17px}}.pair-frame{{flex-wrap:wrap}}.pair-frame-center{{order:-1;width:100%;flex-basis:100%}}.pair-button{{flex:1;text-align:center}}.pair-panel-head{{display:block}}}}</style></head><body><main>
<div class="eyebrow">{eyebrow}</div><h1>{html.escape(str(fixture.get('home')))} vs {html.escape(str(fixture.get('away')))}</h1><p>seed <code>{html.escape(str(fixture.get('seed')))}</code> · <strong class="{'ok' if eligible else 'warn'}">{state}</strong></p><p>{navigation}</p>
<section class="grid"><article class="card"><h2>{baseline_title}</h2><p>{baseline_body}</p></article><article class="card"><h2>{treatment_title}</h2><p>{treatment_body}</p></article><article class="card"><h2>干预范围</h2><p>{html.escape(intervention_explanation)}</p></article></section>
<section class="card"><h2>配对资格检查</h2><ul>{check_rows}</ul></section>
{future_panel}
{propagation_panel}
{society_panel}
{replay_panel}
<section class="card"><h2>处理场减去基线场</h2><div class="scroll"><table><thead><tr><th>指标</th><th>基线</th><th>处理</th><th>差值</th></tr></thead><tbody>{''.join(metric_rows)}</tbody></table></div></section>
<section class="card"><h2>推断边界</h2><p class="muted">{html.escape(str(comparison.get('claim_boundary') or ''))}</p></section>
</main></body></html>"""


def write_paired_comparison_html(
    path: str | Path, comparison: Mapping[str, Any],
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_paired_comparison_html(comparison), encoding="utf-8")
    return target

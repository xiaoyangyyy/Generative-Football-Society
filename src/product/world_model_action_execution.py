"""Bounded official-match evidence for world-model action execution."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any, Mapping


SCHEMA_VERSION = 1
MAX_SOURCE_RECORDS = 96
MAX_EXAMPLES = 5
_ACTIONS = {"hold", "pass", "cross", "shot"}
_DIRECT_LINK = "direct_runtime_identity_match"
_NO_TRAJECTORY = "no_ball_trajectory_by_design"


def _identity(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} is invalid")
    return value


def _number(
    value: Any, *, name: str, minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} is invalid")
    result = float(value)
    if (
        not math.isfinite(result)
        or result < minimum
        or (maximum is not None and result > maximum)
    ):
        raise ValueError(f"{name} is invalid")
    return result


def _signed_number(
    value: Any, *, name: str, absolute_maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} is invalid")
    result = float(value)
    if not math.isfinite(result) or abs(result) > absolute_maximum:
        raise ValueError(f"{name} is invalid")
    return result


def _text(value: Any, *, name: str, limit: int) -> str:
    if not isinstance(value, str) or not value or len(value) > limit:
        raise ValueError(f"{name} is invalid")
    return value


def _unavailable(reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "available": False,
        "reason": reason,
    }


def _validate_source_aggregate(
    adoption: Mapping[str, Any], records: list[Mapping[str, Any]],
) -> tuple[int, bool]:
    opportunities = _integer(
        adoption.get("opportunities"), name="action opportunities",
    )
    influenced = _integer(
        adoption.get("influenced_opportunities"),
        name="influenced opportunities",
    )
    eligible = _integer(
        adoption.get("attribution_eligible_opportunities"),
        name="attribution eligible opportunities",
    )
    changed = _integer(
        adoption.get("counterfactual_action_changes"),
        name="counterfactual action changes",
    )
    _number(
        adoption.get("expected_counterfactual_action_changes"),
        name="expected counterfactual action changes",
    )
    _number(
        adoption.get("mean_recommended_probability_shift"),
        name="mean recommended probability shift", maximum=1.0,
    )
    mean_primary = adoption.get("mean_primary_signal_probability_shift")
    if mean_primary is not None:
        _number(
            mean_primary, name="mean primary signal probability shift",
            maximum=1.0,
        )
    if not (
        len(records) <= opportunities
        and influenced <= opportunities
        and eligible <= opportunities
        and changed <= eligible
    ):
        raise ValueError("world-model action aggregate hierarchy is invalid")
    retained = adoption.get("records_retained")
    if retained is not None and _integer(
        retained, name="retained action records",
    ) != len(records):
        raise ValueError("world-model retained action count is inconsistent")
    truncated = adoption.get("records_truncated")
    expected_truncated = opportunities > len(records)
    if truncated is not None and (
        not isinstance(truncated, bool) or truncated is not expected_truncated
    ):
        raise ValueError("world-model action truncation state is inconsistent")
    return opportunities, expected_truncated


def _validate_links(replay: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    linkage = replay.get("world_model_action_links")
    if not isinstance(linkage, Mapping) or linkage.get("available") is not True:
        return {}
    links = linkage.get("links")
    if not isinstance(links, list) or len(links) > MAX_SOURCE_RECORDS:
        raise ValueError("world-model action links are invalid")
    if (
        linkage.get("causal_claim_authorized") is not False
        or _integer(linkage.get("records"), name="action link records")
        != len(links)
    ):
        raise ValueError("world-model action linkage boundary is invalid")
    by_identity: dict[str, Mapping[str, Any]] = {}
    direct = changed_direct = no_trajectory = unresolved = 0
    for raw in links:
        if not isinstance(raw, Mapping):
            raise ValueError("world-model action link row is invalid")
        identity = _text(
            raw.get("opportunity_id"), name="action link identity", limit=180,
        )
        if identity in by_identity:
            raise ValueError("world-model action link identity is duplicated")
        status = _text(
            raw.get("status"), name="action link status", limit=80,
        )
        direct += int(status == _DIRECT_LINK)
        changed_direct += int(
            status == _DIRECT_LINK and raw.get("policy_changed_action") is True
        )
        no_trajectory += int(status == _NO_TRAJECTORY)
        unresolved += int(status not in {_DIRECT_LINK, _NO_TRAJECTORY})
        by_identity[identity] = raw
    expected = {
        "directly_observed": direct,
        "counterfactual_changed_and_observed": changed_direct,
        "no_trajectory_by_design": no_trajectory,
        "unresolved_or_missing": unresolved,
    }
    if any(
        _integer(linkage.get(key), name=f"action linkage {key}") != value
        for key, value in expected.items()
    ):
        raise ValueError("world-model action linkage aggregate is inconsistent")
    return by_identity


def project_world_model_action_execution(
    report: Mapping[str, Any], *, manager_team: str,
    expected_match_id: str,
) -> dict[str, Any]:
    """Project manager-team action evidence from one official match report."""
    if (
        not isinstance(report, Mapping)
        or report.get("match_id") != expected_match_id
    ):
        return _unavailable("report_identity_mismatch")
    fixture = report.get("fixture")
    if (
        not isinstance(fixture, Mapping)
        or manager_team not in {fixture.get("home"), fixture.get("away")}
    ):
        return _unavailable("manager_team_identity_mismatch")
    layers = report.get("layers")
    world_model = layers.get("world_model") if isinstance(layers, Mapping) else None
    if not isinstance(world_model, Mapping):
        return _unavailable("legacy_report_without_world_model_layer")
    if world_model.get("configured") is not True:
        return _unavailable("world_model_not_configured_for_match")
    if world_model.get("enabled") is not True:
        return _unavailable("world_model_not_observed_at_runtime")
    adoption = world_model.get("action_adoption")
    if not isinstance(adoption, Mapping) or adoption.get("available") is not True:
        reason = (
            adoption.get("reason")
            if isinstance(adoption, Mapping) else None
        )
        return _unavailable(
            str(reason or "world_model_action_opportunities_unavailable")[:160]
        )
    raw_records = adoption.get("records")
    if (
        not isinstance(raw_records, list)
        or len(raw_records) > MAX_SOURCE_RECORDS
        or any(not isinstance(row, Mapping) for row in raw_records)
    ):
        raise ValueError("world-model action records are invalid")
    records = list(raw_records)
    source_opportunities, source_truncated = _validate_source_aggregate(
        adoption, records,
    )
    probability_policy_version = str(
        adoption.get("probability_policy_version") or "legacy_unversioned"
    )[:80]
    replay = report.get("replay")
    replay = replay if isinstance(replay, Mapping) else {}
    links = _validate_links(replay)

    seen: set[str] = set()
    manager_rows = []
    for raw in records:
        opportunity_id = _text(
            raw.get("opportunity_id"), name="action opportunity identity",
            limit=180,
        )
        if opportunity_id in seen:
            raise ValueError("world-model action opportunity identity is duplicated")
        seen.add(opportunity_id)
        team = _text(raw.get("team_id"), name="action team", limit=128)
        t_sec = _number(
            raw.get("t_sec"), name="action timestamp", maximum=6000.0,
        )
        if team != manager_team:
            continue
        influenced = raw.get("influenced")
        eligible = raw.get("attribution_eligible")
        changed = raw.get("policy_changed_action")
        if influenced not in {True, False}:
            raise ValueError("world-model action influence flag is invalid")
        if eligible not in {True, False, None} or changed not in {
            True, False, None,
        }:
            raise ValueError("world-model action attribution flag is invalid")
        recommended = str(raw.get("recommended_action") or "none").lower()
        primary_signal = str(
            raw.get("primary_signal_action") or recommended
        ).lower()
        signal_mode = str(
            raw.get("signal_mode") or "legacy_unclassified"
        ).lower()
        actual = str(raw.get("actual_action") or "none").lower()
        baseline = str(
            raw.get("counterfactual_baseline_action") or "none"
        ).lower()
        if recommended not in _ACTIONS | {"none"}:
            raise ValueError("world-model recommended action is invalid")
        if primary_signal not in _ACTIONS | {"none"} or signal_mode not in {
            "direct_preference", "suppression_only", "none",
            "legacy_unclassified",
        }:
            raise ValueError("world-model primary action signal is invalid")
        if (
            signal_mode == "direct_preference"
            and recommended not in {"pass", "shot", "cross"}
        ) or (
            signal_mode == "suppression_only"
            and (
                recommended != "none"
                or primary_signal not in {"pass", "shot", "cross"}
            )
        ) or (
            signal_mode == "none"
            and (recommended != "none" or primary_signal != "none")
        ):
            raise ValueError("world-model action signal semantics are invalid")
        if actual not in _ACTIONS | {"none"} or baseline not in _ACTIONS | {"none"}:
            raise ValueError("world-model sampled action is invalid")
        if changed is True and (
            eligible is not True
            or influenced is not True
            or actual == "none"
            or baseline == "none"
            or actual == baseline
        ):
            raise ValueError("world-model changed-action semantics are invalid")
        if eligible is True and influenced is not True:
            raise ValueError("world-model attribution eligibility is inconsistent")
        probability_delta = raw.get("recommended_probability_delta")
        probability_delta = (
            None
            if probability_delta is None else
            _signed_number(
                probability_delta,
                name="recommended probability delta", absolute_maximum=1.0,
            )
        )
        primary_probability_delta = raw.get(
            "primary_signal_probability_delta"
        )
        primary_probability_delta = (
            probability_delta
            if primary_probability_delta is None
            else _signed_number(
                primary_probability_delta,
                name="primary signal probability delta",
                absolute_maximum=1.0,
            )
        )
        reference = raw.get("reference_action_effect")
        if reference is not None and not isinstance(reference, Mapping):
            raise ValueError("world-model reference action effect is invalid")
        if isinstance(reference, Mapping):
            reference_delta = _signed_number(
                reference.get("probability_delta"),
                name="reference action probability delta",
                absolute_maximum=1.0,
            )
            reference_projection = {
                "available": True,
                "action": str(reference.get("action") or "").lower(),
                "role": str(reference.get("role") or "")[:80],
                "directly_authorized": reference.get("directly_authorized"),
                "probability_delta": round(reference_delta, 9),
                "received_redistributed_probability": reference.get(
                    "received_redistributed_probability"
                ),
                "realized_as_actual_action": reference.get(
                    "realized_as_actual_action"
                ),
                "policy_changed_to_reference": reference.get(
                    "policy_changed_to_reference"
                ),
            }
            if (
                reference_projection["action"] != "hold"
                or reference_projection["role"]
                != "counterfactual_baseline_only"
                or reference_projection["directly_authorized"] is not False
                or any(
                    not isinstance(reference_projection[key], bool)
                    for key in (
                        "received_redistributed_probability",
                        "realized_as_actual_action",
                        "policy_changed_to_reference",
                    )
                )
                or reference_projection["realized_as_actual_action"]
                is not (actual == "hold")
                or reference_projection["policy_changed_to_reference"]
                is not (changed is True and actual == "hold")
            ):
                raise ValueError("world-model reference action semantics are invalid")
        else:
            reference_projection = {
                "available": False,
                "action": "hold",
                "role": "legacy_reference_effect_unavailable",
                "directly_authorized": False,
                "probability_delta": None,
                "received_redistributed_probability": False,
                "realized_as_actual_action": actual == "hold",
                "policy_changed_to_reference": changed is True and actual == "hold",
            }
        total_variation = raw.get("total_variation_distance")
        total_variation = (
            None
            if total_variation is None else
            _number(
                total_variation,
                name="action total variation", maximum=1.0,
            )
        )
        link = links.get(opportunity_id)
        if link is not None and link.get("team") != manager_team:
            raise ValueError("world-model action link team is inconsistent")
        if link is not None and (
            str(link.get("recommended_action") or "none").lower()
            != recommended
            or str(
                link.get("counterfactual_baseline_action") or "none"
            ).lower() != baseline
            or str(link.get("actual_action") or "none").lower() != actual
            or link.get("policy_changed_action") is not (changed is True)
            or link.get("attribution_eligible") is not (eligible is True)
            or abs(_number(
                link.get("t_sec"), name="action link timestamp",
                maximum=6000.0,
            ) - t_sec) > 1e-6
        ):
            raise ValueError("world-model action record/link evidence is inconsistent")
        link_status = str(
            link.get("status") if isinstance(link, Mapping)
            else "no_link_record"
        )[:80]
        if (
            link_status == _DIRECT_LINK
            and (
                actual not in {"pass", "shot", "cross"}
                or str(link.get("event_type") or "").lower() != actual
            )
        ) or (
            link_status == _NO_TRAJECTORY
            and actual != "hold"
        ):
            raise ValueError("world-model action runtime link semantics are invalid")
        manager_rows.append({
            "opportunity_identity": _identity({
                "match_id": expected_match_id,
                "team": manager_team,
                "opportunity_id": opportunity_id,
            }),
            "t_sec": round(t_sec, 6),
            "clock": f"{int(t_sec // 60)}:{int(t_sec % 60):02d}",
            "recommended_action": recommended,
            "counterfactual_baseline_action": baseline,
            "actual_action": actual,
            "influenced": influenced,
            "attribution_eligible": eligible is True,
            "policy_changed_action": changed is True,
            "recommended_probability_delta": (
                round(probability_delta, 9)
                if probability_delta is not None else None
            ),
            "policy_signal": {
                "probability_policy_version": probability_policy_version,
                "mode": signal_mode,
                "primary_action": primary_signal,
                "primary_probability_delta": (
                    round(primary_probability_delta, 9)
                    if primary_probability_delta is not None else None
                ),
                "direct_recommendation": recommended != "none",
            },
            "reference_action_effect": reference_projection,
            "total_variation_distance": (
                round(total_variation, 9)
                if total_variation is not None else None
            ),
            "runtime_link": {
                "status": link_status,
                "direct_ball_event_identity": link_status == _DIRECT_LINK,
                "event_type": (
                    str(link.get("event_type") or "")[:40]
                    if isinstance(link, Mapping) else ""
                ),
                "event_clock": (
                    str(link.get("event_clock") or "")[:20]
                    if isinstance(link, Mapping) else ""
                ),
            },
            "simulator_local_action_attribution": (
                changed is True and eligible is True
            ),
            "match_outcome_causality": False,
        })
    if not manager_rows:
        return _unavailable(
            "no_manager_team_action_records_retained"
            if not source_truncated else
            "manager_team_action_records_not_retained_or_truncated"
        )
    manager_rows.sort(key=lambda row: (
        row["policy_changed_action"],
        row["influenced"],
        row["total_variation_distance"] or 0.0,
        -row["t_sec"],
    ), reverse=True)
    resolved = [
        row for row in manager_rows if row["actual_action"] in _ACTIONS
    ]
    influenced = [row for row in manager_rows if row["influenced"]]
    eligible = [row for row in manager_rows if row["attribution_eligible"]]
    changed = [row for row in manager_rows if row["policy_changed_action"]]
    direct = [
        row for row in manager_rows
        if row["runtime_link"]["direct_ball_event_identity"]
    ]
    changed_direct = [
        row for row in changed
        if row["runtime_link"]["direct_ball_event_identity"]
    ]
    no_trajectory = [
        row for row in manager_rows
        if row["runtime_link"]["status"] == _NO_TRAJECTORY
    ]
    unresolved = [
        row for row in manager_rows
        if row["runtime_link"]["status"] not in {
            _DIRECT_LINK, _NO_TRAJECTORY,
        }
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "match_id": expected_match_id,
        "team": manager_team,
        "evidence_state": (
            "locally_attributable_action_changes_observed" if changed else
            "probability_influence_without_realized_action_change"
            if influenced else
            "no_nonzero_action_influence_observed"
        ),
        "retained_record_evidence": {
            "records": len(manager_rows),
            "resolved_action_decisions": len(resolved),
            "influenced_decisions": len(influenced),
            "attribution_eligible_decisions": len(eligible),
            "locally_attributable_action_changes": len(changed),
            "direct_ball_event_links": len(direct),
            "changed_direct_ball_event_links": len(changed_direct),
            "decision_only_no_trajectory": len(no_trajectory),
            "unresolved_or_missing_ball_event_links": len(unresolved),
            "source_match_opportunities": source_opportunities,
            "source_records_truncated": source_truncated,
            "manager_record_coverage_complete": not source_truncated,
        },
        "examples": copy.deepcopy(manager_rows[:MAX_EXAMPLES]),
        "examples_truncated": len(manager_rows) > MAX_EXAMPLES,
        "outcome_comparison_performed": False,
        "outcome_effect_estimate": None,
        "causal_effect_authorized": False,
        "claim_boundary": (
            "official-match retained action decisions, shared-uniform simulator-"
            "local action attribution and exact ball-event identity links only; "
            "no trajectory, score, tactic-quality or real-football causal claim"
        ),
    }
    payload["evidence_identity"] = _identity(payload)
    validate_world_model_action_execution(payload)
    return payload


def validate_world_model_action_execution(evidence: Mapping[str, Any]) -> None:
    """Validate a bounded projection without needing the source match report."""
    if not isinstance(evidence, Mapping) or evidence.get(
        "schema_version"
    ) != SCHEMA_VERSION:
        raise ValueError("world-model official action evidence schema is invalid")
    if evidence.get("available") is not True:
        if not isinstance(evidence.get("reason"), str) or not evidence.get("reason"):
            raise ValueError("unavailable world-model action evidence is invalid")
        return
    if (
        not isinstance(evidence.get("match_id"), str)
        or not evidence.get("match_id")
        or not isinstance(evidence.get("team"), str)
        or not evidence.get("team")
        or evidence.get("outcome_comparison_performed") is not False
        or evidence.get("outcome_effect_estimate") is not None
        or evidence.get("causal_effect_authorized") is not False
    ):
        raise ValueError("world-model official action evidence boundary is invalid")
    counts = evidence.get("retained_record_evidence")
    examples = evidence.get("examples")
    if (
        not isinstance(counts, Mapping)
        or not isinstance(examples, list)
        or len(examples) > MAX_EXAMPLES
        or len(examples) != min(
            MAX_EXAMPLES, int(counts.get("records") or 0),
        )
        or evidence.get("examples_truncated")
        is not (int(counts.get("records") or 0) > len(examples))
    ):
        raise ValueError("world-model official action evidence shape is invalid")
    names = (
        "records", "resolved_action_decisions", "influenced_decisions",
        "attribution_eligible_decisions",
        "locally_attributable_action_changes", "direct_ball_event_links",
        "changed_direct_ball_event_links", "decision_only_no_trajectory",
        "unresolved_or_missing_ball_event_links", "source_match_opportunities",
    )
    values = {name: _integer(counts.get(name), name=name) for name in names}
    expected_state = (
        "locally_attributable_action_changes_observed"
        if values["locally_attributable_action_changes"] else
        "probability_influence_without_realized_action_change"
        if values["influenced_decisions"] else
        "no_nonzero_action_influence_observed"
    )
    if not (
        evidence.get("evidence_state") == expected_state
        and
        values["locally_attributable_action_changes"]
        <= values["attribution_eligible_decisions"]
        <= values["records"]
        and values["locally_attributable_action_changes"]
        <= values["influenced_decisions"]
        and values["influenced_decisions"] <= values["records"]
        and values["resolved_action_decisions"] <= values["records"]
        and values["direct_ball_event_links"] <= values["records"]
        and values["changed_direct_ball_event_links"]
        <= min(
            values["direct_ball_event_links"],
            values["locally_attributable_action_changes"],
        )
        and values["decision_only_no_trajectory"] <= values["records"]
        and values["unresolved_or_missing_ball_event_links"]
        <= values["records"]
        and (
            values["direct_ball_event_links"]
            + values["decision_only_no_trajectory"]
            + values["unresolved_or_missing_ball_event_links"]
            == values["records"]
        )
        and values["records"] <= values["source_match_opportunities"]
        and isinstance(counts.get("source_records_truncated"), bool)
        and counts.get("manager_record_coverage_complete")
        is (not counts.get("source_records_truncated"))
    ):
        raise ValueError("world-model official action evidence counts are invalid")
    seen = set()
    changed_examples = 0
    for row in examples:
        if not isinstance(row, Mapping):
            raise ValueError("world-model official action example is invalid")
        identity = _text(
            row.get("opportunity_identity"),
            name="projected action identity", limit=64,
        )
        if (
            len(identity) != 64
            or any(character not in "0123456789abcdef" for character in identity)
            or identity in seen
        ):
            raise ValueError("projected action identity is invalid")
        seen.add(identity)
        if (
            row.get("recommended_action") not in _ACTIONS | {"none"}
            or row.get("counterfactual_baseline_action") not in _ACTIONS | {"none"}
            or row.get("actual_action") not in _ACTIONS | {"none"}
            or not isinstance(row.get("influenced"), bool)
            or not isinstance(row.get("attribution_eligible"), bool)
            or not isinstance(row.get("policy_changed_action"), bool)
            or row.get("match_outcome_causality") is not False
            or row.get("simulator_local_action_attribution")
            is not (
                row.get("policy_changed_action") is True
                and row.get("attribution_eligible") is True
            )
        ):
            raise ValueError("world-model official action example semantics are invalid")
        policy_signal = row.get("policy_signal")
        if policy_signal is not None:
            if (
                not isinstance(policy_signal, Mapping)
                or not isinstance(
                    policy_signal.get("probability_policy_version"), str,
                )
                or policy_signal.get("mode") not in {
                    "direct_preference", "suppression_only", "none",
                    "legacy_unclassified",
                }
                or policy_signal.get("primary_action") not in _ACTIONS | {"none"}
                or policy_signal.get("direct_recommendation")
                is not (row.get("recommended_action") != "none")
            ):
                raise ValueError(
                    "world-model official action signal projection is invalid"
                )
            primary_delta = policy_signal.get("primary_probability_delta")
            if primary_delta is not None:
                _signed_number(
                    primary_delta, name="projected primary signal delta",
                    absolute_maximum=1.0,
                )
        reference = row.get("reference_action_effect")
        if reference is not None and (
            not isinstance(reference, Mapping)
            or not isinstance(reference.get("available"), bool)
            or reference.get("action") != "hold"
            or reference.get("directly_authorized") is not False
            or not isinstance(
                reference.get("received_redistributed_probability"), bool,
            )
            or reference.get("realized_as_actual_action")
            is not (row.get("actual_action") == "hold")
            or reference.get("policy_changed_to_reference")
            is not (
                row.get("policy_changed_action") is True
                and row.get("actual_action") == "hold"
            )
        ):
            raise ValueError(
                "world-model official reference action projection is invalid"
            )
        if row.get("policy_changed_action") is True:
            if (
                row.get("influenced") is not True
                or row.get("actual_action") == row.get(
                    "counterfactual_baseline_action"
                )
            ):
                raise ValueError(
                    "world-model official changed-action example is invalid"
                )
            changed_examples += 1
        _number(
            row.get("t_sec"), name="projected action timestamp",
            maximum=6000.0,
        )
        probability_delta = row.get("recommended_probability_delta")
        if probability_delta is not None:
            _signed_number(
                probability_delta, name="projected probability delta",
                absolute_maximum=1.0,
            )
        total_variation = row.get("total_variation_distance")
        if total_variation is not None:
            _number(
                total_variation, name="projected total variation",
                maximum=1.0,
            )
        runtime = row.get("runtime_link")
        if (
            not isinstance(runtime, Mapping)
            or not isinstance(runtime.get("status"), str)
            or runtime.get("direct_ball_event_identity")
            is not (runtime.get("status") == _DIRECT_LINK)
        ):
            raise ValueError("world-model official action runtime link is invalid")
    if changed_examples != min(
        values["locally_attributable_action_changes"], len(examples),
    ):
        raise ValueError("world-model official action example priority is invalid")
    frozen = copy.deepcopy(dict(evidence))
    observed = frozen.pop("evidence_identity", None)
    if not isinstance(observed, str) or observed != _identity(frozen):
        raise ValueError("world-model official action evidence identity mismatch")


__all__ = [
    "MAX_EXAMPLES", "SCHEMA_VERSION",
    "project_world_model_action_execution",
    "validate_world_model_action_execution",
]

"""Bounded, defensive ball-action replay extraction for Studio reports."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping


MAX_REPLAY_FILE_BYTES = 4 * 1024 * 1024
MAX_SOURCE_EVENTS = 2_000
MAX_RETAINED_EVENTS = 240


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _text(value: Any, maximum: int = 100) -> str:
    text = str(value or "").strip()
    text = "".join(char for char in text if ord(char) >= 32)
    return text[:maximum]


def _point(value: Any) -> tuple[list[float] | None, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None, 0
    x, y = _finite(value[0]), _finite(value[1])
    if x is None or y is None:
        return None, 0
    clipped_x = min(1.0, max(0.0, x))
    clipped_y = min(1.0, max(0.0, y))
    return [clipped_x, clipped_y], int(clipped_x != x) + int(clipped_y != y)


def _evenly_select(indices: list[int], count: int) -> list[int]:
    if count <= 0 or not indices:
        return []
    if len(indices) <= count:
        return list(indices)
    if count == 1:
        return [indices[len(indices) // 2]]
    positions = {
        round(index * (len(indices) - 1) / (count - 1))
        for index in range(count)
    }
    return [indices[position] for position in sorted(positions)]


def _select_events(
    events: list[dict[str, Any]], priority_opportunity_ids: set[str],
) -> list[dict[str, Any]]:
    if len(events) <= MAX_RETAINED_EVENTS:
        return events
    identity_priority = [
        index for index, event in enumerate(events)
        if event.get("wm_action_opportunity_id") in priority_opportunity_ids
    ]
    semantic_priority = [
        index for index, event in enumerate(events)
        if event["type"] == "shot"
        or event["outcome"] in {"GOAL", "SAVED", "INTERCEPTED"}
        if index not in set(identity_priority)
    ]
    if len(identity_priority) >= MAX_RETAINED_EVENTS:
        selected = _evenly_select(identity_priority, MAX_RETAINED_EVENTS)
    else:
        semantic_budget = MAX_RETAINED_EVENTS - len(identity_priority)
        selected_semantic = _evenly_select(semantic_priority, semantic_budget)
        selected = identity_priority + selected_semantic
        priority_set = set(selected)
        remaining = [
            index for index in range(len(events)) if index not in priority_set
        ]
        selected += _evenly_select(
            remaining, MAX_RETAINED_EVENTS - len(selected)
        )
    return [events[index] for index in sorted(set(selected))]


def unavailable_replay(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "available": False,
        "reason": reason,
        "source_events": 0,
        "retained_events": 0,
        "events": [],
    }


def build_match_replay(
    path: str | Path | None, *, root: str | Path,
    home: str, away: str,
    priority_opportunity_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Extract a bounded action trajectory without trusting persisted JSONL."""
    if path is None or not str(path).strip():
        return unavailable_replay("ball_log_unavailable")
    workspace_root = Path(root).resolve()
    allowed_root = (workspace_root / "outputs" / "ball_log").resolve()
    candidate = Path(path)
    resolved = (
        candidate if candidate.is_absolute() else workspace_root / candidate
    ).resolve()
    try:
        resolved.relative_to(allowed_root)
    except ValueError:
        return unavailable_replay("ball_log_outside_workspace")
    if resolved.suffix.lower() != ".jsonl" or not resolved.is_file():
        return unavailable_replay("ball_log_unavailable")
    try:
        if resolved.stat().st_size > MAX_REPLAY_FILE_BYTES:
            return unavailable_replay("ball_log_too_large")
        lines = resolved.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return unavailable_replay("ball_log_unreadable")
    if not lines:
        return unavailable_replay("ball_log_empty")

    invalid_rows = 0
    coordinates_clipped = 0
    source_limit_reached = False
    meta: Mapping[str, Any] = {}
    try:
        first = json.loads(lines[0])
        if isinstance(first, Mapping) and isinstance(first.get("meta"), Mapping):
            meta = first["meta"]
            event_lines = lines[1:]
        else:
            event_lines = lines
    except (json.JSONDecodeError, TypeError):
        event_lines = lines
    if meta and (
        str(meta.get("home")) != home or str(meta.get("away")) != away
    ):
        return unavailable_replay("ball_log_fixture_mismatch")
    if len(event_lines) > MAX_SOURCE_EVENTS:
        event_lines = event_lines[:MAX_SOURCE_EVENTS]
        source_limit_reached = True

    events: list[dict[str, Any]] = []
    for source_index, line in enumerate(event_lines):
        try:
            raw = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            invalid_rows += 1
            continue
        if not isinstance(raw, Mapping) or raw.get("type") not in {
            "pass", "shot", "cross",
        }:
            invalid_rows += 1
            continue
        event_type = str(raw["type"])
        start, clipped = _point(raw.get("from_xy"))
        coordinates_clipped += clipped
        if start is None:
            invalid_rows += 1
            continue
        if event_type == "pass":
            end, clipped = _point(raw.get("land_xy"))
            actor = _text(raw.get("from"))
            target = _text(raw.get("to"))
        elif event_type == "cross":
            end, clipped = _point(raw.get("land_xy"))
            actor = _text(raw.get("crosser"))
            target = _text(raw.get("winner")) or _text(raw.get("contact"))
        else:
            team = _text(raw.get("team"))
            goal_x = 1.0 if team == home else 0.0 if team == away else (
                1.0 if start[0] < 0.5 else 0.0
            )
            end, clipped = [goal_x, 0.5], 0
            actor = _text(raw.get("shooter"))
            target = _text(raw.get("gk"))
        coordinates_clipped += clipped
        if end is None:
            invalid_rows += 1
            continue
        t_sec = _finite(raw.get("t_sec"))
        if t_sec is None:
            invalid_rows += 1
            continue
        t_sec = min(8_000.0, max(0.0, t_sec))
        event = {
            "source_index": source_index,
            "t_sec": t_sec,
            "clock": f"{int(t_sec // 60)}:{int(t_sec % 60):02d}",
            "type": event_type,
            "team": _text(raw.get("team")),
            "actor": actor,
            "target": target,
            "kind": _text(raw.get("kind"), 40),
            "start": start,
            "end": end,
            "outcome": _text(raw.get("outcome"), 40).upper(),
            "p_success": _finite(raw.get("p_success")),
            "xg": _finite(raw.get("xg")),
            "xg_added": _finite(raw.get("xg_added")),
            "wm_action_opportunity_id": _text(
                raw.get("wm_action_opportunity_id"), 180,
            ),
        }
        events.append(event)
    events.sort(key=lambda event: (event["t_sec"], event["source_index"]))
    retained = _select_events(events, {
        _text(identity, 180) for identity in (priority_opportunity_ids or set())
        if _text(identity, 180)
    })
    for event_index, event in enumerate(retained):
        event["event_index"] = event_index
    return {
        "schema_version": 1,
        "available": bool(retained),
        "reason": "available" if retained else "no_valid_ball_actions",
        "source_events": len(events),
        "retained_events": len(retained),
        "source_limit_reached": source_limit_reached,
        "retention_truncated": len(retained) < len(events),
        "invalid_rows": invalid_rows,
        "coordinates_clipped": coordinates_clipped,
        "logger_truncated": bool(meta.get("truncated", False)),
        "events": retained,
    }


def unavailable_action_links(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1, "available": False, "reason": reason,
        "records": 0, "directly_observed": 0,
        "counterfactual_changed_and_observed": 0,
        "no_trajectory_by_design": 0, "unresolved_or_missing": 0,
        "links": [],
    }


def build_world_model_action_links(
    replay: dict[str, Any], adoption: Any,
) -> dict[str, Any]:
    """Link policy records to ball actions only through runtime identity keys."""
    if not isinstance(replay, dict) or not replay.get("available"):
        return unavailable_action_links("action_replay_unavailable")
    if not isinstance(adoption, Mapping) or not adoption.get("available"):
        return unavailable_action_links("world_model_action_records_unavailable")
    records = [
        row for row in list(adoption.get("records") or [])[:96]
        if isinstance(row, Mapping)
    ]
    if not records:
        return unavailable_action_links("world_model_action_records_empty")
    events = [
        event for event in list(replay.get("events") or [])[:MAX_RETAINED_EVENTS]
        if isinstance(event, dict)
    ]
    by_identity: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        identity = _text(event.get("wm_action_opportunity_id"), 180)
        if identity:
            by_identity.setdefault(identity, []).append(event)
    record_identity_counts: dict[str, int] = {}
    for record in records:
        identity = _text(record.get("opportunity_id"), 180)
        if identity:
            record_identity_counts[identity] = (
                record_identity_counts.get(identity, 0) + 1
            )

    links = []
    direct_count = changed_direct_count = no_trajectory_count = 0
    unresolved_count = 0
    for record in records:
        identity = _text(record.get("opportunity_id"), 180)
        actual = _text(record.get("actual_action"), 40).lower()
        team = _text(record.get("team_id"))
        t_sec = _finite(record.get("t_sec"))
        link = {
            "opportunity_id": identity,
            "team": team,
            "t_sec": t_sec,
            "clock": (
                f"{int(max(0.0, t_sec) // 60)}:{int(max(0.0, t_sec) % 60):02d}"
                if t_sec is not None else "unknown"
            ),
            "recommended_action": _text(record.get("recommended_action"), 40).lower(),
            "counterfactual_baseline_action": _text(
                record.get("counterfactual_baseline_action"), 40,
            ).lower(),
            "actual_action": actual,
            "policy_changed_action": bool(record.get("policy_changed_action")),
            "attribution_eligible": bool(record.get("attribution_eligible")),
            "event_index": None, "event_type": None, "event_clock": None,
            "time_delta_seconds": None, "evidence_level": "unavailable",
        }
        candidates = by_identity.get(identity, []) if identity else []
        if identity and record_identity_counts.get(identity, 0) > 1:
            link["status"] = "ambiguous_action_record_identity_collision"
            unresolved_count += 1
        elif actual == "hold":
            link.update({
                "status": "no_ball_trajectory_by_design",
                "evidence_level": "decision_record_only",
            })
            no_trajectory_count += 1
        elif actual not in {"pass", "shot", "cross"}:
            link["status"] = "unresolved_or_unsupported_action"
            unresolved_count += 1
        elif not identity:
            link["status"] = "legacy_record_without_runtime_identity"
            unresolved_count += 1
        elif len(candidates) > 1:
            link["status"] = "ambiguous_runtime_identity_collision"
            unresolved_count += 1
        elif not candidates:
            link["status"] = (
                "action_event_may_be_truncated"
                if replay.get("retention_truncated")
                or replay.get("logger_truncated")
                or replay.get("source_limit_reached")
                else "action_event_missing"
            )
            unresolved_count += 1
        else:
            event = candidates[0]
            event_type = _text(event.get("type"), 40).lower()
            event_team = _text(event.get("team"))
            event_t = _finite(event.get("t_sec"))
            delta = (
                abs(event_t - t_sec)
                if event_t is not None and t_sec is not None else None
            )
            if event_type != actual:
                link["status"] = "runtime_identity_action_mismatch"
                unresolved_count += 1
            elif event_team != team:
                link["status"] = "runtime_identity_team_mismatch"
                unresolved_count += 1
            elif delta is None or delta > 1e-6:
                link["status"] = "runtime_identity_timestamp_mismatch"
                unresolved_count += 1
            else:
                event_index = int(_finite(event.get("event_index")) or 0)
                link.update({
                    "status": "direct_runtime_identity_match",
                    "evidence_level": "direct_runtime_identity",
                    "event_index": event_index,
                    "event_type": event_type,
                    "event_clock": _text(event.get("clock"), 20),
                    "time_delta_seconds": delta,
                })
                event["world_model_link"] = {
                    "opportunity_id": identity,
                    "policy_changed_action": link["policy_changed_action"],
                    "attribution_eligible": link["attribution_eligible"],
                }
                direct_count += 1
                if link["policy_changed_action"]:
                    changed_direct_count += 1
        links.append(link)
    return {
        "schema_version": 1, "available": True,
        "reason": "runtime_identity_linkage_evaluated",
        "records": len(links), "directly_observed": direct_count,
        "counterfactual_changed_and_observed": changed_direct_count,
        "no_trajectory_by_design": no_trajectory_count,
        "unresolved_or_missing": unresolved_count,
        "linkage_rule": "exact_opportunity_id+action+team+timestamp",
        "causal_claim_authorized": False,
        "links": links,
    }

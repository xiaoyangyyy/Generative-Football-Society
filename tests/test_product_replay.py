"""Defensive contract tests for bounded Studio ball-action replay."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from src.product.replay import (
    MAX_REPLAY_FILE_BYTES, MAX_RETAINED_EVENTS, build_match_replay,
    build_world_model_action_links,
)


def _write_log(path, events, *, home="Brazil", away="Argentina", truncated=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"meta": {
        "home": home, "away": away, "n_events": len(events),
        "truncated": truncated,
    }}, *events]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8",
    )


def test_extracts_pass_and_shot_with_fixture_identity_and_goal_direction(tmp_path):
    path = tmp_path / "outputs/ball_log/match.jsonl"
    _write_log(path, [{
        "type": "pass", "t_sec": 61.5, "team": "Brazil",
        "from": "A", "to": "B", "kind": "ground",
        "from_xy": [0.2, 0.4], "land_xy": [0.6, 0.5],
        "p_success": 0.8, "outcome": "COMPLETE",
    }, {
        "type": "shot", "t_sec": 90, "team": "Argentina",
        "shooter": "C", "gk": "D", "kind": "curl",
        "from_xy": [0.7, 0.3], "xg": 0.25, "outcome": "SAVED",
    }])

    replay = build_match_replay(
        path, root=tmp_path, home="Brazil", away="Argentina",
    )

    assert replay["available"] and replay["retained_events"] == 2
    assert replay["events"][0]["clock"] == "1:01"
    assert replay["events"][0]["end"] == [0.6, 0.5]
    assert replay["events"][1]["end"] == [0.0, 0.5]


def test_malformed_rows_and_coordinates_are_bounded_without_breaking_replay(tmp_path):
    path = tmp_path / "outputs/ball_log/corrupt.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("\n".join([
        json.dumps({"meta": {"home": "Brazil", "away": "Argentina"}}),
        "not-json",
        json.dumps({"type": "pass", "t_sec": 1, "from_xy": [0, 0]}),
        json.dumps({
            "type": "pass", "t_sec": -2, "team": "Brazil",
            "from": "A", "to": "B", "from_xy": [-4, 3],
            "land_xy": [2, -1], "outcome": "complete",
        }),
    ]) + "\n", encoding="utf-8")

    replay = build_match_replay(
        path, root=tmp_path, home="Brazil", away="Argentina",
    )

    assert replay["available"] and replay["invalid_rows"] == 2
    assert replay["coordinates_clipped"] == 4
    assert replay["events"][0]["start"] == [0.0, 1.0]
    assert replay["events"][0]["end"] == [1.0, 0.0]
    assert replay["events"][0]["t_sec"] == 0.0


def test_rejects_path_escape_fixture_mismatch_and_oversized_file(tmp_path):
    outside = tmp_path / "outside.jsonl"
    _write_log(outside, [])
    assert build_match_replay(
        outside, root=tmp_path, home="Brazil", away="Argentina",
    )["reason"] == "ball_log_outside_workspace"

    mismatch = tmp_path / "outputs/ball_log/mismatch.jsonl"
    _write_log(mismatch, [], home="France")
    assert build_match_replay(
        mismatch, root=tmp_path, home="Brazil", away="Argentina",
    )["reason"] == "ball_log_fixture_mismatch"

    oversized = tmp_path / "outputs/ball_log/oversized.jsonl"
    oversized.write_bytes(b"x" * (MAX_REPLAY_FILE_BYTES + 1))
    assert build_match_replay(
        oversized, root=tmp_path, home="Brazil", away="Argentina",
    )["reason"] == "ball_log_too_large"


def test_bounded_selection_retains_priority_actions(tmp_path):
    path = tmp_path / "outputs/ball_log/long.jsonl"
    events = [{
        "type": "pass", "t_sec": index, "team": "Brazil",
        "from": "A", "to": "B", "from_xy": [0.1, 0.2],
        "land_xy": [0.2, 0.3], "outcome": "COMPLETE",
    } for index in range(300)]
    events[299] = {
        "type": "shot", "t_sec": 299, "team": "Brazil",
        "shooter": "A", "from_xy": [0.9, 0.5], "outcome": "GOAL",
    }
    events[298]["wm_action_opportunity_id"] = "direct:Brazil:298.000:0"
    _write_log(path, events, truncated=True)

    replay = build_match_replay(
        path, root=tmp_path, home="Brazil", away="Argentina",
        priority_opportunity_ids={"direct:Brazil:298.000:0"},
    )

    assert replay["source_events"] == 300
    assert replay["retained_events"] == MAX_RETAINED_EVENTS
    assert replay["retention_truncated"] and replay["logger_truncated"]
    assert any(event["outcome"] == "GOAL" for event in replay["events"])
    assert any(
        event["wm_action_opportunity_id"] == "direct:Brazil:298.000:0"
        for event in replay["events"]
    )


def test_ball_logger_carries_pending_world_model_runtime_identity(monkeypatch):
    from src.match_engine import ball_path_logger

    monkeypatch.setattr(ball_path_logger, "ball_path_log_enabled", lambda: True)
    state = SimpleNamespace(
        clock_seconds=12.0, ball_path_log=[],
        _wm_pending_direct_action_adoption={
            "opportunity_id": "direct:Brazil:12.000:3",
        },
    )
    carrier = SimpleNamespace(
        name="A", role="CM", team_id="Brazil", player_id="a",
        position=np.array([0.2, 0.3]),
    )
    receiver = SimpleNamespace(
        name="B", role="ST", team_id="Brazil", player_id="b",
        position=np.array([0.7, 0.4]),
    )
    ball_path_logger.record_pass(
        state, carrier=carrier, receiver=receiver, kind="ground",
        target=receiver.position, landed=receiver.position, completed=True,
        success_p=0.8, press=0.1, lane=0.9, intercepted=False,
    )
    assert state.ball_path_log[0]["wm_action_opportunity_id"] == (
        "direct:Brazil:12.000:3"
    )
    ball_path_logger.record_shot(
        state, carrier=carrier, gk=receiver, kind="placed", xg=0.2,
        goal=False, on_target=True, saved=True, dist=0.3,
    )
    assert state.ball_path_log[1]["wm_action_opportunity_id"] == (
        "direct:Brazil:12.000:3"
    )
    ball_path_logger.record_cross(
        state, carrier=carrier, landed=np.array([0.9, 0.5]),
        contact="header", winner=receiver, xg_added=0.08,
    )
    assert state.ball_path_log[2]["type"] == "cross"
    assert state.ball_path_log[2]["wm_action_opportunity_id"] == (
        "direct:Brazil:12.000:3"
    )
    assert state.ball_path_log[2]["winner_id"] == "b"


def test_runtime_identity_links_decision_to_exact_observed_action(tmp_path):
    identity = "direct:Brazil:12.000:3"
    path = tmp_path / "outputs/ball_log/linked.jsonl"
    _write_log(path, [{
        "type": "pass", "t_sec": 12, "team": "Brazil",
        "from": "A", "to": "B", "kind": "ground",
        "from_xy": [0.2, 0.3], "land_xy": [0.7, 0.4],
        "outcome": "COMPLETE", "wm_action_opportunity_id": identity,
    }])
    replay = build_match_replay(
        path, root=tmp_path, home="Brazil", away="Argentina",
    )
    links = build_world_model_action_links(replay, {
        "available": True, "records": [{
            "opportunity_id": identity, "team_id": "Brazil", "t_sec": 12,
            "recommended_action": "pass", "actual_action": "pass",
            "counterfactual_baseline_action": "hold",
            "policy_changed_action": True, "attribution_eligible": True,
        }],
    })

    assert links["directly_observed"] == 1
    assert links["counterfactual_changed_and_observed"] == 1
    assert links["links"][0]["status"] == "direct_runtime_identity_match"
    assert links["links"][0]["event_index"] == 0
    assert replay["events"][0]["world_model_link"][
        "policy_changed_action"
    ] is True
    assert links["causal_claim_authorized"] is False


def test_cross_trajectory_is_replayed_and_identity_linked(tmp_path):
    identity = "direct:Brazil:22.000:5"
    path = tmp_path / "outputs/ball_log/cross.jsonl"
    _write_log(path, [{
        "type": "cross", "t_sec": 22, "team": "Brazil",
        "crosser": "A [RW]", "winner": "B [ST]", "kind": "aerial",
        "from_xy": [0.75, 0.12], "land_xy": [0.91, 0.48],
        "contact": "header", "xg_added": 0.08, "outcome": "HEADER",
        "wm_action_opportunity_id": identity,
    }])
    replay = build_match_replay(
        path, root=tmp_path, home="Brazil", away="Argentina",
    )
    links = build_world_model_action_links(replay, {
        "available": True, "records": [{
            "opportunity_id": identity, "team_id": "Brazil", "t_sec": 22,
            "recommended_action": "cross", "actual_action": "cross",
            "counterfactual_baseline_action": "hold",
            "policy_changed_action": True, "attribution_eligible": True,
        }],
    })

    assert replay["events"][0]["type"] == "cross"
    assert replay["events"][0]["end"] == [0.91, 0.48]
    assert replay["events"][0]["xg_added"] == 0.08
    assert links["directly_observed"] == 1
    assert links["links"][0]["status"] == "direct_runtime_identity_match"


def test_linker_fails_closed_for_legacy_collision_and_nontrajectory_actions():
    replay = {
        "available": True, "retention_truncated": False,
        "logger_truncated": False, "source_limit_reached": False,
        "events": [{
            "event_index": index, "type": "pass", "team": "Brazil",
            "t_sec": 12.0, "clock": "0:12", "wm_action_opportunity_id": "dup",
        } for index in range(2)],
    }
    adoption = {"available": True, "records": [{
        "opportunity_id": "dup", "team_id": "Brazil", "t_sec": 12,
        "actual_action": "pass",
    }, {
        "opportunity_id": "", "team_id": "Brazil", "t_sec": 13,
        "actual_action": "pass",
    }, {
        "opportunity_id": "hold-id", "team_id": "Brazil", "t_sec": 14,
        "actual_action": "hold", "policy_changed_action": True,
    }]}

    links = build_world_model_action_links(replay, adoption)

    assert [link["status"] for link in links["links"]] == [
        "ambiguous_runtime_identity_collision",
        "legacy_record_without_runtime_identity",
        "no_ball_trajectory_by_design",
    ]
    assert links["directly_observed"] == 0
    assert links["no_trajectory_by_design"] == 1
    assert links["unresolved_or_missing"] == 2


def test_linker_rejects_duplicate_records_and_tampered_action_team_or_time():
    events = []
    for index, identity in enumerate(("dup-record", "action", "team", "time")):
        events.append({
            "event_index": index, "type": "pass", "team": "Brazil",
            "t_sec": 12.0, "clock": "0:12",
            "wm_action_opportunity_id": identity,
        })
    replay = {"available": True, "events": events}
    records = [{
        "opportunity_id": "dup-record", "team_id": "Brazil", "t_sec": 12,
        "actual_action": "pass",
    }, {
        "opportunity_id": "dup-record", "team_id": "Brazil", "t_sec": 12,
        "actual_action": "pass",
    }, {
        "opportunity_id": "action", "team_id": "Brazil", "t_sec": 12,
        "actual_action": "shot",
    }, {
        "opportunity_id": "team", "team_id": "Argentina", "t_sec": 12,
        "actual_action": "pass",
    }, {
        "opportunity_id": "time", "team_id": "Brazil", "t_sec": 13,
        "actual_action": "pass",
    }]

    links = build_world_model_action_links(
        replay, {"available": True, "records": records},
    )

    assert [link["status"] for link in links["links"]] == [
        "ambiguous_action_record_identity_collision",
        "ambiguous_action_record_identity_collision",
        "runtime_identity_action_mismatch",
        "runtime_identity_team_mismatch",
        "runtime_identity_timestamp_mismatch",
    ]
    assert links["directly_observed"] == 0
    assert links["unresolved_or_missing"] == 5

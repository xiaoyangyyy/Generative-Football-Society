"""Bounded, non-causal synchronized replay tests for paired matches."""

from __future__ import annotations

from src.product.comparison import (
    MAX_PAIRED_REPLAY_EVENTS_PER_SIDE,
    MAX_PAIRED_REPLAY_FRAMES,
    build_paired_comparison,
    build_synchronized_paired_replay,
    render_paired_comparison_html,
)


def _report(match_id, *, tactic="gegenpress", reuse=False, baseline=None):
    return {
        "match_id": match_id, "studio": {"mode": "research"},
        "fixture": {
            "home": "Brazil", "away": "Argentina", "seed": 77, "fast": True,
        },
        "match_plan": {
            "experience": "tactical_lab", "home_tactic": tactic,
            "away_tactic": "low_block_counter", "reuse_last_seed": reuse,
            "score_path": "physics_official",
            "paired_baseline_match_id": baseline,
        },
        "result": {
            "score": {"home": 1, "away": 0},
            "xg": {"home": 1.1, "away": 0.7},
            "possession": {"home": 0.55, "away": 0.45},
            "passes": {"home": 40, "away": 35},
            "shots": {"home": 8, "away": 5},
        },
        "layers": {
            "psychology": {},
            "world_model": {
                "runtime": {"checkpoint_signature": "sha256:model"},
                "action_adoption": {},
            },
        },
        "integrity": {"accepted": True},
    }


def _event(t_sec, *, team="Brazil", event_type="pass", outcome="COMPLETE"):
    return {
        "type": event_type, "team": team, "actor": "A", "target": "B",
        "kind": "ground" if event_type == "pass" else "placed",
        "outcome": outcome, "t_sec": t_sec,
        "start": [0.2, 0.3], "end": [0.7, 0.4],
    }


def _with_replay(report, events):
    report["replay"] = {
        "available": True, "source_events": len(events),
        "retained_events": len(events), "events": events,
    }
    return report


def _pair(baseline_events, treatment_events):
    baseline = _with_replay(_report("m1"), baseline_events)
    treatment = _with_replay(
        _report("m2", tactic="counter_attack", reuse=True, baseline="m1"),
        treatment_events,
    )
    return baseline, treatment


def test_synchronized_replay_uses_union_clock_without_event_pairing():
    baseline, treatment = _pair(
        [_event(10), _event(30, event_type="shot", outcome="SAVED")],
        [_event(20), _event(30, event_type="shot", outcome="GOAL")],
    )
    replay = build_synchronized_paired_replay(
        baseline, treatment, home="Brazil", away="Argentina",
    )

    assert replay["available"]
    assert [frame["t_sec"] for frame in replay["frames"]] == [10, 20, 30]
    assert replay["frames"][0]["baseline_current_indices"] == [0]
    assert replay["frames"][0]["treatment_current_indices"] == []
    assert replay["frames"][1]["baseline_visible_count"] == 1
    assert replay["frames"][1]["treatment_current_indices"] == [0]
    assert replay["frames"][2]["baseline_current_indices"] == [1]
    assert replay["frames"][2]["treatment_current_indices"] == [1]
    assert replay["summary"]["goals"] == {
        "baseline": 0, "treatment": 1, "delta": 1,
    }
    assert replay["event_correspondence_authorized"] is False
    assert replay["causal_claim_authorized"] is False


def test_comparison_html_renders_script_free_dual_pitch_sequence():
    baseline, treatment = _pair(
        [_event(10), _event(30)],
        [_event(20), _event(30, event_type="shot", outcome="GOAL")],
    )
    comparison = build_paired_comparison(baseline, treatment)
    document = render_paired_comparison_html(comparison)

    assert 'data-testid="paired-replay-panel"' in document
    assert 'id="pair-step-overview" checked' in document
    assert document.count('name="pair-step"') == 4
    assert 'class="pair-side pair-side-baseline"' in document
    assert 'class="pair-side pair-side-treatment"' in document
    assert document.count('<svg viewBox="0 0 1000 640"') == 2
    assert 'for="pair-step-0">开始复盘</label>' in document
    assert 'for="pair-step-overview">上一时点</label>' in document
    assert 'for="pair-step-overview">返回概览</label>' in document
    assert "该时点无新保留动作" in document
    assert "只按比赛时钟同步，不进行动作一一配对" in document
    assert '#pair-filter-shots:checked~.pair-pitches' in document
    assert '<script' not in document.lower()


def test_replay_unavailability_does_not_change_pairing_eligibility():
    baseline = _report("m1")
    treatment = _with_replay(
        _report("m2", tactic="counter_attack", reuse=True, baseline="m1"),
        [_event(20)],
    )
    comparison = build_paired_comparison(baseline, treatment)

    assert comparison["eligibility"]["eligible_for_tactical_attribution"]
    assert not comparison["paired_replay"]["available"]
    document = render_paired_comparison_html(comparison)
    assert "双场播放器不可用" in document
    assert "指标比较与资格检查仍然有效" in document


def test_paired_replay_fails_closed_on_invalid_collection_and_rows():
    baseline = _report("m1")
    baseline["replay"] = {"available": True, "events": "not-an-array"}
    treatment = _with_replay(
        _report("m2", tactic="counter_attack", reuse=True, baseline="m1"),
        [_event(20)],
    )
    replay = build_synchronized_paired_replay(
        baseline, treatment, home="Brazil", away="Argentina",
    )
    assert not replay["available"]
    assert replay["baseline"]["reason"] == "invalid_replay_event_collection"

    baseline["replay"]["events"] = [
        "bad", _event(10, team="France"),
        {**_event(11), "start": [float("nan"), 0.2]}, _event(12),
    ]
    replay = build_synchronized_paired_replay(
        baseline, treatment, home="Brazil", away="Argentina",
    )
    assert replay["available"]
    assert replay["baseline"]["invalid_rows"] == 3
    assert len(replay["baseline"]["events"]) == 1


def test_paired_replay_caps_sides_and_shared_frames_but_keeps_key_shot():
    baseline_events = [_event(index * 2) for index in range(300)]
    treatment_events = [_event(index * 2 + 1) for index in range(300)]
    baseline_events[239] = _event(478, event_type="shot", outcome="GOAL")
    baseline, treatment = _pair(baseline_events, treatment_events)

    replay = build_synchronized_paired_replay(
        baseline, treatment, home="Brazil", away="Argentina",
    )

    assert len(replay["baseline"]["events"]) == MAX_PAIRED_REPLAY_EVENTS_PER_SIDE
    assert len(replay["treatment"]["events"]) == MAX_PAIRED_REPLAY_EVENTS_PER_SIDE
    assert replay["baseline"]["input_truncated"]
    assert replay["treatment"]["input_truncated"]
    assert len(replay["frames"]) == MAX_PAIRED_REPLAY_FRAMES
    assert replay["frames_truncated"]
    assert any(frame["t_sec"] == 478 for frame in replay["frames"])

    comparison = build_paired_comparison(baseline, treatment)
    document = render_paired_comparison_html(comparison)
    assert document.count('name="pair-step"') == 241
    assert 'id="pair-step-239"' in document
    assert 'id="pair-step-240"' not in document
    assert len(document.encode("utf-8")) < 1_200_000


def test_renderer_rebuilds_frames_and_escapes_events_instead_of_trusting_payload():
    malicious = _event(10)
    malicious["actor"] = '</span><script>alert("x")</script>'
    baseline, treatment = _pair([malicious], [_event(20)])
    comparison = build_paired_comparison(baseline, treatment)
    comparison["paired_replay"]["frames"] = [{
        "clock": "</style><script>bad()</script>",
        "baseline_visible_count": 999999,
    }]

    document = render_paired_comparison_html(comparison)

    assert "bad()" not in document
    assert "<script" not in document.lower()
    assert "&lt;/span&gt;&lt;script&gt;alert(&quot;x&quot;)" in document
    assert 'id="pair-step-1"' in document

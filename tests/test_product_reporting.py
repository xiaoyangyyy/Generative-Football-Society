"""Studio match reports explain research policy influence without overclaiming."""

from __future__ import annotations

from src.product.reporting import render_match_html


def _report(*, mode="research", enabled=True, action_adoption=None):
    return {
        "studio": {"mode": mode},
        "fixture": {"home": "Brazil", "away": "Argentina"},
        "result": {
            "score": {"home": 2, "away": 1},
            "xg": {"home": 1.4, "away": 0.8},
            "possession": {"home": 0.55, "away": 0.45},
            "passes": {"home": 40, "away": 35},
            "shots": {"home": 8, "away": 5},
        },
        "layers": {
            "psychology": {},
            "world_model": {
                "configured": mode != "stable",
                "enabled": enabled,
                "shot_probability_source": "physics_xg_prior",
                "action_adoption": action_adoption or {},
                "action_adoption_mechanism": {
                    "execution_state": "ready_not_started",
                    "runs_executed": 0,
                    "fixed_run_budget": 24,
                },
            },
            "cognition": {"enabled": False},
        },
        "integrity": {"accepted": True, "state": "accepted"},
        "timeline": ["12' goal"],
    }


def test_stable_report_explicitly_says_research_policy_is_inactive():
    document = render_match_html(_report(mode="stable", enabled=False))

    assert "World-model action influence" in document
    assert "Inactive in stable mode" in document
    assert "without research-policy influence" in document
    assert 'id="replay-wm"' not in document


def test_research_report_separates_realized_and_expected_changes_and_escapes():
    adoption = {
        "available": True,
        "opportunities": 20,
        "influenced_opportunities": 18,
        "attribution_eligible_opportunities": 17,
        "counterfactual_action_changes": 2,
        "expected_counterfactual_action_changes": 2.7,
        "mean_recommended_probability_shift": 0.0125,
        "records": [{
            "team_id": "<script>alert(1)</script>",
            "t_sec": 120.0,
            "recommended_action": "pass",
            "model_adjustments": {"pass": 0.125, "hold": 0.0},
            "quality_gates": {"pass": {
                "open": True, "confidence": 0.8, "certainty": 0.9,
            }, "shot": {
                "open": False, "reason": "shot_quality_gate_closed",
                "confidence": 0.0, "certainty": 0.0,
            }},
            "counterfactual_baseline_action": "pass",
            "actual_action": "hold",
            "sampling_uniform": 0.75,
            "policy_changed_action": True,
            "attribution_eligible": True,
            "base_probability": {"pass": 0.8},
            "adjusted_probability": {"pass": 0.6},
            "total_variation_distance": 0.2,
        }],
    }

    document = render_match_html(_report(action_adoption=adoption))

    assert "Counterfactual changes</span><strong>2</strong>" in document
    assert "Expected changes</span><strong>2.70</strong>" in document
    assert "Expected changes are probability mass" in document
    assert "does not authorize product or academic promotion" in document
    assert "pass &rarr; hold" in document
    assert "80.0% &rarr; 60.0%" in document
    assert "Gate &amp; model signal" in document
    assert "pass</b> +0.125" in document
    assert "open: confidence 80.0%; certainty 90.0%" in document
    assert "u=0.750" in document
    assert "Quality gate decides whether validated model evidence" in document
    assert "same stored random draw" in document
    assert 'id="wm-filter-all" checked' in document
    assert 'id="wm-filter-changed"' in document
    assert 'id="wm-filter-probability"' in document
    assert 'id="wm-filter-closed"' in document
    assert '<fieldset class="wm-filters"><legend>Decision filter</legend>' in document
    assert "#wm-filter-changed:checked~.table-wrap" in document
    assert document.index('<fieldset class="wm-filters">') < document.index(
        '<div class="table-wrap">'
    ) < document.index("</fieldset>")
    assert "Changed action (1)" in document
    assert "Probability shifted (1)" in document
    assert "all quality gates" in document
    assert "shot_quality_gate_closed" in document
    assert "<script" not in document
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in document
    assert "<script>alert(1)</script>" not in document


def test_manager_report_renders_frozen_lineup_and_escapes_player_names():
    report = _report()
    report["layers"]["management"] = {
        "decision": {
            "team": "Brazil", "tactic": "gegenpress", "rotation": "rotate",
            "lineup": {
                "starters": [f"p{index}" for index in range(11)],
                "bench": ["p11"], "source": "manual",
            },
        },
        "effects": {
            "home": {
                "base_status": 72.0, "effective_status": 69.2,
                "fatigue_load_multiplier": 1.05,
                "lineup": {
                    "starters": [f"p{index}" for index in range(11)],
                    "bench": ["p11"], "source": "manual",
                    "starter_names": ["<img src=x onerror=alert(1)>"]
                    + [f"Player {index}" for index in range(1, 11)],
                    "bench_names": ["Safe substitute"],
                },
            },
        },
        "in_match": {"home": {"outcomes": [{
            "scheduled_minute": 70, "condition": "leading",
            "score_for": 2, "score_against": 1,
            "requested_tactic": "low_block_counter",
            "status": "applied", "reason": "condition_met",
        }]}},
    }
    document = render_match_html(report)
    assert 'data-testid="manager-decision-panel"' in document
    assert "Brazil match decision" in document
    assert "gegenpress" in document and "rotate" in document
    assert "72.00 &rarr; 69.20" in document
    assert "Safe substitute" in document
    assert "In-match instruction outcomes" in document
    assert "low_block_counter" in document and "condition_met" in document
    assert "&lt;img src=x onerror=alert(1)&gt;" in document
    assert "<img src=x onerror=alert(1)>" not in document
    assert "do not establish real-world performance" in document


def test_action_explorer_keeps_all_bounded_records_and_degrades_bad_fields():
    records = []
    for index in range(12):
        records.append({
            "team_id": "home", "t_sec": index * 10,
            "recommended_action": "pass",
            "model_adjustments": {"pass": 0.1},
            "quality_gates": {"pass": {
                "open": index % 2 == 0,
                "reason": "closed" if index % 2 else None,
                "confidence": 0.8, "certainty": 0.9,
            }},
            "counterfactual_baseline_action": "hold",
            "actual_action": "pass", "sampling_uniform": 0.4,
            "base_probability": {"pass": 0.4},
            "adjusted_probability": {"pass": 0.6},
            "policy_changed_action": index < 3,
            "attribution_eligible": True,
            "total_variation_distance": 0.2,
        })
    records.extend(["invalid-row", {  # persisted corruption must not break HTML
        "team_id": "away", "t_sec": float("nan"),
        "recommended_action": "hold", "model_adjustments": ["bad"],
        "quality_gates": ["bad"], "base_probability": ["bad"],
        "adjusted_probability": None, "sampling_uniform": float("inf"),
    }])
    report = _report(action_adoption={
        "available": True, "records": records,
        "opportunities": "corrupt", "influenced_opportunities": -4,
        "expected_counterfactual_action_changes": float("nan"),
        "mean_recommended_probability_shift": float("inf"),
    })
    report["layers"]["world_model"]["action_adoption_mechanism"] = ["bad"]
    document = render_match_html(report)

    assert "All (13)" in document
    assert "Opportunities</span><strong>0</strong>" in document
    assert "Non-zero influence</span><strong>0</strong>" in document
    assert "Expected changes</span><strong>0.00</strong>" in document
    assert "Changed action (3)" in document
    assert "Probability shifted (12)" in document
    assert document.count("<tr class=") == 13
    assert "No action-specific gate record" in document
    assert "u=nan" not in document.lower()
    assert "nan%" not in document.lower()
    assert "u=inf" not in document.lower()
    assert "inf%" not in document.lower()


def test_configured_but_missing_runtime_is_visible_as_integrity_problem():
    document = render_match_html(_report(enabled=False))

    assert "Configured but not observed at runtime" in document
    assert "cannot provide world-model action evidence" in document


def test_tactical_lab_report_exposes_score_path_seed_and_inference_boundary():
    report = _report()
    report["fixture"]["seed"] = 77
    report["match_plan"] = {
        "experience": "tactical_lab",
        "home_tactic": "gegenpress",
        "away_tactic": "low_block_counter",
        "reuse_last_seed": True,
        "score_path": "physics_official",
        "paired_baseline_match_id": "0001-brazil-vs-argentina",
    }
    document = render_match_html(report)
    assert 'data-testid="match-plan-panel"' in document
    assert "高位反抢" in document and "低位防反" in document
    assert "物理射门与扑救产生正式比分" in document
    assert "seed 77" in document
    assert "需与对应基线报告配对比较" in document
    assert "0001-brazil-vs-argentina" in document


def test_world_model_fork_report_exposes_policy_and_pairing_boundary():
    report = _report()
    report["fixture"]["seed"] = 91
    report["match_plan"] = {
        "experience": "world_model_lab",
        "home_tactic": "balanced",
        "away_tactic": "balanced",
        "world_model_policy": "action_policy",
        "world_model_branch_at_sec": 2700.0,
        "reuse_last_seed": True,
        "score_path": "physics_official",
        "paired_baseline_match_id": "fork-baseline-001",
    }
    report["simulation_clock"] = {
        "contract": "authoritative_tick_v2",
        "authoritative_tick_clock": True,
    }

    document = render_match_html(report)

    assert 'data-testid="match-plan-panel"' in document
    assert "世界模型因果分叉" in document
    assert "干预：质量门控动作策略" in document
    assert "seed 91" in document
    assert "Branch minute" in document and "45" in document
    assert "Clock contract" in document
    assert "authoritative_tick_v2" in document
    assert "全部资格检查通过" in document
    assert "模拟器内动作策略开关" in document


def test_action_replay_is_css_only_filterable_and_escapes_persisted_text():
    report = _report()
    report["replay"] = {
        "available": True, "reason": "available", "source_events": 2,
        "retained_events": 2, "events": [{
            "type": "pass", "t_sec": 12, "clock": "0:12",
            "team": "Brazil", "actor": "<script>alert(1)</script>",
            "target": "B", "kind": "ground", "outcome": "COMPLETE",
            "start": [0.1, 0.2], "end": [0.6, 0.5],
            "world_model_link": {
                "opportunity_id": "direct:Brazil:12.000:0",
                "policy_changed_action": True,
            },
        }, {
            "type": "shot", "t_sec": 33, "team": "Argentina",
            "actor": "C", "target": "D", "kind": "curl",
            "outcome": "SAVED", "start": [0.7, 0.4], "end": [0, 0.5],
        }], "world_model_action_links": {
            "available": True, "directly_observed": 1,
            "counterfactual_changed_and_observed": 1,
            "unresolved_or_missing": 0,
            "links": [{
                "opportunity_id": "direct:Brazil:12.000:0",
                "status": "direct_runtime_identity_match", "event_index": 0,
            }],
        },
        "manager_annotations": [{
            "minute": 60, "team": "Brazil<script>",
            "condition": "trailing", "score_for": 0, "score_against": 1,
            "tactic": "gegenpress", "status": "applied",
        }],
    }

    document = render_match_html(report)

    assert 'data-testid="action-replay-panel"' in document
    assert "not video or full-player tracking" in document
    assert '<svg viewBox="0 0 1000 640"' in document
    assert 'id="replay-passes"' in document and 'id="replay-shots"' in document
    assert 'id="replay-wm"' in document
    assert 'id="replay-wm-changed"' in document
    assert "#replay-passes:checked~.replay-stage" in document
    assert "WM changed + observed (1)" in document
    assert "It proves record correspondence, not match-level causal effect" in document
    assert "In-match manager events (1)" in document
    assert "Brazil&lt;script&gt;" in document
    assert "<script" not in document
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in document


def test_action_replay_unavailability_is_explicit_and_non_integrity_claiming():
    report = _report()
    report["replay"] = {
        "available": False, "reason": "ball_log_fixture_mismatch", "events": [],
    }
    document = render_match_html(report)
    assert "Unavailable for this match (ball_log_fixture_mismatch)" in document
    assert "does not affect score integrity" in document


def test_world_model_decision_table_links_only_direct_identity_evidence():
    report = _report(action_adoption={
        "available": True, "records": [{
            "opportunity_id": "direct:Brazil:12.000:0",
            "team_id": "Brazil", "t_sec": 12, "actual_action": "pass",
            "counterfactual_baseline_action": "hold",
            "recommended_action": "pass", "policy_changed_action": True,
            "attribution_eligible": True, "model_adjustments": {"pass": 0.1},
            "quality_gates": {}, "base_probability": {"pass": 0.4},
            "adjusted_probability": {"pass": 0.6},
            "total_variation_distance": 0.2,
        }],
    })
    report["replay"] = {
        "available": True, "source_events": 1, "retained_events": 1,
        "events": [{
            "type": "pass", "team": "Brazil", "actor": "A", "target": "B",
            "kind": "ground", "outcome": "COMPLETE", "t_sec": 12,
            "start": [0.2, 0.3], "end": [0.7, 0.4],
            "world_model_link": {"policy_changed_action": True},
        }],
        "world_model_action_links": {
            "available": True, "directly_observed": 1,
            "counterfactual_changed_and_observed": 1,
            "unresolved_or_missing": 0, "links": [{
                "opportunity_id": "direct:Brazil:12.000:0",
                "status": "direct_runtime_identity_match", "event_index": 0,
            }],
        },
    }
    document = render_match_html(report)
    assert 'href="#action-replay-panel">direct trajectory #1</a>' in document
    assert "Time proximity alone is never treated as a direct match" in document


def test_replay_sequence_has_script_free_boundary_navigation_and_current_frame():
    report = _report()
    report["replay"] = {
        "available": True, "source_events": 3, "retained_events": 3,
        "events": [{
            "type": "pass", "team": "Brazil", "actor": f"A{index}",
            "target": f"B{index}", "kind": "ground", "outcome": "COMPLETE",
            "t_sec": 10 + index, "start": [0.1, 0.2],
            "end": [0.2 + index * 0.1, 0.3],
        } for index in range(3)],
    }

    document = render_match_html(report)

    assert 'id="replay-step-overview" checked' in document
    assert document.count('name="replay-step"') == 4
    assert 'for="replay-step-0">Start sequence</label>' in document
    assert 'class="sequence-frame sequence-frame-0"' in document
    assert 'for="replay-step-overview">Previous</label>' in document
    assert 'for="replay-step-1">Next action</label>' in document
    assert 'for="replay-step-overview">Return to overview</label>' in document
    assert '#replay-step-1:checked~.replay-stage .replay-event:nth-of-type(n+3)' in document
    assert '#replay-step-1:checked~.replay-stage .replay-event:nth-of-type(2)' in document
    assert '#replay-step-1:focus-visible~.replay-timeline' in document
    assert 'aria-label="Jump to retained action"' in document
    assert "<script" not in document


def test_replay_sequence_is_bounded_to_240_and_escapes_marker_titles():
    report = _report()
    events = []
    for index in range(300):
        events.append({
            "type": "pass", "team": "Brazil",
            "actor": '</label><script>alert("x")</script>' if index == 0 else "A",
            "target": "B", "kind": "ground", "outcome": "COMPLETE",
            "t_sec": index, "start": [0.1, 0.2], "end": [0.2, 0.3],
        })
    report["replay"] = {
        "available": True, "source_events": 300,
        "retained_events": 240, "events": events,
    }

    document = render_match_html(report)

    assert document.count('name="replay-step"') == 241
    assert 'id="replay-step-239"' in document
    assert 'id="replay-step-240"' not in document
    assert "Action 240 / 240" in document
    assert "nth-of-type(n+241)" in document
    assert len(document.encode("utf-8")) < 600_000
    assert "<script" not in document
    assert "&lt;/label&gt;&lt;script&gt;alert(&quot;x&quot;)" in document

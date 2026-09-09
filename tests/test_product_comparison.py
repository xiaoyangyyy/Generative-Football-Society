import pytest

from src.product.comparison import (
    PairingError, build_paired_comparison, render_paired_comparison_html,
)
from src.simulation.society_continuity import PUBLIC_BOUNDARY


def _report(match_id, *, tactic="gegenpress", reuse=False, baseline=None):
    return {
        "match_id": match_id,
        "studio": {"mode": "research"},
        "fixture": {"home": "Brazil", "away": "Argentina", "seed": 77, "fast": True},
        "match_plan": {
            "experience": "tactical_lab", "home_tactic": tactic,
            "away_tactic": "low_block_counter", "reuse_last_seed": reuse,
            "score_path": "physics_official",
            "paired_baseline_match_id": baseline,
        },
        "result": {
            "score": {"home": 1, "away": 0}, "xg": {"home": 1.1, "away": 0.7},
            "possession": {"home": 0.55, "away": 0.45},
            "passes": {"home": 40, "away": 35}, "shots": {"home": 8, "away": 5},
        },
        "layers": {
            "psychology": {
                "crowd_field": 0.1, "coach_stress": {"home": 0.2, "away": 0.3},
                "tactical_drift": {"home": 0.02, "away": 0.03},
            },
            "world_model": {
                "runtime": {"checkpoint_signature": "sha256:model"},
                "action_adoption": {
                    "opportunities": 10, "influenced_opportunities": 8,
                    "counterfactual_action_changes": 1,
                    "expected_counterfactual_action_changes": 0.8,
                    "mean_recommended_probability_shift": 0.01,
                },
            },
        },
        "integrity": {"accepted": True},
    }


def _replay_event(
    t_sec, *, event_type="pass", outcome="COMPLETE", link=None,
):
    event = {
        "type": event_type, "team": "Brazil", "actor": "A",
        "target": "B", "kind": "ground", "outcome": outcome,
        "t_sec": t_sec, "start": [0.2, 0.3], "end": [0.6, 0.4],
    }
    if link is not None:
        event["world_model_link"] = link
    return event


def _society_snapshot(marker, *, morale=0.0):
    return {
        "available": True,
        "state_identity": marker * 64,
        "source_transaction_id": f"match-{marker}",
        "memory_records": 2,
        "cognitive_memory_records": 1,
        "beliefs": 1,
        "reflections": 1,
        "tactical_controls": {
            "pressing_intensity": 0.5, "risk_budget": 0.5,
            "line_height": 0.5, "rotation_aggressiveness": 0.5,
        },
        "emotion_profile": {
            "pride": 0.2, "anger": 0.1, "shame": 0.1, "fear": 0.1,
            "determination": 0.6,
        },
        "social_narrative_state": {
            "trust_index": 0.5, "polarization": 0.2,
            "narrative_fatigue": 0.1,
        },
        "psychological_state": {
            "morale": morale, "pressure": 0.0, "trust": 0.0,
            "risk_appetite": 0.0, "conflict": 0.0,
            "audience_activation": 0.0,
        },
        "referee_grievance": 0.05,
        "claim_boundary": PUBLIC_BOUNDARY,
    }


def test_same_seed_single_side_pair_is_eligible_but_seed_local_only():
    baseline = _report("m1")
    treatment = _report("m2", tactic="counter_attack", reuse=True, baseline="m1")
    treatment["result"]["xg"]["home"] = 1.4
    comparison = build_paired_comparison(baseline, treatment)
    assert comparison["intervention"]["scope"] == "single_side_tactical_intervention"
    assert comparison["eligibility"]["eligible_for_tactical_attribution"]
    assert comparison["metrics"]["xg_home"]["delta"] == pytest.approx(0.3)
    assert "not a population effect" in comparison["claim_boundary"]


def test_two_changed_sides_are_labeled_joint_not_individually_attributed():
    baseline = _report("m1")
    treatment = _report("m2", tactic="counter_attack", reuse=True, baseline="m1")
    treatment["match_plan"]["away_tactic"] = "balanced"
    comparison = build_paired_comparison(baseline, treatment)
    assert comparison["intervention"]["scope"] == "joint_tactical_intervention"
    assert comparison["intervention"]["changed_sides"] == ["home", "away"]


@pytest.mark.parametrize("mutation", ["seed", "fixture", "baseline_id", "no_change"])
def test_structurally_invalid_pair_is_rejected(mutation):
    baseline = _report("m1")
    treatment = _report("m2", tactic="counter_attack", reuse=True, baseline="m1")
    if mutation == "seed":
        treatment["fixture"]["seed"] = 78
    elif mutation == "fixture":
        treatment["fixture"]["away"] = "France"
    elif mutation == "baseline_id":
        treatment["match_plan"]["paired_baseline_match_id"] = "other"
    else:
        treatment["match_plan"]["home_tactic"] = "gegenpress"
    with pytest.raises(PairingError):
        build_paired_comparison(baseline, treatment)


def test_cognitive_pair_is_descriptive_because_provider_is_uncontrolled():
    baseline = _report("m1")
    treatment = _report("m2", tactic="counter_attack", reuse=True, baseline="m1")
    baseline["studio"]["mode"] = treatment["studio"]["mode"] = "cognitive"
    comparison = build_paired_comparison(baseline, treatment)
    assert not comparison["eligibility"]["eligible_for_tactical_attribution"]
    assert comparison["eligibility"]["failed_checks"] == [
        "provider_determinism_controlled"
    ]


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        ("fast", "same_fast_configuration"),
        ("integrity", "integrity_accepted_both"),
        ("checkpoint", "same_world_model_identity"),
        ("mode", "same_studio_mode"),
    ],
)
def test_quality_or_identity_drift_keeps_comparison_but_revokes_attribution(
    mutation, failed_check,
):
    baseline = _report("m1")
    treatment = _report("m2", tactic="counter_attack", reuse=True, baseline="m1")
    if mutation == "fast":
        treatment["fixture"]["fast"] = False
    elif mutation == "integrity":
        treatment["integrity"]["accepted"] = False
    elif mutation == "checkpoint":
        treatment["layers"]["world_model"]["runtime"].update({
            "checkpoint_signature": "sha256:other"
        })
    else:
        treatment["studio"]["mode"] = "cognitive"
    comparison = build_paired_comparison(baseline, treatment)
    assert not comparison["eligibility"]["eligible_for_tactical_attribution"]
    assert failed_check in comparison["eligibility"]["failed_checks"]


def test_missing_metric_is_explicitly_none_not_silently_zero():
    baseline = _report("m1")
    treatment = _report("m2", tactic="counter_attack", reuse=True, baseline="m1")
    del treatment["result"]["xg"]["home"]
    comparison = build_paired_comparison(baseline, treatment)
    assert comparison["metrics"]["xg_home"] == {
        "baseline": 1.1, "treatment": None, "delta": None,
    }


def test_comparison_html_escapes_fixture_and_renders_eligibility():
    baseline = _report("m1")
    treatment = _report("m2", tactic="counter_attack", reuse=True, baseline="m1")
    baseline["fixture"]["home"] = treatment["fixture"]["home"] = "<script>x</script>"
    comparison = build_paired_comparison(baseline, treatment)
    comparison["reports"] = {
        "baseline": "outputs/studio/demo/matches/m1.json",
        "treatment": "outputs/studio/demo/matches/m2.json",
    }
    document = render_paired_comparison_html(comparison)
    assert "配对资格通过" in document
    assert "&lt;script&gt;x&lt;/script&gt;" in document
    assert "<script>x</script>" not in document
    assert 'href="m1.html"' in document and 'href="m2.html"' in document


def test_world_model_policy_fork_is_isolated_and_rendered_as_two_worlds():
    baseline = _report("m1")
    treatment = _report("m2", reuse=True, baseline="m1")
    for report, policy in (
        (baseline, "predict_only"), (treatment, "action_policy"),
    ):
        report["match_plan"].update({
            "experience": "world_model_lab",
            "world_model_policy": policy,
        })
    treatment["result"]["shots"]["home"] = 10
    treatment["layers"]["world_model"]["action_adoption"]["records"] = [
        {
            "opportunity_id": "direct:Brazil:12.000:0",
            "team_id": "Brazil", "t_sec": 12,
            "counterfactual_baseline_action": "hold",
            "actual_action": "pass", "recommended_action": "pass",
            "policy_changed_action": True, "attribution_eligible": True,
        },
        {
            "opportunity_id": "direct:Brazil:25.000:1",
            "team_id": "Brazil", "t_sec": 25,
            "counterfactual_baseline_action": "pass",
            "actual_action": "hold", "recommended_action": "hold",
            "policy_changed_action": True, "attribution_eligible": True,
        },
        {
            "opportunity_id": "direct:Brazil:30.000:2",
            "team_id": "Brazil", "t_sec": 30,
            "counterfactual_baseline_action": "hold",
            "actual_action": "cross", "recommended_action": "cross",
            "policy_changed_action": True, "attribution_eligible": True,
        },
    ]
    baseline["replay"] = {"available": True, "events": [
        _replay_event(12), _replay_event(20),
        _replay_event(80, event_type="shot", outcome="SAVED"),
    ]}
    treatment["replay"] = {"available": True, "events": [
        _replay_event(12, link={
            "opportunity_id": "direct:Brazil:12.000:0",
            "policy_changed_action": True,
        }),
        _replay_event(18, event_type="shot", outcome="GOAL"),
        _replay_event(30, event_type="cross", link={
            "opportunity_id": "direct:Brazil:30.000:2",
            "policy_changed_action": True,
        }),
        _replay_event(50),
    ]}
    comparison = build_paired_comparison(baseline, treatment)
    assert comparison["intervention"]["scope"] == "world_model_action_policy"
    assert comparison["intervention"]["changed_sides"] == []
    assert comparison["eligibility"][
        "eligible_for_world_model_policy_attribution"
    ]
    assert not comparison["eligibility"]["eligible_for_tactical_attribution"]
    assert comparison["metrics"]["shots_home"]["delta"] == 2
    propagation = comparison["policy_propagation"]
    assert propagation["status"] == "direct_action_changes_observed"
    assert propagation["summary"]["valid_changed_decisions"] == 3
    assert propagation["summary"]["directly_observed_changes"] == 2
    assert propagation["summary"]["locally_attributable_changes"] == 2
    assert comparison["paired_replay"]["summary"]["crosses"] == {
        "baseline": 0, "treatment": 1, "delta": 1,
    }
    future = comparison["counterfactual_future_summary"]
    assert future["status"] == (
        "local_action_divergence_with_descriptive_future_difference"
    )
    assert [world["policy"] for world in future["worlds"]] == [
        "predict_only", "action_policy",
    ]
    assert future["summary"]["changed_actions"] == 3
    assert future["summary"]["locally_attributable_actions"] == 2
    assert future["summary"]["descriptive_outcome_difference_count"] == 1
    assert future["evidence_ladder"][0] == {
        "stage": "shared_prefix",
        "status": "match_start_controlled",
        "claim_authorized": True,
    }
    assert future["claim_authority"] == {
        "simulator_local_action_attribution": True,
        "downstream_trajectory_causality": False,
        "downstream_society_state_causality": False,
        "match_outcome_causality": False,
        "population_effect": False,
        "real_football_causality": False,
        "world_model_promotion": False,
    }
    first = propagation["decisions"][0]
    assert first["local_policy_attribution_eligible"] is True
    assert first["downstream_windows"]["30s"]["delta"] == {
        "actions": 1, "passes": -1, "crosses": 1, "shots": 1,
        "goals": 1, "turnovers": 0,
    }
    assert first["downstream_windows"]["30s"][
        "additional_policy_changes"
    ] == 2
    assert first["downstream_windows"]["30s"][
        "causal_attribution_authorized"
    ] is False
    cross = next(
        row for row in propagation["decisions"]
        if row["treatment_action"] == "cross"
    )
    assert cross["directly_observed"] is True
    assert cross["local_policy_attribution_eligible"] is True
    assert cross["observed_event_index"] is not None
    document = render_paired_comparison_html(comparison)
    assert "世界模型因果分叉" in document
    assert "基线世界" in document and "干预世界" in document
    assert "MATCH_WM_PLAN=0" in document and "MATCH_WM_PLAN=1" in document
    assert "world-model promotion" in document
    assert 'data-testid="policy-propagation-panel"' in document
    assert 'data-testid="counterfactual-future-summary"' in document
    assert "反事实未来总览" in document
    assert "不授权下游赛果因果" in document
    assert "动作分歧与传播链" in document
    assert "hold → pass" in document
    assert "直接轨迹已绑定" in document
    assert "越过“运行轨迹”后不自动继承因果资格" in document
    assert "只比较两场在同一比赛时钟区间内的事件计数" in document


def test_world_model_fork_exposes_exact_terminal_society_divergence():
    baseline = _report("m1")
    treatment = _report("m2", reuse=True, baseline="m1")
    for report, policy in (
        (baseline, "predict_only"), (treatment, "action_policy"),
    ):
        report["match_plan"].update({
            "experience": "world_model_lab",
            "world_model_policy": policy,
        })
    baseline["layers"]["society"] = {
        "Brazil": _society_snapshot("a"),
        "Argentina": _society_snapshot("b"),
    }
    treatment["layers"]["society"] = {
        "Brazil": _society_snapshot("c", morale=0.3),
        "Argentina": _society_snapshot("d"),
    }

    comparison = build_paired_comparison(baseline, treatment)

    divergence = comparison["society_divergence"]
    assert divergence["available"] is True
    assert divergence["state_change_count"] == 1
    assert divergence["state_changes"][0] == {
        "team": "Brazil",
        "scope": "psychological_state",
        "field": "morale",
        "baseline": 0.0,
        "treatment": 0.3,
    }
    future = comparison["counterfactual_future_summary"]
    assert future["schema_version"] == 2
    assert future["summary"][
        "descriptive_society_state_difference_count"
    ] == 1
    assert next(
        stage for stage in future["evidence_ladder"]
        if stage["stage"] == "action_to_society_state"
    )["claim_authorized"] is False
    assert future["claim_authority"][
        "downstream_society_state_causality"
    ] is False
    document = render_paired_comparison_html(comparison)
    assert 'data-testid="society-divergence-panel"' in document
    assert "复杂系统终局分叉" in document
    assert "Brazil" in document and "士气" in document
    assert "局部动作归因不会自动传递到社会状态" in document


def test_world_model_policy_fork_rejects_tactical_or_policy_cointervention():
    baseline = _report("m1")
    treatment = _report("m2", reuse=True, baseline="m1")
    for report, policy in (
        (baseline, "predict_only"), (treatment, "action_policy"),
    ):
        report["match_plan"].update({
            "experience": "world_model_lab",
            "world_model_policy": policy,
        })
    treatment["match_plan"]["home_tactic"] = "counter_attack"
    with pytest.raises(PairingError, match="keep both tactics fixed"):
        build_paired_comparison(baseline, treatment)
    treatment["match_plan"]["home_tactic"] = baseline["match_plan"]["home_tactic"]
    treatment["match_plan"]["world_model_policy"] = "predict_only"
    with pytest.raises(PairingError, match="predict_only to action_policy"):
        build_paired_comparison(baseline, treatment)


def test_world_model_fork_requires_identical_declared_branch_anchor():
    baseline = _report("m1")
    treatment = _report("m2", reuse=True, baseline="m1")
    for report, policy in (
        (baseline, "predict_only"), (treatment, "action_policy"),
    ):
        report["match_plan"].update({
            "experience": "world_model_lab",
            "world_model_policy": policy,
            "world_model_branch_at_sec": 2700.0,
        })
        report["layers"]["world_model"]["branch_anchor"] = {
            "available": True,
            "requested_sec": 2700.0,
            "actual_sec": 2700.0,
            "state_identity": "a" * 64,
        }
        report["simulation_clock"] = {
            "contract": "authoritative_tick_v2",
            "authoritative_tick_clock": True,
        }
    comparison = build_paired_comparison(baseline, treatment)
    assert comparison["eligibility"][
        "eligible_for_world_model_policy_attribution"
    ]
    assert comparison["intervention"]["branch_anchor"]["verified"]
    point = comparison["counterfactual_future_summary"]["intervention_point"]
    assert point["requested_sec"] == 2700.0
    assert point["actual_sec"] == 2700.0
    assert point["clock"] == "45:00"
    assert point["clock_contract"] == "authoritative_tick_v2"
    assert comparison["counterfactual_future_summary"][
        "evidence_ladder"
    ][0]["status"] == "verified"

    treatment["layers"]["world_model"]["branch_anchor"][
        "state_identity"
    ] = "b" * 64
    comparison = build_paired_comparison(baseline, treatment)
    assert not comparison["eligibility"][
        "eligible_for_world_model_policy_attribution"
    ]
    assert "same_pre_intervention_branch_anchor" in comparison[
        "eligibility"
    ]["failed_checks"]

    treatment["layers"]["world_model"]["branch_anchor"][
        "state_identity"
    ] = "a" * 64
    treatment["simulation_clock"] = {
        "contract": "legacy_affective_offset_v1",
        "authoritative_tick_clock": False,
    }
    comparison = build_paired_comparison(baseline, treatment)
    assert not comparison["eligibility"][
        "eligible_for_world_model_policy_attribution"
    ]
    assert "same_authoritative_product_clock" in comparison[
        "eligibility"
    ]["failed_checks"]
    assert comparison["counterfactual_future_summary"]["status"] == (
        "descriptive_only_ineligible"
    )


def test_world_model_propagation_degrades_corrupt_or_duplicate_evidence():
    baseline = _report("m1")
    treatment = _report("m2", reuse=True, baseline="m1")
    for report, policy in (
        (baseline, "predict_only"), (treatment, "action_policy"),
    ):
        report["match_plan"].update({
            "experience": "world_model_lab",
            "world_model_policy": policy,
        })
    treatment["layers"]["world_model"]["action_adoption"]["records"] = [
        {
            "opportunity_id": "duplicate",
            "team_id": "Brazil", "t_sec": 10,
            "counterfactual_baseline_action": "hold<script>",
            "actual_action": "pass", "policy_changed_action": True,
            "attribution_eligible": True,
        },
        {
            "opportunity_id": "duplicate",
            "team_id": "Brazil", "t_sec": 11,
            "counterfactual_baseline_action": "hold",
            "actual_action": "pass", "policy_changed_action": True,
            "attribution_eligible": True,
        },
        {"policy_changed_action": True, "t_sec": "bad"},
        "corrupt",
    ]
    treatment["replay"] = {"available": True, "events": [
        _replay_event(10, event_type="shot", link={
            "opportunity_id": "duplicate", "policy_changed_action": True,
        }),
    ]}
    comparison = build_paired_comparison(baseline, treatment)
    propagation = comparison["policy_propagation"]
    assert propagation["summary"]["valid_changed_decisions"] == 1
    assert propagation["summary"]["duplicate_opportunity_ids"] == 1
    assert propagation["summary"]["invalid_changed_records"] == 1
    assert propagation["summary"]["directly_observed_changes"] == 0
    assert propagation["summary"]["locally_attributable_changes"] == 0
    assert propagation["downstream_causal_attribution_authorized"] is False
    document = render_paired_comparison_html(comparison)
    assert "双方回放不足" in document
    assert "局部归因 0" in document
    assert "hold&lt;script&gt; → pass" in document
    assert "<script>" not in document


def test_world_model_propagation_is_bounded_and_no_change_is_explicit():
    baseline = _report("m1")
    treatment = _report("m2", reuse=True, baseline="m1")
    for report, policy in (
        (baseline, "predict_only"), (treatment, "action_policy"),
    ):
        report["match_plan"].update({
            "experience": "world_model_lab",
            "world_model_policy": policy,
        })
    comparison = build_paired_comparison(baseline, treatment)
    assert comparison["policy_propagation"]["status"] == (
        "no_realized_action_changes"
    )
    assert comparison["counterfactual_future_summary"]["status"] == (
        "no_realized_action_divergence"
    )
    assert comparison["counterfactual_future_summary"]["summary"][
        "changed_actions"
    ] == 0
    document = render_paired_comparison_html(comparison)
    assert "没有产生可保留的实际动作改变" in document
    assert "策略已介入，但动作未分叉" in document

    treatment["layers"]["world_model"]["action_adoption"]["records"] = [
        {
            "opportunity_id": f"changed-{index}",
            "team_id": "Brazil", "t_sec": index,
            "counterfactual_baseline_action": "hold",
            "actual_action": "pass", "policy_changed_action": True,
            "attribution_eligible": True,
        }
        for index in range(45)
    ]
    comparison = build_paired_comparison(baseline, treatment)
    summary = comparison["policy_propagation"]["summary"]
    assert summary["valid_changed_decisions"] == 45
    assert summary["retained_changed_decisions"] == 40
    assert summary["decisions_truncated"] is True
    document = render_paired_comparison_html(comparison)
    assert document.count('class="prop-decision"') == 40
    assert "仅保留最早 40 条" in document


def test_counterfactual_future_summary_preserves_missing_metrics():
    baseline = _report("m1")
    treatment = _report("m2", reuse=True, baseline="m1")
    for report, policy in (
        (baseline, "predict_only"), (treatment, "action_policy"),
    ):
        report["match_plan"].update({
            "experience": "world_model_lab",
            "world_model_policy": policy,
        })
    del treatment["result"]["xg"]["home"]

    future = build_paired_comparison(
        baseline, treatment,
    )["counterfactual_future_summary"]

    chance = next(
        layer for layer in future["layers"]
        if layer["layer"] == "chance_creation"
    )
    assert chance["observed_metric_count"] == 3
    assert chance["missing_metrics"] == ["xg_home"]
    assert future["summary"]["missing_outcome_metric_count"] == 1
    assert future["evidence_ladder"][-1]["status"] == (
        "incomplete_measurement"
    )

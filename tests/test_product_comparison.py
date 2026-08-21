import copy

import pytest

from src.product.comparison import (
    PairingError, build_paired_comparison, render_paired_comparison_html,
)


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
    if mutation == "seed": treatment["fixture"]["seed"] = 78
    elif mutation == "fixture": treatment["fixture"]["away"] = "France"
    elif mutation == "baseline_id": treatment["match_plan"]["paired_baseline_match_id"] = "other"
    else: treatment["match_plan"]["home_tactic"] = "gegenpress"
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
    if mutation == "fast": treatment["fixture"]["fast"] = False
    elif mutation == "integrity": treatment["integrity"]["accepted"] = False
    elif mutation == "checkpoint": (
        treatment["layers"]["world_model"]["runtime"].update({
            "checkpoint_signature": "sha256:other"
        })
    )
    else: treatment["studio"]["mode"] = "cognitive"
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

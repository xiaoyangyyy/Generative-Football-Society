"""World-model uncertainty separates learnable ignorance from event noise."""

import numpy as np
import pytest

from src.match_engine.world_model.uncertainty import (
    compose_uncertainty,
    ensemble_uncertainty_decomposition,
    uncertainty_decomposition_diagnostics,
    transition_ensemble_uncertainty,
)


def test_total_uncertainty_is_bounded_and_monotone_in_both_components():
    baseline = compose_uncertainty(0.2, 0.3)
    assert baseline == pytest.approx(0.44)
    assert compose_uncertainty(0.4, 0.3) > baseline
    assert compose_uncertainty(0.2, 0.5) > baseline
    assert 0.0 <= compose_uncertainty(-2.0, 5.0) <= 1.0


def test_head_disagreement_and_bernoulli_noise_are_not_conflated():
    disagreement = ensemble_uncertainty_decomposition(
        np.array([0.1, 0.9]),
        np.array([0.1, 0.9]),
        np.array([-0.2, 0.2]),
    )
    noisy_consensus = ensemble_uncertainty_decomposition(
        np.array([0.5, 0.5]),
        np.array([0.5, 0.5]),
        np.array([0.0, 0.0]),
    )

    assert disagreement["epistemic_uncertainty"] > noisy_consensus[
        "epistemic_uncertainty"
    ]
    assert disagreement["aleatoric_uncertainty"] < noisy_consensus[
        "aleatoric_uncertainty"
    ]
    assert disagreement["total_uncertainty"] == pytest.approx(
        compose_uncertainty(
            disagreement["epistemic_uncertainty"],
            disagreement["aleatoric_uncertainty"],
        )
    )


def test_transition_disagreement_requires_independently_trained_members():
    states = np.zeros((3, 1, 307), dtype=float)
    states[1, 0, 200] = 0.10
    states[2, 0, 200] = -0.10
    trained = transition_ensemble_uncertainty(states, trained=True)
    legacy = transition_ensemble_uncertainty(states, trained=False)

    assert trained["members"] == 3
    assert trained["ball_rms_disagreement"] > 0.0
    assert trained["epistemic_uncertainty"] > 0.0
    assert legacy["epistemic_uncertainty"] == 0.0


def test_hierarchical_variance_separates_heads_from_dynamics_members():
    states = np.zeros((2, 1, 307), dtype=float)
    member_shift = ensemble_uncertainty_decomposition(
        np.array([[0.2, 0.2], [0.8, 0.8]]),
        np.full((2, 2), 0.2),
        np.zeros((2, 2)),
        transition_state_samples=states,
        transition_ensemble_trained=True,
    )
    head_shift = ensemble_uncertainty_decomposition(
        np.array([[0.2, 0.8], [0.2, 0.8]]),
        np.full((2, 2), 0.2),
        np.zeros((2, 2)),
        transition_state_samples=states,
        transition_ensemble_trained=True,
    )

    assert member_shift["components"]["pass_epistemic"] == 0.0
    assert member_shift["components"]["pass_transition_epistemic"] > 0.0
    assert head_shift["components"]["pass_epistemic"] > 0.0
    assert head_shift["components"]["pass_transition_epistemic"] == 0.0


def test_decomposition_diagnostics_audit_realized_prediction_errors():
    records = [{
        "multi_horizon_regime_outcomes": {
            "60s": {
                "policy_utility": 0.3,
                "world_model_prediction": {
                    "raw_policy_utility": 0.1,
                    "policy_utility": 0.1,
                    "epistemic_uncertainty": 0.2,
                    "aleatoric_uncertainty": 0.25,
                    "uncertainty": compose_uncertainty(0.2, 0.25),
                },
            },
        },
    }]
    report = uncertainty_decomposition_diagnostics([records])

    assert report["samples"] == 1
    assert report["decomposition_available"]
    assert report["mean_epistemic_uncertainty"] == pytest.approx(0.2)
    assert report["mean_aleatoric_uncertainty"] == pytest.approx(0.25)
    assert report["mean_composition_identity_error"] == pytest.approx(0.0)

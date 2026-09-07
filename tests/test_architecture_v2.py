import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_architecture_v2_machine_audit_passes():
    result = subprocess.run(
        [sys.executable, "scripts/audit_architecture_v2.py"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(
        (ROOT / "data/evaluation/architecture_v2_audit.json").read_text(encoding="utf-8")
    )
    assert report["all_pass"]
    assert report["schema_version"] == 2
    assert not report["violations"]["domain_product_or_training_imports"]
    assert not report["violations"]["infrastructure_upward_imports"]
    assert not report["violations"]["unclassified_training_entrypoints"]
    assert report["checks"][
        "society_agent_psychology_is_dedicated_and_inherited"
    ]
    assert report["checks"][
        "society_agent_facade_has_dedicated_behavior_layers"
    ]
    assert report["checks"][
        "tournament_manager_facade_has_dedicated_lifecycle_and_state"
    ]
    assert report["checks"][
        "product_workspace_session_repository_is_dedicated_and_replay_complete"
    ]
    assert report["checks"][
        "product_workspace_evidence_projection_is_dedicated_and_read_only"
    ]
    assert report["integrity"]["release_artifacts"]["ok"]
    assert report["integrity"]["world_model_identity_chain_verified"]
    assert report["checks"]["studio_exposes_one_guided_end_to_end_workflow"]
    assert report["checks"][
        "manager_future_review_is_identity_bound_and_non_causal"
    ]
    assert report["checks"][
        "manager_future_scenarios_are_replayable_and_product_visible"
    ]
    assert report["checks"][
        "manager_future_action_examples_are_bounded_and_non_causal"
    ]
    assert report["checks"][
        "manager_future_review_closes_to_runtime_selection_without_outcome_claim"
    ]
    assert report["checks"][
        "official_manager_matches_surface_world_model_action_execution"
    ]
    assert report["checks"][
        "manager_product_exposes_one_replayable_world_evolution_thread"
    ]
    assert report["checks"][
        "society_cognition_persists_into_next_match_and_product_thread"
    ]
    assert report["checks"][
        "meta_learning_is_delayed_identity_bound_and_product_visible"
    ]
    assert report["checks"][
        "meta_learning_governance_is_content_free_replayable_and_visible"
    ]
    assert report["checks"][
        "manager_counterfactual_workbench_is_one_replayable_workflow"
    ]
    assert report["checks"][
        "manager_season_has_one_current_and_historical_world_navigator"
    ]
    assert report["checks"][
        "manager_action_transition_matrix_is_accessible_and_honest"
    ]
    assert report["checks"][
        "action_change_expectation_matches_shared_uniform_sampler"
    ]
    assert report["checks"][
        "official_action_v4_preserves_exact_shared_uniform_expectation"
    ]
    assert report["checks"][
        "action_transition_map_drills_into_descriptive_world_propagation"
    ]
    assert report["checks"][
        "all_known_exposed_credentials_require_independent_v2_closure"
    ]
    assert report["checks"]["runtime_random_draws_are_identity_scoped"]
    assert report["checks"][
        "match_runtime_inputs_are_side_isolated_and_root_bound"
    ]
    assert report["checks"]["physics_official_score_path_fails_closed"]
    assert report["checks"][
        "prospective_m2_identity_closes_transitive_runtime_dependencies"
    ]
    assert report["checks"]["tournament_resume_binds_random_world_identity"]
    assert report["checks"][
        "tournament_resume_binds_portable_run_input_identity"
    ]
    assert report["checks"][
        "tournament_resume_restores_dynamic_world_and_receipted_reflection"
    ]
    assert report["checks"][
        "tournament_resume_rolls_back_internal_external_state_after_identity"
    ]
    assert report["checks"][
        "tournament_match_is_one_verified_rollback_complete_transaction"
    ]
    assert report["checks"][
        "public_tournament_lifecycle_shares_the_match_workspace_lease"
    ]
    assert report["checks"][
        "human_studies_are_registered_allocated_and_byte_verified"
    ]
    assert report["checks"][
        "product_value_delivery_is_blinded_scoring_sealed_and_product_visible"
    ]
    assert report["checks"][
        "product_value_participant_session_is_isolated_timed_and_attested"
    ]
    assert report["checks"][
        "release_readiness_separates_code_contract_from_external_results"
    ]
    assert not report["violations"]["direct_global_runtime_rng_draws"]
    assert report["checks"]["formal_experiment_is_preregistered_and_compute_bounded"]
    assert report["checks"]["formal_experiment_fails_closed_on_identity_or_partial_evidence"]

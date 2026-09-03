"""Frozen code-side constants for the preregistered M2 study."""

FROZEN_FIXTURES = [
    ["Mexico", "South Korea"],
    ["Brazil", "Germany"],
    ["France", "England"],
    ["Argentina", "Netherlands"],
    ["Spain", "Morocco"],
    ["Portugal", "Uruguay"],
]

FROZEN_TRAINING_CONFIGURATION = {
    "trainer": "world_model_v9_m2",
    "epochs": 45,
    "batch_size": 128,
    "learning_rate": 0.0003,
    "transition": "gru",
    "transition_ensemble_size": 3,
    "multi_step_loss_weight": 0.25,
    "multi_step_warmup_fraction": 0.2,
    "semantic_event_loss_weight": 0.2,
    "policy_utility_loss_weight": 0.25,
    "dataset_manifest": (
        "data/world_model/dataset_manifest_formal_v8_candidate.json"
    ),
    "ball_log_enabled": False,
    "sealed_test_used": False,
}

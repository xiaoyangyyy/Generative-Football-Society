from __future__ import annotations

import json

import pytest

from src.match_engine.world_model.opponent_contract import OPPONENT_HYPOTHESES
from src.match_engine.world_model.opponent_response import (
    compile_opponent_response_memory,
    load_opponent_response_memory,
    opponent_response_diagnostics,
)


def _posterior(primary: str, probability: float = 0.92):
    rest = (1.0 - probability) / (len(OPPONENT_HYPOTHESES) - 1)
    return {
        name: probability if name == primary else rest
        for name in OPPONENT_HYPOTHESES
    }


def _record(
    created: float,
    belief: str,
    *,
    actual_action: str = "",
    checkpoint: str = "checkpoint-a",
    environment: str = "env-a",
):
    return {
        "team_id": "Home",
        "checkpoint_signature": checkpoint,
        "environment_signature": environment,
        "created_t_sec": created,
        "intervention_actual_action": actual_action or None,
        "opponent_belief_context": {
            "opponent_team_id": "Away",
            "posterior": _posterior(belief),
        },
    }


def _match(
    source: str = "balanced",
    target: str = "low_block",
    *,
    action: str = "pass",
    checkpoint: str = "checkpoint-a",
    environment: str = "env-a",
    repeats: int = 1,
):
    records = []
    for index in range(repeats):
        start = float(index * 120)
        records.extend([
            _record(
                start, source, actual_action=action,
                checkpoint=checkpoint, environment=environment,
            ),
            _record(
                start + 60.0, target,
                checkpoint=checkpoint, environment=environment,
            ),
        ])
    return {
        "world_model_decision_adoption": {"records": records},
    }


def test_response_memory_activates_only_after_match_held_out_gain():
    memory = compile_opponent_response_memory(
        [_match() for _ in range(8)],
        checkpoint_signature="checkpoint-a",
        environment_signature="env-a",
    )
    profile = memory.profiles["pass"]
    prediction = memory.predict(_posterior("balanced"), "pass")

    assert profile.training_matches == 4
    assert profile.validation_matches == 4
    assert profile.validation_skill > 0.02
    assert profile.active
    assert 0.0 < profile.trust <= 0.50
    assert prediction["learned_active"]
    assert prediction["response_posterior"]["low_block"] > (
        prediction["structural_posterior"]["low_block"]
    )
    assert sum(prediction["response_posterior"].values()) == pytest.approx(1.0)
    assert not prediction["causal_interpretation"]


def test_each_match_counts_once_even_with_many_within_match_pairs():
    memory = compile_opponent_response_memory(
        [_match(repeats=8), *[_match() for _ in range(3)]],
        checkpoint_signature="checkpoint-a",
        environment_signature="env-a",
    )
    profile = memory.profiles["pass"]

    assert profile.transition_matches == 4
    assert profile.transition_pairs == 11


def test_failed_validation_keeps_learned_response_at_zero_authority():
    logs = [
        *[_match(target="low_block") for _ in range(4)],
        *[_match(target="balanced") for _ in range(4)],
    ]
    memory = compile_opponent_response_memory(
        logs,
        checkpoint_signature="checkpoint-a",
        environment_signature="env-a",
    )
    profile = memory.profiles["pass"]
    prediction = memory.predict(_posterior("balanced"), "pass")

    assert not profile.active
    assert profile.trust == 0.0
    assert not prediction["learned_active"]
    assert prediction["response_posterior"] == prediction["structural_posterior"]


def test_response_memory_rejects_incompatible_and_unexecuted_actions():
    memory = compile_opponent_response_memory(
        [
            _match(checkpoint="checkpoint-b"),
            _match(environment="env-b"),
            {
                "world_model_decision_adoption": {
                    "records": [
                        _record(0.0, "balanced"),
                        _record(60.0, "low_block"),
                    ],
                },
            },
        ],
        checkpoint_signature="checkpoint-a",
        environment_signature="env-a",
    )

    assert memory.compatible_matches == 0
    assert memory.profiles["pass"].transition_pairs == 0
    assert not memory.predict(_posterior("balanced"), "pass")["learned_active"]


def test_response_loader_ignores_corrupt_logs(tmp_path):
    directory = tmp_path / "data" / "persistence" / "cognitive_log"
    directory.mkdir(parents=True)
    (directory / "valid.json").write_text(json.dumps(_match()), encoding="utf-8")
    (directory / "broken.json").write_text("{bad", encoding="utf-8")

    memory = load_opponent_response_memory(
        tmp_path,
        checkpoint_signature="checkpoint-a",
        environment_signature="env-a",
    )
    assert memory.source_logs == 1
    assert memory.compatible_matches == 1


def test_response_diagnostics_are_match_clustered_and_audit_non_persistence():
    current = _record(0.0, "balanced", actual_action="pass")
    current["opponent_response_context"] = {
        "prediction": {
            "action": "pass",
            "learned_active": True,
            "source": "validated_action_conditioned_response_memory",
            "response_posterior": _posterior("low_block"),
            "structural_posterior": _posterior("balanced"),
        },
    }
    following = _record(60.0, "low_block")
    report = opponent_response_diagnostics([{
        "world_model_decision_adoption": {
            "records": [current, following],
        },
        "cognitive_plans": [{
            "plan": {
                "opponent_response_hypothesis_audit": {
                    "accepted": True,
                    "can_update_response_memory": False,
                    "hypothesis": {"if_action": "pass"},
                },
            },
        }],
    }])

    assert report["available"]
    assert report["realized_predictions"] == 1
    assert report["validated_learned_predictions"] == 1
    assert report["scored_matches"] == 1
    assert report["match_clustered_brier"] < report["structural_baseline_brier"]
    assert report["llm_hypotheses_accepted"] == 1
    assert report["all_llm_hypotheses_non_persistent"]
    assert not report["causal_interpretation"]

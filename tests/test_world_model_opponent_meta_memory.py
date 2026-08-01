from __future__ import annotations

import json

import pytest

from src.match_engine.state import (
    CoachAffectiveState,
    CrowdState,
    MatchAffectiveState,
    RefereeAffectiveState,
    TeamAffectiveState,
)
from src.match_engine.tactical_catalog import TACTICAL_PRESETS
from src.match_engine.world_model.opponent_belief import (
    OPPONENT_HYPOTHESES,
    update_opponent_belief,
)
from src.match_engine.world_model.opponent_meta_memory import (
    compile_opponent_meta_belief_memory,
    load_opponent_meta_belief_memory,
)


def _posterior(primary: str, probability: float = 0.82):
    remainder = (1.0 - probability) / (len(OPPONENT_HYPOTHESES) - 1)
    return {
        name: probability if name == primary else remainder
        for name in OPPONENT_HYPOTHESES
    }


def _log(
    primary: str,
    *,
    snapshots: int = 5,
    checkpoint: str = "checkpoint-a",
    environment: str = "env-a",
    observed_preset: str | None = None,
):
    observed = (
        [
            TACTICAL_PRESETS[observed_preset][feature]
            for feature in (
                "pressing_intensity", "risk_budget", "line_height",
                "rotation_aggressiveness",
            )
        ]
        if observed_preset is not None else None
    )
    return {
        "home": "Home",
        "away": "Away",
        "world_model_decision_adoption": {
            "records": [
                {
                    "checkpoint_signature": checkpoint,
                    "environment_signature": environment,
                    "opponent_belief_context": {
                        "opponent_team_id": "Away",
                        "posterior": _posterior(primary),
                        "observed_feature_vector": observed,
                    },
                }
                for _ in range(snapshots)
            ],
        },
    }


def _state(opponent_preset: str = "balanced"):
    controls = {
        feature: TACTICAL_PRESETS[opponent_preset][feature]
        for feature in (
            "pressing_intensity", "risk_budget", "line_height",
            "rotation_aggressiveness",
        )
    }
    return MatchAffectiveState(
        home=TeamAffectiveState(
            team_id="Home", coach=CoachAffectiveState(team_id="Home"),
        ),
        away=TeamAffectiveState(
            team_id="Away",
            coach=CoachAffectiveState(
                team_id="Away", tactical_current=controls,
            ),
        ),
        referee=RefereeAffectiveState(),
        crowd=CrowdState(),
    )


def test_meta_memory_counts_matches_not_correlated_decision_snapshots():
    memory = compile_opponent_meta_belief_memory(
        [_log("low_block", snapshots=12) for _ in range(3)],
        checkpoint_signature="checkpoint-a",
        environment_signature="env-a",
    )
    profile = memory.profiles["Away"]

    assert profile.matches == 3
    assert profile.decision_snapshots == 36
    assert 0.0 < profile.trust <= 0.45
    assert profile.prior["low_block"] > 1.0 / len(OPPONENT_HYPOTHESES)
    assert sum(profile.prior.values()) == pytest.approx(1.0)


def test_meta_memory_isolates_checkpoint_and_policy_environment():
    memory = compile_opponent_meta_belief_memory(
        [
            _log("low_block"),
            _log("gegenpress", checkpoint="checkpoint-b"),
            _log("gegenpress", environment="env-b"),
        ],
        checkpoint_signature="checkpoint-a",
        environment_signature="env-a",
    )

    assert memory.compatible_matches == 1
    assert memory.profiles["Away"].matches == 1
    assert memory.profiles["Away"].empirical_posterior["low_block"] > 0.8


def test_raw_observation_prevents_recursive_posterior_self_reinforcement():
    memory = compile_opponent_meta_belief_memory(
        [
            _log(
                "low_block",
                observed_preset="gegenpress",
                snapshots=8,
            ),
        ],
        checkpoint_signature="checkpoint-a",
        environment_signature="env-a",
    )
    profile = memory.profiles["Away"]

    assert profile.observation_grounded_matches == 1
    assert profile.posterior_fallback_matches == 0
    assert profile.empirical_posterior["gegenpress"] > (
        profile.empirical_posterior["low_block"]
    )


def test_recent_cross_match_style_shift_quarantines_historical_strength():
    stable = compile_opponent_meta_belief_memory(
        [_log("low_block") for _ in range(4)],
        checkpoint_signature="checkpoint-a",
        environment_signature="env-a",
    ).profiles["Away"]
    shifted = compile_opponent_meta_belief_memory(
        [*[_log("low_block") for _ in range(3)], _log("gegenpress")],
        checkpoint_signature="checkpoint-a",
        environment_signature="env-a",
    ).profiles["Away"]

    assert stable.drift_status == "stable"
    assert shifted.drift_status == "quarantined"
    assert shifted.trust < stable.trust


def test_live_belief_uses_meta_prior_but_current_observation_can_override_it():
    memory = compile_opponent_meta_belief_memory(
        [_log("low_block") for _ in range(6)],
        checkpoint_signature="checkpoint-a",
        environment_signature="env-a",
    )
    state = _state("gegenpress")
    state._wm_opponent_meta_belief_memory = memory
    belief = update_opponent_belief(state, "Home")

    assert belief["meta_prior"]["available"]
    assert belief["meta_prior"]["matches"] == 6
    assert belief["posterior"]["gegenpress"] > belief["posterior"]["low_block"]


def test_loader_ignores_corrupt_logs(tmp_path):
    log_dir = tmp_path / "data" / "persistence" / "cognitive_log"
    log_dir.mkdir(parents=True)
    (log_dir / "good.json").write_text(json.dumps(_log("low_block")), encoding="utf-8")
    (log_dir / "bad.json").write_text("{broken", encoding="utf-8")

    memory = load_opponent_meta_belief_memory(
        tmp_path,
        checkpoint_signature="checkpoint-a",
        environment_signature="env-a",
    )
    assert memory.source_logs == 1
    assert memory.profiles["Away"].matches == 1

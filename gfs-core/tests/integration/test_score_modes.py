from dataclasses import replace
from pathlib import Path

import pytest

from gfs_core.bootstrap.application import CoreApplication
from gfs_core.contracts.micro_config import MicroMatchConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_score_mode_contract_and_legacy_mapping():
    scheduled = MicroMatchConfig()
    assert scheduled.score_mode == "scheduled"
    assert scheduled.physics_goals is False
    physics = replace(scheduled, score_mode="physics")
    assert physics.physics_goals is True
    hybrid = replace(scheduled, use_micro_goals=True)
    assert hybrid.score_mode == "hybrid"
    with pytest.raises(ValueError, match="score_mode"):
        replace(scheduled, score_mode="manual")


def test_physics_score_is_not_overwritten_by_macro_prior(tmp_path):
    app = CoreApplication(PROJECT_ROOT, workspace_root=tmp_path)
    scheduled = app.run_micro_match(fast=True, seed=42)
    assert scheduled.score_mode == "scheduled"
    assert scheduled.score_provenance["official_from_physics"] is False
    physics = app.run_micro_match(fast=True, seed=42, score_mode="physics")
    assert physics.score_mode == "physics"
    assert physics.score_provenance["official_from_physics"] is True
    assert physics.score_provenance["supplement"] == {}
    assert [physics.goals_micro_home, physics.goals_micro_away] == [
        physics.goals_physics_home, physics.goals_physics_away,
    ]
    assert physics.score_provenance["physics_goals"] == [
        physics.goals_micro_home, physics.goals_micro_away,
    ]

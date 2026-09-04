"""Phase 4 — single official score path."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest import mock

import pytest

from src.simulation.score_path import (
    OfficialScoreIntegrityError,
    ScorePathMode,
    finalize_official_score_from_micro,
    physics_official_enabled,
    resolve_score_path_mode,
    scheduled_shots_allowed,
    xg_supplement_allowed,
)


def _env(**kwargs: str):
    return mock.patch.dict(os.environ, kwargs, clear=True)


def test_default_micro_is_physics_official():
    with _env(MATCH_MICRO="1", MATCH_MICRO_SCORE="1"):
        assert resolve_score_path_mode() == ScorePathMode.PHYSICS_OFFICIAL
        assert physics_official_enabled()


def test_micro_replay_when_score_off():
    with _env(MATCH_MICRO="1", MATCH_MICRO_SCORE="0"):
        assert resolve_score_path_mode() == ScorePathMode.MICRO_REPLAY


def test_legacy_poisson():
    with _env(MATCH_LEGACY_POISSON="1", MATCH_MICRO="1"):
        assert resolve_score_path_mode() == ScorePathMode.POISSON_LEGACY


def test_xg_supplement_blocked_on_physics_path():
    with _env(MATCH_MICRO="1", MATCH_MICRO_SCORE="1", XG_SUPPLEMENT="1"):
        assert xg_supplement_allowed() is False
    with _env(MATCH_MICRO="1", MATCH_MICRO_SCORE="0", XG_SUPPLEMENT="1"):
        assert xg_supplement_allowed() is True


def test_scheduled_shots_off_by_default_on_physics():
    with _env(MATCH_MICRO="1", MATCH_MICRO_SCORE="1"):
        assert scheduled_shots_allowed() is False
    with _env(MATCH_MICRO="1", MATCH_MICRO_SCORE="1", MATCH_SCHEDULED_SHOTS="calibrated"):
        assert scheduled_shots_allowed() is True


class _Summary:
    goals_micro_home = 2
    goals_micro_away = 1
    micro_xg_home = 1.8
    micro_xg_away = 0.9
    goals_physics_home = 2
    goals_physics_away = 1
    xg_supplement_applied = False


def test_finalize_official_score_from_micro():
    s1, s2, x1, x2, meta = finalize_official_score_from_micro(
        _Summary(),
        home_micro="Mexico",
        fixture_home="Mexico",
        macro_xg_prior_home=1.2,
        macro_xg_prior_away=0.8,
    )
    assert (s1, s2) == (2, 1)
    assert (x1, x2) == (1.8, 0.9)
    assert meta["source"] == "physics_official"
    assert meta["macro_xg_prior"] == [1.2, 0.8]


def _valid_summary(**overrides):
    values = {
        "goals_micro_home": 2,
        "goals_micro_away": 1,
        "micro_xg_home": 1.8,
        "micro_xg_away": 0.9,
        "goals_physics_home": 2,
        "goals_physics_away": 1,
        "xg_supplement_applied": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _finalize(summary, *, prior_home=1.2, prior_away=0.8):
    return finalize_official_score_from_micro(
        summary,
        home_micro="Mexico",
        fixture_home="Mexico",
        macro_xg_prior_home=prior_home,
        macro_xg_prior_away=prior_away,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("goals_micro_home", True),
        ("goals_micro_home", 1.5),
        ("goals_physics_away", -1),
        ("micro_xg_home", float("nan")),
        ("micro_xg_away", float("inf")),
        ("micro_xg_away", -0.1),
        ("xg_supplement_applied", 0),
    ],
)
def test_physics_official_summary_rejects_invalid_evidence(field, value):
    with pytest.raises(OfficialScoreIntegrityError, match=field):
        _finalize(_valid_summary(**{field: value}))


def test_physics_official_summary_requires_complete_evidence():
    summary = _valid_summary()
    del summary.goals_physics_home

    with pytest.raises(OfficialScoreIntegrityError, match="missing goals_physics_home"):
        _finalize(summary)


def test_physics_official_summary_rejects_supplement_or_goal_mismatch():
    with pytest.raises(OfficialScoreIntegrityError, match="xG supplement"):
        _finalize(_valid_summary(xg_supplement_applied=True))
    with pytest.raises(OfficialScoreIntegrityError, match="do not match"):
        _finalize(_valid_summary(goals_physics_home=1))


@pytest.mark.parametrize("prior", [True, float("nan"), float("inf"), -0.1])
def test_physics_official_summary_rejects_invalid_macro_prior(prior):
    with pytest.raises(
        OfficialScoreIntegrityError, match="macro_xg_prior_home"
    ):
        _finalize(_valid_summary(), prior_home=prior)

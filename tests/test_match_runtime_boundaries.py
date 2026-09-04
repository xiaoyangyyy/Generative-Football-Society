from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import src.match_engine.match_affective_runner as affective_runner
import src.match_engine.match_micro_runner as micro_runner
import src.simulation.match_pipeline as match_pipeline
import src.simulation.tournament_reporting as tournament_reporting
import src.simulation.tournament_scoring as tournament_scoring
from src.match_engine.internal_signals import normalize_internal_match_signals


def _micro_kwargs() -> dict:
    return {
        "xg_prior_home": 1.2,
        "xg_prior_away": 0.9,
        "eff_status_home": 70.0,
        "eff_status_away": 68.0,
        "referee": {"strictness": 0.5},
        "stage_pressure": 0.4,
        "drama_score": 0.3,
        "internal_home": {"coordination": 0.2, "conflict_heat": 0.1},
        "internal_away": {"coordination": 0.8, "conflict_heat": 0.9},
        "seed": 7,
    }


def test_internal_signals_are_side_specific_and_swap_symmetric():
    home = {"coordination": 0.2, "conflict_heat": 0.1}
    away = {"coordination": 0.8, "conflict_heat": 0.9}

    signals = normalize_internal_match_signals(home, away)
    swapped = normalize_internal_match_signals(away, home)

    assert signals.coordination_home == 0.2
    assert signals.coordination_away == 0.8
    assert signals.conflict_home == 0.1
    assert signals.conflict_away == 0.9
    assert swapped.coordination_home == signals.coordination_away
    assert swapped.coordination_away == signals.coordination_home
    assert swapped.conflict_home == signals.conflict_away
    assert swapped.conflict_away == signals.conflict_home


def test_internal_signals_are_finite_bounded_and_have_explicit_defaults():
    signals = normalize_internal_match_signals(
        {"coordination": float("nan"), "conflict_heat": -4.0},
        {"coordination": 3.0, "conflict_heat": float("inf")},
    )

    assert signals.coordination_home == 0.6
    assert signals.coordination_away == 1.0
    assert signals.conflict_home == 0.0
    assert signals.conflict_away == 0.12

    defaults = normalize_internal_match_signals(None, {"coordination": "bad"})
    assert defaults.coordination_home == defaults.coordination_away == 0.6
    assert defaults.conflict_home == defaults.conflict_away == 0.12


def test_micro_initialization_passes_each_teams_conflict_to_schedule(monkeypatch):
    captured = {}
    state = SimpleNamespace()
    monkeypatch.setattr(micro_runner, "build_match_affective_state", lambda *a, **k: state)
    monkeypatch.setattr(micro_runner, "_init_micro_state", lambda *a, **k: None)

    def fake_schedule(*args, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(micro_runner, "build_event_schedule", fake_schedule)
    cfg = SimpleNamespace(
        dt_default=1.0,
        match_seconds=2.0,
        discipline_tick_fouls=True,
    )
    tactical = SimpleNamespace(bootstrap=lambda *args: None)

    result = micro_runner._initialize_micro_match_state(
        home_agent=object(), away_agent=object(), referee={"strictness": 0.5},
        stage_pressure=0.4, neutral_venue=False, eff_h=70.0, eff_a=68.0,
        poss_home=0.5, cfg=cfg, rng=np.random.default_rng(1),
        tactical_engine=tactical, tactical_override_home=None,
        tactical_override_away=None,
        internal_home={"coordination": 0.2, "conflict_heat": 0.1},
        internal_away={"coordination": 0.8, "conflict_heat": 0.9},
        match_seconds=None, goals_home=0, goals_away=0,
        xg_home=1.0, xg_away=1.0, drama_score=0.3, base_dir=".",
    )

    assert result["conflict_h"] == 0.1
    assert result["conflict_a"] == 0.9
    assert captured["internal_conflict_home"] == 0.1
    assert captured["internal_conflict_away"] == 0.9


def test_match_pipeline_forwards_root_to_regulation_and_replay(monkeypatch, tmp_path):
    captured = []
    marker = object()

    def fake_micro(*args, **kwargs):
        captured.append(kwargs)
        return marker

    monkeypatch.setattr(micro_runner, "run_match_micro_simulation", fake_micro)
    agents = (object(), object())

    assert match_pipeline.run_physics_first_micro(
        *agents, **_micro_kwargs(), base_dir=tmp_path,
    ) is marker
    replay_kwargs = _micro_kwargs()
    replay_kwargs.pop("eff_status_home")
    replay_kwargs.pop("eff_status_away")
    assert match_pipeline.run_micro_layer(
        *agents,
        goals_home=1,
        goals_away=0,
        xg_home=replay_kwargs.pop("xg_prior_home"),
        xg_away=replay_kwargs.pop("xg_prior_away"),
        base_dir=tmp_path,
        **replay_kwargs,
    ) is marker

    assert [Path(call["base_dir"]) for call in captured] == [tmp_path, tmp_path]


def test_extra_time_preserves_root(monkeypatch, tmp_path):
    captured = {}
    marker = object()

    def fake_regulation(*args, **kwargs):
        captured.update(kwargs)
        return marker

    monkeypatch.setattr(match_pipeline, "run_physics_first_micro", fake_regulation)
    assert match_pipeline.run_extra_time_micro(
        object(), object(), **_micro_kwargs(), base_dir=tmp_path,
    ) is marker

    assert Path(captured["base_dir"]) == tmp_path
    assert captured["seed"] == 784
    assert captured["match_seconds"] == 30.0 * 60.0


def test_affective_runner_forwards_root_before_simulation(monkeypatch, tmp_path):
    class RootObserved(RuntimeError):
        pass

    def fake_state(*args, **kwargs):
        assert Path(kwargs["base_dir"]) == tmp_path
        raise RootObserved

    monkeypatch.setattr(affective_runner, "build_match_affective_state", fake_state)
    with pytest.raises(RootObserved):
        affective_runner.run_match_affective_simulation(
            object(), object(), goals_home=0, goals_away=0,
            xg_home=1.0, xg_away=1.0, base_dir=tmp_path,
        )


class _ScoringHarness(tournament_scoring.TournamentScoringMixin):
    def __init__(self, base_dir):
        self.base_dir = str(base_dir)

    @staticmethod
    def _map_micro_score(home_micro, t1_name, goals_home, goals_away):
        return goals_home, goals_away


def test_tournament_regulation_and_extra_time_use_manager_root(monkeypatch, tmp_path):
    roots = []
    summary = SimpleNamespace(
        goals_micro_home=1,
        goals_micro_away=0,
        goals_physics_home=1,
        goals_physics_away=0,
        micro_xg_home=0.4,
        micro_xg_away=0.2,
        xg_supplement_applied=False,
    )

    def fake_micro(*args, **kwargs):
        roots.append(Path(kwargs["base_dir"]))
        return summary

    monkeypatch.setattr(tournament_scoring, "expected_match_xg", lambda *a, **k: (1.0, 1.0, {}))
    monkeypatch.setattr(tournament_scoring, "run_physics_first_micro", fake_micro)
    monkeypatch.setattr(tournament_scoring, "run_extra_time_micro", fake_micro)
    monkeypatch.setattr(tournament_scoring, "print_micro_match_logs", lambda *a, **k: None)
    monkeypatch.setattr(
        tournament_scoring,
        "finalize_official_score_from_micro",
        lambda *a, **k: (1, 0, 0.4, 0.2, {}),
    )
    harness = _ScoringHarness(tmp_path)
    kwargs = _micro_kwargs()
    harness._run_physics_official_score(
        ah=object(), aa=object(), t1_name="Home", t2_name="Away",
        stage_name="Group", is_knockout=False, home_micro="Home",
        away_micro="Away", neutral_venue=False, pressure=0.4,
        referee=kwargs["referee"], internal_micro_h=kwargs["internal_home"],
        internal_micro_a=kwargs["internal_away"],
        eff_micro_home=70.0, eff_micro_away=68.0,
        fused_vol_h=0.2, fused_vol_a=0.3, seed=7,
        rng_score=np.random.default_rng(2), drama_pre=0.3,
    )
    result = harness._resolve_knockout_score(
        a1=object(), a2=object(), ah=object(), aa=object(),
        t1_name="Home", t2_name="Away", stage_name="Final",
        is_knockout=True, home_micro="Home", away_micro="Away",
        neutral_venue=True, pressure=0.8, referee=kwargs["referee"],
        internal_micro_h=kwargs["internal_home"],
        internal_micro_a=kwargs["internal_away"],
        eff_status_1=70.0, eff_status_2=68.0,
        eff_micro_home=70.0, eff_micro_away=68.0,
        physics_first=True, xg_prior_h=1.0, xg_prior_a=1.0,
        drama_pre=0.5, seed=7, s1=0, s2=0,
    )

    assert result["went_to_extra_time"] is True
    assert roots == [tmp_path, tmp_path]


def _regulation_kwargs():
    return {
        "a1": object(),
        "a2": object(),
        "ah": object(),
        "aa": object(),
        "t1_name": "Home",
        "t2_name": "Away",
        "stage_name": "Group",
        "is_knockout": False,
        "home_micro": "Home",
        "away_micro": "Away",
        "neutral_venue": False,
        "pressure": 0.4,
        "referee": {"strictness": 0.5},
        "internal_micro_h": {},
        "internal_micro_a": {},
        "eff_status_1": 70.0,
        "eff_status_2": 68.0,
        "eff_micro_home": 70.0,
        "eff_micro_away": 68.0,
        "fused_1": {"volatility": 0.0},
        "fused_2": {"volatility": 0.0},
        "fused_vol_h": 0.0,
        "fused_vol_a": 0.0,
        "match_seed": 7,
    }


def test_physics_official_failure_never_falls_back_to_macro(
    monkeypatch, tmp_path,
):
    monkeypatch.setenv("MATCH_MICRO", "1")
    monkeypatch.setenv("MATCH_MICRO_SCORE", "1")
    monkeypatch.setenv("MATCH_MICRO_STRICT", "0")
    monkeypatch.setattr(
        tournament_scoring,
        "expected_match_xg",
        lambda *a, **k: (1.0, 1.0, {}),
    )
    monkeypatch.setattr(
        tournament_scoring,
        "run_physics_first_micro",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("corrupt roster")),
    )
    macro_called = False

    def macro(*args, **kwargs):
        nonlocal macro_called
        macro_called = True
        return 0, 0, 0.0, 0.0, {}

    harness = _ScoringHarness(tmp_path)
    monkeypatch.setattr(harness, "_run_macro_regulation_score", macro)

    with pytest.raises(
        tournament_scoring.PhysicsOfficialScoreError,
        match="corrupt roster",
    ):
        harness._resolve_regulation_score(**_regulation_kwargs())

    assert macro_called is False


def test_physics_official_rejects_nonfinite_prior_before_micro(
    monkeypatch, tmp_path,
):
    monkeypatch.setenv("MATCH_MICRO", "1")
    monkeypatch.setenv("MATCH_MICRO_SCORE", "1")
    monkeypatch.setattr(
        tournament_scoring,
        "expected_match_xg",
        lambda *a, **k: (float("nan"), 1.0, {}),
    )
    micro_called = False

    def micro(*args, **kwargs):
        nonlocal micro_called
        micro_called = True

    monkeypatch.setattr(
        tournament_scoring, "run_physics_first_micro", micro
    )
    harness = _ScoringHarness(tmp_path)

    with pytest.raises(
        tournament_scoring.PhysicsOfficialScoreError,
        match="macro_xg_prior_home",
    ):
        harness._resolve_regulation_score(**_regulation_kwargs())

    assert micro_called is False


def test_macro_score_path_remains_an_explicit_choice(monkeypatch, tmp_path):
    monkeypatch.setenv("MATCH_MICRO", "0")
    monkeypatch.setenv("MATCH_MICRO_SCORE", "0")
    monkeypatch.setenv("MATCH_LEGACY_POISSON", "0")
    harness = _ScoringHarness(tmp_path)
    monkeypatch.setattr(
        harness,
        "_run_macro_regulation_score",
        lambda **kwargs: (1, 2, 0.8, 1.1, {}),
    )
    monkeypatch.setattr(
        tournament_scoring,
        "run_physics_first_micro",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("physics must not run")
        ),
    )

    result = harness._resolve_regulation_score(**_regulation_kwargs())

    assert result["score_path"].value == "macro_unified"
    assert result["physics_first"] is False
    assert (result["s1"], result["s2"]) == (1, 2)


def test_physics_extra_time_rejects_unproven_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(
        tournament_scoring,
        "run_extra_time_micro",
        lambda *a, **k: SimpleNamespace(
            goals_micro_home=1,
            goals_micro_away=0,
            goals_physics_home=0,
            goals_physics_away=0,
            micro_xg_home=0.3,
            micro_xg_away=0.1,
            xg_supplement_applied=False,
        ),
    )
    harness = _ScoringHarness(tmp_path)
    kwargs = _micro_kwargs()

    with pytest.raises(
        tournament_scoring.PhysicsOfficialScoreError,
        match="extra time.*do not match",
    ):
        harness._resolve_knockout_score(
            a1=object(), a2=object(), ah=object(), aa=object(),
            t1_name="Home", t2_name="Away", stage_name="Final",
            is_knockout=True, home_micro="Home", away_micro="Away",
            neutral_venue=True, pressure=0.8, referee=kwargs["referee"],
            internal_micro_h=kwargs["internal_home"],
            internal_micro_a=kwargs["internal_away"],
            eff_status_1=70.0, eff_status_2=68.0,
            eff_micro_home=70.0, eff_micro_away=68.0,
            physics_first=True, xg_prior_h=1.0, xg_prior_a=1.0,
            drama_pre=0.5, seed=7, s1=0, s2=0,
        )


class _ReportingHarness(tournament_reporting.TournamentReportingMixin):
    def __init__(self, base_dir):
        self.base_dir = str(base_dir)


def test_tournament_micro_and_affective_replays_use_manager_root(monkeypatch, tmp_path):
    roots = []

    def fake_micro(*args, **kwargs):
        roots.append(Path(kwargs["base_dir"]))
        return object()

    def fake_affective(*args, **kwargs):
        roots.append(Path(kwargs["base_dir"]))
        return SimpleNamespace(
            final_psi=0.0,
            home_coach_stress=0.1,
            away_coach_stress=0.2,
            ref_strictness_mean=0.5,
            tactical_drift_home=0.0,
            tactical_drift_away=0.0,
        )

    monkeypatch.setattr(tournament_reporting, "run_micro_layer", fake_micro)
    monkeypatch.setattr(tournament_reporting, "print_micro_match_logs", lambda *a, **k: None)
    monkeypatch.setattr(tournament_reporting, "env_bool", lambda *a, **k: True)
    monkeypatch.setattr(affective_runner, "run_match_affective_simulation", fake_affective)
    harness = _ReportingHarness(tmp_path)
    common = {
        "a1": object(), "a2": object(), "t1_name": "Home",
        "t2_name": "Away", "stage_name": "Group", "s1": 1, "s2": 0,
        "xg1": 1.0, "xg2": 0.8, "referee": {"strictness": 0.5},
        "pressure": 0.4, "drama_score": 0.3,
        "internal_1": {}, "internal_2": {}, "seed": 7,
        "micro_summary": None,
    }
    harness._run_report_replay(micro_replay_mode=True, **common)
    harness._run_report_replay(micro_replay_mode=False, **common)

    assert roots == [tmp_path, tmp_path]

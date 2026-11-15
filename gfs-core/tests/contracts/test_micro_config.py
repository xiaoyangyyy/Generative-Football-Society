from gfs_core.contracts.micro import MicroMatchConfig, MicroMatchSummary


def test_fast_demo_has_stable_coarse_runtime_settings():
    config = MicroMatchConfig.fast_demo()
    assert config.dt_default == 30.0
    assert config.grid_nx == 16
    assert config.grid_ny == 11
    assert config.lane_samples == 5
    assert config.enable_passing is True
    assert config.enable_spatial is True


def test_research_configuration_is_not_part_of_core_contract():
    config = MicroMatchConfig()
    assert not hasattr(config, "enable_continuous_event_clock")
    assert not hasattr(config, "enable_subtick_reception_queue")
    assert not hasattr(config, "enable_hierarchical_policy")


def test_core_summary_has_no_research_evidence_fields():
    summary = MicroMatchSummary(
        home_team="Brazil",
        away_team="Argentina",
        ticks=0,
        final_psi=0.0,
        home_coach_stress=0.0,
        away_coach_stress=0.0,
        ref_strictness_mean=0.0,
        home_emotion_mean={},
        away_emotion_mean={},
        icon_shock_home=0.0,
        icon_shock_away=0.0,
        tactical_drift_home=0.0,
        tactical_drift_away=0.0,
        controversy_integral=0.0,
    )
    assert not hasattr(summary, "world_model_runtime")
    assert not hasattr(summary, "cognitive_plans")
    assert not hasattr(summary, "continuous_clock")
    assert summary.simulation_clock == {}

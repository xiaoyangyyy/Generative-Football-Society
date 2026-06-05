from src.data_engine.entity_dynamics import (
    PLAYER_CHANNEL_NAMES,
    PLAYER_CONDITION_KEYS,
    build_player_dynamics_payload,
    player_channel_affinities,
    player_condition_equilibrium,
    player_exogenous_inputs,
)
from src.data_engine.player_loader import PlayerProfile, build_player_profile_from_observables


def test_player_has_full_dynamics_block():
    p = build_player_profile_from_observables(
        player_id="tm_1",
        name="Test Player",
        team_id="Brazil",
        role="ST",
        market_value_eur=50_000_000,
        international_caps=40,
        height_cm=182,
        squad_max_mv=80_000_000,
        age=25,
    )
    assert set(p["condition"].keys()) == set(PLAYER_CONDITION_KEYS)
    assert abs(sum(p["channel_affinities"].values()) - 1.0) < 1e-5
    assert p["primary_channel"] in PLAYER_CHANNEL_NAMES
    assert "u0" in p["dynamics_input"]
    assert len(p["role_embedding"]) == 8
    assert "pass_skill" in p["abilities"]


def test_gk_channel_bias():
    u = player_exogenous_inputs(
        log_market_value=15.0,
        international_caps=10,
        height_cm=190,
        squad_max_mv=50_000_000,
        age=28,
    )
    cond = player_condition_equilibrium(u, "GK")
    aff = player_channel_affinities(cond, "GK")
    assert aff["sweeper_keeper"] > aff["box_finisher"]


def test_player_profile_roundtrip():
    raw = build_player_profile_from_observables(
        player_id="tm_2",
        name="Mid",
        team_id="Spain",
        role="CM",
        market_value_eur=30_000_000,
        international_caps=20,
        height_cm=178,
        squad_max_mv=60_000_000,
    )
    prof = PlayerProfile.from_dict({**raw, "team_id": "Spain"})
    assert prof.dynamics_model == "entity_dynamics_v1"
    assert prof.condition["technical_quality"] > 0

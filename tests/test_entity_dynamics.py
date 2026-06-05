import numpy as np

from src.data_engine.entity_dynamics import (
    coach_mental_equilibrium,
    coach_exogenous_inputs,
    player_ability_field,
    preset_affinities,
    role_embedding,
    squad_observables,
    team_state_vector,
)


def test_role_embedding_smooth():
    e_gk = role_embedding("GK")
    e_st = role_embedding("ST")
    assert not np.allclose(e_gk, e_st)


def test_player_abilities_bounded_sigmoid():
    u = squad_observables(log_market_value=16.0, international_caps=50, height_cm=182, squad_max_mv=80_000_000)
    a = player_ability_field(u, "ST")
    assert 0.0 < a["tech"] < 1.0
    assert a["gk"] < 0.5
    a_gk = player_ability_field(u, "GK")
    assert a_gk["gk"] > a["gk"]


def test_coach_equilibrium_coupled():
    u = coach_exogenous_inputs(tenure_years=8, fifa_ranking=5, reputation_prior=0.7)
    m = coach_mental_equilibrium(u)
    assert set(m.keys()) == {
        "experience",
        "tactical_knowledge",
        "pressure_handling",
        "adaptability",
        "discipline",
        "motivation",
    }
    aff = preset_affinities(m)
    assert abs(sum(aff.values()) - 1.0) < 1e-5


def test_team_state_vector():
    squad = [{"tech": 0.7, "pace": 0.8, "press": 0.6, "aerial": 0.5, "gk": 0.1}] * 11
    coach = {"motivation": 0.7, "discipline": 0.6, "experience": 0.8}
    x = team_state_vector(squad, coach, fifa_ranking=10)
    assert "attack" in x and -1.0 <= x["attack"] <= 1.0

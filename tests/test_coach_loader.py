import json
import os

from src.data_engine.coach_loader import (
    CoachProfile,
    attach_coaches_to_agents,
    coach_authority_from_profile,
    derive_mental_attributes,
    load_coaches_json,
)
from src.simulation.agent import SocietyAgent


def test_derive_mental_long_tenure():
    m = derive_mental_attributes(in_charge_since=2012, fifa_ranking=5, coach_name="Didier Deschamps")
    assert m["experience"] > 0.7
    assert m["pressure_handling"] > 0.3


def test_coach_authority_bounded():
    prof = CoachProfile(
        team_id="France",
        name="Didier Deschamps",
        mental=derive_mental_attributes(in_charge_since=2012, fifa_ranking=5, coach_name="Didier Deschamps"),
    )
    a = coach_authority_from_profile(prof)
    assert 0.25 <= a <= 0.96


def test_wc2026_coaches_file_if_present():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "coaches", "wc2026_coaches.json")
    if not os.path.exists(path):
        return
    coaches = load_coaches_json(path)
    assert len(coaches) >= 40
    assert coaches["Brazil"].name == "Carlo Ancelotti"


def test_attach_coaches_to_agent():
    agent = SocietyAgent("Brazil", {"tier": "Core", "final_status_score": 80})
    prof = CoachProfile(
        team_id="Brazil",
        name="Carlo Ancelotti",
        preferred_preset="possession_control",
        mental={"experience": 0.9, "tactical_knowledge": 0.92, "pressure_handling": 0.8, "discipline": 0.7},
    )
    attach_coaches_to_agents({"Brazil": agent}, {"Brazil": prof})
    assert agent.coach_name == "Carlo Ancelotti"
    assert agent.coach_profile.preferred_preset == "possession_control"

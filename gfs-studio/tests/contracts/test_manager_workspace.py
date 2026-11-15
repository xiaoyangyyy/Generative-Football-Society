import json
from pathlib import Path
from types import SimpleNamespace

from gfs_studio.api import web_gateway
from gfs_studio.plugins.cognition.gateway.provider import GatewayConfig, LLMGateway, RuleBasedProvider
from gfs_studio.plugins.cognition.gateway.service import CognitionService
from gfs_studio.plugins.cognition.roles.service import RoleService
from gfs_studio.plugins.product.manager.workspace import ManagerWorkspace

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def summary():
    return SimpleNamespace(
        home_team="Brazil",
        away_team="Argentina",
        ticks=180,
        shots_home=0,
        shots_away=4,
        micro_xg_home=0.03,
        micro_xg_away=0.81,
        goals_micro_home=0,
        goals_micro_away=3,
        crosses_attempted=4,
        player_stats={"Brazil": {"p1": {"minutes": 75.0}}},
    )


def test_manager_dossier_has_four_bounded_sections():
    cognition = LLMGateway(RuleBasedProvider(), GatewayConfig())
    workspace = ManagerWorkspace(None, RoleService(CognitionService(cognition)))
    dossier = workspace.review(summary())
    assert set(dossier) == {
        "manager_choice",
        "reviewed_future",
        "advisory_reflection",
        "official_action",
        "society_transition",
        "causal_claim",
        "outcome_improvement_claim",
    }
    assert dossier["manager_choice"]["recommended_action"] == "reduce_low_value_shots"
    assert dossier["manager_choice"]["hold_reference"] is True
    assert dossier["official_action"]["score"] == [0, 3]
    assert dossier["society_transition"]["players"]["rotation_recommended"] is True
    assert dossier["society_transition"]["media"]["causal_claim"] is False
    assert dossier["causal_claim"] is False
    assert dossier["outcome_improvement_claim"] is False
    assert cognition.status()["matches"]


def test_reflection_identity_changes_with_match_evidence():
    cognition = LLMGateway(RuleBasedProvider(), GatewayConfig())
    workspace = ManagerWorkspace(None, RoleService(CognitionService(cognition)))
    workspace.review(summary())
    first = cognition.history[-1]["request_id"]
    changed = summary()
    changed.goals_micro_away += 1
    workspace.review(changed)
    assert cognition.history[-1]["request_id"] != first


def test_manager_briefing_http_runs_real_match():
    gateway = web_gateway(PROJECT_ROOT)
    response = gateway.handle(
        "POST",
        "/api/manager/briefing",
        host="127.0.0.1",
        body=json.dumps({"home": "Brazil", "away": "Argentina", "seed": 42}).encode(),
    )
    assert response.status == 200
    payload = json.loads(response.body)
    assert payload["official_action"]["score"] == [0, 3]
    assert payload["official_action"]["scope"] == "official_simulator_record"
    assert payload["reviewed_future"]["scope"] == "simulator_local_descriptive_comparison"
    assert payload["reviewed_future"]["comparison_contract"] == "same_seed_and_same_initial_data"
    assert len(payload["reviewed_future"]["candidates"]) == 4
    assert {candidate["seed"] for candidate in payload["reviewed_future"]["candidates"]} == {42, 43}
    assert payload["reviewed_future"]["evaluation"]["status"] == "complete"
    assert payload["reviewed_future"]["evaluation"]["outcome_improvement_claim"] is False
    assert payload["manager_choice"]["requires_human_review"] is True
    assert payload["manager_choice"]["official_simulation_changed"] is False
    assert payload["advisory_reflection"]["authority"] == "advisory_only"
    assert payload["causal_claim"] is False
    gateway.close()

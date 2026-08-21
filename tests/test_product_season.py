import copy

import pytest

from src.product.season import (
    ClubResourcePlan,
    SeasonPlan,
    archive_completed_season,
    build_round_robin,
    manager_board_review,
    manager_career_profile,
    manager_season_profile,
    matchday_command_center,
    new_season_state,
    next_matchday,
    opponent_preparation,
    season_standings,
    validate_manager_career_transition,
    validate_season_state,
)
from src.product.sporting_director import SportingDirective
from src.simulation.squad_registry import RecruitmentMove, RecruitmentPlan
from src.simulation.player_lifecycle import RetentionPlan
from src.simulation.player_market import FreeAgentPlan


def _state(teams=("A", "B", "C", "D"), legs=1):
    return new_season_state(
        SeasonPlan(teams=teams, legs=legs),
        season_id="season-0001", seed=42, created_at="2026-01-01T00:00:00Z",
    )


def test_plan_rejects_duplicate_unknown_manager_and_unbounded_team_counts():
    with pytest.raises(ValueError, match="unique"):
        SeasonPlan(teams=("Brazil", "brazil", "A", "B"))
    with pytest.raises(ValueError, match="manager_team"):
        SeasonPlan(teams=("A", "B", "C", "D"), manager_team="E")
    with pytest.raises(ValueError, match="4-16"):
        SeasonPlan(teams=("A", "B", "C"))
    with pytest.raises(ValueError, match="points target"):
        SeasonPlan(
            teams=("A", "B", "C", "D"), manager_team="A",
            manager_objective="points_target",
        )


def test_club_resource_plan_is_fixed_budget_and_has_no_current_match_bonus():
    with pytest.raises(ValueError, match="total 6"):
        ClubResourcePlan(recovery=4, medical=2, sports_science=2)
    with pytest.raises(ValueError, match="integers from 0 to 4"):
        ClubResourcePlan(recovery=True, medical=2, sports_science=3)
    with pytest.raises(ValueError, match="requires manager_team"):
        SeasonPlan(
            teams=("A", "B", "C", "D"),
            manager_resources=ClubResourcePlan(4, 1, 1),
        )

    plan = SeasonPlan(
        teams=("A", "B", "C", "D"), manager_team="A",
        manager_resources=ClubResourcePlan(4, 1, 1),
    )
    payload = plan.as_dict()["manager_resources"]
    assert payload["budget"] == 6
    assert payload["effects"] == {
        "baseline_rest_units": 1.0,
        "manager_rest_units": 1.48,
        "medical_recovery_credit_per_matchday": 0.15,
        "fatigue_load_factor": 0.96,
        "current_match_status_bonus": 0.0,
    }
    assert SeasonPlan.from_payload(plan.as_dict()) == plan
    tampered = copy.deepcopy(plan.as_dict())
    tampered["manager_resources"]["effects"]["fatigue_load_factor"] = 0.1
    with pytest.raises(ValueError, match="effects do not match"):
        SeasonPlan.from_payload(tampered)
    with pytest.raises(ValueError, match="requires manager_team"):
        SeasonPlan(
            teams=("A", "B", "C", "D"), manager_objective="champion",
        )


def test_recruitment_plan_is_frozen_and_requires_matching_season_evidence():
    recruitment = RecruitmentPlan(
        "market-0002-deadbeefdead",
        (RecruitmentMove("candidate-1", "player-1"),),
    )
    with pytest.raises(ValueError, match="requires manager_team"):
        SeasonPlan(
            teams=("A", "B", "C", "D"),
            manager_recruitment=recruitment,
        )
    plan = SeasonPlan(
        teams=("A", "B", "C", "D"), manager_team="A",
        manager_recruitment=recruitment,
    )
    assert SeasonPlan.from_payload(plan.as_dict()) == plan
    with pytest.raises(ValueError, match="transaction identity"):
        new_season_state(
            plan, season_id="season-0002", seed=42,
            created_at="2026-01-01T00:00:00Z",
        )
    transaction = {
        "season_id": "season-0002", "team": "A",
        "plan": recruitment.as_dict(),
    }
    state = new_season_state(
        plan, season_id="season-0002", seed=42,
        created_at="2026-01-01T00:00:00Z",
        recruitment_transaction=transaction,
    )
    validate_season_state(state)
    state["recruitment_transaction"]["team"] = "B"
    with pytest.raises(ValueError, match="transaction identity"):
        validate_season_state(state)


def test_retention_plan_is_frozen_and_requires_manager_control():
    retention = RetentionPlan("lifecycle-0002-deadbeef-deadbeefdead", ("p1",))
    with pytest.raises(ValueError, match="requires manager_team"):
        SeasonPlan(teams=("A", "B", "C", "D"), manager_retention=retention)
    plan = SeasonPlan(
        teams=("A", "B", "C", "D"), manager_team="A",
        manager_retention=retention,
    )
    assert SeasonPlan.from_payload(plan.as_dict()) == plan


def test_free_agent_plan_is_frozen_and_requires_manager_control():
    signing = FreeAgentPlan("free-market-0003-deadbeef-deadbeefdead", "free", "out")
    with pytest.raises(ValueError, match="requires manager_team"):
        SeasonPlan(teams=("A", "B", "C", "D"), manager_free_agent=signing)
    plan = SeasonPlan(
        teams=("A", "B", "C", "D"), manager_team="A",
        manager_free_agent=signing,
    )
    assert SeasonPlan.from_payload(plan.as_dict()) == plan


def test_sporting_directive_is_frozen_and_requires_manager_control():
    directive = SportingDirective(
        "a" * 64, philosophy="youth_pathway", risk_level="high",
        priority_roles=("CM", "ST"),
    )
    with pytest.raises(ValueError, match="requires manager_team"):
        SeasonPlan(
            teams=("A", "B", "C", "D"),
            manager_sporting_directive=directive,
        )
    plan = SeasonPlan(
        teams=("A", "B", "C", "D"), manager_team="A",
        manager_sporting_directive=directive,
    )
    assert SeasonPlan.from_payload(plan.as_dict()) == plan


@pytest.mark.parametrize("count,legs", [(4, 1), (5, 1), (6, 2)])
def test_round_robin_has_exact_pair_coverage_and_no_team_double_booked(count, legs):
    teams = tuple(f"T{i}" for i in range(count))
    fixtures = build_round_robin(SeasonPlan(teams=teams, legs=legs))
    assert len(fixtures) == count * (count - 1) // 2 * legs
    for matchday in {fixture["matchday"] for fixture in fixtures}:
        participants = [
            team for fixture in fixtures if fixture["matchday"] == matchday
            for team in (fixture["home"], fixture["away"])
        ]
        assert len(participants) == len(set(participants))
    unordered = [frozenset((fixture["home"], fixture["away"])) for fixture in fixtures]
    assert all(unordered.count(pair) == legs for pair in set(unordered))


def test_standings_are_recomputed_exactly_once_from_completed_fixtures():
    state = _state()
    first, second = state["fixtures"][:2]
    for fixture, score in ((first, (2, 1)), (second, (0, 0))):
        fixture.update({
            "state": "completed", "match_id": fixture["fixture_id"],
            "report": f"outputs/{fixture['fixture_id']}.json",
            "score": {"home": score[0], "away": score[1]},
        })
    rows = season_standings(state)
    assert sum(row["played"] for row in rows) == 4
    assert sum(row["points"] for row in rows) == 5
    assert rows[0]["points"] == 3
    assert next_matchday(state) == 2


def test_validation_fails_closed_on_fixture_or_result_tampering():
    state = _state()
    tampered = copy.deepcopy(state)
    tampered["fixtures"][0]["home"] = "intruder"
    with pytest.raises(ValueError, match="identity"):
        validate_season_state(tampered)
    invalid = copy.deepcopy(state)
    invalid["fixtures"][0]["state"] = "completed"
    with pytest.raises(ValueError, match="valid result"):
        validate_season_state(invalid)


def test_all_completed_season_has_no_next_matchday():
    state = _state()
    for fixture in state["fixtures"]:
        fixture.update({
            "state": "completed", "match_id": fixture["fixture_id"],
            "report": f"outputs/{fixture['fixture_id']}.json",
            "score": {"home": 0, "away": 0},
        })
    assert next_matchday(state) is None
    validate_season_state(state)


def test_matchday_command_center_requires_only_the_current_fixture_decision():
    teams = ("A", "B", "C", "D", "E")
    schedule = build_round_robin(SeasonPlan(teams=teams))
    first_day_teams = {
        team for fixture in schedule if fixture["matchday"] == 1
        for team in (fixture["home"], fixture["away"])
    }
    bye_team = next(team for team in teams if team not in first_day_teams)
    state = new_season_state(
        SeasonPlan(teams=teams, manager_team=bye_team),
        season_id="season-bye", seed=7, created_at="2026-01-01T00:00:00Z",
    )

    command = matchday_command_center(state)

    assert command["phase"] == "ready_to_advance"
    assert command["is_bye_matchday"] is True
    assert command["current_fixture"] is None
    assert command["next_fixture"]["matchday"] > 1
    assert command["decision_required"] is False
    assert command["primary_action"] == {"id": "advance_matchday", "enabled": True}


def test_matchday_command_center_derives_stakes_form_and_last_debrief():
    teams = ("A", "B", "C", "D")
    base_schedule = build_round_robin(SeasonPlan(teams=teams))
    manager = base_schedule[0]["home"]
    state = new_season_state(
        SeasonPlan(teams=teams, manager_team=manager),
        season_id="season-command", seed=9, created_at="2026-01-01T00:00:00Z",
    )
    for fixture in state["fixtures"]:
        if fixture["matchday"] != 1:
            continue
        home_goals = 2 if fixture["home"] == manager else 0
        away_goals = 1 if fixture["home"] == manager else 0
        fixture.update({
            "state": "completed",
            "match_id": fixture["fixture_id"],
            "report": f"outputs/{fixture['fixture_id']}.json",
            "dashboard": f"outputs/{fixture['fixture_id']}.html",
            "score": {"home": home_goals, "away": away_goals},
        })

    command = matchday_command_center(state)

    assert command["phase"] == "decision_required"
    assert command["current_matchday"] == 2
    assert command["current_fixture"]["manager_position"] == 1
    assert command["last_result"]["outcome"] == "win"
    assert command["last_result"]["points_earned"] == 3
    assert command["last_result"]["dashboard"].endswith(".html")
    assert command["recent_form"] == "W"
    assert command["primary_action"]["id"] == "submit_decision"


def test_matchday_command_center_does_not_call_a_partially_completed_day_a_bye():
    teams = ("A", "B", "C", "D")
    schedule = build_round_robin(SeasonPlan(teams=teams))
    managed_fixture = schedule[0]
    state = new_season_state(
        SeasonPlan(teams=teams, manager_team=managed_fixture["home"]),
        season_id="season-partial", seed=11, created_at="2026-01-01T00:00:00Z",
    )
    first = state["fixtures"][0]
    first.update({
        "state": "completed", "match_id": first["fixture_id"],
        "report": "outputs/played.json", "score": {"home": 1, "away": 0},
    })
    state["fixtures"][1]["state"] = "failed"

    command = matchday_command_center(state)

    assert command["current_matchday"] == 1
    assert command["is_bye_matchday"] is False
    assert command["current_fixture"]["state"] == "completed"
    assert command["decision_required"] is False
    assert command["phase"] == "ready_to_resume"
    assert command["primary_action"]["id"] == "resume_matchday"


def test_manager_profile_tracks_frozen_objective_decisions_and_archive():
    plan = SeasonPlan(
        teams=("A", "B", "C", "D"), manager_team="A",
        manager_objective="champion",
    )
    state = new_season_state(
        plan, season_id="season-career", seed=17,
        created_at="2026-01-01T00:00:00Z",
    )
    for fixture in state["fixtures"]:
        manager_home = fixture["home"] == "A"
        manager_away = fixture["away"] == "A"
        fixture.update({
            "state": "completed", "match_id": fixture["fixture_id"],
            "report": f"outputs/{fixture['fixture_id']}.json",
            "dashboard": f"outputs/{fixture['fixture_id']}.html",
            "score": (
                {"home": 2, "away": 0} if manager_home
                else {"home": 0, "away": 2} if manager_away
                else {"home": 0, "away": 0}
            ),
        })
        if manager_home or manager_away:
            fixture["manager_decision"] = {
                "team": "A", "tactic": "gegenpress", "rotation": "balanced",
            }
    state["state"] = "complete"

    profile = manager_season_profile(state)
    archived = archive_completed_season(state)

    assert profile["objective"]["status"] == "achieved"
    assert profile["objective"]["target_position"] == 1
    assert profile["record"] == {
        "win": 3, "draw": 0, "loss": 0, "matches": 3, "points": 9,
    }
    assert profile["decision_identity"]["dominant_tactic"] == "gegenpress"
    assert len(profile["journal"]) == 3
    assert all(entry["points_earned"] == 3 for entry in profile["journal"])
    assert archived["season_id"] == "season-career"
    assert archived["manager_profile"] == profile


def test_points_objective_can_be_mathematically_unreachable_without_prediction():
    state = new_season_state(
        SeasonPlan(
            teams=("A", "B", "C", "D"), manager_team="A",
            manager_objective="points_target", manager_points_target=7,
        ),
        season_id="season-points", seed=19,
        created_at="2026-01-01T00:00:00Z",
    )
    managed = [
        fixture for fixture in state["fixtures"]
        if "A" in {fixture["home"], fixture["away"]}
    ]
    for fixture in managed[:2]:
        fixture.update({
            "state": "completed", "match_id": fixture["fixture_id"],
            "report": f"outputs/{fixture['fixture_id']}.json",
            "score": (
                {"home": 0, "away": 1}
                if fixture["home"] == "A" else {"home": 1, "away": 0}
            ),
            "manager_decision": {"team": "A"},
        })

    objective = manager_season_profile(state)["objective"]

    assert objective["remaining_matches"] == 1
    assert objective["maximum_points"] == 3
    assert objective["status"] == "unreachable"


def _completed_manager_season(*, season_id, win):
    state = new_season_state(
        SeasonPlan(
            teams=("A", "B", "C", "D"), manager_team="A",
            manager_objective="top_half",
        ),
        season_id=season_id, seed=29,
        created_at="2026-01-01T00:00:00Z",
    )
    for fixture in state["fixtures"]:
        manager_home = fixture["home"] == "A"
        manager_away = fixture["away"] == "A"
        score = {"home": 0, "away": 0}
        if manager_home:
            score = {"home": 2 if win else 0, "away": 0 if win else 2}
        elif manager_away:
            score = {"home": 0 if win else 2, "away": 2 if win else 0}
        fixture.update({
            "state": "completed", "match_id": fixture["fixture_id"],
            "report": f"outputs/{fixture['fixture_id']}.json", "score": score,
        })
        if manager_home or manager_away:
            fixture["manager_decision"] = {
                "team": "A", "tactic": "balanced", "rotation": "balanced",
            }
    state["state"] = "complete"
    return state


def test_board_review_is_reproducible_game_policy_not_causal_claim():
    archive = archive_completed_season(
        _completed_manager_season(season_id="season-win", win=True)
    )
    review = archive["board_review"]

    assert review == manager_board_review(
        archive["manager_profile"], league_size=4,
    )
    assert review["components"] == {
        "objective": 18, "league_finish": 8, "decision_stewardship": 4,
        "season_commitments": -1,
        "player_role_promises": 0,
    }
    assert review["confidence_delta"] == 29
    assert review["reputation_delta"] == 10
    assert "not a real-world manager rating" in review["claim_boundary"]

    tampered = copy.deepcopy(archive["manager_profile"])
    tampered["record"]["points"] = 99
    with pytest.raises(ValueError, match="incomplete"):
        manager_board_review(tampered, league_size=4)


def test_board_review_treats_unavailable_player_promise_as_neutral():
    profile = manager_season_profile(
        _completed_manager_season(season_id="season-excused", win=True)
    )
    baseline = manager_board_review(profile, league_size=4)
    profile["player_role_promises"] = {
        "schema_version": 1,
        "available": True,
        "final": True,
        "contract_id": "promise-contract",
        "entries": [{"player_id": "young", "status": "excused"}],
        "board_consequence": {
            "fulfilled_confidence": 1,
            "missed_confidence": -2,
            "reputation_change": 0,
        },
    }

    review = manager_board_review(profile, league_size=4)

    assert review["components"]["player_role_promises"] == 0
    assert review["confidence_delta"] == baseline["confidence_delta"]
    assert review["reputation_delta"] == baseline["reputation_delta"]
    assert review["evidence"]["player_promise_statuses"] == {
        "young": "excused",
    }


def test_manager_career_has_consequences_and_resets_board_on_club_change():
    archives = [
        archive_completed_season(
            _completed_manager_season(season_id=f"season-loss-{index}", win=False)
        )
        for index in (1, 2)
    ]
    first = manager_career_profile(archives[:1])
    dismissed = manager_career_profile(archives)

    assert first["employment_status"] == "under_review"
    assert first["next_contract"] == {
        "same_club_allowed": True,
        "same_club_points_growth_required": 3,
        "baseline_points": 0,
        "reason": "under_review_requires_ambitious_objective",
    }
    assert dismissed["employment_status"] == "dismissed"
    assert dismissed["board_confidence"] == 10
    assert dismissed["reputation"] == 32
    with pytest.raises(ValueError, match="dismissed"):
        validate_manager_career_transition(
            SeasonPlan(teams=("A", "B", "C", "D"), manager_team="A"),
            dismissed,
        )

    new_club = new_season_state(
        SeasonPlan(teams=("A", "B", "C", "D"), manager_team="B"),
        season_id="season-new-club", seed=31,
        created_at="2026-01-01T00:00:00Z",
    )
    moved = manager_career_profile(archives, current_season=new_club)
    assert moved["current_team"] == "B"
    assert moved["appointment_state"] == "new_appointment"
    assert moved["appointments"] == 2
    assert moved["board_confidence"] == 60
    assert moved["reputation"] == 32
    assert moved["next_contract"]["same_club_allowed"] is True


def test_under_review_same_club_points_target_cannot_shrink():
    career = manager_career_profile([
        archive_completed_season(
            _completed_manager_season(season_id="season-loss", win=False)
        )
    ])
    with pytest.raises(ValueError, match="at least 3"):
        validate_manager_career_transition(
            SeasonPlan(
                teams=("A", "B", "C", "D"), manager_team="A",
                manager_objective="points_target", manager_points_target=2,
            ),
            career,
        )
    validate_manager_career_transition(
        SeasonPlan(
            teams=("A", "B", "C", "D"), manager_team="A",
            manager_objective="points_target", manager_points_target=3,
        ),
        career,
    )


def test_completed_current_season_previews_review_without_folding_it_twice():
    current = _completed_manager_season(
        season_id="season-pending-review", win=False,
    )
    career = manager_career_profile([], current_season=current)

    assert career["completed_seasons"] == 0
    assert career["appointments"] == 1
    assert career["board_confidence"] == 60
    assert career["reputation"] == 50
    assert career["pending_board_review"] is True
    assert career["pending_review"]["confidence_delta"] == -25
    assert career["projected_after_review"] == {
        "board_confidence": 35,
        "reputation": 41,
        "employment_status": "under_review",
    }
    assert career["next_contract"]["same_club_points_growth_required"] == 3


def test_opponent_preparation_requires_prior_dominance_and_never_reads_current_choice():
    state = new_season_state(
        SeasonPlan(teams=("A", "B", "C", "D"), manager_team="A"),
        season_id="season-adaptation", seed=43,
        created_at="2026-01-01T00:00:00Z",
    )
    managed = [
        fixture for fixture in state["fixtures"]
        if "A" in {fixture["home"], fixture["away"]}
    ]
    first = opponent_preparation(state, managed[0]["fixture_id"])
    assert first["state"] == "insufficient_evidence"
    assert first["selected_tactic"] == "team_identity"
    assert first["evidence"]["observation_count"] == 0

    for fixture in managed[:2]:
        fixture.update({
            "state": "completed", "match_id": fixture["fixture_id"],
            "report": f"outputs/{fixture['fixture_id']}.json",
            "score": {"home": 0, "away": 0},
            "manager_decision": {
                "team": "A", "tactic": "gegenpress", "rotation": "balanced",
            },
        })
    managed[2]["manager_decision"] = {
        "team": "A", "tactic": "tiki_taka", "rotation": "balanced",
    }
    prepared = opponent_preparation(state, managed[2]["fixture_id"])

    assert prepared["state"] == "adapted"
    assert prepared["selected_tactic"] == "direct_vertical"
    assert prepared["evidence"]["tactic_counts"] == {"gegenpress": 2}
    assert prepared["evidence"]["dominant_share"] == 1.0
    assert all(
        item["fixture_id"] != managed[2]["fixture_id"]
        for item in prepared["evidence"]["observations"]
    )
    assert "not learned" in prepared["claim_boundary"]

    state["fixtures"][state["fixtures"].index(managed[2])][
        "opponent_preparation"
    ] = prepared
    validate_season_state(state)
    tampered = copy.deepcopy(state)
    target = next(
        item for item in tampered["fixtures"]
        if item["fixture_id"] == managed[2]["fixture_id"]
    )
    target["opponent_preparation"]["selected_tactic"] = "balanced"
    with pytest.raises(ValueError, match="opponent preparation"):
        validate_season_state(tampered)


def test_opponent_preparation_falls_back_when_prior_pattern_is_ambiguous():
    state = new_season_state(
        SeasonPlan(teams=("A", "B", "C", "D"), manager_team="A"),
        season_id="season-ambiguous", seed=47,
        created_at="2026-01-01T00:00:00Z",
    )
    managed = [
        fixture for fixture in state["fixtures"]
        if "A" in {fixture["home"], fixture["away"]}
    ]
    for fixture, tactic in zip(managed[:2], ("gegenpress", "tiki_taka")):
        fixture.update({
            "state": "completed", "match_id": fixture["fixture_id"],
            "report": f"outputs/{fixture['fixture_id']}.json",
            "score": {"home": 0, "away": 0},
            "manager_decision": {
                "team": "A", "tactic": tactic, "rotation": "balanced",
            },
        })

    prepared = opponent_preparation(state, managed[2]["fixture_id"])
    assert prepared["state"] == "ambiguous_pattern"
    assert prepared["selected_tactic"] == "team_identity"
    assert prepared["evidence"]["dominant_share"] == 0.5

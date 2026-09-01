"""Deterministic, recoverable competition contract for GFS Studio."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.simulation.lineup import LineupSelection
from src.simulation.squad_registry import RecruitmentPlan
from src.simulation.player_lifecycle import RetentionPlan
from src.simulation.player_market import FreeAgentPlan
from src.product.club_strategy import validate_club_strategy_snapshot
from src.product.sporting_director import (
    SportingDirective, validate_sporting_brief, validate_sporting_choices,
)
from src.product.club_timeline import (
    ClubEventChoice, derive_club_situation, validate_club_timeline,
)
from src.product.season_commitments import (
    SeasonCommitmentPlan, build_commitment_contract, commitment_progress,
)
from src.product.world_state_evidence import (
    validate_fixture_world_state_transition, validate_team_state_snapshot,
)
from src.product.decision_advice import (
    validate_manager_advice_adoption, validate_manager_decision_advice,
)
from src.product.player_promises import (
    player_promise_progress, validate_player_promise_contract,
)
from src.match_engine.manager_plan import InMatchPlan


SEASON_SCHEMA_VERSION = 1
MIN_SEASON_TEAMS = 4
MAX_SEASON_TEAMS = 16
MANAGER_ROTATIONS = {"strongest", "balanced", "rotate"}
MANAGER_OBJECTIVES = {"champion", "top_half", "points_target"}
BOARD_CONFIDENCE_START = 60
MANAGER_REPUTATION_START = 50
CLUB_RESOURCE_BUDGET = 6
CLUB_RESOURCE_MAX_PER_AREA = 4
OPPONENT_ADAPTATION_MIN_OBSERVATIONS = 2
OPPONENT_ADAPTATION_MIN_SHARE = 0.60
OPPONENT_RESPONSE_POLICY = {
    "balanced": "gegenpress",
    "tiki_taka": "low_block_counter",
    "gegenpress": "direct_vertical",
    "counter_attack": "balanced",
    "low_block_counter": "tiki_taka",
    "direct_vertical": "low_block_counter",
}


def _clean_team(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("season team names must be non-empty strings")
    return value.strip()


@dataclass(frozen=True)
class ClubResourcePlan:
    recovery: int = 2
    medical: int = 2
    sports_science: int = 2

    def __post_init__(self) -> None:
        values = (self.recovery, self.medical, self.sports_science)
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            or not 0 <= value <= CLUB_RESOURCE_MAX_PER_AREA
            for value in values
        ):
            raise ValueError(
                f"club resource allocations must be integers from 0 to "
                f"{CLUB_RESOURCE_MAX_PER_AREA}"
            )
        if sum(values) != CLUB_RESOURCE_BUDGET:
            raise ValueError(
                f"club resource allocations must total {CLUB_RESOURCE_BUDGET}"
            )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ClubResourcePlan":
        if not isinstance(payload, Mapping):
            raise ValueError("club resource plan must be an object")
        plan = cls(
            recovery=payload.get("recovery", 2),
            medical=payload.get("medical", 2),
            sports_science=payload.get("sports_science", 2),
        )
        if payload.get("schema_version", 1) != 1:
            raise ValueError("unsupported club resource plan schema")
        if payload.get("budget", CLUB_RESOURCE_BUDGET) != CLUB_RESOURCE_BUDGET:
            raise ValueError("club resource budget identity mismatch")
        if (
            payload.get("effects") is not None
            and payload.get("effects") != club_resource_effects(plan)
        ):
            raise ValueError("club resource effects do not match allocations")
        return plan

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "budget": CLUB_RESOURCE_BUDGET,
            "recovery": self.recovery,
            "medical": self.medical,
            "sports_science": self.sports_science,
            "effects": club_resource_effects(self),
            "claim_boundary": (
                "frozen simulated club operations budget; effects alter bounded "
                "carryover mechanics and do not buy a win probability"
            ),
        }


def club_resource_effects(plan: ClubResourcePlan) -> dict[str, Any]:
    return {
        "baseline_rest_units": 1.0,
        "manager_rest_units": round(1.0 + 0.12 * plan.recovery, 6),
        "medical_recovery_credit_per_matchday": round(0.15 * plan.medical, 6),
        "fatigue_load_factor": round(1.0 - 0.04 * plan.sports_science, 6),
        "current_match_status_bonus": 0.0,
    }


@dataclass(frozen=True)
class SeasonPlan:
    teams: tuple[str, ...]
    legs: int = 1
    fast: bool = False
    manager_team: str | None = None
    manager_objective: str | None = None
    manager_points_target: int | None = None
    manager_resources: ClubResourcePlan | None = None
    manager_recruitment: RecruitmentPlan | None = None
    manager_retention: RetentionPlan | None = None
    manager_free_agent: FreeAgentPlan | None = None
    manager_sporting_directive: SportingDirective | None = None
    manager_commitments: SeasonCommitmentPlan | None = None

    def __post_init__(self) -> None:
        teams = tuple(_clean_team(team) for team in self.teams)
        object.__setattr__(self, "teams", teams)
        if not MIN_SEASON_TEAMS <= len(teams) <= MAX_SEASON_TEAMS:
            raise ValueError(
                f"season requires {MIN_SEASON_TEAMS}-{MAX_SEASON_TEAMS} teams"
            )
        folded = [team.casefold() for team in teams]
        if len(set(folded)) != len(folded):
            raise ValueError("season teams must be unique")
        if self.legs not in {1, 2}:
            raise ValueError("season legs must be 1 or 2")
        if not isinstance(self.fast, bool):
            raise ValueError("season fast must be boolean")
        if self.manager_team is not None:
            manager = _clean_team(self.manager_team)
            matches = [team for team in teams if team.casefold() == manager.casefold()]
            if not matches:
                raise ValueError("manager_team must belong to the season")
            object.__setattr__(self, "manager_team", matches[0])
            objective = self.manager_objective or "top_half"
            if objective not in MANAGER_OBJECTIVES:
                raise ValueError("unsupported manager objective")
            object.__setattr__(self, "manager_objective", objective)
            resources = self.manager_resources or ClubResourcePlan()
            if not isinstance(resources, ClubResourcePlan):
                raise ValueError("manager_resources must be a ClubResourcePlan")
            object.__setattr__(self, "manager_resources", resources)
            if (
                self.manager_recruitment is not None
                and not isinstance(self.manager_recruitment, RecruitmentPlan)
            ):
                raise ValueError("manager_recruitment must be a RecruitmentPlan")
            if (
                self.manager_retention is not None
                and not isinstance(self.manager_retention, RetentionPlan)
            ):
                raise ValueError("manager_retention must be a RetentionPlan")
            if (
                self.manager_free_agent is not None
                and not isinstance(self.manager_free_agent, FreeAgentPlan)
            ):
                raise ValueError("manager_free_agent must be a FreeAgentPlan")
            if (
                self.manager_sporting_directive is not None
                and not isinstance(
                    self.manager_sporting_directive, SportingDirective,
                )
            ):
                raise ValueError(
                    "manager_sporting_directive must be a SportingDirective"
                )
            commitments = self.manager_commitments or SeasonCommitmentPlan()
            if not isinstance(commitments, SeasonCommitmentPlan):
                raise ValueError(
                    "manager_commitments must be a SeasonCommitmentPlan"
                )
            object.__setattr__(self, "manager_commitments", commitments)
            if objective == "points_target":
                target = self.manager_points_target
                maximum = (len(teams) - 1) * self.legs * 3
                if (
                    isinstance(target, bool) or not isinstance(target, int)
                    or not 1 <= target <= maximum
                ):
                    raise ValueError(
                        f"manager points target must be between 1 and {maximum}"
                    )
            elif self.manager_points_target is not None:
                raise ValueError("manager points target requires points_target objective")
        elif (
            self.manager_objective is not None
            or self.manager_points_target is not None
            or self.manager_resources is not None
            or self.manager_recruitment is not None
            or self.manager_retention is not None
            or self.manager_free_agent is not None
            or self.manager_sporting_directive is not None
            or self.manager_commitments is not None
        ):
            raise ValueError(
                "manager objective, resources, recruitment, retention, or free agent requires manager_team"
            )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SeasonPlan":
        if not isinstance(payload, Mapping):
            raise ValueError("season plan must be an object")
        teams = payload.get("teams")
        if not isinstance(teams, Sequence) or isinstance(teams, (str, bytes)):
            raise ValueError("season teams must be an array")
        return cls(
            teams=tuple(teams),
            legs=payload.get("legs", 1),
            fast=payload.get("fast", False),
            manager_team=payload.get("manager_team"),
            manager_objective=payload.get("manager_objective"),
            manager_points_target=payload.get("manager_points_target"),
            manager_resources=(
                ClubResourcePlan.from_payload(payload["manager_resources"])
                if payload.get("manager_resources") is not None else None
            ),
            manager_recruitment=(
                RecruitmentPlan.from_payload(payload["manager_recruitment"])
                if payload.get("manager_recruitment") is not None else None
            ),
            manager_retention=(
                RetentionPlan.from_payload(payload["manager_retention"])
                if payload.get("manager_retention") is not None else None
            ),
            manager_free_agent=(
                FreeAgentPlan.from_payload(payload["manager_free_agent"])
                if payload.get("manager_free_agent") is not None else None
            ),
            manager_sporting_directive=(
                SportingDirective.from_payload(
                    payload["manager_sporting_directive"],
                )
                if payload.get("manager_sporting_directive") is not None
                else None
            ),
            manager_commitments=(
                SeasonCommitmentPlan.from_payload(
                    payload["manager_commitments"],
                )
                if payload.get("manager_commitments") is not None else None
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SEASON_SCHEMA_VERSION,
            "teams": list(self.teams),
            "legs": self.legs,
            "fast": self.fast,
            "manager_team": self.manager_team,
            "manager_objective": self.manager_objective,
            "manager_points_target": self.manager_points_target,
            "manager_resources": (
                self.manager_resources.as_dict()
                if self.manager_resources is not None else None
            ),
            "manager_recruitment": (
                self.manager_recruitment.as_dict()
                if self.manager_recruitment is not None else None
            ),
            "manager_retention": (
                self.manager_retention.as_dict()
                if self.manager_retention is not None else None
            ),
            "manager_free_agent": (
                self.manager_free_agent.as_dict()
                if self.manager_free_agent is not None else None
            ),
            "manager_sporting_directive": (
                self.manager_sporting_directive.as_dict()
                if self.manager_sporting_directive is not None else None
            ),
            "manager_commitments": (
                self.manager_commitments.as_dict()
                if self.manager_commitments is not None else None
            ),
            "claim_boundary": (
                "simulated competition experience; standings describe this seeded "
                "season and do not estimate real-world team strength"
            ),
        }


@dataclass(frozen=True)
class ManagerDecision:
    team: str
    tactic: str = "team_identity"
    rotation: str = "balanced"
    lineup: LineupSelection | None = None
    in_match_plan: InMatchPlan | None = None
    club_event_choice: ClubEventChoice | None = None

    def __post_init__(self) -> None:
        from src.product.match_plan import PLAYABLE_TACTICS

        object.__setattr__(self, "team", _clean_team(self.team))
        if self.tactic not in PLAYABLE_TACTICS:
            raise ValueError("unsupported manager tactic")
        if self.rotation not in MANAGER_ROTATIONS:
            raise ValueError("unsupported manager rotation")
        if self.in_match_plan is not None and self.in_match_plan.team != self.team:
            raise ValueError("in-match plan team identity mismatch")
        if (
            self.club_event_choice is not None
            and not isinstance(self.club_event_choice, ClubEventChoice)
        ):
            raise ValueError("club_event_choice must be a ClubEventChoice")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ManagerDecision":
        if not isinstance(payload, Mapping):
            raise ValueError("manager decision must be an object")
        return cls(
            team=payload.get("team"),
            tactic=payload.get("tactic", "team_identity"),
            rotation=payload.get("rotation", "balanced"),
            lineup=(
                LineupSelection.from_payload(payload["lineup"])
                if payload.get("lineup") is not None else None
            ),
            in_match_plan=(
                InMatchPlan.from_payload(payload["in_match_plan"])
                if payload.get("in_match_plan") is not None else None
            ),
            club_event_choice=(
                ClubEventChoice.from_payload(payload["club_event_choice"])
                if payload.get("club_event_choice") is not None else None
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "team": self.team,
            "tactic": self.tactic,
            "rotation": self.rotation,
            "lineup": self.lineup.as_dict() if self.lineup is not None else None,
            "in_match_plan": (
                self.in_match_plan.as_dict()
                if self.in_match_plan is not None else None
            ),
            "club_event_choice": (
                self.club_event_choice.as_dict()
                if self.club_event_choice is not None else None
            ),
            "claim_boundary": (
                "gameplay intervention in one simulated season fixture; "
                "not causal evidence or a real-world performance estimate"
            ),
        }


def build_round_robin(plan: SeasonPlan) -> list[dict[str, Any]]:
    """Circle-method schedule with stable matchday and fixture identities."""
    rotation: list[str | None] = list(plan.teams)
    if len(rotation) % 2:
        rotation.append(None)
    round_count = len(rotation) - 1
    half = len(rotation) // 2
    first_leg: list[list[tuple[str, str]]] = []
    for round_index in range(round_count):
        pairs: list[tuple[str, str]] = []
        for index in range(half):
            left = rotation[index]
            right = rotation[-1 - index]
            if left is None or right is None:
                continue
            if (round_index + index) % 2:
                home, away = right, left
            else:
                home, away = left, right
            pairs.append((home, away))
        first_leg.append(pairs)
        rotation = [rotation[0], rotation[-1], *rotation[1:-1]]

    rounds = list(first_leg)
    if plan.legs == 2:
        rounds.extend([[(away, home) for home, away in pairs] for pairs in first_leg])

    fixtures: list[dict[str, Any]] = []
    for matchday, pairs in enumerate(rounds, start=1):
        for order, (home, away) in enumerate(pairs, start=1):
            fixture_id = f"md{matchday:02d}-fx{order:02d}"
            fixtures.append({
                "fixture_id": fixture_id,
                "matchday": matchday,
                "order": order,
                "home": home,
                "away": away,
                "state": "scheduled",
                "match_id": None,
                "report": None,
                "score": None,
                "manager_decision": None,
            })
    return fixtures


def new_season_state(
    plan: SeasonPlan, *, season_id: str, seed: int, created_at: str,
    recruitment_transaction: Mapping[str, Any] | None = None,
    club_strategies: Mapping[str, Any] | None = None,
    sporting_brief: Mapping[str, Any] | None = None,
    sporting_evaluation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(season_id, str) or not season_id.strip():
        raise ValueError("season_id must be a non-empty string")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**31 - 1:
        raise ValueError("season seed must be a 32-bit non-negative integer")
    state = {
        "schema_version": SEASON_SCHEMA_VERSION,
        "season_id": season_id,
        "state": "active",
        "seed": seed,
        "created_at": created_at,
        "updated_at": created_at,
        "revision": 0,
        "plan": plan.as_dict(),
        "fixtures": build_round_robin(plan),
        "recruitment_transaction": (
            dict(recruitment_transaction)
            if recruitment_transaction is not None else None
        ),
        "club_strategies": (
            dict(club_strategies) if club_strategies is not None else None
        ),
        "sporting_brief": (
            dict(sporting_brief) if sporting_brief is not None else None
        ),
        "sporting_evaluation": (
            dict(sporting_evaluation)
            if sporting_evaluation is not None else None
        ),
        "club_timeline": {"schema_version": 1, "events": []},
        "season_commitments": None,
        "player_role_promises": None,
    }
    state["season_commitments"] = build_commitment_contract(state)
    validate_season_state(state)
    return state


def validate_season_state(state: Mapping[str, Any]) -> None:
    """Fail closed when schedule identity or a completed result was tampered with."""
    if not isinstance(state, Mapping) or state.get("schema_version") != SEASON_SCHEMA_VERSION:
        raise ValueError("unsupported season schema")
    if state.get("state") not in {"active", "complete"}:
        raise ValueError("invalid season state")
    plan = SeasonPlan.from_payload(state.get("plan") or {})
    expected_commitments = build_commitment_contract(state)
    if state.get("season_commitments") != expected_commitments:
        raise ValueError("season commitment contract source replay mismatch")
    player_promises = state.get("player_role_promises")
    if player_promises is not None:
        validate_player_promise_contract(player_promises)
        if (
            player_promises.get("season_id") != state.get("season_id")
            or player_promises.get("team") != plan.manager_team
            or player_promises.get("total_managed_matches")
            != (len(plan.teams) - 1) * plan.legs
        ):
            raise ValueError("player promise season identity mismatch")
    sporting_brief = state.get("sporting_brief")
    sporting_evaluation = state.get("sporting_evaluation")
    if plan.manager_sporting_directive is None:
        if sporting_brief is not None or sporting_evaluation is not None:
            raise ValueError("season has sporting evidence without a directive")
    else:
        if (
            not isinstance(sporting_brief, Mapping)
            or not isinstance(sporting_evaluation, Mapping)
        ):
            raise ValueError("season sporting evidence is unavailable")
        validate_sporting_brief(sporting_brief)
        expected_sporting = validate_sporting_choices(
            plan.manager_sporting_directive, sporting_brief,
            recruitment=(
                plan.manager_recruitment.as_dict()
                if plan.manager_recruitment is not None else None
            ),
            retention=(
                plan.manager_retention.as_dict()
                if plan.manager_retention is not None else None
            ),
            free_agent=(
                plan.manager_free_agent.as_dict()
                if plan.manager_free_agent is not None else None
            ),
        )
        if (
            sporting_brief.get("target_season_id") != state.get("season_id")
            or sporting_brief.get("team") != plan.manager_team
            or dict(sporting_evaluation) != expected_sporting
        ):
            raise ValueError("season sporting evidence mismatch")
    club_strategies = state.get("club_strategies")
    if club_strategies is not None:
        validate_club_strategy_snapshot(club_strategies, teams=plan.teams)
        if club_strategies.get("season_id") != state.get("season_id"):
            raise ValueError("club strategy season identity mismatch")
    recruitment = state.get("recruitment_transaction")
    if plan.manager_recruitment is None:
        if recruitment is not None:
            raise ValueError("season has recruitment evidence without a plan")
    elif (
        not isinstance(recruitment, Mapping)
        or recruitment.get("season_id") != state.get("season_id")
        or recruitment.get("team") != plan.manager_team
        or recruitment.get("plan") != plan.manager_recruitment.as_dict()
    ):
        raise ValueError("season recruitment transaction identity mismatch")
    expected = build_round_robin(plan)
    fixtures = state.get("fixtures")
    if not isinstance(fixtures, list) or len(fixtures) != len(expected):
        raise ValueError("season schedule identity mismatch")
    for actual, frozen in zip(fixtures, expected):
        if not isinstance(actual, Mapping) or any(
            actual.get(key) != frozen.get(key)
            for key in ("fixture_id", "matchday", "order", "home", "away")
        ):
            raise ValueError("season fixture identity mismatch")
        fixture_state = actual.get("state")
        if fixture_state not in {"scheduled", "running", "completed", "failed"}:
            raise ValueError("invalid season fixture state")
        if fixture_state == "completed":
            score = actual.get("score")
            if (
                not isinstance(score, Mapping)
                or any(isinstance(score.get(side), bool) or not isinstance(score.get(side), int) or score.get(side) < 0 for side in ("home", "away"))
                or not actual.get("match_id")
                or not actual.get("report")
            ):
                raise ValueError("completed season fixture lacks a valid result")
        before_state = actual.get("world_state_before")
        if before_state is not None:
            if fixture_state not in {"running", "failed"}:
                raise ValueError("pre-match world state requires an unfinished attempt")
            if not isinstance(before_state, Mapping) or set(before_state) != {
                actual.get("home"), actual.get("away"),
            }:
                raise ValueError("fixture pre-match world-state coverage mismatch")
            for team in (actual["home"], actual["away"]):
                validate_team_state_snapshot(before_state[team], team=team)
        transition = actual.get("world_state_transition")
        if transition is not None:
            if fixture_state != "completed":
                raise ValueError("world-state transition requires a completed fixture")
            validate_fixture_world_state_transition(
                transition, season_id=str(state.get("season_id") or ""),
                fixture_id=actual["fixture_id"], match_id=actual["match_id"],
                home=actual["home"], away=actual["away"],
            )
            recovered_days = state.get("recovered_matchdays") or []
            if bool(transition.get("recovery_complete")) != (
                actual["matchday"] in recovered_days
            ):
                raise ValueError("fixture recovery-state evidence is inconsistent")
        decision_payload = actual.get("manager_decision")
        advice = actual.get("manager_decision_advice")
        adoption = actual.get("manager_advice_adoption")
        future_reviews = actual.get("manager_future_reviews")
        if future_reviews is not None:
            from src.product.manager_future_review import (
                MAX_REVIEWS_PER_FIXTURE,
                validate_manager_future_review,
            )

            if (
                not isinstance(future_reviews, list)
                or not 1 <= len(future_reviews) <= MAX_REVIEWS_PER_FIXTURE
                or plan.manager_team is None
                or plan.manager_team not in {
                    actual.get("home"), actual.get("away"),
                }
            ):
                raise ValueError("manager future review history is invalid")
            task_ids = set()
            context_ids = set()
            for review in future_reviews:
                validate_manager_future_review(
                    review,
                    season_id=str(state.get("season_id") or ""),
                    fixture_id=str(actual.get("fixture_id") or ""),
                    matchday=int(actual.get("matchday") or 0),
                    manager_team=plan.manager_team,
                )
                if (
                    review["task_id"] in task_ids
                    or review["context_identity"] in context_ids
                ):
                    raise ValueError("manager future review is duplicated")
                task_ids.add(review["task_id"])
                context_ids.add(review["context_identity"])
        if advice is not None:
            validate_manager_decision_advice(advice)
            expected_seed = (
                int(state.get("seed", 0))
                + int(actual["matchday"]) * 100
                + int(actual["order"])
            )
            if (
                plan.manager_team is None
                or plan.manager_team not in {actual.get("home"), actual.get("away")}
                or advice.get("season_id") != state.get("season_id")
                or advice.get("fixture_id") != actual.get("fixture_id")
                or advice.get("home") != actual.get("home")
                or advice.get("away") != actual.get("away")
                or advice.get("manager_team") != plan.manager_team
                or advice.get("match_seed") != expected_seed
                or int(advice.get("issued_revision", -1))
                > int(state.get("revision", 0))
            ):
                raise ValueError("world-model manager advice fixture binding mismatch")
        elif adoption is not None:
            raise ValueError("manager advice adoption requires advice evidence")
        if decision_payload is not None:
            decision = ManagerDecision.from_payload(decision_payload)
            if (
                plan.manager_team is None
                or decision.team != plan.manager_team
                or decision.team not in {actual.get("home"), actual.get("away")}
            ):
                raise ValueError("manager decision fixture identity mismatch")
            availability = actual.get("player_promise_availability")
            if isinstance(player_promises, Mapping):
                expected_player_ids = {
                    row["player_id"] for row in player_promises["players"]
                }
                if (
                    not (
                        isinstance(availability, Mapping)
                        or (
                            not expected_player_ids
                            and player_promises.get("control")
                            == "deterministic_compatibility"
                            and availability is None
                        )
                    )
                    or (
                        isinstance(availability, Mapping)
                        and set(availability) != expected_player_ids
                    )
                    or any(
                        not isinstance(value, bool)
                        for value in (
                            availability.values()
                            if isinstance(availability, Mapping) else []
                        )
                    )
                ):
                    raise ValueError(
                        "manager decision player availability evidence mismatch"
                    )
            elif availability is not None:
                raise ValueError(
                    "player availability evidence requires a frozen promise contract"
                )
            if adoption is not None:
                validate_manager_advice_adoption(
                    adoption, advice=advice, selected_tactic=decision.tactic,
                )
        elif adoption is not None:
            raise ValueError("manager advice adoption requires a frozen decision")
        if future_reviews is not None and decision_payload is None:
            raise ValueError("manager future review requires a frozen decision")
    for fixture in fixtures:
        stored = fixture.get("opponent_preparation")
        if stored is None:
            continue
        expected = _derive_opponent_preparation(state, fixture)
        if stored != expected:
            raise ValueError("opponent preparation does not match prior season evidence")
    validate_club_timeline(state)


def _build_opponent_preparation(
    *, fixture_id: str, manager_team: str, opponent: str,
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    frozen_observations = [dict(item) for item in observations]
    counts: dict[str, int] = {}
    for item in frozen_observations:
        tactic = str(item["tactic"])
        counts[tactic] = counts.get(tactic, 0) + 1
    observed = len(frozen_observations)
    dominant = (
        sorted(counts, key=lambda tactic: (-counts[tactic], tactic))[0]
        if counts else None
    )
    dominant_count = counts.get(str(dominant), 0) if dominant is not None else 0
    share = dominant_count / observed if observed else 0.0
    if observed < OPPONENT_ADAPTATION_MIN_OBSERVATIONS:
        state_name = "insufficient_evidence"
        selected = "team_identity"
        reason = "minimum_prior_observations_not_met"
    elif dominant == "team_identity":
        state_name = "native_pattern"
        selected = "team_identity"
        reason = "dominant_manager_tactic_has_no_override"
    elif dominant not in OPPONENT_RESPONSE_POLICY:
        state_name = "unsupported_pattern"
        selected = "team_identity"
        reason = "dominant_tactic_has_no_response_policy"
    elif share < OPPONENT_ADAPTATION_MIN_SHARE:
        state_name = "ambiguous_pattern"
        selected = "team_identity"
        reason = "dominant_tactic_share_below_threshold"
    else:
        state_name = "adapted"
        selected = OPPONENT_RESPONSE_POLICY[dominant]
        reason = "bounded_response_policy_applied"
    return {
        "schema_version": 1,
        "available": True,
        "fixture_id": fixture_id,
        "manager_team": manager_team,
        "opponent": opponent,
        "state": state_name,
        "selected_tactic": selected,
        "reason": reason,
        "evidence": {
            "source": "earlier_completed_managed_fixtures",
            "observations": frozen_observations,
            "observation_count": observed,
            "tactic_counts": dict(sorted(counts.items())),
            "dominant_tactic": dominant,
            "dominant_count": dominant_count,
            "dominant_share": round(share, 6),
        },
        "thresholds": {
            "minimum_observations": OPPONENT_ADAPTATION_MIN_OBSERVATIONS,
            "minimum_dominant_share": OPPONENT_ADAPTATION_MIN_SHARE,
        },
        "policy": {
            "id": "bounded_opponent_response_v1",
            "observed_tactic": dominant,
            "response_tactic": selected,
        },
        "claim_boundary": (
            "deterministic in-game opponent preparation from earlier simulated "
            "season decisions; response mappings are design rules, not learned or "
            "empirically proven tactical counters"
        ),
    }


def _derive_opponent_preparation(
    state: Mapping[str, Any], target: Mapping[str, Any],
) -> dict[str, Any]:
    plan = SeasonPlan.from_payload(state["plan"])
    team = plan.manager_team
    fixture_id = str(target.get("fixture_id") or "")
    if team is None or team not in {target.get("home"), target.get("away")}:
        return {
            "schema_version": 1, "available": False,
            "reason": "fixture_has_no_managed_team", "fixture_id": fixture_id,
        }
    opponent = target["away"] if target["home"] == team else target["home"]
    target_order = (int(target["matchday"]), int(target["order"]))
    earlier = [
        fixture for fixture in state["fixtures"]
        if team in {fixture["home"], fixture["away"]}
        and (int(fixture["matchday"]), int(fixture["order"])) < target_order
        and fixture["state"] == "completed"
    ]
    observations = []
    for fixture in earlier:
        payload = fixture.get("manager_decision")
        if not isinstance(payload, Mapping):
            continue
        observations.append({
            "fixture_id": fixture["fixture_id"],
            "matchday": fixture["matchday"],
            "tactic": ManagerDecision.from_payload(payload).tactic,
        })
    return _build_opponent_preparation(
        fixture_id=fixture_id, manager_team=team, opponent=opponent,
        observations=observations,
    )


def archived_opponent_preparation(
    manager_team: str, entry: Mapping[str, Any],
    prior_entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Rebuild one archived preparation from earlier journal decisions."""
    if (
        not isinstance(entry, Mapping)
        or not isinstance(prior_entries, Sequence)
        or isinstance(prior_entries, (str, bytes))
    ):
        raise ValueError("invalid archived opponent preparation evidence")
    observations = [
        {
            "fixture_id": prior["fixture_id"],
            "matchday": prior["matchday"],
            "tactic": prior["tactic"],
        }
        for prior in prior_entries
        if isinstance(prior, Mapping) and prior.get("tactic") != "unrecorded"
    ]
    return _build_opponent_preparation(
        fixture_id=str(entry.get("fixture_id") or ""),
        manager_team=manager_team,
        opponent=str(entry.get("opponent") or ""),
        observations=observations,
    )


def opponent_preparation(
    state: Mapping[str, Any], fixture_id: str,
) -> dict[str, Any]:
    """Derive the opponent's bounded preparation without reading the current choice."""
    validate_season_state(state)
    target = next((
        fixture for fixture in state["fixtures"]
        if fixture.get("fixture_id") == fixture_id
    ), None)
    if target is None:
        raise ValueError("opponent preparation fixture identity mismatch")
    return _derive_opponent_preparation(state, target)


def season_standings(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_season_state(state)
    teams = SeasonPlan.from_payload(state["plan"]).teams
    rows = {
        team: {"team": team, "played": 0, "won": 0, "drawn": 0, "lost": 0,
               "goals_for": 0, "goals_against": 0, "goal_difference": 0, "points": 0}
        for team in teams
    }
    for fixture in state["fixtures"]:
        if fixture["state"] != "completed":
            continue
        home, away = fixture["home"], fixture["away"]
        home_goals, away_goals = fixture["score"]["home"], fixture["score"]["away"]
        for team, goals_for, goals_against in (
            (home, home_goals, away_goals), (away, away_goals, home_goals),
        ):
            row = rows[team]
            row["played"] += 1
            row["goals_for"] += goals_for
            row["goals_against"] += goals_against
            if goals_for > goals_against:
                row["won"] += 1
                row["points"] += 3
            elif goals_for == goals_against:
                row["drawn"] += 1
                row["points"] += 1
            else:
                row["lost"] += 1
    for row in rows.values():
        row["goal_difference"] = row["goals_for"] - row["goals_against"]
    ordered = sorted(
        rows.values(),
        key=lambda row: (-row["points"], -row["goal_difference"], -row["goals_for"], row["team"].casefold()),
    )
    for position, row in enumerate(ordered, start=1):
        row["position"] = position
    return ordered


def manager_season_profile(state: Mapping[str, Any]) -> dict[str, Any]:
    """Derive an auditable objective and decision journal for one season."""
    validate_season_state(state)
    plan = SeasonPlan.from_payload(state["plan"])
    team = plan.manager_team
    if team is None:
        return {
            "schema_version": 1, "available": False,
            "reason": "season_has_no_manager_team",
        }
    standings = season_standings(state)
    row = next(item for item in standings if item["team"] == team)
    managed = [
        fixture for fixture in state["fixtures"]
        if team in {fixture["home"], fixture["away"]}
    ]
    completed = [fixture for fixture in managed if fixture["state"] == "completed"]
    journal = []
    tactic_usage: dict[str, int] = {}
    rotation_usage: dict[str, int] = {}
    record = {"win": 0, "draw": 0, "loss": 0}
    timeline_by_fixture = {
        row["event"]["fixture_id"]: row
        for row in (state.get("club_timeline") or {}).get("events") or []
        if isinstance(row, Mapping) and isinstance(row.get("event"), Mapping)
    }
    for fixture in completed:
        score = fixture["score"]
        home_side = fixture["home"] == team
        goals_for = score["home"] if home_side else score["away"]
        goals_against = score["away"] if home_side else score["home"]
        outcome = (
            "win" if goals_for > goals_against
            else "draw" if goals_for == goals_against else "loss"
        )
        points = 3 if outcome == "win" else 1 if outcome == "draw" else 0
        record[outcome] += 1
        raw_decision = fixture.get("manager_decision")
        decision = (
            ManagerDecision.from_payload(raw_decision)
            if isinstance(raw_decision, Mapping) else None
        )
        tactic = decision.tactic if decision is not None else "unrecorded"
        rotation = decision.rotation if decision is not None else "unrecorded"
        tactic_usage[tactic] = tactic_usage.get(tactic, 0) + 1
        rotation_usage[rotation] = rotation_usage.get(rotation, 0) + 1
        instructions = (
            decision.in_match_plan.instructions
            if decision is not None and decision.in_match_plan is not None else ()
        )
        journal.append({
            "fixture_id": fixture["fixture_id"],
            "matchday": fixture["matchday"],
            "order": fixture["order"],
            "opponent": fixture["away"] if home_side else fixture["home"],
            "venue": "home" if home_side else "away",
            "score": dict(score),
            "goals_for": goals_for,
            "goals_against": goals_against,
            "outcome": outcome,
            "points_earned": points,
            "tactic": tactic,
            "rotation": rotation,
            "lineup_source": (
                decision.lineup.source
                if decision is not None and decision.lineup is not None
                else "team_level"
            ),
            "lineup": (
                decision.lineup.as_dict()
                if decision is not None and decision.lineup is not None
                else None
            ),
            "planned_instruction_count": len(instructions),
            "match_id": fixture.get("match_id"),
            "report": fixture.get("report"),
            "dashboard": fixture.get("dashboard"),
            "opponent_preparation": (
                dict(fixture["opponent_preparation"])
                if isinstance(fixture.get("opponent_preparation"), Mapping)
                else None
            ),
            "club_event": (
                dict(timeline_by_fixture[fixture["fixture_id"]])
                if fixture["fixture_id"] in timeline_by_fixture else None
            ),
            "player_promise_availability": copy.deepcopy(
                fixture.get("player_promise_availability")
            ),
            "world_state_transition": copy.deepcopy(
                fixture.get("world_state_transition")
            ),
            "manager_decision_advice": copy.deepcopy(
                fixture.get("manager_decision_advice")
            ),
            "manager_advice_adoption": copy.deepcopy(
                fixture.get("manager_advice_adoption")
            ),
            "manager_future_reviews": copy.deepcopy(
                fixture.get("manager_future_reviews") or []
            ),
        })

    objective = str(plan.manager_objective or "top_half")
    target_position = (
        1 if objective == "champion"
        else (len(plan.teams) + 1) // 2 if objective == "top_half"
        else None
    )
    target_points = plan.manager_points_target if objective == "points_target" else None
    current_met = (
        row["position"] <= target_position
        if target_position is not None else row["points"] >= int(target_points or 0)
    )
    remaining_matches = len(managed) - len(completed)
    maximum_points = row["points"] + 3 * remaining_matches
    all_complete = remaining_matches == 0
    if all_complete:
        objective_status = "achieved" if current_met else "missed"
    elif target_points is not None and maximum_points < target_points:
        objective_status = "unreachable"
    elif current_met:
        objective_status = "currently_meeting"
    else:
        objective_status = "in_progress"
    dominant_tactic = (
        sorted(tactic_usage, key=lambda key: (-tactic_usage[key], key))[0]
        if tactic_usage else None
    )
    dominant_rotation = (
        sorted(rotation_usage, key=lambda key: (-rotation_usage[key], key))[0]
        if rotation_usage else None
    )
    profile = {
        "schema_version": 1,
        "available": True,
        "team": team,
        "season_id": state["season_id"],
        "seed": state["seed"],
        "objective": {
            "kind": objective,
            "league_size": len(plan.teams),
            "target_position": target_position,
            "target_points": target_points,
            "current_position": row["position"],
            "current_points": row["points"],
            "remaining_matches": remaining_matches,
            "maximum_points": maximum_points,
            "current_snapshot_met": current_met,
            "status": objective_status,
        },
        "record": {**record, "matches": len(completed), "points": row["points"]},
        "decision_identity": {
            "dominant_tactic": dominant_tactic,
            "dominant_rotation": dominant_rotation,
            "tactic_usage": dict(sorted(tactic_usage.items())),
            "rotation_usage": dict(sorted(rotation_usage.items())),
        },
        "journal": journal,
        "journal_limit": len(managed),
        "claim_boundary": (
            "objective progress and journal are descriptive records from this "
            "simulated season; tactic frequency does not establish causal effectiveness"
        ),
    }
    profile["commitments"] = commitment_progress(state, profile["objective"])
    contract = state.get("player_role_promises")
    profile["player_role_promises"] = (
        player_promise_progress(
            contract, fixtures=state.get("fixtures") or [],
            final=state.get("state") == "complete",
        )
        if isinstance(contract, Mapping) else {
            "schema_version": 1, "available": False,
            "reason": "not_frozen", "entries": [],
        }
    )
    return profile


def manager_board_review(
    profile: Mapping[str, Any], *, league_size: int | None = None,
) -> dict[str, Any]:
    """Score one completed simulated season using an explicit gameplay policy."""
    if not isinstance(profile, Mapping) or profile.get("available") is not True:
        raise ValueError("board review requires a managed season profile")
    objective = profile.get("objective")
    record = profile.get("record")
    journal = profile.get("journal")
    commitments = profile.get("commitments")
    player_promises = profile.get("player_role_promises")
    if not isinstance(objective, Mapping) or objective.get("status") not in {
        "achieved", "missed",
    }:
        raise ValueError("board review requires a completed objective")
    resolved_size = league_size if league_size is not None else objective.get("league_size")
    position = objective.get("current_position")
    matches = record.get("matches") if isinstance(record, Mapping) else None
    record_values = (
        {key: record.get(key) for key in ("win", "draw", "loss", "matches", "points")}
        if isinstance(record, Mapping) else {}
    )
    record_is_valid = (
        len(record_values) == 5
        and all(
            not isinstance(value, bool) and isinstance(value, int) and value >= 0
            for value in record_values.values()
        )
        and record_values.get("win", 0) + record_values.get("draw", 0)
        + record_values.get("loss", 0) == record_values.get("matches")
        and 3 * record_values.get("win", 0) + record_values.get("draw", 0)
        == record_values.get("points")
    )
    if (
        isinstance(resolved_size, bool) or not isinstance(resolved_size, int)
        or resolved_size < MIN_SEASON_TEAMS
        or isinstance(position, bool) or not isinstance(position, int)
        or not 1 <= position <= resolved_size
        or not record_is_valid
        or isinstance(matches, bool) or not isinstance(matches, int) or matches < 1
        or not isinstance(journal, list) or len(journal) != matches
        or not isinstance(commitments, Mapping)
        or commitments.get("final") is not True
        or commitments.get("recorded_matches") != matches
    ):
        raise ValueError("board review profile is incomplete")
    journal_outcomes = {
        outcome: sum(
            isinstance(entry, Mapping) and entry.get("outcome") == outcome
            for entry in journal
        )
        for outcome in ("win", "draw", "loss")
    }
    if any(journal_outcomes[outcome] != record_values[outcome] for outcome in journal_outcomes):
        raise ValueError("board review journal and record disagree")

    objective_delta = 18 if objective["status"] == "achieved" else -20
    position_delta = round(
        8 * (1 - 2 * ((position - 1) / max(1, resolved_size - 1)))
    )
    recorded = sum(
        isinstance(entry, Mapping)
        and entry.get("tactic") != "unrecorded"
        and entry.get("rotation") != "unrecorded"
        for entry in journal
    )
    decision_coverage = recorded / matches
    stewardship_delta = 4 if decision_coverage == 1 else 2 if decision_coverage >= 0.8 else 0
    commitment_entries = commitments.get("entries")
    if (
        not isinstance(commitment_entries, list)
        or [entry.get("id") for entry in commitment_entries if isinstance(entry, Mapping)]
        != ["board_objective", "tactical_identity", "squad_stewardship"]
        or any(
            not isinstance(entry, Mapping)
            or entry.get("status") not in {"fulfilled", "missed", "excused"}
            for entry in commitment_entries
        )
    ):
        raise ValueError("board review commitment evidence is incomplete")
    policy_entries = commitment_entries[1:]
    commitment_policy = commitments.get("board_consequence")
    if commitment_policy != {
        "fulfilled_confidence": 3, "missed_confidence": -4,
        "fulfilled_reputation": 1, "missed_reputation": -1,
    }:
        raise ValueError("board review commitment consequence is invalid")
    commitment_confidence = sum(
        commitment_policy[f"{entry['status']}_confidence"]
        for entry in policy_entries
    )
    commitment_reputation = sum(
        commitment_policy[f"{entry['status']}_reputation"]
        for entry in policy_entries
    )
    player_promise_confidence = 0
    player_promise_statuses: dict[str, str] = {}
    player_promise_contract_id = None
    if isinstance(player_promises, Mapping) and player_promises.get("available") is not False:
        if player_promises.get("final") is not True:
            raise ValueError("board review player promises are unsettled")
        promise_entries = player_promises.get("entries")
        if not isinstance(promise_entries, list) or any(
            not isinstance(entry, Mapping)
            or entry.get("status") not in {"fulfilled", "missed", "excused"}
            for entry in promise_entries
        ):
            raise ValueError("board review player promise evidence is incomplete")
        promise_policy = player_promises.get("board_consequence")
        if promise_policy != {
            "fulfilled_confidence": 1, "missed_confidence": -2,
            "reputation_change": 0,
        }:
            raise ValueError("board review player promise consequence is invalid")
        player_promise_confidence = sum(
            promise_policy[f"{entry['status']}_confidence"]
            for entry in promise_entries
            if entry["status"] in {"fulfilled", "missed"}
        )
        player_promise_statuses = {
            str(entry["player_id"]): str(entry["status"])
            for entry in promise_entries
        }
        player_promise_contract_id = player_promises.get("contract_id")
    confidence_delta = max(
        -28, min(
            30, objective_delta + position_delta + stewardship_delta
            + commitment_confidence + player_promise_confidence,
        )
    )
    reputation_delta = max(
        -9, min(10, (6 if objective["status"] == "achieved" else -5)
                    + int(position_delta / 2) + commitment_reputation)
    )
    grade = (
        "outstanding" if confidence_delta >= 20
        else "positive" if confidence_delta >= 8
        else "mixed" if confidence_delta >= -7
        else "negative" if confidence_delta >= -18
        else "critical"
    )
    return {
        "schema_version": 1,
        "team": profile.get("team"),
        "season_id": profile.get("season_id"),
        "grade": grade,
        "confidence_delta": confidence_delta,
        "reputation_delta": reputation_delta,
        "components": {
            "objective": objective_delta,
            "league_finish": position_delta,
            "decision_stewardship": stewardship_delta,
            "season_commitments": commitment_confidence,
            "player_role_promises": player_promise_confidence,
        },
        "evidence": {
            "objective_status": objective["status"],
            "final_position": position,
            "league_size": resolved_size,
            "recorded_decisions": recorded,
            "managed_matches": matches,
            "decision_coverage": round(decision_coverage, 6),
            "commitment_contract_id": commitments.get("contract_id"),
            "commitment_statuses": {
                entry["id"]: entry["status"] for entry in commitment_entries
            },
            "player_promise_contract_id": player_promise_contract_id,
            "player_promise_statuses": player_promise_statuses,
        },
        "claim_boundary": (
            "deterministic gameplay policy over simulated season records; this is "
            "not a real-world manager rating or evidence that decisions caused results"
        ),
    }


def _employment_status(confidence: int) -> str:
    if confidence >= 75:
        return "secure"
    if confidence >= 45:
        return "stable"
    if confidence >= 20:
        return "under_review"
    return "dismissed"


def manager_career_profile(
    archives: Sequence[Mapping[str, Any]],
    *, current_season: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fold validated archives into one bounded, reproducible manager career view."""
    if isinstance(archives, (str, bytes)) or not isinstance(archives, Sequence):
        raise ValueError("manager career archives must be a sequence")
    confidence = BOARD_CONFIDENCE_START
    reputation = MANAGER_REPUTATION_START
    current_team: str | None = None
    completed_tenure = 0
    appointments = 0
    completed_seasons = 0
    objectives_achieved = 0
    totals = {"win": 0, "draw": 0, "loss": 0, "matches": 0, "points": 0}
    last_review: dict[str, Any] | None = None
    last_points = 0
    reviews = []
    for archive in archives:
        if not isinstance(archive, Mapping):
            raise ValueError("invalid manager career archive")
        plan = SeasonPlan.from_payload(archive.get("plan") or {})
        profile = archive.get("manager_profile")
        if plan.manager_team is None:
            continue
        review = manager_board_review(profile, league_size=len(plan.teams))
        stored_review = archive.get("board_review")
        if stored_review is not None and stored_review != review:
            raise ValueError("archived board review does not match season evidence")
        team = plan.manager_team
        if team != current_team:
            current_team = team
            confidence = BOARD_CONFIDENCE_START
            completed_tenure = 0
            appointments += 1
        confidence = max(0, min(100, confidence + review["confidence_delta"]))
        reputation = max(0, min(100, reputation + review["reputation_delta"]))
        completed_tenure += 1
        completed_seasons += 1
        objectives_achieved += review["evidence"]["objective_status"] == "achieved"
        record = profile["record"]
        for key in totals:
            totals[key] += int(record[key])
        last_points = int(record["points"])
        last_review = review
        reviews.append(review)

    inherited_status = _employment_status(confidence)
    appointment_state = "no_appointment"
    pending_board_review = False
    pending_review: dict[str, Any] | None = None
    projected_after_review: dict[str, Any] | None = None
    active_objective = None
    if current_season is not None:
        validate_season_state(current_season)
        current_plan = SeasonPlan.from_payload(current_season["plan"])
        active_objective = current_plan.manager_objective
        if current_plan.manager_team is not None:
            if current_plan.manager_team != current_team:
                current_team = current_plan.manager_team
                confidence = BOARD_CONFIDENCE_START
                completed_tenure = 0
                appointments += 1
                appointment_state = "new_appointment"
            else:
                appointment_state = "continuing"
            pending_board_review = current_season.get("state") == "complete"
            if pending_board_review:
                current_profile = manager_season_profile(current_season)
                pending_review = manager_board_review(
                    current_profile, league_size=len(current_plan.teams),
                )
                projected_confidence = max(
                    0, min(100, confidence + pending_review["confidence_delta"])
                )
                projected_reputation = max(
                    0, min(100, reputation + pending_review["reputation_delta"])
                )
                projected_after_review = {
                    "board_confidence": projected_confidence,
                    "reputation": projected_reputation,
                    "employment_status": _employment_status(projected_confidence),
                }

    status = _employment_status(confidence) if current_team is not None else "unappointed"
    contract_status = (
        projected_after_review["employment_status"]
        if projected_after_review is not None
        else status if current_season is not None else inherited_status
    )
    contract_points = (
        int(manager_season_profile(current_season)["record"]["points"])
        if pending_board_review and current_season is not None else last_points
    )
    same_club_allowed = contract_status != "dismissed"
    points_growth_required = 3 if contract_status == "under_review" else 0
    return {
        "schema_version": 1,
        "available": current_team is not None or completed_seasons > 0,
        "current_team": current_team,
        "appointment_state": appointment_state,
        "employment_status": status,
        "board_confidence": confidence,
        "reputation": reputation,
        "completed_seasons": completed_seasons,
        "completed_tenure_seasons": completed_tenure,
        "appointments": appointments,
        "objectives_achieved": objectives_achieved,
        "career_record": totals,
        "active_objective": active_objective,
        "pending_board_review": pending_board_review,
        "pending_review": pending_review,
        "projected_after_review": projected_after_review,
        "last_review": last_review,
        "reviews": reviews,
        "next_contract": {
            "same_club_allowed": same_club_allowed,
            "same_club_points_growth_required": points_growth_required,
            "baseline_points": (
                contract_points
                if last_review is not None or pending_board_review else None
            ),
            "reason": (
                "dismissed_after_retained_season_reviews" if not same_club_allowed
                else "under_review_requires_ambitious_objective"
                if points_growth_required else "current_season_in_progress"
                if current_season is not None and not pending_board_review
                else "no_additional_constraint"
            ),
        },
        "history_scope": {
            "retained_seasons": len(archives),
            "rolling_window": True,
        },
        "claim_boundary": (
            "board confidence and reputation are transparent game-state scores folded "
            "from retained simulated seasons; they are not real-world evaluations"
        ),
    }


def validate_manager_career_transition(
    plan: SeasonPlan, career: Mapping[str, Any],
) -> None:
    """Enforce the previous board verdict before opening a new managed season."""
    team = plan.manager_team
    previous_team = career.get("current_team")
    if team is None or previous_team is None or team != previous_team:
        return
    contract = career.get("next_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("manager career contract is invalid")
    if contract.get("same_club_allowed") is not True:
        raise ValueError("manager was dismissed and must change club")
    growth = contract.get("same_club_points_growth_required", 0)
    if isinstance(growth, bool) or not isinstance(growth, int) or growth < 0:
        raise ValueError("manager career points requirement is invalid")
    if growth and plan.manager_objective == "points_target":
        maximum = (len(plan.teams) - 1) * plan.legs * 3
        baseline = int(contract.get("baseline_points") or 0)
        required = min(maximum, baseline + growth)
        if int(plan.manager_points_target or 0) < required:
            raise ValueError(
                f"under-review manager points target must be at least {required}"
            )


def archive_completed_season(state: Mapping[str, Any]) -> dict[str, Any]:
    """Create a bounded immutable summary before a completed season is replaced."""
    validate_season_state(state)
    if any(fixture["state"] != "completed" for fixture in state["fixtures"]):
        raise ValueError("only a completed season can be archived")
    profile = manager_season_profile(state)
    plan = SeasonPlan.from_payload(state["plan"])
    return {
        "schema_version": 1,
        "season_id": state["season_id"],
        "seed": state["seed"],
        "completed_at": state.get("updated_at"),
        "plan": dict(state["plan"]),
        "final_standings": season_standings(state),
        "manager_profile": profile,
        "board_review": (
            manager_board_review(profile, league_size=len(plan.teams))
            if profile.get("available") else None
        ),
        "recruitment_transaction": (
            dict(state["recruitment_transaction"])
            if state.get("recruitment_transaction") is not None else None
        ),
        "club_strategies": (
            dict(state["club_strategies"])
            if state.get("club_strategies") is not None else None
        ),
        "sporting_brief": (
            dict(state["sporting_brief"])
            if state.get("sporting_brief") is not None else None
        ),
        "sporting_evaluation": (
            dict(state["sporting_evaluation"])
            if state.get("sporting_evaluation") is not None else None
        ),
        "club_timeline": copy.deepcopy(state.get("club_timeline")),
        "season_commitments": copy.deepcopy(state.get("season_commitments")),
        "player_role_promises": copy.deepcopy(state.get("player_role_promises")),
        "player_promise_outcomes": None,
        "fixture_count": len(state["fixtures"]),
        "claim_boundary": "immutable descriptive summary of one simulated season",
    }


def next_matchday(state: Mapping[str, Any]) -> int | None:
    validate_season_state(state)
    pending = [
        fixture["matchday"] for fixture in state["fixtures"]
        if fixture["state"] != "completed"
    ]
    return min(pending) if pending else None


def matchday_command_center(state: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the manager's single authoritative matchday journey.

    This is a read model, not additional persisted season state. Rebuilding it from
    the frozen schedule and completed results prevents the UI from inventing a
    second, potentially contradictory notion of match progress.
    """
    validate_season_state(state)
    plan = SeasonPlan.from_payload(state["plan"])
    manager_team = plan.manager_team
    current_matchday = next_matchday(state)
    if manager_team is None:
        return {
            "schema_version": 1,
            "enabled": False,
            "phase": "spectator",
            "manager_team": None,
            "current_matchday": current_matchday,
            "primary_action": {"id": "advance_matchday", "enabled": current_matchday is not None},
            "claim_boundary": "derived from the persisted simulated season only",
        }

    fixtures = list(state["fixtures"])
    managed = [
        fixture for fixture in fixtures
        if manager_team in {fixture["home"], fixture["away"]}
    ]
    pending = [fixture for fixture in managed if fixture["state"] != "completed"]
    next_fixture = pending[0] if pending else None
    current_fixture = next((
        fixture for fixture in managed
        if fixture["matchday"] == current_matchday
    ), None)
    completed = [fixture for fixture in managed if fixture["state"] == "completed"]
    last_fixture = completed[-1] if completed else None
    standings = {row["team"]: row for row in season_standings(state)}
    current_day_fixtures = [
        fixture for fixture in fixtures
        if fixture["matchday"] == current_matchday
    ]
    is_resume_matchday = any(
        fixture["state"] != "scheduled" for fixture in current_day_fixtures
    )

    def fixture_context(fixture: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if fixture is None:
            return None
        home = fixture["home"]
        opponent = fixture["away"] if home == manager_team else home
        manager_row = standings[manager_team]
        opponent_row = standings[opponent]
        return {
            "fixture_id": fixture["fixture_id"],
            "matchday": fixture["matchday"],
            "home": home,
            "away": fixture["away"],
            "venue": "home" if home == manager_team else "away",
            "opponent": opponent,
            "manager_position": manager_row["position"],
            "manager_points": manager_row["points"],
            "opponent_position": opponent_row["position"],
            "opponent_points": opponent_row["points"],
            "state": fixture["state"],
            "attempts": int(fixture.get("attempts", 0)),
            "decision": fixture.get("manager_decision"),
            "opponent_preparation": (
                fixture.get("opponent_preparation")
                or _derive_opponent_preparation(state, fixture)
            ),
        }

    def result_context(fixture: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if fixture is None:
            return None
        score = fixture["score"]
        goals_for = score["home"] if fixture["home"] == manager_team else score["away"]
        goals_against = score["away"] if fixture["home"] == manager_team else score["home"]
        outcome = "win" if goals_for > goals_against else "draw" if goals_for == goals_against else "loss"
        return {
            **fixture_context(fixture),
            "score": dict(score),
            "goals_for": goals_for,
            "goals_against": goals_against,
            "outcome": outcome,
            "points_earned": 3 if outcome == "win" else 1 if outcome == "draw" else 0,
            "match_id": fixture.get("match_id"),
            "report": fixture.get("report"),
            "dashboard": fixture.get("dashboard"),
        }

    recent_results = [result_context(fixture) for fixture in completed[-5:]]
    form = "".join({"win": "W", "draw": "D", "loss": "L"}[item["outcome"]] for item in recent_results)
    decision_required = (
        current_fixture is not None
        and current_fixture["state"] != "completed"
        and current_fixture.get("manager_decision") is None
    )
    current_situation = (
        derive_club_situation(state, current_fixture["fixture_id"])
        if current_fixture is not None else None
    )
    situation_resolved = bool(
        current_situation is not None and any(
            row.get("event", {}).get("fixture_id") == current_fixture["fixture_id"]
            for row in (state.get("club_timeline") or {}).get("events") or []
            if isinstance(row, Mapping)
        )
    )
    if current_matchday is None:
        phase = "season_complete"
        action = {"id": "review_season", "enabled": True}
    elif decision_required:
        phase = "decision_required"
        action = {"id": "submit_decision", "enabled": True}
    elif is_resume_matchday:
        phase = "ready_to_resume"
        action = {"id": "resume_matchday", "enabled": True}
    else:
        phase = "ready_to_advance"
        action = {"id": "advance_matchday", "enabled": True}

    return {
        "schema_version": 1,
        "enabled": True,
        "phase": phase,
        "manager_team": manager_team,
        "current_matchday": current_matchday,
        "is_bye_matchday": current_matchday is not None and current_fixture is None,
        "is_resume_matchday": is_resume_matchday,
        "decision_required": decision_required,
        "club_situation": (
            current_situation if current_situation is not None and not situation_resolved
            else None
        ),
        "current_fixture": fixture_context(current_fixture),
        "next_fixture": fixture_context(next_fixture),
        "last_result": result_context(last_fixture),
        "recent_form": form or "-",
        "recent_results": recent_results,
        "club_resources": (
            plan.manager_resources.as_dict()
            if plan.manager_resources is not None else None
        ),
        "primary_action": action,
        "journey": [
            {"id": "briefing", "status": "complete" if next_fixture else "not_applicable"},
            {
                "id": "club_situation",
                "status": (
                    "action_required" if current_situation is not None and not situation_resolved
                    else "complete" if situation_resolved else "not_applicable"
                ),
            },
            {
                "id": "decision",
                "status": (
                    "action_required" if decision_required
                    else "complete" if current_fixture is not None
                    else "not_applicable"
                ),
            },
            {
                "id": "matchday",
                "status": "blocked" if decision_required else "ready" if current_matchday is not None else "complete",
            },
            {"id": "debrief", "status": "available" if last_fixture else "pending"},
        ],
        "claim_boundary": (
            "deterministic read model derived from the persisted simulated season; "
            "positions, form and outcomes are not real-world estimates"
        ),
    }

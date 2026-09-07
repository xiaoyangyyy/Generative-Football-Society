"""Load and replay-validate persisted product workspace sessions."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from src.data_engine.roster_loader import load_roster_json, roster_path_for_team
from src.product.club_finance import (
    build_season_finance_settlement,
    evidence_identity,
    recruitment_allowance,
    validate_finance_registry,
)
from src.product.club_strategy import (
    build_season_club_strategies,
    resolve_fixture_club_strategy,
)
from src.product.league_ecosystem import (
    ai_club_recruitment_decision,
    validate_league_ecosystem,
)
from src.product.player_development import (
    build_development_transaction,
    validate_development_registry,
    verify_participation_report_files,
)
from src.product.player_promises import (
    PlayerPromisePlan,
    build_player_promise_contract,
)
from src.product.scouting_outcomes import (
    build_scouting_outcome,
    resolve_signing_observation,
    validate_scouting_outcome_registry,
)
from src.product.season import SeasonPlan, validate_season_state
from src.product.sporting_reviews import (
    build_sporting_review,
    validate_sporting_review_registry,
)
from src.simulation.player_lifecycle import (
    RetentionPlan,
    build_lifecycle_transaction,
    retirement_age_for_player,
    validate_lifecycle_registry,
)
from src.simulation.player_market import (
    FreeAgentPlan,
    ai_free_agent_decision,
    execute_free_agent_signing,
    market_player_quality,
    market_preview_from_window,
    open_market_window,
    pool_entry_from_player,
    validate_market_registry,
)
from src.simulation.scouting import (
    scouting_identity,
    scouting_observation,
    validate_scouting_registry,
)
from src.simulation.squad_registry import (
    replay_squad_registry_through,
    validate_squad_registry,
)


def load_workspace_session(
    *,
    root: Path,
    session_path: Path,
    season_history_view: Callable[[Mapping[str, Any]], list[dict[str, Any]]],
    sporting_brief_for_transition: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    session = json.loads(session_path.read_text(encoding="utf-8"))
    if not isinstance(session, dict) or session.get("schema_version") != 1:
        raise ValueError("invalid product session")
    validate_development_registry(session.get("player_development"))
    validate_lifecycle_registry(session.get("player_lifecycle"))
    validate_market_registry(session.get("player_market"))
    validate_scouting_registry(session.get("scouting_registry"))
    validate_scouting_outcome_registry(session.get("scouting_outcomes"))
    validate_sporting_review_registry(session.get("sporting_reviews"))
    validate_squad_registry(
        root,
        session.get("squad_registry"),
        development_registry=session.get("player_development"),
        lifecycle_registry=session.get("player_lifecycle"),
        market_registry=session.get("player_market"),
    )
    validate_finance_registry(session.get("finance_registry"))
    validate_league_ecosystem(session.get("league_ecosystem"))
    season = session.get("season")
    if isinstance(season, Mapping):
        validate_season_state(season)
        plan = SeasonPlan.from_payload(season["plan"])
        if plan.manager_recruitment is not None:
            club = (
                (session.get("squad_registry") or {})
                .get("clubs", {})
                .get(str(plan.manager_team), {})
            )
            matches = (
                [
                    transaction
                    for transaction in club.get("transactions", [])
                    if transaction.get("season_id") == season.get("season_id")
                ]
                if isinstance(club, Mapping)
                else []
            )
            if (
                len(matches) != 1
                or season.get("recruitment_transaction") != matches[0]
            ):
                raise ValueError(
                    "active season recruitment registry identity mismatch"
                )
    season_history_view(session)
    archives = {
        str(archive.get("season_id") or ""): archive
        for archive in session.get("season_history") or []
        if isinstance(archive, Mapping)
    }
    finance_clubs = (session.get("finance_registry") or {}).get("clubs") or {}
    squad_clubs = (session.get("squad_registry") or {}).get("clubs") or {}
    development_clubs = (session.get("player_development") or {}).get("clubs") or {}
    lifecycle_clubs = (session.get("player_lifecycle") or {}).get("clubs") or {}
    for scouting_report in (session.get("scouting_registry") or {}).get(
        "reports"
    ) or []:
        target_id = str(scouting_report.get("target_season_id") or "")
        target_index = int(target_id.split("-")[-1])
        prior_market = {
            "schema_version": 1,
            "transitions": [
                copy.deepcopy(item)
                for item in (session.get("player_market") or {}).get(
                    "transitions", []
                )
                if int(str(item["target_season_id"]).split("-")[-1]) < target_index
            ],
        }
        window = open_market_window(
            prior_market,
            target_season_id=target_id,
        )
        team = str(scouting_report.get("team") or "")
        preview = market_preview_from_window(window, team=team)
        player_id = str(scouting_report.get("player_id") or "")
        entry = (window.get("available") or {}).get(player_id)
        if (
            entry is None
            or scouting_report.get("market_id") != preview["market_id"]
            or scouting_report.get("candidate_entry_identity")
            != scouting_identity(entry)
            or scouting_report.get("observation")
            != scouting_observation(
                market_id=preview["market_id"],
                team=team,
                player_id=player_id,
                true_quality=market_player_quality(entry["player"]),
                level="scouted",
            )
        ):
            raise ValueError("scouting report source replay mismatch")
    for lifecycle_team, club in lifecycle_clubs.items():
        base_roster = load_roster_json(
            roster_path_for_team(
                str(root),
                lifecycle_team,
            )
        )
        if base_roster is None:
            raise ValueError("lifecycle source roster is unavailable")
        transactions = club.get("transactions") or []
        for transaction in transactions:
            source_id = str(transaction.get("source_season_id") or "")
            target_id = str(transaction.get("target_season_id") or "")
            if source_id not in archives:
                continue
            target_index = int(target_id.split("-")[-1])
            prior_lifecycle = {
                "schema_version": 1,
                "clubs": {
                    lifecycle_team: {
                        "transactions": [
                            copy.deepcopy(item)
                            for item in transactions
                            if int(str(item["target_season_id"]).split("-")[-1])
                            < target_index
                        ]
                    }
                },
            }
            development_transactions = (
                development_clubs.get(lifecycle_team) or {}
            ).get("transactions", [])
            prior_development = {
                "schema_version": 1,
                "clubs": {
                    lifecycle_team: {
                        "transactions": [
                            copy.deepcopy(item)
                            for item in development_transactions
                            if int(str(item["target_season_id"]).split("-")[-1])
                            < target_index
                        ]
                    }
                },
            }
            source_roster = replay_squad_registry_through(
                base_roster,
                team=lifecycle_team,
                registry=session.get("squad_registry"),
                season_id=source_id,
                development_registry=session.get("player_development"),
                lifecycle_registry=session.get("player_lifecycle"),
                market_registry=session.get("player_market"),
            )
            target_roster = replay_squad_registry_through(
                base_roster,
                team=lifecycle_team,
                registry=session.get("squad_registry"),
                season_id=target_id,
                development_registry=prior_development,
                lifecycle_registry=prior_lifecycle,
                market_registry=session.get("player_market"),
            )
            raw_plan = transaction.get("plan")
            retention = (
                RetentionPlan.from_payload(raw_plan)
                if raw_plan is not None
                else None
            )
            expected, _result = build_lifecycle_transaction(
                source_roster,
                target_roster,
                team=lifecycle_team,
                target_season_id=target_id,
                plan=retention,
                control=str(transaction.get("control") or ""),
            )
            if transaction != expected:
                raise ValueError("player lifecycle source replay mismatch")
    for development_team, club in development_clubs.items():
        base_roster = load_roster_json(
            roster_path_for_team(
                str(root),
                development_team,
            )
        )
        if base_roster is None:
            raise ValueError("development source roster is unavailable")
        transactions = club.get("transactions") or []
        for transaction in transactions:
            source_id = str(transaction.get("source_season_id") or "")
            target_id = str(transaction.get("target_season_id") or "")
            source_archive = archives.get(source_id)
            if source_archive is None:
                continue
            evidence_by_team = source_archive.get(
                "player_development_evidence",
            )
            if (
                not isinstance(evidence_by_team, Mapping)
                or development_team not in evidence_by_team
            ):
                raise ValueError("development archive evidence is unavailable")
            participation = evidence_by_team[development_team]
            verify_participation_report_files(root, participation)
            source_roster = replay_squad_registry_through(
                base_roster,
                team=development_team,
                registry=session.get("squad_registry"),
                season_id=source_id,
                development_registry=session.get("player_development"),
                lifecycle_registry=session.get("player_lifecycle"),
                market_registry=session.get("player_market"),
            )
            target_index = int(target_id.split("-")[-1])
            prior_development = {
                "schema_version": 1,
                "clubs": {
                    development_team: {
                        "transactions": [
                            copy.deepcopy(item)
                            for item in transactions
                            if int(str(item["target_season_id"]).split("-")[-1])
                            < target_index
                        ],
                    },
                },
            }
            target_roster = replay_squad_registry_through(
                base_roster,
                team=development_team,
                registry=session.get("squad_registry"),
                season_id=target_id,
                development_registry=prior_development,
                lifecycle_registry=session.get("player_lifecycle"),
                market_registry=session.get("player_market"),
            )
            expected, _result = build_development_transaction(
                source_archive,
                source_roster,
                target_roster,
                participation,
                team=development_team,
                target_season_id=target_id,
            )
            if transaction != expected:
                raise ValueError("player development source replay mismatch")
    for team, club in finance_clubs.items():
        for entry in club.get("entries") or []:
            evidence = entry.get("evidence") or {}
            entry_season_id = str(entry.get("season_id") or "")
            if entry.get("type") == "recruitment_charge":
                squad_entry = squad_clubs.get(team) or {}
                matches = [
                    transaction
                    for transaction in squad_entry.get("transactions") or []
                    if transaction.get("season_id") == entry_season_id
                ]
                if (
                    len(matches) != 1
                    or evidence.get("spent") != matches[0].get("spent")
                    or evidence.get("squad_transaction_identity")
                    != evidence_identity(matches[0])
                ):
                    raise ValueError("recruitment finance evidence mismatch")
            elif entry_season_id in archives:
                base_roster = load_roster_json(
                    roster_path_for_team(
                        str(root),
                        team,
                    )
                )
                historical_roster = (
                    replay_squad_registry_through(
                        base_roster,
                        team=team,
                        registry=session.get("squad_registry"),
                        season_id=entry_season_id,
                        development_registry=session.get("player_development"),
                        lifecycle_registry=session.get("player_lifecycle"),
                        market_registry=session.get("player_market"),
                    )
                    if base_roster is not None
                    else None
                )
                expected = build_season_finance_settlement(
                    archives[entry_season_id],
                    historical_roster,
                    team=team,
                )
                if evidence != expected:
                    raise ValueError("season finance evidence mismatch")
    season_sources = dict(archives)
    if isinstance(season, Mapping):
        season_sources[str(season.get("season_id") or "")] = season
    for promise_season_id, promise_state in season_sources.items():
        contract = promise_state.get("player_role_promises")
        if not isinstance(contract, Mapping):
            continue
        promise_plan = SeasonPlan.from_payload(promise_state.get("plan") or {})
        team = str(promise_plan.manager_team or "")
        base_roster = (
            load_roster_json(
                roster_path_for_team(
                    str(root),
                    team,
                )
            )
            if team
            else None
        )
        historical_roster = (
            replay_squad_registry_through(
                base_roster,
                team=team,
                registry=session.get("squad_registry"),
                season_id=promise_season_id,
                development_registry=session.get("player_development"),
                lifecycle_registry=session.get("player_lifecycle"),
                market_registry=session.get("player_market"),
            )
            if base_roster is not None
            else None
        )
        frozen_plan = (
            PlayerPromisePlan.from_payload(
                {
                    "schema_version": 1,
                    "promises": [
                        {
                            "player_id": row.get("player_id"),
                            "role": row.get("promised_role"),
                        }
                        for row in contract.get("players") or []
                    ],
                }
            )
            if contract.get("control") == "manager"
            else None
        )
        expected_contract = build_player_promise_contract(
            season_id=promise_season_id,
            team=team,
            total_managed_matches=(len(promise_plan.teams) - 1) * promise_plan.legs,
            roster=historical_roster,
            plan=frozen_plan,
            control=str(contract.get("control") or ""),
        )
        if contract != expected_contract:
            raise ValueError("player promise contract source replay mismatch")
    for target_id, target_state in season_sources.items():
        target_plan = SeasonPlan.from_payload(target_state.get("plan") or {})
        if target_plan.manager_sporting_directive is None:
            continue
        target_index = int(str(target_id).split("-")[-1])
        source_id = f"season-{target_index - 1:04d}"
        source_archive = archives.get(source_id)
        if source_archive is None:
            continue
        expected_brief = sporting_brief_for_transition(
            session,
            source_archive,
            team=str(target_plan.manager_team),
            target_season_id=target_id,
        )
        if target_state.get("sporting_brief") != expected_brief:
            raise ValueError("sporting brief source replay mismatch")
    for lifecycle_team, club in lifecycle_clubs.items():
        for transaction in club.get("transactions") or []:
            target_state = season_sources.get(
                str(transaction.get("target_season_id") or "")
            )
            if target_state is None:
                continue
            target_plan = SeasonPlan.from_payload(target_state["plan"])
            manager_controlled = target_plan.manager_team == lifecycle_team
            expected_plan = (
                target_plan.manager_retention.as_dict()
                if manager_controlled and target_plan.manager_retention is not None
                else None
            )
            if (
                transaction.get("control")
                != ("manager" if manager_controlled else "ai")
                or transaction.get("plan") != expected_plan
            ):
                raise ValueError("player lifecycle season plan mismatch")
    for market_transition in (session.get("player_market") or {}).get(
        "transitions"
    ) or []:
        source_id = str(market_transition.get("source_season_id") or "")
        target_id = str(market_transition.get("target_season_id") or "")
        target_state = season_sources.get(str(target_id))
        if target_state is None:
            continue
        target_plan = SeasonPlan.from_payload(target_state["plan"])
        manager_signings = [
            signing
            for signing in market_transition.get("signings") or []
            if signing.get("control") == "manager"
        ]
        expected_manager_plan = (
            target_plan.manager_free_agent.as_dict()
            if target_plan.manager_free_agent is not None
            else None
        )
        if len(manager_signings) != (1 if expected_manager_plan else 0) or (
            expected_manager_plan is not None
            and (
                manager_signings[0].get("team") != target_plan.manager_team
                or manager_signings[0].get("plan") != expected_manager_plan
            )
        ):
            raise ValueError("global market season plan mismatch")
        source_archive = archives.get(source_id)
        if source_archive is None:
            continue
        source_plan = SeasonPlan.from_payload(source_archive["plan"])
        continuing = sorted(set(source_plan.teams) & set(target_plan.teams))
        target_index = int(target_id.split("-")[-1])
        prior_market = {
            "schema_version": 1,
            "transitions": [
                copy.deepcopy(item)
                for item in (session.get("player_market") or {}).get(
                    "transitions", []
                )
                if int(str(item["target_season_id"]).split("-")[-1]) < target_index
            ],
        }
        replay_window = open_market_window(
            prior_market,
            target_season_id=target_id,
        )
        replay_rosters = {}
        for market_team in continuing:
            base_roster = load_roster_json(
                roster_path_for_team(
                    str(root),
                    market_team,
                )
            )
            if base_roster is None:
                continue
            prior_development = {
                "schema_version": 1,
                "clubs": {
                    market_team: {
                        "transactions": [
                            copy.deepcopy(item)
                            for item in (
                                (development_clubs.get(market_team) or {}).get(
                                    "transactions", []
                                )
                            )
                            if int(str(item["target_season_id"]).split("-")[-1])
                            < target_index
                        ]
                    }
                },
            }
            prior_lifecycle = {
                "schema_version": 1,
                "clubs": {
                    market_team: {
                        "transactions": [
                            copy.deepcopy(item)
                            for item in (
                                (lifecycle_clubs.get(market_team) or {}).get(
                                    "transactions", []
                                )
                            )
                            if int(str(item["target_season_id"]).split("-")[-1])
                            < target_index
                        ]
                    }
                },
            }
            replay_rosters[market_team] = replay_squad_registry_through(
                base_roster,
                team=market_team,
                registry=session.get("squad_registry"),
                season_id=target_id,
                development_registry=prior_development,
                lifecycle_registry=prior_lifecycle,
                market_registry=prior_market,
            )
        expected_signings = []
        if target_plan.manager_free_agent is not None:
            manager_team = str(target_plan.manager_team or "")
            replay_window, replay_rosters[manager_team], signing = (
                execute_free_agent_signing(
                    replay_window,
                    replay_rosters[manager_team],
                    team=manager_team,
                    plan=target_plan.manager_free_agent,
                    control="manager",
                )
            )
            expected_signings.append(signing)
        expected_ai_decisions = []
        for market_team in (
            team
            for team in continuing
            if team != target_plan.manager_team and team in replay_rosters
        ):
            decision = ai_free_agent_decision(
                replay_window,
                replay_rosters[market_team],
                team=market_team,
            )
            expected_ai_decisions.append(decision)
            if decision["plan"] is not None:
                replay_window, replay_rosters[market_team], signing = (
                    execute_free_agent_signing(
                        replay_window,
                        replay_rosters[market_team],
                        team=market_team,
                        plan=FreeAgentPlan.from_payload(decision["plan"]),
                        control="ai",
                    )
                )
                expected_signings.append(signing)
        if (
            market_transition.get("signings") != expected_signings
            or market_transition.get("ai_decisions") != expected_ai_decisions
        ):
            raise ValueError("global market source replay mismatch")
        expected_entries = [
            pool_entry_from_player(
                signing["outgoing"],
                origin_team=str(signing["team"]),
                entered_season_id=target_id,
                retirement_age=retirement_age_for_player(
                    str(signing["team"]),
                    str(signing["outgoing"]["player_id"]),
                ),
                reason="squad_replacement",
            )
            for signing in expected_signings
        ]
        for lifecycle_team in continuing:
            matches = [
                item
                for item in (
                    (lifecycle_clubs.get(lifecycle_team) or {}).get(
                        "transactions", []
                    )
                )
                if item.get("target_season_id") == target_id
            ]
            if len(matches) != 1:
                continue
            for exit_row in matches[0]["exits"]:
                if exit_row["reason"] == "contract_released":
                    expected_entries.append(
                        pool_entry_from_player(
                            exit_row["player"],
                            origin_team=lifecycle_team,
                            entered_season_id=target_id,
                            retirement_age=int(exit_row["retirement_age"]),
                            reason="contract_released",
                        )
                    )
        if market_transition.get("entries") != expected_entries:
            raise ValueError("global market entry source mismatch")
    for outcome in (session.get("scouting_outcomes") or {}).get("outcomes") or []:
        season_id = str(outcome.get("season_id") or "")
        source_archive = archives.get(season_id)
        if source_archive is None:
            continue
        transitions = [
            item
            for item in (session.get("player_market") or {}).get("transitions", [])
            if item.get("target_season_id") == season_id
        ]
        if len(transitions) != 1:
            raise ValueError("scouting outcome market source is unavailable")
        market_transition = transitions[0]
        signings = [
            item
            for item in market_transition.get("signings") or []
            if evidence_identity(item) == outcome.get("signing_identity")
        ]
        if len(signings) != 1:
            raise ValueError("scouting outcome signing source mismatch")
        signing = signings[0]
        team = str(signing.get("team") or "")
        participation = (
            source_archive.get("player_development_evidence") or {}
        ).get(team)
        finance_entries = (finance_clubs.get(team) or {}).get("entries") or []
        settlements = [
            entry.get("evidence")
            for entry in finance_entries
            if entry.get("type") == "season_settlement"
            and entry.get("season_id") == season_id
        ]
        if not isinstance(participation, Mapping) or len(settlements) != 1:
            raise ValueError("scouting outcome season evidence is unavailable")
        observation, observation_source = resolve_signing_observation(
            signing,
            market_transition,
            session.get("scouting_registry"),
        )
        expected = build_scouting_outcome(
            signing,
            observation=observation,
            observation_source=observation_source,
            archive=source_archive,
            participation=participation,
            settlement=settlements[0],
        )
        if outcome != expected:
            raise ValueError("scouting outcome source replay mismatch")
    for review in (session.get("sporting_reviews") or {}).get("reviews") or []:
        season_id = str(review.get("season_id") or "")
        source_archive = archives.get(season_id)
        if source_archive is None:
            continue
        source_plan = SeasonPlan.from_payload(source_archive["plan"])
        team = str(source_plan.manager_team or "")
        participation = (
            source_archive.get("player_development_evidence") or {}
        ).get(team)
        settlements = [
            entry.get("evidence")
            for entry in (finance_clubs.get(team) or {}).get("entries") or []
            if entry.get("type") == "season_settlement"
            and entry.get("season_id") == season_id
        ]
        base_roster = load_roster_json(
            roster_path_for_team(
                str(root),
                team,
            )
        )
        completed_roster = (
            replay_squad_registry_through(
                base_roster,
                team=team,
                registry=session.get("squad_registry"),
                season_id=season_id,
                development_registry=session.get("player_development"),
                lifecycle_registry=session.get("player_lifecycle"),
                market_registry=session.get("player_market"),
            )
            if base_roster is not None
            else None
        )
        signings = [
            signing
            for transition in (session.get("player_market") or {}).get(
                "transitions", []
            )
            if transition.get("target_season_id") == season_id
            for signing in transition.get("signings") or []
            if signing.get("team") == team
        ]
        if (
            not isinstance(participation, Mapping)
            or len(settlements) != 1
            or completed_roster is None
        ):
            raise ValueError("sporting review source evidence is unavailable")
        expected_review = build_sporting_review(
            source_archive,
            completed_roster,
            participation=participation,
            settlement=settlements[0],
            free_agent_signings=signings,
        )
        if review != expected_review:
            raise ValueError("sporting review source replay mismatch")
    ecosystem = session.get("league_ecosystem") or {}
    for event in ecosystem.get("transitions") or []:
        source_id = str(event.get("from_season_id") or "")
        target_id = str(event.get("to_season_id") or "")
        source_archive = archives.get(source_id)
        target_state = season_sources.get(target_id)
        if source_archive is None or target_state is None:
            continue
        source_plan = SeasonPlan.from_payload(source_archive["plan"])
        target_plan = SeasonPlan.from_payload(target_state["plan"])
        participating = sorted(set(source_plan.teams) & set(target_plan.teams))
        controlled_team = (
            target_plan.manager_team
            if target_plan.manager_team in participating
            else None
        )
        expected_clubs = []
        for ai_team in (team for team in participating if team != controlled_team):
            finance_entries = (finance_clubs.get(ai_team) or {}).get(
                "entries"
            ) or []
            settlement_matches = [
                entry
                for entry in finance_entries
                if entry.get("type") == "season_settlement"
                and entry.get("season_id") == source_id
            ]
            if len(settlement_matches) != 1:
                raise ValueError("league ecosystem settlement evidence mismatch")
            settlement_entry = settlement_matches[0]
            base_roster = load_roster_json(
                roster_path_for_team(
                    str(root),
                    ai_team,
                )
            )
            historical_roster = (
                replay_squad_registry_through(
                    base_roster,
                    team=ai_team,
                    registry=session.get("squad_registry"),
                    season_id=source_id,
                    development_registry=session.get("player_development"),
                    lifecycle_registry=session.get("player_lifecycle"),
                    market_registry=session.get("player_market"),
                )
                if base_roster is not None
                else None
            )
            decision = ai_club_recruitment_decision(
                source_archive,
                historical_roster,
                team=ai_team,
                next_season_index=int(target_id.split("-")[-1]),
                allowance=recruitment_allowance(
                    int(settlement_entry["balance_after"]),
                ),
            )
            transactions = [
                transaction
                for transaction in (squad_clubs.get(ai_team) or {}).get(
                    "transactions", []
                )
                if transaction.get("season_id") == target_id
            ]
            if decision["plan"] is None:
                if transactions:
                    raise ValueError("unexpected AI recruitment transaction")
                transaction_identity = None
            else:
                if (
                    len(transactions) != 1
                    or transactions[0].get("plan") != decision["plan"]
                ):
                    raise ValueError("AI recruitment policy mismatch")
                transaction_identity = evidence_identity(transactions[0])
            expected_clubs.append(
                {
                    "team": ai_team,
                    "settlement_identity": evidence_identity(
                        settlement_entry["evidence"],
                    ),
                    "decision": decision,
                    "squad_transaction_identity": transaction_identity,
                }
            )
        expected_event = {
            "schema_version": 1,
            "from_season_id": source_id,
            "to_season_id": target_id,
            "archive_identity": evidence_identity(source_archive),
            "manager_controlled_team": controlled_team,
            "participating_teams": participating,
            "ai_clubs": expected_clubs,
            "claim_boundary": (
                "deterministic bounded AI-club evolution from simulated "
                "standings, game-credit finance, and fictional markets"
            ),
        }
        if dict(event) != expected_event:
            raise ValueError("league ecosystem source replay mismatch")
    strategy_states = list(archives.values())
    if isinstance(season, Mapping):
        strategy_states.append(season)
    ecosystem_transitions = list(ecosystem.get("transitions") or [])
    for strategy_state in strategy_states:
        snapshot = strategy_state.get("club_strategies")
        if snapshot is None:
            continue
        target_id = str(strategy_state.get("season_id") or "")
        target_plan = SeasonPlan.from_payload(strategy_state["plan"])
        source_id = snapshot.get("source_season_id")
        source_archive = None
        previous_snapshot = None
        transition = None
        if source_id is not None:
            source_archive = archives.get(str(source_id))
            if source_archive is None:
                continue
            previous_snapshot = source_archive.get("club_strategies")
            transition_matches = [
                item
                for item in ecosystem_transitions
                if item.get("from_season_id") == source_id
                and item.get("to_season_id") == target_id
            ]
            if len(transition_matches) != 1:
                raise ValueError("club strategy ecosystem source mismatch")
            transition = transition_matches[0]
        strategy_rosters = {}
        recruitment_moves = {}
        for strategy_team in sorted(target_plan.teams):
            base_roster = load_roster_json(
                roster_path_for_team(
                    str(root),
                    strategy_team,
                )
            )
            strategy_rosters[strategy_team] = (
                replay_squad_registry_through(
                    base_roster,
                    team=strategy_team,
                    registry=session.get("squad_registry"),
                    season_id=target_id,
                    development_registry=session.get("player_development"),
                    lifecycle_registry=session.get("player_lifecycle"),
                    market_registry=session.get("player_market"),
                )
                if base_roster is not None
                else None
            )
            target_transactions = [
                transaction
                for transaction in (squad_clubs.get(strategy_team) or {}).get(
                    "transactions", []
                )
                if transaction.get("season_id") == target_id
            ]
            recruitment_moves[strategy_team] = sum(
                len(transaction.get("moves") or [])
                for transaction in target_transactions
            )
        expected_snapshot = build_season_club_strategies(
            season_id=target_id,
            teams=target_plan.teams,
            manager_team=target_plan.manager_team,
            rosters=strategy_rosters,
            source_archive=source_archive,
            previous_snapshot=previous_snapshot,
            ecosystem_transition=transition,
            recruitment_moves=recruitment_moves,
        )
        if snapshot != expected_snapshot:
            raise ValueError("club strategy source replay mismatch")
        fixtures = strategy_state.get("fixtures")
        if not isinstance(fixtures, list):
            continue
        fixtures_by_id = {
            str(fixture.get("fixture_id") or ""): fixture
            for fixture in fixtures
            if isinstance(fixture, Mapping)
        }
        for match_record in session.get("matches") or []:
            match_plan = (
                match_record.get("match_plan")
                if isinstance(match_record, Mapping)
                else None
            )
            competition = (
                match_plan.get("competition")
                if isinstance(match_plan, Mapping)
                else None
            )
            if (
                not isinstance(competition, Mapping)
                or competition.get("season_id") != target_id
            ):
                continue
            fixture = fixtures_by_id.get(str(competition.get("fixture_id") or ""))
            if fixture is None:
                raise ValueError("club strategy match fixture mismatch")
            expected_fixture_strategy = resolve_fixture_club_strategy(
                snapshot,
                home=str(fixture["home"]),
                away=str(fixture["away"]),
                manager_decision=fixture.get("manager_decision"),
                opponent_preparation=fixture.get("opponent_preparation"),
            )
            if (
                match_plan.get("club_strategy") != expected_fixture_strategy
                or match_plan.get("home_tactic")
                != expected_fixture_strategy["home_tactic"]
                or match_plan.get("away_tactic")
                != expected_fixture_strategy["away_tactic"]
            ):
                raise ValueError("completed match club strategy mismatch")
    return session


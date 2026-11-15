from __future__ import annotations

import random
from copy import deepcopy

from gfs_core.contracts import PluginContext, PluginMetadata
from gfs_core.kernel.identity import content_identity
from gfs_core.kernel.random import derive_seed

from .advancement.bracket import build_qualified_entries, build_r32_pairings
from .checkpointing.carryover import CrossMatchService, TeamCarryover
from .engines import MacroMatchEngine
from .scheduling.official import OFFICIAL_GROUP_MATCHDAYS, validate_groups

ROUNDS = (
    ("Round of 32", 32),
    ("Round of 16", 16),
    ("Quarter-Finals", 8),
    ("Semi-Finals", 4),
    ("Final", 2),
)


class TournamentRunner:
    def __init__(self, *, groups, engine, transactions, checkpointing, status, tactics, cross_match=None, root_seed=42):
        validate_groups(groups)
        self.groups = deepcopy(groups)
        self.engine = engine
        self.transactions = transactions
        self.checkpointing = checkpointing
        self.status = status
        self.tactics = tactics
        self.cross_match = cross_match or CrossMatchService()
        self.memory = self.cross_match.memory
        self.root_seed = int(root_seed)
        self.standings = {
            group: {
                team: {"pts": 0, "gf": 0, "ga": 0, "gd": 0}
                for team in teams
            }
            for group, teams in groups.items()
        }
        self.phase = "group"
        self.completed_matches = {}
        self.match_index = 0
        self.qualified_teams = []
        self.r32_fixtures = []
        self.current_round = None
        self.round_fixtures = []
        self.round_winners = []
        self.final_result = {}
        self.carryover = {
            team: TeamCarryover(team)
            for teams in groups.values() for team in teams
        }
        self._group_schedule = [
            (group, home, away, matchday)
            for group, matchdays in OFFICIAL_GROUP_MATCHDAYS.items()
            for matchday, fixtures in enumerate(matchdays, start=1)
            for home, away in fixtures
        ]
        self._group_index = 0
        self.scenario_id = self._scenario_identity()

    def play_next(self):
        fixture = self._peek_fixture()
        if fixture is None:
            return None
        stage, home, away, _ = fixture
        key = _match_key(self.scenario_id, stage, home, away)
        if key in self.completed_matches:
            raise ValueError(f"match already committed: {key}")
        resource = _TournamentResource(self)
        return self.transactions.execute(
            key,
            resource,
            lambda: self._play(fixture, key),
        )

    def run(self):
        while self.play_next() is not None:
            pass
        return dict(self.final_result)

    def resume(self):
        checkpoint = self.checkpointing.load()
        self._restore(checkpoint)
        return self.run()

    def snapshot(self):
        return {
            "schema_version": 2,
            "root_seed": self.root_seed,
            "scenario_id": self.scenario_id,
            "base_status": self._base_status(),
            "memory": self.memory.snapshot() if self.memory is not None else None,
            "standings": deepcopy(self.standings),
            "phase": self.phase,
            "completed_matches": dict(self.completed_matches),
            "match_index": self.match_index,
            "qualified_teams": list(self.qualified_teams),
            "r32_fixtures": list(self.r32_fixtures),
            "current_round": self.current_round,
            "round_fixtures": list(self.round_fixtures),
            "round_winners": list(self.round_winners),
            "final_result": dict(self.final_result),
            "carryover": {
                team: value.to_dict() for team, value in self.carryover.items()
            },
            "group_index": self._group_index,
        }

    def _restore(self, checkpoint):
        if checkpoint.get("scenario_id") != self.scenario_id:
            raise ValueError("tournament scenario identity mismatch")
        if int(checkpoint["root_seed"]) != self.root_seed:
            raise ValueError("tournament checkpoint root seed mismatch")
        if checkpoint.get("base_status") is not None and checkpoint["base_status"] != self._base_status():
            raise ValueError("tournament base team status has changed")
        if self.memory is not None and checkpoint.get("memory") is not None:
            self.memory.restore(checkpoint["memory"])
        self.standings = deepcopy(checkpoint["standings"])
        self.phase = checkpoint["phase"]
        self.completed_matches = dict(checkpoint["completed_matches"])
        self.match_index = int(checkpoint["match_index"])
        self.qualified_teams = list(checkpoint["qualified_teams"])
        self.r32_fixtures = [tuple(value) for value in checkpoint["r32_fixtures"]]
        self.current_round = checkpoint["current_round"]
        self.round_fixtures = [tuple(value) for value in checkpoint["round_fixtures"]]
        self.round_winners = list(checkpoint["round_winners"])
        self.final_result = dict(checkpoint["final_result"])
        self.carryover = {
            team: TeamCarryover.from_dict(value)
            for team, value in checkpoint["carryover"].items()
        }
        self._group_index = int(checkpoint["group_index"])

    def _scenario_identity(self) -> str:
        return content_identity({
            "schema_version": 1,
            "groups": self.groups,
            "root_seed": self.root_seed,
            "base_status": self._base_status(),
            "tactics_identity": self.tactics.identity,
            "data_identity": self.tactics.data_identity,
            "engine": self.engine.name,
            "engine_config": getattr(self.engine, "config", None),
            "carryover_policy": {
                "fatigue_status_penalty": self.cross_match.fatigue_status_penalty,
                "morale_status_scale": self.cross_match.morale_status_scale,
                "memory_status_scale": self.cross_match.memory_status_scale,
            },
        })

    def profile_for(self, team: str):
        profile = self.tactics.profile(team, self.status.table.loc[team])
        return self.cross_match.apply_to_profile(profile, self.carryover[team])

    def _base_status(self) -> dict[str, float]:
        return {
            team: float(self.status.table.loc[team, "final_status_score"])
            for teams in self.groups.values()
            for team in teams
        }

    def _play(self, fixture, key):
        stage, home, away, matchday = fixture
        seed = derive_seed(
            self.root_seed,
            "group_fixture" if self.phase == "group" else "knockout_fixture",
            stage,
            matchday,
            home,
            away,
        )
        result = self.engine.play(
            self.profile_for(home),
            self.profile_for(away),
            stage=stage,
            match_id=key,
            seed=seed,
            knockout=self.phase != "group",
        )
        if result.home_goals > result.away_goals:
            winner = home
        elif result.away_goals > result.home_goals:
            winner = away
        elif self.phase == "group":
            winner = "draw"
        else:
            home_penalties, away_penalties = self.engine.penalties(seed)
            winner = home if home_penalties > away_penalties else away
        self._commit(fixture, key, result, winner)
        return {
            "key": key,
            "home": home,
            "away": away,
            "winner": winner,
            "score": [result.home_goals, result.away_goals],
            "xg": [result.home_xg, result.away_xg],
            "engine": result.engine,
        }

    def _commit(self, fixture, key, result, winner):
        stage, home, away, _ = fixture
        if self.phase == "group":
            self._group_index += 1
            self._apply_group_result(home, away, result.home_goals, result.away_goals)
        else:
            self.round_winners.append(winner)
        self.completed_matches[key] = winner
        self.match_index += 1
        for team, won, lost in (
            (home, winner == home, winner == away),
            (away, winner == away, winner == home),
        ):
            carryover = self.carryover[team]
            if result.summary is not None:
                self.cross_match.settle(carryover, result.summary, key)
            else:
                is_home = team == home
                self.cross_match.settle_outcome(
                    carryover,
                    won=won,
                    lost=lost,
                    match_id=key,
                    opponent=away if is_home else home,
                    score=(
                        result.home_goals if is_home else result.away_goals,
                        result.away_goals if is_home else result.home_goals,
                    ),
                    xg=(
                        result.home_xg if is_home else result.away_xg,
                        result.away_xg if is_home else result.home_xg,
                    ),
                    engine=result.engine,
                )
        if stage == "Final":
            runner_up = away if winner == home else home
            self.final_result = {"champion": winner, "runner_up": runner_up}
            self.phase = "complete"

    def _peek_fixture(self):
        if self.phase == "complete":
            return None
        if self.phase == "group":
            if self._group_index < len(self._group_schedule):
                return self._group_schedule[self._group_index]
            self._resolve_advancements()
        if self.current_round is None:
            self.current_round = "Round of 32"
            self.round_fixtures = list(self.r32_fixtures)
            self.round_winners = []
        while self.current_round is not None:
            if self.round_fixtures:
                home, away = self.round_fixtures.pop(0)
                return self.current_round, home, away, 0
            self.qualified_teams = list(self.round_winners)
            next_index = next(
                (index for index, (name, _) in enumerate(ROUNDS) if name == self.current_round),
                None,
            )
            if next_index is None or next_index + 1 >= len(ROUNDS):
                if not self.final_result:
                    raise RuntimeError("tournament ended without a champion")
                self.phase = "complete"
                return None
            next_round, count = ROUNDS[next_index + 1]
            self.current_round = next_round
            self.round_fixtures = [
                (self.qualified_teams[index], self.qualified_teams[index + 1])
                for index in range(0, count, 2)
            ]
            self.round_winners = []
        return None

    def _resolve_advancements(self):
        entries = build_qualified_entries(self.groups, self.standings)
        self.qualified_teams = [entry.team for entry in entries]
        self.r32_fixtures = build_r32_pairings(
            entries,
            rng=random.Random(derive_seed(self.root_seed, "round_of_32_pairings")),
        )
        self.phase = "knockout"

    def _apply_group_result(self, home, away, home_goals, away_goals):
        group = next(
            name for name, teams in self.groups.items()
            if home in teams and away in teams
        )
        home_record = self.standings[group][home]
        away_record = self.standings[group][away]
        home_record["gf"] += home_goals
        home_record["ga"] += away_goals
        away_record["gf"] += away_goals
        away_record["ga"] += home_goals
        home_record["gd"] = home_record["gf"] - home_record["ga"]
        away_record["gd"] = away_record["gf"] - away_record["ga"]
        if home_goals > away_goals:
            home_record["pts"] += 3
        elif away_goals > home_goals:
            away_record["pts"] += 3
        else:
            home_record["pts"] += 1
            away_record["pts"] += 1


class _TournamentResource:
    def __init__(self, runner) -> None:
        self.runner = runner

    def snapshot(self):
        return self.runner.snapshot()

    def restore(self, snapshot):
        self.runner._restore(snapshot)

    def verify(self, result):
        key = result["key"]
        if self.runner.completed_matches.get(key) != result["winner"]:
            raise RuntimeError("tournament result was not committed")
        if self.runner.match_index != len(self.runner.completed_matches):
            raise RuntimeError("tournament match index does not match results")
        for team in (result["home"], result["away"]):
            if key not in self.runner.carryover[team].settled_match_ids:
                raise RuntimeError("tournament carryover was not committed")
            if self.runner.memory is not None and not self.runner.memory.has_match(key, team):
                raise RuntimeError("tournament memory was not committed")


class TournamentPlugin:
    metadata = PluginMetadata(
        id="gfs.core.tournament",
        version="0.1.0",
        api=1,
        contract="gfs_core.contracts.tournament",
        provides=("tournament-runner",),
        requires=(
            "gfs.core.tournament.scheduling",
            "gfs.core.tournament.checkpointing",
            "gfs.core.match.scoring",
            "gfs.core.persistence.transactions",
            "gfs.core.status.confidence",
            "gfs.core.data.tactics",
            "gfs.core.society.memory",
        ),
    )

    def __init__(self, groups, transactions, checkpointing, *, engine=None, root_seed=42):
        self.groups = groups
        self.transactions = transactions
        self.checkpointing = checkpointing
        self.engine = engine
        self.root_seed = root_seed
        self._runner = None

    def activate(self, context: PluginContext) -> None:
        engine = self.engine or MacroMatchEngine(context.require("scoring"))
        self._runner = TournamentRunner(
            groups=self.groups,
            engine=engine,
            transactions=self.transactions,
            checkpointing=self.checkpointing,
            status=context.require("sample-confidence"),
            tactics=context.require("tactics"),
            cross_match=CrossMatchService(memory=context.require("memory")),
            root_seed=self.root_seed,
        )
        context.provide("tournament-runner", self._runner)

    def deactivate(self) -> None:
        self._runner = None


def _match_key(scenario_id, stage, home, away):
    first, second = sorted((home, away))
    return f"{scenario_id}::{stage}::{first}::{second}"

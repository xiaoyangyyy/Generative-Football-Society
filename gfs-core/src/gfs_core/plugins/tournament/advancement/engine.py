from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from ..random import named_rng


class WorldCupEngine:
    def __init__(self, groups, score_simulator) -> None:
        self.groups = groups
        self.score_simulator = score_simulator

    def champion(self, stats: pd.DataFrame, *, rng: np.random.Generator) -> str:
        standings = self._group_stage(stats, rng)
        advancing = self._qualified_teams(standings, stats)
        return self._knockout_stage(advancing, stats, rng)

    def monte_carlo(
        self,
        stats: pd.DataFrame,
        *,
        simulations: int,
        seed: int,
    ) -> list[dict[str, object]]:
        if simulations < 1:
            raise ValueError("simulations must be positive")
        wins = {team: 0 for teams in self.groups.values() for team in teams}
        for index in range(simulations):
            champion = self.champion(
                stats, rng=named_rng(seed, "monte_carlo", index),
            )
            wins[champion] += 1
        rows = [
            {
                "team": team,
                "wins": count,
                "probability": count / simulations,
                "status_score": self._status(team, stats),
            }
            for team, count in sorted(wins.items())
        ]
        return sorted(rows, key=lambda row: row["probability"], reverse=True)

    def _group_stage(self, stats, rng):
        standings = {}
        for group_name, teams in self.groups.items():
            record = {
                team: {"pts": 0, "gf": 0, "ga": 0}
                for team in teams
            }
            for home, away in combinations(teams, 2):
                winner, score = self._play(home, away, stats, False, rng)
                record[home]["gf"] += score.home_goals
                record[home]["ga"] += score.away_goals
                record[away]["gf"] += score.away_goals
                record[away]["ga"] += score.home_goals
                if winner == "home":
                    record[home]["pts"] += 3
                elif winner == "away":
                    record[away]["pts"] += 3
                else:
                    record[home]["pts"] += 1
                    record[away]["pts"] += 1
            standings[group_name] = sorted(
                (
                    {
                        "team": team,
                        **values,
                        "gd": values["gf"] - values["ga"],
                    }
                    for team, values in record.items()
                ),
                key=lambda row: (
                    row["pts"], row["gd"], row["gf"],
                    self._status(row["team"], stats),
                ),
                reverse=True,
            )
        return standings

    def _qualified_teams(self, standings, stats):
        advancing = []
        thirds = []
        for ranking in standings.values():
            advancing.extend([ranking[0]["team"], ranking[1]["team"]])
            thirds.append(ranking[2])
        thirds.sort(
            key=lambda row: (
                row["pts"], row["gd"], row["gf"],
                self._status(row["team"], stats),
            ),
            reverse=True,
        )
        best_thirds = [row["team"] for row in thirds[:8]]
        advancing.extend(best_thirds)
        return advancing

    def _knockout_stage(self, advancing, stats, rng):
        pool = [advancing[index] for index in rng.permutation(len(advancing))]
        while len(pool) > 1:
            next_pool = []
            for index in range(0, len(pool), 2):
                home = pool[index]
                away = pool[index + 1]
                winner, _ = self._play(home, away, stats, True, rng)
                next_pool.append(home if winner == "home" else away)
            pool = next_pool
        return pool[0]

    def _play(self, home, away, stats, knockout, rng):
        score = self.score_simulator.simulate(
            self._status(home, stats),
            self._status(away, stats),
            is_knockout=knockout,
            rng=rng,
        )
        if score.home_goals > score.away_goals:
            return "home", score
        if score.away_goals > score.home_goals:
            return "away", score
        if not knockout:
            return "draw", score
        home_penalties, away_penalties = self.score_simulator.penalties(rng)
        return ("home" if home_penalties > away_penalties else "away"), score

    @staticmethod
    def _status(team: str, stats: pd.DataFrame) -> float:
        if team in stats.index:
            return float(stats.loc[team]["final_status_score"])
        return 30.0

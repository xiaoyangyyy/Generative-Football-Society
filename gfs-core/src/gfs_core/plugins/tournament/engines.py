from __future__ import annotations

import numpy as np

from gfs_core.contracts.tournament import TournamentMatchResult


class MacroMatchEngine:
    name = "macro"

    def __init__(self, scoring) -> None:
        self.scoring = scoring

    def play(self, home, away, *, stage, match_id, seed, knockout):
        del stage, match_id
        score = self.scoring.simulate(
            home.status_score,
            away.status_score,
            is_knockout=knockout,
            rng=np.random.default_rng(seed),
        )
        return TournamentMatchResult(
            score.home_goals,
            score.away_goals,
            score.home_xg,
            score.away_xg,
            self.name,
        )

    def penalties(self, seed: int) -> tuple[int, int]:
        return self.scoring.penalties(np.random.default_rng(seed + 17))


class MicroMatchEngine:
    name = "micro"

    def __init__(self, scoring, match_runner, config=None, memory=None, memory_as_of=None) -> None:
        self.scoring = scoring
        self.match_runner = match_runner
        self.config = config
        self.memory = memory
        self.memory_as_of = memory_as_of

    def play(self, home, away, *, stage, match_id, seed, knockout):
        prior = self.scoring.simulate(
            home.status_score,
            away.status_score,
            is_knockout=knockout,
            rng=np.random.default_rng(seed),
        )
        summary = self.match_runner.run(
            home,
            away,
            goals_home=prior.home_goals,
            goals_away=prior.away_goals,
            xg_home=prior.home_xg,
            xg_away=prior.away_xg,
            config=self.config,
            seed=seed,
            stage_pressure=0.75 if knockout else 0.3,
            stage_name=stage,
            authoritative_clock=True,
            memory=self.memory,
            memory_as_of=self.memory_as_of,
        )
        return TournamentMatchResult(
            summary.goals_micro_home,
            summary.goals_micro_away,
            summary.micro_xg_home,
            summary.micro_xg_away,
            self.name,
            summary,
        )

    def penalties(self, seed: int) -> tuple[int, int]:
        return self.scoring.penalties(np.random.default_rng(seed + 17))

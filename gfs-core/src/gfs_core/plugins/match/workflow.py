from dataclasses import replace

from gfs_core.contracts import PluginContext, PluginMetadata
from gfs_core.contracts.micro_config import MicroMatchConfig
from gfs_core.kernel.random import named_rng


class MatchWorkflow:
    def __init__(self, status, tactics, scoring, match_runner, memory=None) -> None:
        self.status = status
        self.tactics = tactics
        self.scoring = scoring
        self.match_runner = match_runner
        self.memory = memory

    def run(self, home: str, away: str, *, seed: int = 42, fast: bool = False, config=None, stage_name: str = "match", authoritative_clock: bool = False, home_tactics=None, away_tactics=None, score_mode: str | None = None, memory_as_of: int | None = None, **options):
        config = config or (MicroMatchConfig.fast_demo() if fast else MicroMatchConfig())
        if score_mode is not None:
            config = replace(config, score_mode=score_mode)
        home_profile = self.tactics.profile(home, self.status.table.loc[home])
        away_profile = self.tactics.profile(away, self.status.table.loc[away])
        if home_tactics:
            home_profile = replace(home_profile, tactical={**home_profile.tactical, **home_tactics})
        if away_tactics:
            away_profile = replace(away_profile, tactical={**away_profile.tactical, **away_tactics})
        prior = self.scoring.simulate(
            home_profile.status_score,
            away_profile.status_score,
            is_knockout=False,
            rng=named_rng(seed, "micro_match", home, away, "macro_prior"),
        )
        options["data_identity"] = self.tactics.data_identity
        return self.match_runner.run(
            home_profile,
            away_profile,
            goals_home=prior.home_goals,
            goals_away=prior.away_goals,
            xg_home=prior.home_xg,
            xg_away=prior.away_xg,
            referee={"strictness": 0.58, "bias_t1": 0.04},
            config=config,
            seed=seed,
            stage_name=stage_name,
            authoritative_clock=authoritative_clock,
            memory=self.memory,
            memory_as_of=memory_as_of,
            **options,
        )


class MatchWorkflowPlugin:
    metadata = PluginMetadata(
        id="gfs.core.match.workflow",
        version="0.1.0",
        api=1,
        contract="gfs_core.contracts.match",
        provides=("match-workflow",),
        requires=(
            "gfs.core.match",
            "gfs.core.status.confidence",
            "gfs.core.data.tactics",
            "gfs.core.society.memory",
        ),
    )

    def activate(self, context: PluginContext) -> None:
        context.provide("match-workflow", MatchWorkflow(
            context.require("sample-confidence"),
            context.require("tactics"),
            context.require("scoring"),
            context.require("match-runner"),
            context.require("memory"),
        ))

    def deactivate(self) -> None:
        pass

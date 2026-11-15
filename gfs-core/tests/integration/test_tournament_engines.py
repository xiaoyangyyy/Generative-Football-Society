from pathlib import Path

from gfs_core.bootstrap.builtin_plugins import build_micro_registry
from gfs_core.contracts.micro_config import MicroMatchConfig
from gfs_core.kernel import PluginKernel
from gfs_core.plugins.tournament.engines import MicroMatchEngine

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_micro_match_engine_uses_the_same_tournament_contract():
    kernel = PluginKernel(build_micro_registry(
        PROJECT_ROOT, config=MicroMatchConfig.fast_demo(),
    ))
    try:
        kernel.start(["gfs.core.status.confidence", "gfs.core.data.tactics", "gfs.core.match"])
        snapshot = kernel.context.require("sample-confidence")
        tactics = kernel.context.require("tactics")
        engine = MicroMatchEngine(
            kernel.context.require("scoring"),
            kernel.context.require("match-runner"),
            MicroMatchConfig.fast_demo(),
        )
        result = engine.play(
            tactics.profile("Brazil", snapshot.table.loc["Brazil"]),
            tactics.profile("Argentina", snapshot.table.loc["Argentina"]),
            stage="test",
            match_id="micro-contract-test",
            seed=42,
            knockout=False,
        )
        assert result.engine == "micro"
        assert result.summary is not None
        assert result.summary.ticks == 180
        assert [result.home_goals, result.away_goals] == [
            result.summary.goals_micro_home,
            result.summary.goals_micro_away,
        ]
        assert [result.home_xg, result.away_xg] == [
            result.summary.micro_xg_home,
            result.summary.micro_xg_away,
        ]
    finally:
        kernel.stop()

from gfs_core.contracts import PluginContext, PluginMetadata


class ManagerWorkspace:
    def __init__(self, workflows, roles) -> None:
        self.workflows = workflows
        self.roles = roles

    def briefing(self, home: str, away: str, seed: int) -> dict:
        comparison = self.workflows.compare_tactics(home, away, seed)
        return self.review(
            comparison["baseline"], comparison["candidates"], comparison["evaluation"],
        )

    def review(self, summary, candidates=None, evaluation=None) -> dict:
        coach = self.roles.coach.review(summary, candidates, evaluation)
        return {
            "manager_choice": coach["manager_choice"],
            "reviewed_future": coach["reviewed_future"],
            "advisory_reflection": coach["advisory_reflection"],
            "official_action": {
                "score": [summary.goals_micro_home, summary.goals_micro_away],
                "xg": [summary.micro_xg_home, summary.micro_xg_away],
                "shots": [summary.shots_home, summary.shots_away],
                "crosses": summary.crosses_attempted,
                "scope": "official_simulator_record",
            },
            "society_transition": {
                "players": self.roles.player.review(summary),
                "media": self.roles.media.review(summary),
                "scope": "descriptive_persistent_simulator_state",
            },
            "causal_claim": False,
            "outcome_improvement_claim": False,
        }


class ProductManagerPlugin:
    metadata = PluginMetadata(
        id="gfs.studio.product.manager",
        version="0.1.0",
        api=1,
        contract="gfs_studio.contracts.product",
        provides=("manager",),
        requires=("gfs.studio.product.studio", "gfs.studio.cognition.roles"),
    )

    def __init__(self, workspace) -> None:
        self._workspace = workspace

    def activate(self, context: PluginContext) -> None:
        context.provide("manager", self._workspace)

    def deactivate(self) -> None:
        pass

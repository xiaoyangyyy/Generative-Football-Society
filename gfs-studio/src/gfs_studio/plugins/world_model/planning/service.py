from gfs_core.contracts import PluginContext, PluginMetadata


class PlanningService:
    def __init__(self, runtime) -> None:
        self.runtime = runtime

    def forecast(self, observation, action, *, steps: int | None = None, kind: str | None = None) -> dict:
        kind = kind or self.runtime.action_kind(action)
        output = self.runtime.imagine(observation, action, steps=steps)
        return {
            "next_observation": output.next_obs,
            "pass_success": output.pass_success,
            "progress_delta": output.progress_delta,
            "shot_goal_prob": output.shot_goal_prob,
            "uncertainty": output.uncertainty,
            "authority": self.runtime.planner_authority(observation, kind=kind),
            "deployed": False,
        }


class PlanningPlugin:
    metadata = PluginMetadata(
        id="gfs.studio.world-model.planning",
        version="0.1.0",
        api=1,
        contract="gfs_studio.contracts.world_model",
        provides=("planning",),
        requires=("gfs.studio.world-model.latent",),
    )

    def __init__(self, runtime) -> None:
        self._runtime = runtime
        self.service = PlanningService(runtime)

    def activate(self, context: PluginContext) -> None:
        context.provide("planning", self.service)

    def deactivate(self) -> None:
        pass

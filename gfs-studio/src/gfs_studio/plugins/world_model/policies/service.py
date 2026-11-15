from gfs_core.contracts import PluginContext, PluginMetadata


class PolicyService:
    def __init__(self, planning) -> None:
        self.planning = planning

    def recommend(self, observation, action, *, baseline_action: str = "pass") -> dict:
        forecast = self.planning.forecast(observation, action)
        authorized = forecast["authority"]["authorized"]
        if not authorized:
            return {
                "action": baseline_action,
                "authority": False,
                "baseline_preserved": True,
                "outcome_improvement_claim": False,
            }
        return {
            "action": "pass" if forecast["pass_success"] >= 0.5 else "hold",
            "authority": True,
            "baseline_preserved": False,
            "outcome_improvement_claim": False,
            "uncertainty": forecast["uncertainty"],
            "prediction": {
                "candidate_id": "world_model_forecast",
                "utility_estimate": forecast["pass_success"] - forecast["uncertainty"],
                "samples": 1,
                "scope": "model_prediction_not_observed_match_or_causal_effect",
            },
        }

    def assess_case(self, case, observation, action) -> dict:
        forecast = self.planning.forecast(observation, action)
        return {
            **case.as_dict(),
            "candidate_id": f"{self.planning.runtime.action_kind(action)}_forecast",
            "authorized": forecast["authority"]["authorized"],
            "predicted_metrics": {
                "pass_success": forecast["pass_success"],
                "progress_delta": forecast["progress_delta"],
                "shot_goal_prob": forecast["shot_goal_prob"],
                "uncertainty": forecast["uncertainty"],
            },
            "observed_simulation_metrics": None,
            "scope": "model_prediction_separate_from_observed_simulation",
        }


class PoliciesPlugin:
    metadata = PluginMetadata(
        id="gfs.studio.world-model.policies",
        version="0.1.0",
        api=1,
        contract="gfs_studio.contracts.world_model",
        provides=("policies",),
        requires=("gfs.studio.world-model.planning",),
    )

    def __init__(self, planning) -> None:
        self._planning = planning

    def activate(self, context: PluginContext) -> None:
        context.provide("policies", PolicyService(self._planning))

    def deactivate(self) -> None:
        pass

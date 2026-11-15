from statistics import fmean

from gfs_core.contracts import PluginContext, PluginMetadata


class ComparisonService:
    def __init__(self, workflows) -> None:
        self._workflows = workflows

    def compare(self, baseline: dict, treatment: dict) -> dict:
        _validate_result(baseline)
        _validate_result(treatment)
        if (baseline["home"], baseline["away"]) != (treatment["home"], treatment["away"]):
            raise ValueError("comparison requires the same teams")
        return {
            "home": baseline["home"],
            "away": baseline["away"],
            "goal_delta": _delta(baseline, treatment, "score"),
            "xg_delta": _delta(baseline, treatment, "xg"),
            "pass_delta": _delta(baseline, treatment, "passes"),
            "shot_delta": _delta(baseline, treatment, "shots"),
            "pass_completion_delta": _delta(
                baseline, treatment, "pass_completion",
            ),
            "scope": "descriptive_comparison_not_causal_or_significant",
            "causal_claim": False,
        }

    def compare_seeds(self, home: str, away: str, baseline_seed: int, treatment_seed: int) -> dict:
        baseline = self._workflows.micro_match(home, away, baseline_seed)
        treatment = self._workflows.micro_match(home, away, treatment_seed)
        return {
            **self.compare(baseline, treatment),
            "baseline_seed": baseline_seed,
            "treatment_seed": treatment_seed,
            "scope": "seed_sensitivity_only_not_policy_or_causal_effect",
        }

    def aggregate(self, comparisons: list[dict]) -> dict:
        if not comparisons:
            raise ValueError("at least one comparison is required")
        metrics = (
            "goal_delta", "xg_delta", "pass_delta", "shot_delta",
            "pass_completion_delta",
        )
        means = {}
        for metric in metrics:
            for side in (0, 1):
                means[f"{metric}_{side}"] = fmean(
                    comparison[metric][side] for comparison in comparisons
                )
        return {
            "comparisons": len(comparisons),
            "mean_deltas": means,
            "scope": "descriptive_aggregate_no_causal_or_significance_claim",
        }


class ComparisonPlugin:
    metadata = PluginMetadata(
        id="gfs.studio.product.comparison",
        version="0.1.0",
        api=1,
        contract="gfs_studio.contracts.product",
        provides=("comparison",),
        requires=("gfs.studio.product.studio",),
    )

    def activate(self, context: PluginContext) -> None:
        context.provide("comparison", ComparisonService(
            context.require("studio-workflows"),
        ))

    def deactivate(self) -> None:
        pass


def _validate_result(result: dict) -> None:
    if not isinstance(result, dict):
        raise ValueError("match comparison requires result objects")
    for name in ("home", "away"):
        if not isinstance(result.get(name), str) or not result[name]:
            raise ValueError("match comparison requires team names")
    for name in ("score", "xg", "passes", "shots", "pass_completion"):
        values = result.get(name)
        if not isinstance(values, (list, tuple)) or len(values) != 2:
            raise ValueError(f"match comparison requires two values for {name}")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise ValueError(f"match comparison requires numeric {name}")
    if any(value < 0 for value in result["score"] + result["xg"]):
        raise ValueError("goals and xG cannot be negative")


def _delta(baseline: dict, treatment: dict, name: str) -> list[float]:
    return [
        treatment[name][side] - baseline[name][side]
        for side in (0, 1)
    ]

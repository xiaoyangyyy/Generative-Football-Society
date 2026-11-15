from dataclasses import is_dataclass

from gfs_core.kernel.identity import content_identity


class CoachRole:
    def __init__(self, cognition) -> None:
        self.cognition = cognition

    def review(self, summary, candidates=None, evaluation=None) -> dict:
        context_identity = content_identity({
            "summary": summary if is_dataclass(summary) else vars(summary),
            "candidates": candidates,
            "evaluation": evaluation,
        })
        seed = candidates[0]["seed"] if candidates else 42
        reflection = self.cognition.reflect(
            role="coach",
            topic=f"Review {summary.home_team} versus {summary.away_team}",
            match_id=f"micro:{summary.home_team}:{summary.away_team}:{context_identity[:24]}",
            seed=seed,
        )
        if candidates:
            if evaluation is None:
                best = max(candidates, key=lambda candidate: candidate["utility"])
                best_id = best["id"]
                expected = best["xg"]
            else:
                best_id = evaluation["best_candidate"]
                expected = evaluation["candidates"][best_id]["mean_xg"] if best_id else [
                    summary.micro_xg_home, summary.micro_xg_away,
                ]
            choice = {
                "recommended_action": best_id or "preserve_current_vector",
                "tactic": best_id or "preserve_current_vector",
                "hold_reference": True,
                "requires_human_review": True,
                "official_simulation_changed": False,
                "evaluation_status": evaluation["status"] if evaluation else "legacy_single_observation",
            }
            future = {
                "expected_micro_xg": expected[0],
                "expected_micro_xg_away": expected[1],
                "candidates": candidates,
                "evaluation": evaluation,
                "comparison_contract": "same_seed_and_same_initial_data",
                "scope": "simulator_local_descriptive_comparison",
            }
        else:
            low_value = summary.shots_home == 0 and summary.micro_xg_home < 0.1
            choice = {
                "recommended_action": (
                    "reduce_low_value_shots" if low_value else "preserve_current_vector"
                ),
                "tactic": "balanced",
                "hold_reference": True,
            }
            future = {
                "expected_micro_xg": summary.micro_xg_home,
                "expected_micro_xg_away": summary.micro_xg_away,
                "scope": "simulator_local_descriptive_comparison",
            }
        return {
            "manager_choice": choice,
            "reviewed_future": future,
            "advisory_reflection": reflection,
        }

from __future__ import annotations

import math
from statistics import mean, stdev


class CandidateEvaluator:
    def evaluate(self, observations: list[dict]) -> dict:
        if not observations:
            raise ValueError("candidate evaluation requires observations")
        grouped = {}
        unique = {}
        for observation in observations:
            key = (observation["id"], observation["seed"])
            if key in unique:
                if unique[key] != observation:
                    raise ValueError("candidate seed identity conflicts with its content")
                continue
            unique[key] = observation
            grouped.setdefault(observation["id"], []).append(observation)
        summaries = {
            name: _summarize(records)
            for name, records in sorted(grouped.items())
        }
        seed_sets = {frozenset(row["seeds"]) for row in summaries.values()}
        complete = (
            all(row["samples"] >= 2 for row in summaries.values())
            and len(seed_sets) == 1
        )
        paired = _paired_deltas(grouped)
        best = max(summaries, key=lambda name: summaries[name]["mean_utility"]) if complete else None
        return {
            "status": "complete" if complete else "insufficient_paired_samples",
            "best_candidate": best,
            "candidates": summaries,
            "paired_comparisons": paired,
            "scope": "paired_descriptive_simulator_comparison_not_causal",
            "causal_claim": False,
            "outcome_improvement_claim": False,
        }


def candidate_record(name, summary, tactics, seed):
    utility = (
        summary.micro_xg_home
        - 0.5 * summary.micro_xg_away
        + 0.01 * (summary.shots_home - summary.shots_away)
    )
    if isinstance(seed, bool) or not isinstance(seed, int) or not math.isfinite(utility):
        raise ValueError("candidate seed and utility must be finite and valid")
    return {
        "id": name,
        "tactics": dict(tactics),
        "seed": seed,
        "score": [summary.goals_micro_home, summary.goals_micro_away],
        "xg": [summary.micro_xg_home, summary.micro_xg_away],
        "shots": [summary.shots_home, summary.shots_away],
        "passes": [summary.passes_home, summary.passes_away],
        "utility": utility,
        "scope": "same_seed_descriptive_simulator_comparison",
    }


def _summarize(records):
    utilities = [record["utility"] for record in records]
    return {
        "samples": len(records),
        "seeds": sorted({record["seed"] for record in records}),
        "mean_utility": mean(utilities),
        "utility_std": stdev(utilities) if len(utilities) > 1 else 0.0,
        "utility_range": [min(utilities), max(utilities)],
        "mean_xg": [mean(record["xg"][side] for record in records) for side in (0, 1)],
        "mean_shots": [mean(record["shots"][side] for record in records) for side in (0, 1)],
    }


def _paired_deltas(grouped):
    baseline = {
        record["seed"]: record["utility"]
        for record in grouped.get("preserve_current_vector", [])
    }
    output = {}
    for name, records in grouped.items():
        if name == "preserve_current_vector":
            continue
        deltas = [
            record["utility"] - baseline[record["seed"]]
            for record in records if record["seed"] in baseline
        ]
        output[name] = {
            "pairs": len(deltas),
            "mean_utility_delta": mean(deltas) if deltas else None,
            "utility_delta_std": stdev(deltas) if len(deltas) > 1 else 0.0 if deltas else None,
        }
    return output

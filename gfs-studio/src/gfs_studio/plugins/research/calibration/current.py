from __future__ import annotations

import math
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np

from gfs_core.contracts.micro_config import MicroMatchConfig
from gfs_core.kernel.project import data_identity, source_identities


class CurrentSimulationEvaluator:
    def __init__(self, match_workflow, evidence) -> None:
        self.match_workflow = match_workflow
        self.evidence = evidence

    def evaluate(self, *, fixtures=None, sample_indices=None, fast: bool = False) -> dict:
        baseline = self.evidence.read_frozen("m0_formal_baseline_v1.json")
        fixture_names = baseline["fixtures"] if fixtures is None else fixtures
        samples = list(sample_indices if sample_indices is not None else range(3))
        if not samples or any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in samples):
            raise ValueError("sample indices must be non-negative integers")
        fixtures = [tuple(name.split("_vs_", 1)) for name in fixture_names]
        if any(len(fixture) != 2 for fixture in fixtures):
            raise ValueError("fixture must contain home and away teams")
        rows = []
        config = MicroMatchConfig.fast_demo() if fast else MicroMatchConfig()
        for home, away in fixtures:
            for sample_index in samples:
                seed = 260806 + sample_index
                summary = self.match_workflow.run(
                    home,
                    away,
                    seed=seed,
                    config=config,
                )
                rows.append(_metric_row(summary, home, away, seed, fast))
        metrics = _compare(rows, baseline)
        diagnostic_loss = float(np.mean([
            row["diagnostic_z"] ** 2 for row in metrics.values()
        ])) if metrics else math.inf
        protocol_complete = (
            not fast
            and fixture_names == baseline["fixtures"]
            and samples == list(range(3))
            and len(rows) == int(baseline["samples_total"])
        )
        report = {
            "schema_version": 1,
            "state": "completed",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_identity": source_identities(self.evidence.root)["simulation"],
            "source_identities": source_identities(self.evidence.root),
            "data_identity": data_identity(self.evidence.root),
            "historical_baseline": "m0_formal_baseline_v1.json",
            "protocol": {
                "fixtures": fixture_names,
                "sample_indices": samples,
                "seed_start": 260806,
                "runs_total": len(rows),
                "engine": "micro_fast_demo" if fast else "micro_full_90min",
            },
            "protocol_complete": protocol_complete,
            "matches_executed": len(rows),
            "metrics": metrics,
            "diagnostic_loss": diagnostic_loss,
            "regressed_metrics": sorted(
                name for name, row in metrics.items()
                if row["diagnostic_score"] < row["historical_baseline_score"]
            ),
            "rows": rows,
            "scope": "current_descriptive_external_metric_comparison_not_preregistered_confirmatory",
        }
        digest = self.evidence.write_current("simulation_evaluation.json", report)
        receipt = self.evidence.issue_receipt(f"simulation-{digest}", report)
        return {**report, "receipt": asdict(receipt)}


def _metric_row(summary, home, away, seed, fast) -> dict:
    passes = summary.passes_home + summary.passes_away
    shots = summary.shots_home + summary.shots_away
    goals = summary.goals_micro_home + summary.goals_micro_away
    interceptions = summary.pass_intercepts_home + summary.pass_intercepts_away
    return {
        "fixture": f"{home}_vs_{away}",
        "home": home,
        "away": away,
        "seed": seed,
        "engine": "micro_fast_demo" if fast else "micro_full_90min",
        "possession_team": [summary.possession_home, 1.0 - summary.possession_home],
        "passes_team": [summary.passes_home, summary.passes_away],
        "shots_team": [summary.shots_home, summary.shots_away],
        "goals_team": [summary.goals_micro_home, summary.goals_micro_away],
        "pass_completion": (summary.pass_completion_home + summary.pass_completion_away) / 2,
        "interceptions_per_pass": interceptions / max(1, passes),
        "passes_per_team_match": passes / 2,
        "through_share": (summary.through_balls_home + summary.through_balls_away) / max(1, passes),
        "long_pass_share": (summary.long_passes_home + summary.long_passes_away) / max(1, passes),
        "shots_per_team_match": shots / 2,
        "shots_on_target_rate": (summary.shots_on_target_home + summary.shots_on_target_away) / max(1, shots),
        "goals_per_team_match": goals / 2,
        "fouls_committed_per_team_match": (summary.fouls_committed_home + summary.fouls_committed_away) / 2,
        "yellow_cards_per_team_match": (summary.yellow_cards_home + summary.yellow_cards_away) / 2,
        "red_cards_per_team_match": (summary.red_cards_home + summary.red_cards_away) / 2,
        "possession_share": summary.possession_home,
        "crosses_per_team_match": summary.crosses_attempted / 2,
        "headers_per_team_match": summary.headers_attempted / 2,
        "tackles_per_team_match": (summary.tackles_home + summary.tackles_away) / 2,
        "micro_xg_per_team_match": (summary.micro_xg_home + summary.micro_xg_away) / 2,
        "goals_to_micro_xg_ratio": goals / max(1e-9, summary.micro_xg_home + summary.micro_xg_away),
    }


def _compare(rows, baseline) -> dict:
    targets = dict(baseline["report"])
    targets.update(baseline.get("soft_constraints", {}))
    metrics = {}
    for name, target in targets.items():
        if name in {"possession_passes_correlation", "shots_goals_correlation"}:
            first, second = {
                "possession_passes_correlation": ("possession", "passes"),
                "shots_goals_correlation": ("shots", "goals"),
            }[name]
            left = np.asarray([row[f"{first}_team"] for row in rows], dtype=float).ravel()
            right = np.asarray([row[f"{second}_team"] for row in rows], dtype=float).ravel()
            observed = float(np.corrcoef(left, right)[0, 1]) if len(left) > 1 and left.std() and right.std() else 0.0
            sample_count = len(left)
        else:
            values = np.asarray([row[name] for row in rows], dtype=float)
            observed = float(values.mean())
            sample_count = len(values)
        scale = float(target["scale"])
        z_score = abs(observed - float(target["target_mean"])) / max(scale, 1e-12)
        metrics[name] = {
            "observed": observed,
            "target": float(target["target_mean"]),
            "scale": scale,
            "diagnostic_z": z_score,
            "diagnostic_score": float(np.exp(-0.5 * z_score**2)),
            "samples": sample_count,
            "historical_baseline_pass": bool(target.get("pass", False)),
            "historical_baseline_score": float(target.get("score", 0.0)),
        }
    return metrics

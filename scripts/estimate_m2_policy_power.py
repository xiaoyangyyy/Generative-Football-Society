#!/usr/bin/env python3
"""Reproduce the variance-only sample budget used by the M2 protocol."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import stdev


DEFAULT_INPUT = Path(
    "data/evaluation/action_outcome_v1/progress.json"
)


def _keyed(rows: list[dict]) -> dict[tuple[str, int], dict]:
    output = {}
    for row in rows:
        key = (str(row["fixture"]), int(row["sample_index"]))
        if key in output:
            raise ValueError(f"duplicate historical unit: {key}")
        output[key] = row
    return output


def estimate(
    payload: dict,
    *,
    meaningful_delta: float = 0.10,
    z_alpha: float = 1.96,
    z_power: float = 0.84,
) -> dict:
    """Use historical paired variance, never its candidate mean effect."""
    arms = payload.get("arms") or {}
    baseline = _keyed((arms.get("M0") or {}).get("rows") or [])
    candidate = _keyed((arms.get("M1") or {}).get("rows") or [])
    if len(baseline) < 3 or set(baseline) != set(candidate):
        raise ValueError("historical M0/M1 units are incomplete or unpaired")
    differences = []
    for key in sorted(baseline):
        base = baseline[key]
        model = candidate[key]
        base_margin = float(
            base["micro_xg_home"] - base["micro_xg_away"]
        )
        model_margin = float(
            model["micro_xg_home"] - model["micro_xg_away"]
        )
        differences.append(model_margin - base_margin)
    paired_sd = stdev(differences)
    required = math.ceil(
        ((z_alpha + z_power) * paired_sd / meaningful_delta) ** 2
    )
    return {
        "schema_version": 1,
        "role": "prospective_variance_planning_only",
        "historical_units": len(differences),
        "historical_home_xg_margin_paired_sd": paired_sd,
        "meaningful_delta": meaningful_delta,
        "two_sided_alpha": 0.05,
        "target_power": 0.80,
        "normal_approximation_required_units": required,
        "historical_candidate_mean_effect_used": False,
        "limitations": [
            "M1 was a bilateral legacy policy whereas M2 is one-sided.",
            "The normal approximation does not replace final clustered bootstrap.",
            "The estimate fixes budget only and is not M2 efficacy evidence."
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--meaningful-delta", type=float, default=0.10)
    args = parser.parse_args()
    if args.meaningful_delta <= 0.0:
        raise SystemExit("--meaningful-delta must be positive")
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    print(json.dumps(
        estimate(payload, meaningful_delta=args.meaningful_delta),
        indent=2,
    ))


if __name__ == "__main__":
    main()

"""Aggregate transition calibration and randomized policy-bridge evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.match_engine.world_model.online_evaluation import (  # noqa: E402
    aggregate_online_calibration,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate online world-model calibration and policy-bridge effects."
        ),
    )
    parser.add_argument(
        "--log-dir",
        default=str(ROOT / "data/persistence/cognitive_log"),
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "reports/evaluation/online_world_model.json"),
    )
    parser.add_argument("--min-transitions", type=int, default=50)
    parser.add_argument("--min-policy-arm", type=int, default=8)
    parser.add_argument("--min-residual-samples", type=int, default=20)
    parser.add_argument(
        "--required-policy-horizon",
        type=float,
        default=0.0,
        help="Outcome horizon in seconds; 0 means the immediate transition.",
    )
    parser.add_argument(
        "--require-policy-effect",
        action="store_true",
        help="Fail readiness until randomized treatment and control arms pass.",
    )
    parser.add_argument(
        "--require-outcome-calibration",
        action="store_true",
        help="Fail readiness until realized utility forecasts are calibrated.",
    )
    parser.add_argument(
        "--require-uncertainty-decomposition",
        action="store_true",
        help=(
            "Fail readiness until enough realized forecasts carry a valid "
            "epistemic/aleatoric decomposition."
        ),
    )
    parser.add_argument(
        "--require-transition-ensemble",
        action="store_true",
        help=(
            "Fail readiness until enough realized forecasts come from a "
            "trained transition ensemble."
        ),
    )
    parser.add_argument(
        "--require-opponent-belief",
        action="store_true",
        help=(
            "Fail readiness until enough decisions carry a grounded opponent "
            "belief and hypothesis-conditioned counterfactuals."
        ),
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    logs = []
    if log_dir.is_dir():
        for path in sorted(log_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("world_model_online_calibration"):
                logs.append(payload)
    if not logs:
        print(
            "No online calibration logs found. Run micro matches with "
            "MATCH_WORLD_MODEL=1 first.",
            file=sys.stderr,
        )
        return 2
    report = aggregate_online_calibration(
        logs,
        min_transitions=args.min_transitions,
        min_policy_arm=args.min_policy_arm,
        require_policy_effect=args.require_policy_effect,
        required_policy_horizon_s=args.required_policy_horizon,
        min_residual_samples=args.min_residual_samples,
        require_outcome_calibration=args.require_outcome_calibration,
        require_uncertainty_decomposition=(
            args.require_uncertainty_decomposition
        ),
        require_transition_ensemble=args.require_transition_ensemble,
        require_opponent_belief=args.require_opponent_belief,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

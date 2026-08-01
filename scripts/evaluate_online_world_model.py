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
    parser.add_argument(
        "--require-opponent-meta-belief",
        action="store_true",
        help=(
            "Fail readiness until enough decisions use a compatible "
            "cross-match opponent prior."
        ),
    )
    parser.add_argument(
        "--require-opponent-change-detection",
        action="store_true",
        help=(
            "Fail readiness until enough decisions carry the numeric change "
            "detector contract and every LLM change claim is non-controlling."
        ),
    )
    parser.add_argument(
        "--require-opponent-response-model",
        action="store_true",
        help=(
            "Fail readiness until enough realized predictions come from a "
            "held-out-validated opponent response model."
        ),
    )
    parser.add_argument(
        "--require-two-step-trajectory-planning",
        action="store_true",
        help=(
            "Fail readiness until enough decisions use budgeted predicted-state "
            "continuations backed by grouped two-step holdout gain."
        ),
    )
    parser.add_argument(
        "--require-llm-semantic-critic",
        action="store_true",
        help=(
            "Fail readiness until enough realized LLM residual critiques use "
            "match-held-out authority without mutating world-model forecasts."
        ),
    )
    parser.add_argument(
        "--require-llm-semantic-events",
        action="store_true",
        help=(
            "Fail readiness until paired LLM/neural semantic-event forecasts "
            "are realized across at least four matches and remain shadow-only."
        ),
    )
    parser.add_argument(
        "--require-learned-semantic-events",
        action="store_true",
        help=(
            "Fail readiness until validated learned event heads and their "
            "bounded fusion beat projection across at least four matches."
        ),
    )
    parser.add_argument(
        "--require-llm-event-options",
        action="store_true",
        help=(
            "Fail readiness until shadow event-conditioned options resolve "
            "events and observe later actions across at least four matches."
        ),
    )
    parser.add_argument(
        "--require-llm-event-option-values",
        action="store_true",
        help=(
            "Fail readiness until naturally matched continuation branches "
            "have calibrated policy-utility predictions across four matches."
        ),
    )
    parser.add_argument(
        "--require-llm-contrastive-faithfulness",
        action="store_true",
        help=(
            "Fail readiness until model-checked LLM explanations are "
            "directionally faithful across at least four matches."
        ),
    )
    parser.add_argument(
        "--require-llm-contrastive-repair",
        action="store_true",
        help=(
            "Fail readiness until one-shot explanation repairs improve "
            "world-model faithfulness across at least four matches."
        ),
    )
    parser.add_argument(
        "--require-llm-risk-certificates",
        action="store_true",
        help=(
            "Require realized conservative LLM chance-constraint certificates."
        ),
    )
    parser.add_argument(
        "--require-llm-distributional-decisions",
        action="store_true",
        help=(
            "Require calibrated member-utility distributions and faithful "
            "LLM distributional action claims."
        ),
    )
    parser.add_argument(
        "--require-llm-risk-preferences",
        action="store_true",
        help=(
            "Require realized, model-checked multi-horizon LLM risk "
            "preferences over calibrated predictive scenarios."
        ),
    )
    parser.add_argument(
        "--require-temporal-path-calibration",
        action="store_true",
        help=(
            "Require empirically coupled temporal paths to match or beat "
            "their same-marginal comonotonic benchmark on realized paths."
        ),
    )
    parser.add_argument(
        "--require-opponent-information-queries",
        action="store_true",
        help=(
            "Require calibrated, model-ranked LLM questions about observable "
            "opponent tactical information."
        ),
    )
    parser.add_argument(
        "--require-opponent-information-adaptation",
        action="store_true",
        help=(
            "Require realized, model-consistent LLM adaptations to prior "
            "opponent-information feedback to improve their declared metric."
        ),
    )
    parser.add_argument(
        "--require-llm-deliberation-focus",
        action="store_true",
        help=(
            "Require the coach LLM to follow the model-ranked deliberation "
            "agenda without extra, missing, or rejected optional contracts."
        ),
    )
    parser.add_argument(
        "--require-llm-deliberation-compute-value",
        action="store_true",
        help=(
            "Require balanced randomized evidence that the final shadow "
            "compute credit produces useful model-internal artifacts."
        ),
    )
    parser.add_argument(
        "--require-llm-task-encouragement",
        action="store_true",
        help=(
            "Require balanced post-action randomized task encouragement and "
            "at least 50 percent exact LLM portfolio compliance."
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
        require_opponent_meta_belief=args.require_opponent_meta_belief,
        require_opponent_change_detection=(
            args.require_opponent_change_detection
        ),
        require_opponent_response_model=args.require_opponent_response_model,
        require_two_step_trajectory_planning=(
            args.require_two_step_trajectory_planning
        ),
        require_llm_semantic_critic=args.require_llm_semantic_critic,
        require_llm_semantic_events=args.require_llm_semantic_events,
        require_learned_semantic_events=(
            args.require_learned_semantic_events
        ),
        require_llm_event_options=args.require_llm_event_options,
        require_llm_event_option_values=(
            args.require_llm_event_option_values
        ),
        require_llm_contrastive_faithfulness=(
            args.require_llm_contrastive_faithfulness
        ),
        require_llm_contrastive_repair=(
            args.require_llm_contrastive_repair
        ),
        require_llm_risk_certificates=args.require_llm_risk_certificates,
        require_llm_distributional_decisions=(
            args.require_llm_distributional_decisions
        ),
        require_llm_risk_preferences=args.require_llm_risk_preferences,
        require_temporal_path_calibration=(
            args.require_temporal_path_calibration
        ),
        require_opponent_information_queries=(
            args.require_opponent_information_queries
        ),
        require_opponent_information_adaptation=(
            args.require_opponent_information_adaptation
        ),
        require_llm_deliberation_focus=(
            args.require_llm_deliberation_focus
        ),
        require_llm_deliberation_compute_value=(
            args.require_llm_deliberation_compute_value
        ),
        require_llm_task_encouragement=(
            args.require_llm_task_encouragement
        ),
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

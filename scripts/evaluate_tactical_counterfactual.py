"""Matched-seed micro-simulation evaluation for a tactical policy choice."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.match_engine.match_micro_runner import run_match_micro_simulation  # noqa: E402
from src.match_engine.micro_config import MicroMatchConfig  # noqa: E402
from src.match_engine.tactical_catalog import (  # noqa: E402
    TACTICAL_PRESETS,
    resolve_tactical_preset,
)
from src.match_engine.tactical_profile import build_tactical_vector_for_agent  # noqa: E402
from src.simulation.fusion_audit import (  # noqa: E402
    evaluate_matched_seed_tactical_policy,
)
from src.simulation.counterfactual_evidence import (  # noqa: E402
    append_counterfactual_evidence,
)
from src.simulation.world_cup_runner import build_world_and_tournament  # noqa: E402


def _utility(summary) -> float:
    """Attack/territory utility; declared explicitly for reproducibility."""
    goal_diff = float(summary.goals_micro_home - summary.goals_micro_away)
    xg_diff = float(summary.micro_xg_home - summary.micro_xg_away)
    possession_edge = 2.0 * (float(summary.possession_home) - 0.5)
    return goal_diff + 0.35 * xg_diff + 0.15 * possession_edge


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare a tactical preset with balanced under paired seeds.",
    )
    parser.add_argument("--team", required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--baseline", default="balanced")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--root-seed", type=int, default=42)
    parser.add_argument("--match-seconds", type=float, default=900.0)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Do not append the result to persistent fusion evidence.",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "reports/evaluation/tactical_counterfactual.json"),
    )
    args = parser.parse_args()

    baseline = resolve_tactical_preset(args.baseline)
    treatment = resolve_tactical_preset(args.treatment)
    if baseline == treatment:
        parser.error("baseline and treatment resolve to the same tactical preset")
    engine, _, _ = build_world_and_tournament(str(ROOT))
    if args.team not in engine.agents or args.opponent not in engine.agents:
        available = ", ".join(sorted(engine.agents)[:12])
        parser.error(f"unknown team; examples: {available}")
    original_home = engine.agents[args.team]
    original_away = engine.agents[args.opponent]
    opponent_policy = build_tactical_vector_for_agent(original_away)
    config = MicroMatchConfig.fast_demo() if args.fast else MicroMatchConfig()
    config.use_micro_goals = True

    def run(seed: int, preset: str) -> float:
        home = copy.deepcopy(original_home)
        away = copy.deepcopy(original_away)
        summary = run_match_micro_simulation(
            home,
            away,
            goals_home=0,
            goals_away=0,
            xg_home=1.0,
            xg_away=1.0,
            referee={"strictness": 0.55, "bias_t1": 0.0},
            stage_pressure=0.5,
            drama_score=0.3,
            config=config,
            seed=seed,
            writeback_agents=False,
            neutral_venue=True,
            stage_name="tactical_counterfactual",
            match_seconds=float(args.match_seconds),
            tactical_override_home=TACTICAL_PRESETS[preset],
            tactical_override_away=opponent_policy,
        )
        return _utility(summary)

    report = evaluate_matched_seed_tactical_policy(
        run,
        baseline_preset=baseline,
        treatment_preset=treatment,
        root_seed=args.root_seed,
        samples=args.samples,
    )
    report.update({
        "team": args.team,
        "opponent": args.opponent,
        "match_seconds": float(args.match_seconds),
        "fast_mode": bool(args.fast),
        "root_seed": int(args.root_seed),
        "utility": "goal_diff + 0.35*xg_diff + 0.15*possession_edge",
        "opponent_policy": "held_fixed_from_current_agent_profile",
    })
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not args.no_register:
        append_counterfactual_evidence(ROOT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

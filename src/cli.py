"""Unified command line interface for Generative Football Society."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src import app
from src.product import ProductWorkspace, StudioConfig


def _set_flag(name: str, enabled: bool) -> None:
    os.environ[name] = "1" if enabled else "0"


def cmd_status(args: argparse.Namespace) -> int:
    stats, _, _ = app.load_status_table(args.base_dir)
    rows = stats.sort_values(by="final_status_score", ascending=False).head(args.top)
    cols = [
        "total_matches",
        "historical_base",
        "modern_power",
        "final_status_score",
        "global_exposure_gate",
        "sample_confidence",
        "tier",
    ]
    print(rows[cols].to_string())
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        rows[cols].to_csv(out)
        print(f"\nSaved: {out}")
    return 0


def cmd_micro(args: argparse.Namespace) -> int:
    summary = app.run_micro_match(
        args.home,
        args.away,
        base_dir=args.base_dir,
        seed=args.seed,
        fast=args.fast,
    )
    print(f"{args.home} vs {args.away}")
    print(f"Score: {summary.goals_micro_home}-{summary.goals_micro_away}")
    print(f"xG: {summary.micro_xg_home:.2f}-{summary.micro_xg_away:.2f}")
    print(
        "Passes: "
        f"{summary.passes_home} ({summary.pass_completion_home:.1%}) / "
        f"{summary.passes_away} ({summary.pass_completion_away:.1%})"
    )
    print(f"Shots: {summary.shots_home}-{summary.shots_away}")
    print(f"Possession home: {summary.possession_home:.1%}")
    return 0


def cmd_tournament(args: argparse.Namespace) -> int:
    _set_flag("MATCH_MICRO", args.micro)
    if args.micro:
        _set_flag("MATCH_MICRO_SCORE", args.micro_score)
    tournament = app.run_full_tournament(
        base_dir=args.base_dir,
        resume=args.resume,
        seed=args.seed,
        require_tactics=args.require_tactics,
    )
    if getattr(tournament, "final_result", None):
        champion = tournament.final_result.get("champion")
        if champion:
            print(f"Champion: {champion}")
    return 0


def cmd_monte_carlo(args: argparse.Namespace) -> int:
    rows = app.run_monte_carlo(args.num_sims, base_dir=args.base_dir, seed=args.seed)
    for idx, row in enumerate(rows[: args.top], start=1):
        print(
            f"{idx:2d}. {row['team']:<24} "
            f"{row['probability'] * 100:6.2f}% "
            f"wins={row['wins']:<5} status={row['status_score']:.1f}"
        )
    return 0


def cmd_studio_init(args: argparse.Namespace) -> int:
    workspace = ProductWorkspace.create(
        args.base_dir, StudioConfig(name=args.name, mode=args.mode, seed=args.seed),
    )
    print(f"Studio initialized: {workspace.config.name} ({workspace.config.mode})")
    print(f"Session: {workspace.session_path}")
    return 0


def cmd_studio_status(args: argparse.Namespace) -> int:
    import json
    print(json.dumps(ProductWorkspace.load(args.base_dir).status(), ensure_ascii=False, indent=2))
    return 0


def cmd_studio_match(args: argparse.Namespace) -> int:
    workspace = ProductWorkspace.load(args.base_dir)
    report = workspace.run_match(args.home, args.away, fast=args.fast)
    result = report["result"]
    print(f"{args.home} {result['score']['home']}-{result['score']['away']} {args.away}")
    print(f"xG {result['xg']['home']:.2f}-{result['xg']['away']:.2f}")
    print(f"Report: {report['report_path']}")
    print(f"Dashboard: {report['dashboard_path']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gfs", description="Generative Football Society")
    parser.add_argument("--base-dir", default=str(app.project_root()), help="Project root directory")
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="Compute team status rankings")
    p_status.add_argument("--top", type=int, default=20)
    p_status.add_argument("--out", default="")
    p_status.set_defaults(func=cmd_status)

    p_micro = sub.add_parser("micro", help="Run one micro match")
    p_micro.add_argument("--home", default="Brazil")
    p_micro.add_argument("--away", default="Argentina")
    p_micro.add_argument("--seed", type=int, default=42)
    p_micro.add_argument("--fast", action="store_true")
    p_micro.set_defaults(func=cmd_micro)

    p_tournament = sub.add_parser("tournament", help="Run the full tournament")
    p_tournament.add_argument("--resume", action="store_true")
    p_tournament.add_argument("--seed", type=int, default=None)
    p_tournament.add_argument("--micro", action="store_true", help="Enable micro match engine")
    p_tournament.add_argument(
        "--micro-score",
        action="store_true",
        help="Use micro physics goals as the official score when --micro is set",
    )
    p_tournament.add_argument("--require-tactics", action="store_true")
    p_tournament.set_defaults(func=cmd_tournament)

    p_mc = sub.add_parser("monte-carlo", help="Run lightweight tournament Monte Carlo")
    p_mc.add_argument("-n", "--num-sims", type=int, default=1000)
    p_mc.add_argument("--top", type=int, default=10)
    p_mc.add_argument("--seed", type=int, default=None)
    p_mc.set_defaults(func=cmd_monte_carlo)

    p_studio = sub.add_parser("studio", help="Run the cohesive GFS product workspace")
    studio_sub = p_studio.add_subparsers(dest="studio_command", required=True)
    p_studio_init = studio_sub.add_parser("init", help="Create or replace a studio session")
    p_studio_init.add_argument("--name", default="My GFS Studio")
    p_studio_init.add_argument("--mode", choices=("stable", "research", "cognitive"), default="stable")
    p_studio_init.add_argument("--seed", type=int, default=42)
    p_studio_init.set_defaults(func=cmd_studio_init)
    p_studio_status = studio_sub.add_parser("status", help="Show mode readiness and evidence")
    p_studio_status.set_defaults(func=cmd_studio_status)
    p_studio_match = studio_sub.add_parser("match", help="Run a match inside the current studio")
    p_studio_match.add_argument("--home", default="Brazil")
    p_studio_match.add_argument("--away", default="Argentina")
    p_studio_match.add_argument("--fast", action="store_true")
    p_studio_match.set_defaults(func=cmd_studio_match)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

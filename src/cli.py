"""Unified command line interface for Generative Football Society."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src import app
from src.product import (
    ProductControlPlane, ProductWorkspace, ProspectivePilot, StudioConfig,
)


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
        replace=args.replace,
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
    if not report["integrity"]["accepted"]:
        print("Integrity: degraded (" + ", ".join(report["integrity"]["blockers"]) + ")")
    return 0


def cmd_studio_run(args: argparse.Namespace) -> int:
    """Complete the normal product journey through one safe entry point."""
    session_path = Path(args.base_dir) / "data/persistence/product_session.json"
    if session_path.is_file() and not args.replace:
        workspace = ProductWorkspace.load(args.base_dir)
        requested = {
            "name": args.name, "mode": args.mode, "seed": args.seed,
        }
        actual = {
            "name": workspace.config.name,
            "mode": workspace.config.mode,
            "seed": workspace.config.seed,
        }
        mismatches = [
            key for key, value in requested.items()
            if value is not None and value != actual[key]
        ]
        if mismatches:
            raise ValueError(
                "existing Studio configuration differs for "
                + ", ".join(mismatches)
                + "; use --replace to reset explicitly"
            )
    else:
        workspace = ProductWorkspace.create(
            args.base_dir,
            StudioConfig(
                name=args.name or "My GFS Studio",
                mode=args.mode or "stable",
                seed=42 if args.seed is None else args.seed,
            ),
            replace=args.replace,
        )
    readiness = workspace.readiness()
    if not readiness["ready"]:
        raise RuntimeError(
            "Studio is blocked before simulation: "
            + ", ".join(readiness["blockers"])
        )
    report = workspace.run_match(args.home, args.away, fast=args.fast)
    result = report["result"]
    print(f"{args.home} {result['score']['home']}-{result['score']['away']} {args.away}")
    print(f"Report: {report['report_path']}")
    print(f"Dashboard: {report['dashboard_path']}")
    print("Workflow: review")
    return 0


def cmd_studio_pilot(args: argparse.Namespace) -> int:
    import json
    pilot = ProspectivePilot(ProductWorkspace.load(args.base_dir))
    result = pilot.execute() if args.execute else pilot.preflight()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not args.execute:
        print("Preflight only: no external provider calls were made.")
    else:
        print(f"Pilot report: {pilot.report_path}")
    return 0


def cmd_studio_provider(args: argparse.Namespace) -> int:
    """Validate local provider configuration without making a network call."""
    import json
    from src.simulation.llm_gateway import provider_preflight

    result = provider_preflight()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("Preflight only: no client was created and no external call was made.")
    return 0 if result["ready"] else 2


def cmd_studio_web(args: argparse.Namespace) -> int:
    """Serve the loopback-only Studio Web Beta."""
    from src.product import create_product_web_server

    server = create_product_web_server(
        args.base_dir, host=args.host, port=args.port,
    )
    print(f"GFS Studio Web Beta: http://{args.host}:{server.server_port}")
    print("Local access only. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nWeb Beta stopped.")
    finally:
        server.server_close()
    return 0


def cmd_studio_backup(args: argparse.Namespace) -> int:
    """Create an integrity-checked archive of the current Studio."""
    import json
    from src.product import ProductRecovery

    result = ProductRecovery(args.base_dir).create_backup(
        args.out, overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_studio_verify_backup(args: argparse.Namespace) -> int:
    import json
    from src.product import ProductRecovery

    result = ProductRecovery.verify_backup(args.bundle)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_studio_restore(args: argparse.Namespace) -> int:
    """Restore a verified archive behind an explicit replacement boundary."""
    import json
    from src.product import ProductRecovery

    result = ProductRecovery(args.base_dir).restore_backup(
        args.bundle, replace=args.replace,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_studio_jobs(args: argparse.Namespace) -> int:
    import json
    print(json.dumps(
        ProductControlPlane(args.base_dir).snapshot(), ensure_ascii=False, indent=2,
    ))
    return 0


def cmd_studio_stop_job(args: argparse.Namespace) -> int:
    path = ProductControlPlane(args.base_dir).request_stop(args.run_id)
    print(f"Cooperative stop requested: {path}")
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
    p_studio_init.add_argument(
        "--replace", action="store_true",
        help="Explicitly replace the existing persisted Studio session",
    )
    p_studio_init.set_defaults(func=cmd_studio_init)
    p_studio_status = studio_sub.add_parser("status", help="Show mode readiness and evidence")
    p_studio_status.set_defaults(func=cmd_studio_status)
    p_studio_match = studio_sub.add_parser("match", help="Run a match inside the current studio")
    p_studio_match.add_argument("--home", default="Brazil")
    p_studio_match.add_argument("--away", default="Argentina")
    p_studio_match.add_argument("--fast", action="store_true")
    p_studio_match.set_defaults(func=cmd_studio_match)
    p_studio_run = studio_sub.add_parser(
        "run", help="Create/load Studio, verify readiness, run, and report one match",
    )
    p_studio_run.add_argument("--name", default=None)
    p_studio_run.add_argument(
        "--mode", choices=("stable", "research", "cognitive"), default=None,
    )
    p_studio_run.add_argument("--seed", type=int, default=None)
    p_studio_run.add_argument("--home", default="Brazil")
    p_studio_run.add_argument("--away", default="Argentina")
    p_studio_run.add_argument("--fast", action="store_true")
    p_studio_run.add_argument(
        "--replace", action="store_true",
        help="Explicitly replace an existing Studio before running",
    )
    p_studio_run.set_defaults(func=cmd_studio_run)
    p_studio_pilot = studio_sub.add_parser(
        "pilot", help="Preflight or execute the frozen live-provider pilot",
    )
    p_studio_pilot.add_argument(
        "--execute", action="store_true",
        help="Make real provider calls; without this flag the command is read-only",
    )
    p_studio_pilot.set_defaults(func=cmd_studio_pilot)
    p_studio_provider = studio_sub.add_parser(
        "provider", help="Validate LLM provider configuration without any API call",
    )
    p_studio_provider.set_defaults(func=cmd_studio_provider)
    p_studio_web = studio_sub.add_parser(
        "web", help="Serve the loopback-only Studio Web Beta",
    )
    p_studio_web.add_argument("--host", default="127.0.0.1")
    p_studio_web.add_argument("--port", type=int, default=8765)
    p_studio_web.set_defaults(func=cmd_studio_web)
    p_studio_backup = studio_sub.add_parser(
        "backup", help="Create an integrity-checked Studio backup",
    )
    p_studio_backup.add_argument("--out", required=True)
    p_studio_backup.add_argument("--overwrite", action="store_true")
    p_studio_backup.set_defaults(func=cmd_studio_backup)
    p_studio_verify_backup = studio_sub.add_parser(
        "verify-backup", help="Verify a Studio backup without restoring it",
    )
    p_studio_verify_backup.add_argument("bundle")
    p_studio_verify_backup.set_defaults(func=cmd_studio_verify_backup)
    p_studio_restore = studio_sub.add_parser(
        "restore", help="Restore a verified Studio backup",
    )
    p_studio_restore.add_argument("bundle")
    p_studio_restore.add_argument(
        "--replace", action="store_true",
        help="Explicitly replace existing files referenced by the backup",
    )
    p_studio_restore.set_defaults(func=cmd_studio_restore)
    p_studio_jobs = studio_sub.add_parser(
        "jobs", help="Show training jobs and model/LLM decision artifacts",
    )
    p_studio_jobs.set_defaults(func=cmd_studio_jobs)
    p_studio_stop = studio_sub.add_parser(
        "stop-job", help="Request a safe stop at the next epoch boundary",
    )
    p_studio_stop.add_argument("run_id")
    p_studio_stop.set_defaults(func=cmd_studio_stop_job)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

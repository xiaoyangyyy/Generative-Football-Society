from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

import pandas as pd

from gfs_core.contracts import PluginContext, PluginMetadata


class CliService:
    def __init__(self, snapshot, tournament, presenter, micro_workflow=None) -> None:
        self._snapshot = snapshot
        self._tournament = tournament
        self._presenter = presenter
        self._micro_workflow = micro_workflow

    def execute(self, arguments: Sequence[str]) -> str:
        parser = _build_parser()
        args = parser.parse_args(list(arguments))
        if args.command == "status":
            rows = self._presenter.status(self._snapshot, args.top)
        elif args.command == "monte-carlo":
            results = self._tournament.monte_carlo(
                self._snapshot.table,
                simulations=args.simulations,
                seed=args.seed,
            )
            rows = self._presenter.monte_carlo(results)
        else:
            if self._micro_workflow is None:
                raise ValueError("micro match workflow is unavailable")
            summary = self._micro_workflow.run_micro_match(
                args.home,
                args.away,
                seed=args.seed,
                fast=args.fast,
            )
            rows = [self._presenter.micro_match(summary)]
        if args.json:
            return json.dumps(rows, ensure_ascii=True, indent=2, allow_nan=False)
        return pd.DataFrame(rows).to_string(index=False)


class CliPlugin:
    metadata = PluginMetadata(
        id="gfs.core.presentation.cli",
        version="0.1.0",
        api=1,
        contract="gfs_core.contracts.presentation",
        provides=("cli",),
        requires=(
            "gfs.core.presentation.json",
            "gfs.core.tournament.advancement",
            "gfs.core.match",
        ),
    )

    def __init__(self, micro_workflow=None) -> None:
        self._micro_workflow = micro_workflow

    def activate(self, context: PluginContext) -> None:
        context.provide(
            "cli",
            CliService(
                context.require("sample-confidence"),
                context.require("monte-carlo"),
                context.require("json-reports"),
                self._micro_workflow,
            ),
        )

    def deactivate(self) -> None:
        pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gfs-core")
    commands = parser.add_subparsers(dest="command", required=True)

    status = commands.add_parser("status")
    status.add_argument("--top", type=int, default=20)
    status.add_argument("--json", action="store_true")

    monte_carlo = commands.add_parser("monte-carlo")
    monte_carlo.add_argument("-n", "--simulations", type=int, default=1000)
    monte_carlo.add_argument("--seed", type=int, default=42)
    monte_carlo.add_argument("--json", action="store_true")

    micro = commands.add_parser("micro")
    micro.add_argument("--home", default="Brazil")
    micro.add_argument("--away", default="Argentina")
    micro.add_argument("--fast", action="store_true")
    micro.add_argument("--seed", type=int, default=42)
    micro.add_argument("--json", action="store_true")
    return parser

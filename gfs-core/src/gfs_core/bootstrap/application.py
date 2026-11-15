from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from gfs_core.bootstrap.builtin_plugins import (
    build_cli_registry,
    build_micro_registry,
    build_monte_carlo_registry,
    build_status_registry,
)
from gfs_core.contracts.micro_config import MicroMatchConfig
from gfs_core.contracts.status import StatusSnapshot
from gfs_core.kernel import PluginKernel

_STATUS_PLUGIN = "gfs.core.status.confidence"
_MONTE_CARLO_PLUGIN = "gfs.core.tournament.advancement"
_CLI_PLUGIN = "gfs.core.presentation.cli"


class CoreApplication:
    def __init__(
        self,
        project_root: str | Path | None = None,
        *,
        current_year: int = 2026,
        confidence_threshold: int = 80,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.project_root = Path(project_root) if project_root else discover_project_root()
        self.workspace_root = workspace_root
        self.current_year = current_year
        self.confidence_threshold = confidence_threshold

    def load_status(self) -> StatusSnapshot:
        registry = build_status_registry(
            self.project_root,
            current_year=self.current_year,
            confidence_threshold=self.confidence_threshold,
        )
        kernel = PluginKernel(registry)
        try:
            kernel.start([_STATUS_PLUGIN])
            return kernel.context.require("sample-confidence")
        finally:
            kernel.stop()

    def run_monte_carlo(
        self, *, simulations: int = 1000, seed: int = 42,
    ) -> list[dict[str, object]]:
        registry = build_monte_carlo_registry(
            self.project_root,
            current_year=self.current_year,
            confidence_threshold=self.confidence_threshold,
        )
        kernel = PluginKernel(registry)
        try:
            kernel.start([_MONTE_CARLO_PLUGIN])
            stats = kernel.context.require("sample-confidence").table
            return kernel.context.require("monte-carlo").monte_carlo(
                stats, simulations=simulations, seed=seed,
            )
        finally:
            kernel.stop()

    def run_micro_match(
        self,
        home: str = "Brazil",
        away: str = "Argentina",
        *,
        seed: int = 42,
        fast: bool = False,
        config: MicroMatchConfig | None = None,
        stage_name: str = "match",
        authoritative_clock: bool = False,
        checkpoint: str | None = None,
        resume: str | None = None,
        checkpoint_interval: int = 0,
        stop_at_tick: int | None = None,
        trace_sink=None,
        trace_match_id: str | None = None,
        score_mode: str | None = None,
        memory_as_of: int | None = None,
    ):
        config = config or (
            MicroMatchConfig.fast_demo() if fast else MicroMatchConfig()
        )
        checkpoint = checkpoint or resume
        registry = build_micro_registry(
            self.project_root,
            current_year=self.current_year,
            confidence_threshold=self.confidence_threshold,
            config=config,
            workspace_root=self.workspace_root,
        )
        kernel = PluginKernel(registry)
        try:
            selected = ["gfs.core.match.workflow"]
            if checkpoint is not None or resume is not None:
                selected.append("gfs.core.persistence.micro-checkpointing")
            kernel.start(selected)
            checkpoint_service = (
                kernel.context.require("micro-checkpoints")
                if checkpoint is not None or resume is not None else None
            )
            resume_document = (
                checkpoint_service.load(resume) if resume is not None else None
            )
            return kernel.context.require("match-workflow").run(
                home,
                away,
                seed=seed,
                config=config,
                stage_name=stage_name,
                authoritative_clock=authoritative_clock,
                score_mode=score_mode,
                memory_as_of=memory_as_of,
                resume_checkpoint=resume_document,
                checkpoint_sink=(
                    lambda document: checkpoint_service.save(checkpoint, document)
                    if checkpoint is not None else None
                ),
                checkpoint_interval=checkpoint_interval,
                stop_at_tick=stop_at_tick,
                trace_sink=trace_sink,
                trace_match_id=trace_match_id,
            )
        finally:
            kernel.stop()

    def run_cli(self, arguments: Sequence[str]) -> str:
        registry = build_cli_registry(
            self.project_root,
            current_year=self.current_year,
            confidence_threshold=self.confidence_threshold,
            micro_workflow=self,
        )
        kernel = PluginKernel(registry)
        try:
            kernel.start([_CLI_PLUGIN])
            return kernel.context.require("cli").execute(arguments)
        finally:
            kernel.stop()


def discover_project_root() -> Path:
    configured = os.environ.get("GFS_PROJECT_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "data" / "raw" / "results.csv").is_file():
            return candidate
    return Path(__file__).resolve().parents[3]

from __future__ import annotations

from pathlib import Path

from gfs_core.bootstrap.bundles import BundlePlugin
from gfs_core.contracts.micro_config import MicroMatchConfig
from gfs_core.contracts import PluginMetadata
from gfs_core.kernel import PluginRegistry
from gfs_core.plugins.data.squads.rosters import RostersPlugin
from gfs_core.plugins.data.squads.coaches import CoachesPlugin
from gfs_core.plugins.data.squads.tactics import TacticsPlugin
from gfs_core.plugins.data.sources.results import ResultsPlugin
from gfs_core.plugins.match.actions.plugin import ActionsPlugin
from gfs_core.plugins.match.physics.plugin import PhysicsPlugin
from gfs_core.plugins.match.plugin import MatchPlugin
from gfs_core.plugins.match.workflow import MatchWorkflowPlugin
from gfs_core.plugins.match.scoring.plugin import ScoringPlugin
from gfs_core.plugins.match.spatial.plugin import SpatialPlugin
from gfs_core.plugins.presentation.cli.commands import CliPlugin
from gfs_core.plugins.presentation.json.plugin import JsonPresentationPlugin
from gfs_core.plugins.society.agents.agent import AgentsPlugin
from gfs_core.plugins.society.psychology.plugin import PsychologyPlugin
from gfs_core.plugins.society.memory.service import MemoryPlugin
from gfs_core.plugins.status.confidence import ConfidencePlugin
from gfs_core.plugins.status.ranking import RankingPlugin
from gfs_core.plugins.tournament.advancement.plugin import AdvancementPlugin
from gfs_core.plugins.tournament.scheduling.groups import WORLD_CUP_2026_GROUPS
from gfs_core.plugins.tournament.checkpointing.service import CheckpointPlugin
from gfs_core.plugins.tournament.runner import TournamentPlugin
from gfs_core.plugins.persistence.recovery.service import RecoveryPlugin
from gfs_core.plugins.persistence.repositories.workspace import WorkspacePlugin
from gfs_core.plugins.persistence.checkpointing.micro import MicroCheckpointPlugin
from gfs_core.plugins.persistence.transactions.match import (
    MatchTransactionService,
    TransactionPlugin,
)
from gfs_core.plugins.tournament.scheduling.plugin import SchedulingPlugin


def build_status_registry(
    project_root: str | Path,
    *,
    current_year: int = 2026,
    confidence_threshold: int = 80,
) -> PluginRegistry:
    return PluginRegistry([
        ResultsPlugin(project_root),
        RankingPlugin(current_year=current_year),
        ConfidencePlugin(threshold=confidence_threshold),
    ])


def build_monte_carlo_registry(
    project_root: str | Path,
    *,
    current_year: int = 2026,
    confidence_threshold: int = 80,
) -> PluginRegistry:
    status = build_status_registry(
        project_root,
        current_year=current_year,
        confidence_threshold=confidence_threshold,
    )
    return PluginRegistry([
        *status,
        ScoringPlugin(),
        SchedulingPlugin(),
        AdvancementPlugin(),
    ])


def build_micro_registry(
    project_root: str | Path,
    *,
    current_year: int = 2026,
    confidence_threshold: int = 80,
    config: MicroMatchConfig | None = None,
    workspace_root: str | Path | None = None,
) -> PluginRegistry:
    status = build_status_registry(
        project_root,
        current_year=current_year,
        confidence_threshold=confidence_threshold,
    )
    workspace = WorkspacePlugin(workspace_root or project_root)
    return PluginRegistry([
        *status,
        RostersPlugin(project_root),
        TacticsPlugin(project_root),
        SpatialPlugin(),
        AgentsPlugin(),
        PhysicsPlugin(),
        ActionsPlugin(config, base_dir=project_root),
        PsychologyPlugin(),
        ScoringPlugin(),
        MatchPlugin(),
        MatchWorkflowPlugin(),
        MemoryPlugin(workspace_root or project_root),
        workspace,
        MicroCheckpointPlugin(workspace._repository),
    ])


def build_cli_registry(
    project_root: str | Path,
    *,
    current_year: int = 2026,
    confidence_threshold: int = 80,
    micro_workflow=None,
) -> PluginRegistry:
    micro = build_micro_registry(
        project_root,
        current_year=current_year,
        confidence_threshold=confidence_threshold,
    )
    return PluginRegistry([
        *micro,
        SchedulingPlugin(),
        AdvancementPlugin(),
        JsonPresentationPlugin(),
        CliPlugin(micro_workflow),
    ])


def build_stable_registry(project_root: str | Path, *, workspace_root: str | Path | None = None) -> PluginRegistry:
    registry = build_micro_registry(project_root, workspace_root=workspace_root)
    workspace = registry.get("gfs.core.persistence.workspace")
    checkpoint = CheckpointPlugin(workspace._repository)
    registry.register(TransactionPlugin(workspace._repository))
    registry.register(RecoveryPlugin(workspace._repository))
    registry.register(checkpoint)
    registry.register(SchedulingPlugin())
    registry.register(AdvancementPlugin())
    registry.register(TournamentPlugin(
        WORLD_CUP_2026_GROUPS,
        MatchTransactionService(workspace._repository),
        checkpoint,
    ))
    registry.register(CoachesPlugin(project_root))
    registry.register(JsonPresentationPlugin())
    registry.register(CliPlugin())
    for metadata, capability, bindings in _bundle_definitions():
        registry.register(BundlePlugin(metadata, capability, bindings))
    return registry


def _bundle_definitions():
    return [
        (
            PluginMetadata(
                "gfs.core.data", "0.1.0", 1, "gfs_core.contracts.data",
                provides=("data-bundle",),
                requires=("gfs.core.data.results", "gfs.core.data.rosters", "gfs.core.data.coaches", "gfs.core.data.tactics"),
            ),
            "data-bundle",
            {"results": "results", "rosters": "rosters", "coaches": "coaches", "tactics": "tactics"},
        ),
        (
            PluginMetadata(
                "gfs.core.status", "0.1.0", 1, "gfs_core.contracts.status",
                provides=("status-bundle",),
                requires=("gfs.core.status.ranking", "gfs.core.status.confidence"),
            ),
            "status-bundle",
            {"snapshot": "sample-confidence"},
        ),
        (
            PluginMetadata(
                "gfs.core.society", "0.1.0", 1, "gfs_core.contracts.society",
                provides=("society-bundle",),
                requires=("gfs.core.society.agents", "gfs.core.society.psychology", "gfs.core.society.memory"),
            ),
            "society-bundle",
            {"agents": "agents", "psychology": "psychology", "memory": "memory"},
        ),
        (
            PluginMetadata(
                "gfs.core.persistence", "0.1.0", 1, "gfs_core.contracts.persistence",
                provides=("persistence-bundle",),
                requires=("gfs.core.persistence.workspace", "gfs.core.persistence.transactions", "gfs.core.persistence.recovery", "gfs.core.persistence.micro-checkpointing"),
            ),
            "persistence-bundle",
            {"workspace": "workspace-repository", "transactions": "match-transactions", "recovery": "recovery", "micro_checkpoints": "micro-checkpoints"},
        ),
        (
            PluginMetadata(
                "gfs.core.presentation", "0.1.0", 1, "gfs_core.contracts.presentation",
                provides=("presentation-bundle",),
                requires=("gfs.core.presentation.cli", "gfs.core.presentation.json"),
            ),
            "presentation-bundle",
            {"cli": "cli", "json": "json-reports"},
        ),
    ]

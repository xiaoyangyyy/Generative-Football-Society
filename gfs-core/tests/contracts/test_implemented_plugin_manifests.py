from pathlib import Path

from gfs_core.kernel import load_manifest
from gfs_core.plugins.data.squads.rosters import RostersPlugin
from gfs_core.plugins.data.squads.tactics import TacticsPlugin
from gfs_core.plugins.data.sources.results import ResultsPlugin
from gfs_core.plugins.match.actions.plugin import ActionsPlugin
from gfs_core.plugins.match.physics.plugin import PhysicsPlugin
from gfs_core.plugins.match.plugin import MatchPlugin
from gfs_core.plugins.match.workflow import MatchWorkflowPlugin
from gfs_core.plugins.match.scoring.plugin import ScoringPlugin
from gfs_core.plugins.match.spatial.plugin import SpatialPlugin
from gfs_core.plugins.persistence.recovery.service import RecoveryPlugin
from gfs_core.plugins.persistence.repositories.workspace import WorkspacePlugin
from gfs_core.plugins.persistence.checkpointing.micro import MicroCheckpointPlugin
from gfs_core.plugins.persistence.transactions.match import TransactionPlugin
from gfs_core.plugins.presentation.cli.commands import CliPlugin
from gfs_core.plugins.presentation.json.plugin import JsonPresentationPlugin
from gfs_core.plugins.society.agents.agent import AgentsPlugin
from gfs_core.plugins.society.psychology.plugin import PsychologyPlugin
from gfs_core.plugins.status.confidence import ConfidencePlugin
from gfs_core.plugins.status.ranking import RankingPlugin
from gfs_core.plugins.tournament.advancement.plugin import AdvancementPlugin
from gfs_core.plugins.tournament.checkpointing.service import CheckpointPlugin
from gfs_core.plugins.tournament.runner import TournamentPlugin
from gfs_core.plugins.tournament.scheduling.plugin import SchedulingPlugin
from gfs_core.plugins.data.squads.coaches import CoachesPlugin
from gfs_core.plugins.society.memory.service import MemoryPlugin

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_ROOT = PROJECT_ROOT / "src" / "gfs_core" / "plugins"


def test_implemented_plugins_match_their_published_manifests():
    expected = {
        ResultsPlugin.metadata: PLUGIN_ROOT / "data" / "sources" / "plugin.toml",
        RostersPlugin.metadata: PLUGIN_ROOT / "data" / "squads" / "rosters.plugin.toml",
        TacticsPlugin.metadata: PLUGIN_ROOT / "data" / "squads" / "tactics.plugin.toml",
        RankingPlugin.metadata: PLUGIN_ROOT / "status" / "ranking.plugin.toml",
        ConfidencePlugin.metadata: PLUGIN_ROOT / "status" / "confidence.plugin.toml",
        ScoringPlugin.metadata: PLUGIN_ROOT / "match" / "scoring" / "plugin.toml",
        SpatialPlugin.metadata: PLUGIN_ROOT / "match" / "spatial" / "plugin.toml",
        PhysicsPlugin.metadata: PLUGIN_ROOT / "match" / "physics" / "plugin.toml",
        ActionsPlugin.metadata: PLUGIN_ROOT / "match" / "actions" / "plugin.toml",
        MatchPlugin.metadata: PLUGIN_ROOT / "match" / "plugin.toml",
        MatchWorkflowPlugin.metadata: PLUGIN_ROOT / "match" / "workflow.plugin.toml",
        AgentsPlugin.metadata: PLUGIN_ROOT / "society" / "agents" / "plugin.toml",
        PsychologyPlugin.metadata: PLUGIN_ROOT / "society" / "psychology" / "plugin.toml",
        CoachesPlugin.metadata: PLUGIN_ROOT / "data" / "squads" / "coaches.plugin.toml",
        MemoryPlugin.metadata: PLUGIN_ROOT / "society" / "memory" / "plugin.toml",
        SchedulingPlugin.metadata: PLUGIN_ROOT / "tournament" / "scheduling.plugin.toml",
        AdvancementPlugin.metadata: PLUGIN_ROOT / "tournament" / "advancement.plugin.toml",
        CheckpointPlugin.metadata: PLUGIN_ROOT / "tournament" / "checkpointing.plugin.toml",
        TournamentPlugin.metadata: PLUGIN_ROOT / "tournament" / "plugin.toml",
        WorkspacePlugin.metadata: PLUGIN_ROOT / "persistence" / "workspace.plugin.toml",
        TransactionPlugin.metadata: PLUGIN_ROOT / "persistence" / "transactions.plugin.toml",
        RecoveryPlugin.metadata: PLUGIN_ROOT / "persistence" / "recovery.plugin.toml",
        MicroCheckpointPlugin.metadata: PLUGIN_ROOT / "persistence" / "micro_checkpointing.plugin.toml",
        CliPlugin.metadata: PLUGIN_ROOT / "presentation" / "cli.plugin.toml",
        JsonPresentationPlugin.metadata: PLUGIN_ROOT / "presentation" / "json.plugin.toml",
    }
    for metadata, path in expected.items():
        assert load_manifest(path) == metadata

from pathlib import Path

from gfs_core.kernel import load_manifest
from gfs_studio.plugins.evidence.receipts.store import ReceiptsPlugin
from gfs_studio.plugins.evidence.registry.identity import EvidenceRegistryPlugin
from gfs_studio.plugins.evidence.release.gates import ReleasePlugin
from gfs_studio.plugins.evidence.plugin import EvidenceBundlePlugin
from gfs_studio.plugins.research.experiments.protocol import ExperimentsPlugin
from gfs_studio.plugins.research.calibration.study import CalibrationPlugin
from gfs_studio.plugins.research.ablation.study import AblationPlugin
from gfs_studio.plugins.research.reproduction.verifier import ReproductionPlugin
from gfs_studio.plugins.research.plugin import ResearchBundlePlugin
from gfs_studio.plugins.world_model.plugin import LatentWorldModelPlugin
from gfs_studio.plugins.training.trainers.plugin import TrainersPlugin
from gfs_studio.plugins.training.datasets.plugin import DatasetPlugin
from gfs_studio.plugins.training.datasets.trace_export import TraceExportPlugin
from gfs_studio.plugins.training.evaluation.plugin import EvaluationPlugin
from gfs_studio.plugins.training.checkpoints.registry import CheckpointPlugin
from gfs_studio.plugins.training.plugin import TrainingBundlePlugin
from gfs_studio.plugins.cognition.gateway.service import CognitionPlugin
from gfs_studio.plugins.cognition.reflection.service import ReflectionPlugin
from gfs_studio.plugins.cognition.plugin import CognitionBundlePlugin
from gfs_studio.plugins.product.studio.workflow import ProductStudioPlugin
from gfs_studio.plugins.web.gateway.server import WebPlugin
from gfs_studio.plugins.web.api.routes import ApiPlugin
from gfs_studio.plugins.web.views.pages import ViewsPlugin
from gfs_studio.plugins.web.authentication.sessions import AuthenticationPlugin
from gfs_studio.plugins.web.plugin import WebBundlePlugin
from gfs_studio.plugins.world_model.bundle import WorldModelBundlePlugin
from gfs_studio.plugins.cognition.roles.service import RolesPlugin
from gfs_studio.plugins.product.manager.workspace import ProductManagerPlugin
from gfs_studio.plugins.product.recovery.service import ProductRecoveryPlugin
from gfs_studio.plugins.product.tasks.service import TaskPlugin
from gfs_studio.plugins.product.comparison.service import ComparisonPlugin
from gfs_studio.plugins.product.plugin import ProductBundlePlugin
from gfs_studio.plugins.participants.sessions.service import SessionsPlugin
from gfs_studio.plugins.participants.delivery.packets import DeliveryPlugin
from gfs_studio.plugins.participants.scoring.seal import ScoringPlugin
from gfs_studio.plugins.participants.plugin import ParticipantsBundlePlugin
from gfs_studio.plugins.world_model.planning.service import PlanningPlugin
from gfs_studio.plugins.world_model.policies.service import PoliciesPlugin
from gfs_studio.plugins.world_model.frame.service import FrameModelPlugin

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_ROOT = PROJECT_ROOT / "src" / "gfs_studio" / "plugins"


def test_implemented_studio_plugins_match_manifests():
    expected = {
        ReceiptsPlugin.metadata: PLUGIN_ROOT / "evidence" / "receipts.plugin.toml",
        EvidenceRegistryPlugin.metadata: PLUGIN_ROOT / "evidence" / "registry.plugin.toml",
        ReleasePlugin.metadata: PLUGIN_ROOT / "evidence" / "release.plugin.toml",
        EvidenceBundlePlugin.metadata: PLUGIN_ROOT / "evidence" / "plugin.toml",
        ExperimentsPlugin.metadata: PLUGIN_ROOT / "research" / "experiments.plugin.toml",
        CalibrationPlugin.metadata: PLUGIN_ROOT / "research" / "calibration.plugin.toml",
        AblationPlugin.metadata: PLUGIN_ROOT / "research" / "ablation.plugin.toml",
        ReproductionPlugin.metadata: PLUGIN_ROOT / "research" / "reproduction.plugin.toml",
        ResearchBundlePlugin.metadata: PLUGIN_ROOT / "research" / "plugin.toml",
        LatentWorldModelPlugin.metadata: PLUGIN_ROOT / "world_model" / "latent.plugin.toml",
        TrainersPlugin.metadata: PLUGIN_ROOT / "training" / "trainers.plugin.toml",
        DatasetPlugin.metadata: PLUGIN_ROOT / "training" / "datasets.plugin.toml",
        TraceExportPlugin.metadata: PLUGIN_ROOT / "training" / "trace_export.plugin.toml",
        EvaluationPlugin.metadata: PLUGIN_ROOT / "training" / "evaluation.plugin.toml",
        CheckpointPlugin.metadata: PLUGIN_ROOT / "training" / "checkpoints.plugin.toml",
        TrainingBundlePlugin.metadata: PLUGIN_ROOT / "training" / "plugin.toml",
        CognitionPlugin.metadata: PLUGIN_ROOT / "cognition" / "gateway.plugin.toml",
        ReflectionPlugin.metadata: PLUGIN_ROOT / "cognition" / "reflection.plugin.toml",
        CognitionBundlePlugin.metadata: PLUGIN_ROOT / "cognition" / "plugin.toml",
        ProductStudioPlugin.metadata: PLUGIN_ROOT / "product" / "studio.plugin.toml",
        WebPlugin.metadata: PLUGIN_ROOT / "web" / "gateway.plugin.toml",
        ApiPlugin.metadata: PLUGIN_ROOT / "web" / "api.plugin.toml",
        ViewsPlugin.metadata: PLUGIN_ROOT / "web" / "views.plugin.toml",
        AuthenticationPlugin.metadata: PLUGIN_ROOT / "web" / "authentication.plugin.toml",
        WebBundlePlugin.metadata: PLUGIN_ROOT / "web" / "plugin.toml",
        WorldModelBundlePlugin.metadata: PLUGIN_ROOT / "world_model" / "plugin.toml",
        RolesPlugin.metadata: PLUGIN_ROOT / "cognition" / "roles.plugin.toml",
        ProductManagerPlugin.metadata: PLUGIN_ROOT / "product" / "manager.plugin.toml",
        ProductRecoveryPlugin.metadata: PLUGIN_ROOT / "product" / "recovery.plugin.toml",
        TaskPlugin.metadata: PLUGIN_ROOT / "product" / "tasks" / "plugin.toml",
        ComparisonPlugin.metadata: PLUGIN_ROOT / "product" / "comparison.plugin.toml",
        ProductBundlePlugin.metadata: PLUGIN_ROOT / "product" / "plugin.toml",
        SessionsPlugin.metadata: PLUGIN_ROOT / "participants" / "sessions.plugin.toml",
        DeliveryPlugin.metadata: PLUGIN_ROOT / "participants" / "delivery.plugin.toml",
        ScoringPlugin.metadata: PLUGIN_ROOT / "participants" / "scoring.plugin.toml",
        ParticipantsBundlePlugin.metadata: PLUGIN_ROOT / "participants" / "plugin.toml",
        PlanningPlugin.metadata: PLUGIN_ROOT / "world_model" / "planning.plugin.toml",
        PoliciesPlugin.metadata: PLUGIN_ROOT / "world_model" / "policies.plugin.toml",
        FrameModelPlugin.metadata: PLUGIN_ROOT / "world_model" / "frame.plugin.toml",
    }
    assert set(expected.values()) == set(PLUGIN_ROOT.rglob("*.toml"))
    for metadata, path in expected.items():
        assert load_manifest(path) == metadata

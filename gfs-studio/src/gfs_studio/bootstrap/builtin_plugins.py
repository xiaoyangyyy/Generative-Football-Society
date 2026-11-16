from pathlib import Path

from gfs_core.bootstrap.builtin_plugins import build_micro_registry
from gfs_core.kernel import PluginRegistry
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
from gfs_studio.plugins.world_model.runtime import WorldModelRuntime
from gfs_studio.plugins.world_model.planning.service import PlanningPlugin
from gfs_studio.plugins.world_model.policies.service import PoliciesPlugin
from gfs_studio.plugins.world_model.frame.service import FrameModelPlugin
from gfs_studio.plugins.training.trainers.plugin import TrainersPlugin
from gfs_studio.plugins.training.datasets.plugin import DatasetPlugin
from gfs_studio.plugins.training.datasets.trace_export import TraceExportPlugin
from gfs_studio.plugins.training.evaluation.plugin import EvaluationPlugin
from gfs_studio.plugins.training.checkpoints.registry import CheckpointPlugin
from gfs_studio.plugins.training.plugin import TrainingBundlePlugin
from gfs_studio.plugins.cognition.gateway.service import CognitionPlugin
from gfs_studio.plugins.cognition.reflection.service import ReflectionPlugin
from gfs_studio.plugins.cognition.plugin import CognitionBundlePlugin
from gfs_studio.plugins.product.studio.workflow import (
    ProductStudioPlugin,
    ProductWorkflows,
)
from gfs_studio.plugins.product.manager.workspace import (
    ManagerWorkspace,
    ProductManagerPlugin,
)
from gfs_studio.plugins.product.recovery.service import ProductRecoveryPlugin
from gfs_studio.plugins.product.tasks.service import TaskPlugin
from gfs_studio.plugins.product.comparison.service import ComparisonPlugin
from gfs_studio.plugins.product.plugin import ProductBundlePlugin
from gfs_studio.plugins.cognition.roles.service import RoleService, RolesPlugin
from gfs_studio.plugins.participants.sessions.service import SessionsPlugin
from gfs_studio.plugins.participants.delivery.packets import DeliveryPlugin
from gfs_studio.plugins.participants.scoring.seal import ScoringPlugin
from gfs_studio.plugins.participants.plugin import ParticipantsBundlePlugin
from gfs_studio.plugins.web.gateway.server import WebPlugin
from gfs_studio.plugins.web.api.routes import ApiPlugin
from gfs_studio.plugins.web.views.pages import ViewsPlugin
from gfs_studio.plugins.web.authentication.sessions import AuthenticationPlugin
from gfs_studio.plugins.web.plugin import WebBundlePlugin
from gfs_studio.plugins.world_model.bundle import WorldModelBundlePlugin


def build_evidence_registry(project_root: str | Path) -> PluginRegistry:
    evidence = ReceiptsPlugin(project_root)
    return PluginRegistry([
        evidence,
        ExperimentsPlugin(evidence._store),
        SessionsPlugin(project_root),
        ScoringPlugin(),
        EvidenceRegistryPlugin(),
        ReleasePlugin(),
        EvidenceBundlePlugin(),
    ])


def build_research_registry(project_root: str | Path) -> PluginRegistry:
    return PluginRegistry([
        *build_micro_registry(project_root.parent / "gfs-core"),
        *build_evidence_registry(project_root),
        LatentWorldModelPlugin(),
        EvaluationPlugin(),
        CalibrationPlugin(),
        AblationPlugin(),
        ReproductionPlugin(),
        ResearchBundlePlugin(),
    ])


def build_world_model_registry(project_root: str | Path) -> PluginRegistry:
    del project_root
    return PluginRegistry([LatentWorldModelPlugin()])


def build_training_registry(project_root: str | Path, *, core_root: str | Path | None = None) -> PluginRegistry:
    return PluginRegistry([
        *build_micro_registry(core_root or project_root.parent / "gfs-core"),
        ReceiptsPlugin(project_root),
        LatentWorldModelPlugin(),
        DatasetPlugin(),
        TraceExportPlugin(),
        TrainersPlugin(project_root),
        EvaluationPlugin(),
        CheckpointPlugin(project_root),
        TrainingBundlePlugin(),
    ])


def build_cognition_registry(project_root: str | Path) -> PluginRegistry:
    cognition = CognitionPlugin()
    return PluginRegistry([
        ReceiptsPlugin(project_root),
        cognition,
        RolesPlugin(cognition.service),
        ReflectionPlugin(project_root),
        CognitionBundlePlugin(),
    ])


def build_participants_registry(project_root: str | Path) -> PluginRegistry:
    return PluginRegistry([
        ReceiptsPlugin(project_root),
        SessionsPlugin(project_root),
        DeliveryPlugin(),
        ScoringPlugin(),
        ParticipantsBundlePlugin(),
    ])


def build_product_registry(project_root: str | Path) -> PluginRegistry:
    from .application import StudioApplication

    core_root = project_root.parent / "gfs-core"
    application = StudioApplication(project_root)
    workflows = ProductWorkflows(
        project_root, overview=application.overview, closer=application.close,
    )
    cognition = CognitionPlugin()
    roles = RoleService(cognition.service)
    manager = ManagerWorkspace(workflows, roles)
    recovery = ProductRecoveryPlugin(project_root)
    tasks = TaskPlugin(workflows)
    return PluginRegistry([
        *build_micro_registry(core_root),
        ReceiptsPlugin(project_root),
        ProductStudioPlugin(project_root, workflows),
        cognition,
        RolesPlugin(cognition.service, roles),
        ProductManagerPlugin(manager),
        recovery,
        tasks,
        ComparisonPlugin(),
        ProductBundlePlugin(),
    ])


def build_web_registry(project_root: str | Path) -> PluginRegistry:
    product = build_product_registry(project_root)
    workflows = product.get("gfs.studio.product.tasks").service.workflows
    return PluginRegistry([
        *product,
        LatentWorldModelPlugin(),
        WebPlugin(workflows),
        ApiPlugin(),
        ViewsPlugin(),
        AuthenticationPlugin(),
        WebBundlePlugin(),
    ])


def build_studio_registry(project_root: str | Path) -> PluginRegistry:
    registry = PluginRegistry()
    for factory in (
        build_planning_registry,
        build_web_registry,
        build_training_registry,
        build_research_registry,
        build_cognition_registry,
        build_participants_registry,
    ):
        for plugin in factory(project_root):
            if plugin.metadata.id not in {item.id for item in registry.metadata}:
                registry.register(plugin)
    return registry


def build_recovery_registry(project_root: str | Path) -> PluginRegistry:
    return PluginRegistry([
        *build_micro_registry(project_root.parent / "gfs-core"),
        ReceiptsPlugin(project_root),
        ProductStudioPlugin(project_root),
        ProductRecoveryPlugin(project_root),
    ])


def build_task_registry(project_root: str | Path) -> PluginRegistry:
    workflows = ProductWorkflows(project_root)
    return PluginRegistry([
        *build_micro_registry(project_root.parent / "gfs-core"),
        ReceiptsPlugin(project_root),
        ProductStudioPlugin(project_root, workflows),
        TaskPlugin(workflows, cancel_on_close=False),
    ])


def build_planning_registry(project_root: str | Path) -> PluginRegistry:
    runtime = WorldModelRuntime.load(
        Path(project_root) / "data" / "world-model" / "latent_wm_rollout_calibrated_candidate.pt"
    )
    planning = PlanningPlugin(runtime)
    return PluginRegistry([
        *build_micro_registry(project_root.parent / "gfs-core"),
        LatentWorldModelPlugin(),
        planning,
        PoliciesPlugin(planning.service),
        FrameModelPlugin(project_root),
        WorldModelBundlePlugin(),
    ])

from gfs_core.contracts.match import MatchWorkflow
from gfs_core.contracts.persistence import MicroCheckpointStore
from gfs_core.contracts.society import SocietyMemory
from gfs_core.kernel.workspace import WorkspaceRepository
from gfs_core.plugins.match.workflow import MatchWorkflow as CoreMatchWorkflow
from gfs_core.plugins.persistence.checkpointing.micro import MicroCheckpointService
from gfs_core.plugins.society.memory.service import SocietyMemory as CoreMemory
from gfs_studio.contracts.evidence import EvidenceStore
from gfs_studio.contracts.product import ProductWorkflow, TaskStore
from gfs_studio.contracts.training import CheckpointRegistry, ModelEvaluator, Trainer
from gfs_studio.contracts.world_model import CheckpointCodec, FrameCatalog, TraceCodec
from gfs_studio.plugins.evidence.receipts.store import EvidenceStore as Store
from gfs_studio.plugins.product.studio.workflow import ProductWorkflows
from gfs_studio.plugins.product.tasks.service import TaskService
from gfs_studio.plugins.training.checkpoints.registry import CheckpointRegistry as Artifacts
from gfs_studio.plugins.training.evaluation.validator import CandidateEvaluator
from gfs_studio.plugins.training.trainers.plugin import TrainingService
from gfs_studio.plugins.world_model.frame.service import FrameModelService
from gfs_studio.plugins.world_model.plugin import CheckpointCodec as Codec
from gfs_studio.plugins.world_model.trace_codec import WorldModelTraceCodec


def test_capability_implementations_satisfy_lightweight_protocols(tmp_path):
    assert isinstance(CoreMatchWorkflow(None, None, None, None), MatchWorkflow)
    assert isinstance(CoreMemory(tmp_path), SocietyMemory)
    assert isinstance(MicroCheckpointService(WorkspaceRepository(tmp_path)), MicroCheckpointStore)
    assert isinstance(ProductWorkflows(tmp_path), ProductWorkflow)
    tasks = TaskService(tmp_path, None)
    assert isinstance(tasks, TaskStore)
    tasks.close()
    assert isinstance(TrainingService(tmp_path, None, None), Trainer)
    assert isinstance(CandidateEvaluator, ModelEvaluator)
    assert isinstance(Artifacts(tmp_path, None), CheckpointRegistry)
    assert isinstance(Codec(), CheckpointCodec)
    assert isinstance(WorldModelTraceCodec(), TraceCodec)
    assert isinstance(FrameModelService(tmp_path), FrameCatalog)
    assert isinstance(Store(tmp_path), EvidenceStore)

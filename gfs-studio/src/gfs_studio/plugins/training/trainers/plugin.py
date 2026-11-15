import hashlib
import re
from pathlib import Path

from gfs_core.contracts import PluginContext, PluginMetadata
from gfs_core.kernel.identity import content_identity
from gfs_studio.contracts.training import TrainingResult
from .determinism import reproducible
from .job import TrainingJob


class TrainingService:
    def __init__(self, project_root: str | Path, checkpoints, evaluator) -> None:
        self.project_root = Path(project_root)
        self.checkpoints = checkpoints
        self.evaluator = evaluator
        self.code_identity = _source_identity(self.project_root)

    def run(self, model, dataset, run_name: str, **configuration) -> TrainingResult:
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", run_name):
            raise ValueError("invalid training run name")
        run_dir = self.project_root / "data" / "training" / "runs" / run_name
        with reproducible(configuration.get("run_seed", 42)):
            return TrainingJob(
                model,
                dataset,
                run_dir,
                self.checkpoints,
                self.evaluator,
                self.code_identity,
            ).run(**configuration)


class TrainersPlugin:
    metadata = PluginMetadata(
        id="gfs.studio.training.trainers",
        version="0.1.0",
        api=1,
        contract="gfs_studio.contracts.training",
        provides=("trainers",),
        requires=(
            "gfs.studio.world-model.latent",
            "gfs.studio.training.evaluation",
        ),
    )

    def __init__(self, project_root: str | Path) -> None:
        self._project_root = project_root

    def activate(self, context: PluginContext) -> None:
        context.provide("trainers", TrainingService(
            self._project_root,
            context.require("checkpoint-codec"),
            context.require("model-evaluation"),
        ))

    def deactivate(self) -> None:
        pass


def _source_identity(project_root: Path) -> str:
    files = [project_root / "pyproject.toml"]
    files.extend((project_root / "src" / "gfs_studio").rglob("*.py"))
    core_root = project_root.parent / "gfs-core"
    if core_root.is_dir():
        files.append(core_root / "pyproject.toml")
        files.extend((core_root / "src" / "gfs_core").rglob("*.py"))
    identities = {}
    for path in sorted(files, key=lambda item: str(item)):
        if path.is_file():
            identities[str(path.relative_to(project_root.parent))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return content_identity(identities)

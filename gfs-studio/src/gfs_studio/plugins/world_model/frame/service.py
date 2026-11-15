import hashlib
import json
from pathlib import Path

from gfs_core.contracts import PluginContext, PluginMetadata


class FrameModelService:
    def __init__(self, project_root: str | Path) -> None:
        self.root = Path(project_root).resolve()
        self.manifest_path = self.root / "data" / "frame-world" / "manifest.json"

    def manifest(self) -> dict:
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        matches = payload.get("matches", [])
        if not isinstance(matches, list) or not matches:
            raise ValueError("frame-world manifest has no matches")
        missing = []
        invalid = []
        for match in matches:
            if not isinstance(match.get("sha256"), str) or len(match["sha256"]) != 64:
                raise ValueError("frame-world artifact identity is invalid")
            if not match.get("provider") or not match.get("match_id"):
                raise ValueError("frame-world match identity is incomplete")
            path = self._confined(match.get("file", ""))
            if not path.is_file():
                missing.append(match["match_id"])
            elif _file_hash(path) != match["sha256"]:
                invalid.append(match["match_id"])
        return {
            "candidate": payload.get("candidate", payload.get("version", "unknown")),
            "matches": len(matches),
            "providers": sorted({match["provider"] for match in matches}),
            "artifact_verified": not missing and not invalid,
            "missing_artifacts": sorted(missing),
            "invalid_artifacts": sorted(invalid),
            "scope": "verified_artifact_catalog_not_frame_inference",
        }

    def _confined(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise ValueError("frame-world artifact escapes project root") from error
        return path


class FrameModelPlugin:
    metadata = PluginMetadata(
        id="gfs.studio.world-model.frame",
        version="0.1.0",
        api=1,
        contract="gfs_studio.contracts.world_model",
        provides=("frame-models",),
        requires=(),
    )

    def __init__(self, project_root: str | Path) -> None:
        self._project_root = project_root

    def activate(self, context: PluginContext) -> None:
        context.provide("frame-models", FrameModelService(self._project_root))

    def deactivate(self) -> None:
        pass


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

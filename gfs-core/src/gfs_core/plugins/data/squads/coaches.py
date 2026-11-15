from __future__ import annotations

import json
from pathlib import Path

from gfs_core.contracts import PluginContext, PluginMetadata


class CoachesRepository:
    def __init__(self, project_root: str | Path) -> None:
        path = Path(project_root) / "data" / "coaches" / "wc2026_coaches.json"
        self.profiles = json.loads(path.read_text(encoding="utf-8"))

    def profile(self, team: str) -> dict | None:
        profile = self.profiles.get(team)
        return dict(profile) if profile is not None else None


class CoachesPlugin:
    metadata = PluginMetadata(
        id="gfs.core.data.coaches",
        version="0.1.0",
        api=1,
        contract="gfs_core.contracts.data",
        provides=("coaches",),
    )

    def __init__(self, project_root: str | Path) -> None:
        self._repository = CoachesRepository(project_root)

    def activate(self, context: PluginContext) -> None:
        context.provide("coaches", self._repository)

    def deactivate(self) -> None:
        pass

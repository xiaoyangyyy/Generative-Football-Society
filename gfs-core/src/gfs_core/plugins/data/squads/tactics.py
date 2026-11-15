from __future__ import annotations

import json
from pathlib import Path

from gfs_core.contracts import PluginContext, PluginMetadata
from gfs_core.contracts.society import TeamProfile
from gfs_core.kernel.identity import content_identity
from gfs_core.kernel.project import data_identity


class TacticsRepository:
    def __init__(self, project_root: str | Path) -> None:
        root = Path(project_root)
        self.root = root
        tactics_path = root / "data" / "tactics_final_en.json"
        coaches_path = root / "data" / "coaches" / "wc2026_coaches.json"
        self.tactics = json.loads(tactics_path.read_text(encoding="utf-8")) if tactics_path.is_file() else {}
        self.coaches = json.loads(coaches_path.read_text(encoding="utf-8")) if coaches_path.is_file() else {}

    @property
    def identity(self) -> str:
        return content_identity({"tactics": self.tactics, "coaches": self.coaches})

    @property
    def data_identity(self) -> str:
        return data_identity(self.root)

    def profile(self, team: str, status_row) -> TeamProfile:
        tactical = self.tactics.get(team, {})
        coach = self.coaches.get(team, {})
        exposure = status_row.get("global_exp", 50.0)
        return TeamProfile(
            team_name=team,
            status_score=float(status_row.get("final_status_score", 50.0)),
            formation=str(tactical.get("formation", "4-4-2")),
            tactical={},
            emotion_profile={
                "pride": 0.2,
                "anger": 0.1,
                "fear": 0.2,
                "determination": 0.4,
            },
            psychology_profile={"confidence": 0.5},
            media_exposure=float(exposure) / 100.0,
            coach_name=str(coach.get("name", "Head Coach")),
            referee_grievance=0.05,
        )


class TacticsPlugin:
    metadata = PluginMetadata(
        id="gfs.core.data.tactics",
        version="0.1.0",
        api=1,
        contract="gfs_core.contracts.data",
        provides=("tactics",),
    )

    def __init__(self, project_root: str | Path) -> None:
        self._repository = TacticsRepository(project_root)

    def activate(self, context: PluginContext) -> None:
        context.provide("tactics", self._repository)

    def deactivate(self) -> None:
        pass

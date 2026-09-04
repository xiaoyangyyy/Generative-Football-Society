"""Provider-neutral semantic action labels aligned to the frame clock."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

ACTION_NAMES = ("pass", "shot", "pressure")


@dataclass(frozen=True)
class TimedAction:
    provider: str
    match_id: str
    action: str
    start_s: float
    end_s: float
    actor_id: str | None = None
    team_id: str | None = None
    x: float | None = None
    y: float | None = None
    dx: float | None = None
    dy: float | None = None
    confidence: float = 1.0
    supervision: str = "strong"
    target_id: str | None = None
    source_event_id: str | None = None

    def __post_init__(self):
        if self.action not in ACTION_NAMES:
            raise ValueError(f"unknown action: {self.action}")
        if self.end_s < self.start_s or not 0 < self.confidence <= 1:
            raise ValueError("invalid action interval")


def write_actions(path: Path, actions: list[TimedAction]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(asdict(x), separators=(",", ":")) + "\n" for x in actions),
        encoding="utf-8",
    )


def read_actions(path: Path) -> list[TimedAction]:
    return [
        TimedAction(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]

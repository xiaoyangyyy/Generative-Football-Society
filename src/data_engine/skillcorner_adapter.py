"""Streaming adapter for SkillCorner open broadcast tracking JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

from src.data_engine.tracking_contract import TrackingFrame


def _timestamp_seconds(value: object) -> float:
    text = str(value)
    if ":" not in text:
        return float(text)
    parts = [float(part) for part in text.split(":")]
    return sum(part * multiplier for part, multiplier in zip(reversed(parts), (1, 60, 3600)))


def load_skillcorner_frames(match_path: str | Path, tracking_path: str | Path, *, sample_every: int = 1, limit: int | None = None) -> Iterator[TrackingFrame]:
    metadata = json.loads(Path(match_path).read_text(encoding="utf-8"))
    home_id = int(metadata["home_team"]["id"])
    player_team = {}
    for player in metadata["players"]:
        team = "home" if int(player["team_id"]) == home_id else "away"
        for identifier in (player.get("id"), player.get("trackable_object")):
            if identifier is not None:
                player_team[str(identifier)] = team
    yielded = 0
    with Path(tracking_path).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index % max(1, sample_every):
                continue
            raw = json.loads(line)
            if raw.get("period") is None or raw.get("frame") is None or raw.get("timestamp") is None:
                continue
            teams = {"home": {}, "away": {}}
            for player in raw.get("player_data", []):
                key = str(player.get("player_id"))
                team = player_team.get(key)
                if team and player.get("x") is not None and player.get("y") is not None:
                    teams[team][key] = np.array([player["x"], player["y"]], dtype=float)
            ball_data = raw.get("ball_data") or {}
            ball = None
            if ball_data.get("x") is not None and ball_data.get("y") is not None:
                ball = np.array([ball_data["x"], ball_data["y"]], dtype=float)
            yield TrackingFrame(
                provider="skillcorner", competition=str(metadata["competition_edition"]["competition"]["name"]),
                match_id=str(metadata["id"]), period=int(raw["period"]), frame_id=int(raw["frame"]),
                timestamp_s=_timestamp_seconds(raw["timestamp"]), home=teams["home"], away=teams["away"], ball=ball,
            )
            yielded += 1
            if limit is not None and yielded >= limit:
                return

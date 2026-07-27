#!/usr/bin/env python3
"""Build compact pass/shot supervision from seven synchronized IDSSE matches."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from kloppy import sportec

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_engine.dataset_registry import write_json_atomic

MATCHES = ("J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9")
ATTRIBUTE = re.compile(r'(\w+)="([^"]*)"')


def _file(raw: Path, prefix: str, match: str) -> Path:
    values = list(raw.glob(f"{prefix}*{match}.xml"))
    if len(values) != 1:
        raise FileNotFoundError(f"expected one {prefix} file for {match}")
    return values[0]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _segment_distance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    delta = end - start
    denom = max(float(delta @ delta), 1e-12)
    t = np.clip(((points - start) @ delta) / denom, 0.0, 1.0)
    return np.linalg.norm(points - (start + t[:, None] * delta), axis=1)


def _physics_prior(distance: float, lane: float, pressure: float, forward: float) -> float:
    logit = 2.4 - 4.2 * distance + 3.0 * lane + 1.1 * pressure - 0.6 * max(0.0, forward)
    return float(np.clip(1.0 / (1.0 + np.exp(-logit)), 0.03, 0.98))


def _milliseconds(value) -> float:
    if hasattr(value, "total_seconds"):
        return float(value.total_seconds() * 1000.0)
    return float(value)


def _epoch_ms(value) -> float:
    return float(value.timestamp() * 1000.0)


def _tracking_at_events(path: Path, targets: dict[int, float]) -> tuple[dict[int, dict[str, float]], int]:
    ball_times, ball_frames = [], []
    in_ball = False
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<FrameSet"):
                attrs = dict(ATTRIBUTE.findall(line))
                in_ball = attrs.get("TeamId") == "BALL"
            elif line.startswith("</FrameSet"):
                in_ball = False
            elif in_ball and line.startswith("<Frame "):
                attrs = dict(ATTRIBUTE.findall(line))
                ball_frames.append(int(attrs["N"]))
                ball_times.append(_epoch_ms(datetime.fromisoformat(attrs["T"])))
    times = np.asarray(ball_times)
    frames = np.asarray(ball_frames)
    event_frame = {}
    for event_index, target in targets.items():
        position = int(np.searchsorted(times, target))
        choices = [max(0, min(len(times) - 1, position - 1)), max(0, min(len(times) - 1, position))]
        best = min(choices, key=lambda index: abs(float(times[index]) - target))
        event_frame[event_index] = int(frames[best])
    wanted = set(event_frame.values())
    by_frame = {frame: {} for frame in wanted}
    person = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<FrameSet"):
                attrs = dict(ATTRIBUTE.findall(line))
                person = attrs.get("PersonId") if attrs.get("TeamId") != "BALL" else None
            elif line.startswith("</FrameSet"):
                person = None
            elif person and line.startswith("<Frame "):
                attrs = dict(ATTRIBUTE.findall(line))
                frame = int(attrs["N"])
                if frame in wanted:
                    by_frame[frame][f"{person}_x"] = float(attrs["X"]) / 105.0 + 0.5
                    by_frame[frame][f"{person}_y"] = float(attrs["Y"]) / 68.0 + 0.5
                    by_frame[frame][f"{person}_s"] = float(attrs.get("S", 0.0))
    return {event_index: by_frame[frame] for event_index, frame in event_frame.items()}, len(ball_frames)


def _derive_match(raw: Path, match: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    meta = _file(raw, "DFL_02", match)
    event_path = _file(raw, "DFL_03", match)
    tracking_path = _file(raw, "DFL_04", match)
    dataset = sportec.load_event(event_data=event_path, meta_data=meta)
    events = dataset.to_df()
    player_team = {player.player_id: team.team_id for team in dataset.metadata.teams for player in team.players}
    player_ids = sorted(player_team)
    pass_rows, shot_rows = [], []
    selected = events[events.event_type.isin(["PASS", "SHOT"])].copy()
    period_start = {period.id: _epoch_ms(period.start_timestamp) for period in dataset.metadata.periods}
    targets = {index: period_start[int(event.period_id)] + _milliseconds(event.timestamp) for index, event in selected.iterrows() if not pd.isna(event.timestamp)}
    event_frames, tracking_frame_count = _tracking_at_events(tracking_path, targets)
    for event_index, event in selected.iterrows():
        period = int(event.period_id)
        frame = event_frames.get(event_index)
        if frame is None:
            continue
        event_ms = _milliseconds(event.timestamp)
        if event.event_type == "SHOT":
            shot_rows.append({"provider": "sportec", "match_id": match, "event_id": str(event.event_id), "period": period, "timestamp_ms": event_ms, "x": event.coordinates_x, "y": event.coordinates_y, "goal": int(str(event.result).upper() == "GOAL"), "on_target": int(str(event.result).upper() in {"GOAL", "SAVED"})})
            continue
        passer = str(event.player_id)
        team = player_team.get(passer)
        if team is None or pd.isna(frame.get(f"{passer}_x")):
            continue
        start = np.array([frame[f"{passer}_x"], frame[f"{passer}_y"]], dtype=float)
        teammates = [pid for pid in player_ids if player_team[pid] == team and pid != passer and pd.notna(frame.get(f"{pid}_x"))]
        opponents = [pid for pid in player_ids if player_team[pid] != team and pd.notna(frame.get(f"{pid}_x"))]
        if len(teammates) < 5 or len(opponents) < 5:
            continue
        defenders = np.array([[frame[f"{pid}_x"], frame[f"{pid}_y"]] for pid in opponents])
        defender_speeds = np.nan_to_num(np.array([frame.get(f"{pid}_s", 0.0) for pid in opponents], dtype=float))
        direction = 1.0 if start[0] < 0.5 else -1.0
        receiver = str(event.receiver_player_id) if pd.notna(event.receiver_player_id) else ""
        if not receiver and pd.notna(event.end_coordinates_x) and pd.notna(event.end_coordinates_y):
            intended = np.array([event.end_coordinates_x, event.end_coordinates_y], dtype=float)
            receiver = min(
                teammates,
                key=lambda pid: float(np.linalg.norm(np.array([frame[f"{pid}_x"], frame[f"{pid}_y"]]) - intended)),
            )
        success = int(bool(event.success))
        event_id = f"sportec-{match}-{event.event_id}"
        for candidate in teammates:
            target = np.array([frame[f"{candidate}_x"], frame[f"{candidate}_y"]], dtype=float)
            distance = float(np.linalg.norm(target - start))
            defender_distance = float(np.linalg.norm(defenders - target, axis=1).min())
            lane = float(_segment_distance(defenders, start, target).min())
            candidate_speed = float(np.nan_to_num(frame.get(f"{candidate}_s", 0.0)))
            pass_time = max(0.15, distance * 105.0 / 18.0)
            receiver_arrival = distance * 105.0 / max(2.0, candidate_speed)
            defender_arrival = defender_distance * 105.0 / max(2.0, float(defender_speeds.max(initial=2.0)))
            forward = direction * float(target[0] - start[0])
            prior = _physics_prior(distance, lane, defender_distance, forward)
            pass_rows.append({"provider": "sportec", "match_id": match, "event_id": event_id, "period": period, "timestamp_ms": event_ms, "passer": passer, "candidate": candidate, "is_receiver": int(candidate == receiver), "completion": success, "physics_prior": prior, "distance": distance, "forward_progress": forward, "lateral_change": abs(float(target[1] - start[1])), "receiver_defender_distance": defender_distance, "lane_clearance": lane, "receiver_arrival_margin": defender_arrival - receiver_arrival, "lane_intercept_margin": lane * 68.0 - pass_time * float(defender_speeds.max(initial=0.0)), "body_alignment": np.nan, "pressure_change": 0.0, "preferred_foot_alignment": np.nan})
    manifest = {"match_id": match, "events": int(len(events)), "tracking_frames": tracking_frame_count, "passes": int(selected.event_type.eq("PASS").sum()), "shots": int(selected.event_type.eq("SHOT").sum()), "source_sha256": {path.name: _sha256(path) for path in (meta, event_path, tracking_path)}}
    return pd.DataFrame(pass_rows), pd.DataFrame(shot_rows), manifest


def main() -> int:
    raw = ROOT / "data/external/sportec/raw"
    output = ROOT / "data/external/sportec/derived"
    output.mkdir(parents=True, exist_ok=True)
    manifests = []
    for match in MATCHES:
        pass_frame, shot_frame, report = _derive_match(raw, match)
        pass_frame.to_csv(output / f"{match}_passes.csv", index=False, lineterminator="\n")
        shot_frame.to_csv(output / f"{match}_shots.csv", index=False, lineterminator="\n")
        manifests.append(report)
        print(json.dumps(report))
    write_json_atomic(output / "manifest.json", {"schema_version": 1, "provider": "sportec", "dataset": "IDSSE", "license": "CC BY 4.0", "matches": manifests, "usage": "synchronized real event/tracking supervision; group only by match_id"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

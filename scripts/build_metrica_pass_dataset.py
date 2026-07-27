#!/usr/bin/env python3
"""Build a compact receiver-choice dataset from synchronized Metrica tracking."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_engine.dataset_registry import write_json_atomic


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tracking_at_frames(path: Path, wanted: set[int]) -> dict[int, dict[str, np.ndarray]]:
    result: dict[int, dict[str, np.ndarray]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        next(reader)
        labels = next(reader)
        player_columns = [(index, labels[index]) for index in range(3, len(labels) - 1, 2) if labels[index]]
        for row in reader:
            if len(row) < 5:
                continue
            try:
                frame = int(row[1])
            except ValueError:
                continue
            if frame not in wanted:
                continue
            positions = {}
            for index, player in player_columns:
                try:
                    x, y = float(row[index]), float(row[index + 1])
                except (ValueError, IndexError):
                    continue
                if math.isfinite(x) and math.isfinite(y):
                    positions[player] = np.array([x, y], dtype=float)
            result[frame] = positions
    return result


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    delta = end - start
    denom = float(delta @ delta)
    if denom <= 1e-12:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(((point - start) @ delta) / denom, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * delta)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Metrica sample-data repository root")
    parser.add_argument("--game", type=int, default=1)
    parser.add_argument("--out", default=str(ROOT / "data/external/metrica/pass_candidates_game1.csv"))
    args = parser.parse_args()
    game_dir = Path(args.source) / "data" / f"Sample_Game_{args.game}"
    prefix = f"Sample_Game_{args.game}"
    event_path = game_dir / f"{prefix}_RawEventsData.csv"
    tracking_paths = {
        "Home": game_dir / f"{prefix}_RawTrackingData_Home_Team.csv",
        "Away": game_dir / f"{prefix}_RawTrackingData_Away_Team.csv",
    }
    events = pd.read_csv(event_path)
    passes = events[(events["Type"] == "PASS") & events["From"].notna() & events["To"].notna()].copy()
    passes["Start Frame"] = passes["Start Frame"].astype(int)
    wanted = set(passes["Start Frame"].tolist())
    tracking = {team: _tracking_at_frames(path, wanted) for team, path in tracking_paths.items()}

    orientation = {}
    for (team, period), frame in passes.groupby(["Team", "Period"]):
        delta = (frame["End X"] - frame["Start X"]).dropna()
        orientation[(team, int(period))] = 1.0 if float(delta.median()) >= 0 else -1.0

    rows = []
    dropped = defaultdict(int)
    for event_index, event in passes.iterrows():
        team, period, frame = str(event["Team"]), int(event["Period"]), int(event["Start Frame"])
        own = tracking[team].get(frame, {})
        opponents = tracking["Away" if team == "Home" else "Home"].get(frame, {})
        passer, receiver = str(event["From"]), str(event["To"])
        if passer not in own or receiver not in own or len(opponents) < 7:
            dropped["missing_tracking"] += 1
            continue
        start = own[passer]
        direction = orientation[(team, period)]
        event_id = f"g{args.game}-p{period}-f{frame}-{event_index}"
        candidates = [(name, pos) for name, pos in own.items() if name != passer]
        if len(candidates) < 7:
            dropped["few_candidates"] += 1
            continue
        for candidate, target in candidates:
            distance = float(np.linalg.norm(target - start))
            nearest_defender = min(float(np.linalg.norm(defender - target)) for defender in opponents.values())
            lane_clearance = min(_point_segment_distance(defender, start, target) for defender in opponents.values())
            rows.append({
                "event_id": event_id,
                "game": args.game,
                "period": period,
                "team": team,
                "frame": frame,
                "passer": passer,
                "candidate": candidate,
                "is_receiver": int(candidate == receiver),
                "distance": distance,
                "forward_progress": direction * float(target[0] - start[0]),
                "lateral_change": abs(float(target[1] - start[1])),
                "receiver_defender_distance": nearest_defender,
                "lane_clearance": lane_clearance,
                "target_x": float(target[0]),
                "target_y": float(target[1]),
            })
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(rows)
    valid_events = data.groupby("event_id")["is_receiver"].sum()
    valid_ids = set(valid_events[valid_events == 1].index)
    data = data[data["event_id"].isin(valid_ids)].copy()
    data.to_csv(output, index=False, lineterminator="\n")
    manifest = {
        "schema_version": 1,
        "dataset": f"metrica-receiver-choice-game{args.game}",
        "source": "https://github.com/metrica-sports/sample-data",
        "acknowledgement": "Metrica Sports sample tracking and event data",
        "source_files": {path.name: _sha256(path) for path in [event_path, *tracking_paths.values()]},
        "derived_file": output.name,
        "derived_sha256": _sha256(output),
        "events": len(valid_ids),
        "candidate_rows": len(data),
        "dropped": dict(dropped),
        "split_policy": "game-level split is defined by evaluate_receiver_ranker.py",
        "usage": "receiver ranking only; never completion labels or world-model transition training",
    }
    write_json_atomic(output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

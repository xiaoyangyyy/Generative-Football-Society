#!/usr/bin/env python3
"""Fetch matching StatsBomb events and derive compact 360 pass/shot data."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_engine.dataset_registry import write_json_atomic

SOURCE_360 = Path(r"C:\tmp\statsbomb-open-data\data\three-sixty")
RAW = ROOT / "data/external/statsbomb360/raw"
DERIVED = ROOT / "data/external/statsbomb360/derived"
EVENT_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/{match_id}.json"
FRAME_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/three-sixty/{match_id}.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _download(match_id: str) -> tuple[str, str]:
    target = RAW / "events" / f"{match_id}.json"
    if target.is_file() and target.stat().st_size > 1000:
        return match_id, "cached"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(EVENT_URL.format(match_id=match_id), timeout=90) as response:
                payload = response.read()
            if len(payload) < 1000:
                raise ValueError("event payload is unexpectedly small")
            temporary = target.with_suffix(".json.part")
            temporary.write_bytes(payload)
            temporary.replace(target)
            return match_id, "downloaded"
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("unreachable")


def _valid_json(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1000:
        return False
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except (OSError, ValueError):
        return False


def _repair_frame(match_id: str) -> tuple[str, str]:
    target = RAW / "three-sixty" / f"{match_id}.json"
    if _valid_json(target):
        return match_id, "valid"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(FRAME_URL.format(match_id=match_id), timeout=180) as response:
                payload = response.read()
            json.loads(payload)
            temporary = target.with_suffix(".json.part")
            temporary.write_bytes(payload)
            temporary.replace(target)
            return match_id, "repaired"
        except Exception:
            if attempt == 3:
                return match_id, "invalid_upstream"
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError("unreachable")


def _segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    delta = end - start
    denom = max(float(delta @ delta), 1e-12)
    t = float(np.clip(((point - start) @ delta) / denom, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * delta)))


def _prior(distance: float, lane: float, target_space: float, under_pressure: float) -> float:
    logit = 2.2 - 0.035 * distance + 0.10 * lane + 0.08 * target_space - 0.55 * under_pressure
    return float(np.clip(1.0 / (1.0 + np.exp(-logit)), 0.03, 0.98))


def _derive(match_id: str) -> tuple[list[dict], list[dict], dict]:
    events_path = RAW / "events" / f"{match_id}.json"
    frames_path = RAW / "three-sixty" / f"{match_id}.json"
    events = json.loads(events_path.read_text(encoding="utf-8"))
    frames = {
        str(frame["event_uuid"]): frame
        for frame in json.loads(frames_path.read_text(encoding="utf-8"))
    }
    passes, shots = [], []
    for event in events:
        frame = frames.get(str(event.get("id")))
        if frame is None or not event.get("location"):
            continue
        kind = (event.get("type") or {}).get("name")
        start = np.asarray(event["location"][:2], dtype=float)
        freeze = frame.get("freeze_frame") or []
        defenders = [
            np.asarray(item["location"][:2], dtype=float)
            for item in freeze if not item.get("teammate") and item.get("location")
        ]
        if not defenders:
            continue
        if kind == "Pass":
            payload = event.get("pass") or {}
            if not payload.get("end_location"):
                continue
            target = np.asarray(payload["end_location"][:2], dtype=float)
            distance = float(np.linalg.norm(target - start))
            target_space = min(float(np.linalg.norm(point - target)) for point in defenders)
            lane = min(_segment_distance(point, start, target) for point in defenders)
            under_pressure = float(bool(event.get("under_pressure")))
            completion = int(not payload.get("outcome"))
            passes.append({
                "provider": "statsbomb360", "match_id": match_id,
                "event_id": str(event["id"]), "completion": completion,
                "physics_prior": _prior(distance, lane, target_space, under_pressure),
                "distance": distance / 120.0,
                "forward_progress": float(target[0] - start[0]) / 120.0,
                "lateral_change": abs(float(target[1] - start[1])) / 80.0,
                "receiver_defender_distance": target_space / 120.0,
                "lane_clearance": lane / 80.0,
                "receiver_arrival_margin": np.nan,
                "lane_intercept_margin": lane / 80.0,
                "body_alignment": np.nan,
                "pressure_change": under_pressure,
                "preferred_foot_alignment": float((payload.get("body_part") or {}).get("name") in {"Left Foot", "Right Foot"}),
            })
        elif kind == "Shot":
            payload = event.get("shot") or {}
            outcome = (payload.get("outcome") or {}).get("name", "")
            keeper = next(
                (np.asarray(item["location"][:2], dtype=float) for item in freeze if item.get("keeper") and not item.get("teammate")),
                None,
            )
            shots.append({
                "provider": "statsbomb360", "match_id": match_id,
                "event_id": str(event["id"]), "x": start[0] / 120.0, "y": start[1] / 80.0,
                "xg": float(payload.get("statsbomb_xg", 0.0)),
                "goal": int(outcome == "Goal"),
                "on_target": int(outcome in {"Goal", "Saved", "Saved to Post"}),
                "nearest_defender": min(float(np.linalg.norm(point - start)) for point in defenders) / 120.0,
                "keeper_distance": float(np.linalg.norm(keeper - start)) / 120.0 if keeper is not None else np.nan,
            })
    return passes, shots, {
        "match_id": match_id, "events": len(events), "frames360": len(frames),
        "passes": len(passes), "shots": len(shots),
        "sha256": {"events": _sha256(events_path), "three_sixty": _sha256(frames_path)},
    }


def main() -> int:
    if not SOURCE_360.is_dir():
        raise SystemExit(f"missing StatsBomb 360 source: {SOURCE_360}")
    (RAW / "events").mkdir(parents=True, exist_ok=True)
    (RAW / "three-sixty").mkdir(parents=True, exist_ok=True)
    DERIVED.mkdir(parents=True, exist_ok=True)
    source_files = sorted(SOURCE_360.glob("*.json"))
    match_ids = [path.stem for path in source_files]
    for source in source_files:
        target = RAW / "three-sixty" / source.name
        if not target.exists():
            shutil.copy2(source, target)
    with ThreadPoolExecutor(max_workers=8) as pool:
        repaired = list(pool.map(_repair_frame, match_ids))
    changed = [match_id for match_id, status in repaired if status == "repaired"]
    invalid = [match_id for match_id, status in repaired if status == "invalid_upstream"]
    match_ids = [match_id for match_id, status in repaired if status != "invalid_upstream"]
    print(f"validated 360 files={len(source_files)} repaired={changed} invalid_upstream={invalid}", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_download, match_id) for match_id in match_ids]
        for count, future in enumerate(as_completed(futures), 1):
            match_id, status = future.result()
            if count % 25 == 0 or count == len(futures):
                print(f"events {count}/{len(futures)} latest={match_id} {status}", flush=True)
    pass_rows, shot_rows, manifests = [], [], []
    for count, match_id in enumerate(match_ids, 1):
        match_passes, match_shots, report = _derive(match_id)
        pass_rows.extend(match_passes)
        shot_rows.extend(match_shots)
        manifests.append(report)
        if count % 50 == 0 or count == len(match_ids):
            print(f"derived {count}/{len(match_ids)} passes={len(pass_rows)} shots={len(shot_rows)}", flush=True)
    pd.DataFrame(pass_rows).to_csv(DERIVED / "passes.csv", index=False, lineterminator="\n")
    pd.DataFrame(shot_rows).to_csv(DERIVED / "shots.csv", index=False, lineterminator="\n")
    write_json_atomic(DERIVED / "manifest.json", {
        "schema_version": 1, "provider": "statsbomb360",
        "source": "StatsBomb Open Data", "matches": manifests,
        "excluded_invalid_upstream": invalid,
        "usage": "event-level freeze-frame completion and shot supervision; not continuous tracking",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

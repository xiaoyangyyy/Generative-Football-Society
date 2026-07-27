#!/usr/bin/env python3
"""Fetch SkillCorner tracking and derive pass, receiver, and spatial summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"C:\tmp\skillcorner-opendata\data\matches")
RAW = ROOT / "data/external/skillcorner/raw"
DERIVED = ROOT / "data/external/skillcorner/derived"
LFS_BATCH_URL = "https://github.com/SkillCorner/opendata.git/info/lfs/objects/batch"
DOWNLOADS: dict[str, tuple[str, dict[str, str], int]] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(match_id: str) -> tuple[str, int]:
    global DOWNLOADS
    target = RAW / match_id / f"{match_id}_tracking_extrapolated.jsonl"
    if target.is_file() and target.stat().st_size > 50_000_000:
        return match_id, target.stat().st_size
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".jsonl.part")
    for attempt in range(6):
        try:
            url, headers, expected_size = DOWNLOADS[match_id]
            offset = temporary.stat().st_size if temporary.exists() else 0
            if offset > expected_size:
                temporary.unlink()
                offset = 0
            headers = dict(headers)
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=180) as response:
                if offset and response.status != 206:
                    temporary.unlink(missing_ok=True)
                    raise RuntimeError("LFS host ignored resume range")
                with temporary.open("ab" if offset else "wb") as handle:
                    downloaded = offset
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if downloaded % (10 * 1024 * 1024) < len(chunk):
                            print(f"tracking-progress match={match_id} bytes={downloaded}/{expected_size}", flush=True)
            if temporary.stat().st_size != expected_size:
                raise ValueError(f"tracking size mismatch: {temporary.stat().st_size} != {expected_size}")
            temporary.replace(target)
            return match_id, target.stat().st_size
        except Exception as error:
            print(f"tracking-retry match={match_id} attempt={attempt + 1} partial={temporary.stat().st_size if temporary.exists() else 0} error={error}", flush=True)
            if attempt == 5:
                raise
            try:
                DOWNLOADS.update(_lfs_downloads([match_id]))
            except Exception as refresh_error:
                print(f"tracking-url-refresh match={match_id} error={refresh_error}", flush=True)
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError("unreachable")


def _lfs_downloads(match_ids: list[str]) -> dict[str, tuple[str, dict[str, str], int]]:
    repo = Path(r"C:\tmp\skillcorner-opendata")
    objects, by_oid = [], {}
    for match_id in match_ids:
        path = f"data/matches/{match_id}/{match_id}_tracking_extrapolated.jsonl"
        pointer = subprocess.check_output(
            ["git", "-c", f"safe.directory={repo.as_posix()}", "show", f"HEAD:{path}"],
            cwd=repo, text=True, encoding="utf-8",
        )
        values = dict(line.split(" ", 1) for line in pointer.splitlines() if " " in line)
        oid, size = values["oid"].removeprefix("sha256:"), int(values["size"])
        objects.append({"oid": oid, "size": size})
        by_oid[oid] = match_id
    payload = json.dumps({"operation": "download", "transfers": ["basic"], "objects": objects}).encode()
    request = urllib.request.Request(
        LFS_BATCH_URL, data=payload, method="POST",
        headers={"Accept": "application/vnd.git-lfs+json", "Content-Type": "application/vnd.git-lfs+json", "User-Agent": "gfs-v7-import"},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        batch = json.load(response)
    result = {}
    for item in batch["objects"]:
        action = item.get("actions", {}).get("download")
        if not action:
            raise RuntimeError(f"LFS object unavailable: {item}")
        result[by_oid[item["oid"]]] = (action["href"], action.get("header", {}), int(item["size"]))
    return result


def _number(row: pd.Series, name: str, default: float = np.nan) -> float:
    value = row.get(name, default)
    return float(value) if pd.notna(value) else default


def _derive_events(match_id: str, path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(path, low_memory=False)
    attempted = data[data.pass_outcome.notna()].copy()
    pass_rows = []
    for _, row in attempted.iterrows():
        start_x = _number(row, "player_in_possession_x_start", _number(row, "x_start"))
        start_y = _number(row, "player_in_possession_y_start", _number(row, "y_start"))
        target_x = _number(row, "player_targeted_x_pass", _number(row, "x_end"))
        target_y = _number(row, "player_targeted_y_pass", _number(row, "y_end"))
        pass_rows.append({
            "provider": "skillcorner", "match_id": str(match_id), "event_id": str(row.event_id),
            "completion": int(str(row.pass_outcome).lower() == "successful"),
            "physics_prior": _number(row, "player_targeted_xpass_completion", _number(row, "xpass_completion", 0.5)),
            "distance": _number(row, "pass_distance", 0.0) / 105.0,
            "forward_progress": (target_x - start_x) / 105.0,
            "lateral_change": abs(target_y - start_y) / 68.0,
            "receiver_defender_distance": max(0.0, 5.0 - _number(row, "n_player_targeted_opponents_within_5m_start", 2.5)) / 5.0,
            "lane_clearance": np.nan,
            "receiver_arrival_margin": _number(row, "speed_difference", 0.0) / 10.0,
            "lane_intercept_margin": _number(row, "n_opponents_bypassed", 0.0) / 11.0,
            "body_alignment": float(np.cos(np.deg2rad(_number(row, "pass_angle", 0.0)))),
            "pressure_change": (_number(row, "n_opponents_ahead_pass_reception", 0.0) - _number(row, "n_opponents_ahead_player_in_possession_pass_moment", 0.0)) / 11.0,
            "preferred_foot_alignment": float(not bool(row.get("is_header", False))),
        })
    options = data[data.event_type.eq("passing_option")].copy()
    receiver_rows = []
    for _, row in options.iterrows():
        group = row.get("associated_player_possession_event_id")
        if pd.isna(group):
            continue
        receiver_rows.append({
            "provider": "skillcorner", "match_id": str(match_id),
            "event_id": f"skillcorner-{match_id}-{group}", "candidate": str(row.get("player_id", "")),
            "is_receiver": int(bool(row.get("targeted", False))),
            "distance": _number(row, "pass_distance", _number(row, "interplayer_distance", 0.0)) / 105.0,
            "forward_progress": _number(row, "n_opponents_overtaken", 0.0) / 11.0,
            "lateral_change": abs(_number(row, "interplayer_angle", 0.0)) / 180.0,
            "receiver_defender_distance": max(0.0, 5.0 - _number(row, "n_player_targeted_opponents_within_5m_start", 2.5)) / 5.0,
            "lane_clearance": np.nan,
            "receiver_arrival_margin": _number(row, "speed_difference", 0.0) / 10.0,
            "lane_intercept_margin": _number(row, "n_opponents_overtaken", 0.0) / 11.0,
            "body_alignment": float(np.cos(np.deg2rad(_number(row, "interplayer_angle", 0.0)))),
            "pressure_change": 0.0,
            "preferred_foot_alignment": 1.0,
        })
    shots = data[data.lead_to_shot.fillna(False)].copy()
    shot_rows = pd.DataFrame({
        "provider": "skillcorner", "match_id": str(match_id),
        "event_id": shots.event_id.astype(str),
        "goal": shots.lead_to_goal.fillna(False).astype(int),
        "x": shots.x_end, "y": shots.y_end,
        "xshot": shots.xshot_player_possession_end,
    })
    return pd.DataFrame(pass_rows), pd.DataFrame(receiver_rows), shot_rows


def _derive_tracking(match_id: str, path: Path, sample_every: int = 10) -> tuple[pd.DataFrame, dict]:
    rows, total = [], 0
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            total += 1
            if index % sample_every:
                continue
            frame = json.loads(line)
            players = frame.get("player_data") or []
            ball = frame.get("ball_data") or {}
            detected = sum(bool(player.get("is_detected")) for player in players)
            rows.append({
                "provider": "skillcorner", "match_id": str(match_id),
                "period": frame.get("period"), "frame": frame.get("frame"),
                "timestamp": frame.get("timestamp"), "players": len(players),
                "detected_players": detected, "extrapolated_players": len(players) - detected,
                "ball_x": ball.get("x"), "ball_y": ball.get("y"),
                "possession_group": (frame.get("possession") or {}).get("group"),
            })
    return pd.DataFrame(rows), {"frames": total, "sampled_frames": len(rows)}


def main() -> int:
    global DOWNLOADS
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-only", action="store_true")
    parser.add_argument("--match-id", action="append", default=[])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    match_ids = sorted(path.name for path in SOURCE.iterdir() if path.is_dir())
    download_ids = match_ids
    if args.match_id:
        unknown = set(args.match_id) - set(match_ids)
        if unknown:
            raise ValueError(f"unknown match ids: {sorted(unknown)}")
        download_ids = args.match_id
    if not args.events_only:
        DOWNLOADS = _lfs_downloads(download_ids)
    RAW.mkdir(parents=True, exist_ok=True)
    DERIVED.mkdir(parents=True, exist_ok=True)
    for match_id in match_ids:
        target = RAW / match_id
        target.mkdir(parents=True, exist_ok=True)
        for suffix in ("_match.json", "_dynamic_events.csv", "_phases_of_play.csv"):
            source = SOURCE / match_id / f"{match_id}{suffix}"
            if source.is_file() and not (target / source.name).exists():
                shutil.copy2(source, target / source.name)
    if not args.events_only:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            for count, (match_id, size) in enumerate(pool.map(_download, download_ids), 1):
                print(f"tracking {count}/{len(download_ids)} match={match_id} bytes={size}", flush=True)
    if args.download_only:
        return 0
    passes, receivers, shots, tracking, manifests = [], [], [], [], []
    for match_id in match_ids:
        base = RAW / match_id
        pass_frame, receiver_frame, shot_frame = _derive_events(match_id, base / f"{match_id}_dynamic_events.csv")
        tracking_path = base / f"{match_id}_tracking_extrapolated.jsonl"
        if tracking_path.is_file():
            tracking_frame, tracking_report = _derive_tracking(match_id, tracking_path)
        else:
            tracking_frame, tracking_report = pd.DataFrame(), {"frames": 0, "sampled_frames": 0, "tracking_status": "network_unavailable"}
        passes.append(pass_frame)
        receivers.append(receiver_frame)
        shots.append(shot_frame)
        if not tracking_frame.empty:
            tracking.append(tracking_frame)
        for partial in base.glob("*.part"):
            partial.unlink(missing_ok=True)
        files = list(base.iterdir())
        manifests.append({
            "match_id": match_id, **tracking_report,
            "passes": len(pass_frame), "receiver_candidates": len(receiver_frame), "shots": len(shot_frame),
            "sha256": {path.name: _sha256(path) for path in files},
        })
        print(json.dumps(manifests[-1]), flush=True)
    pd.concat(passes, ignore_index=True).to_csv(DERIVED / "passes.csv", index=False, lineterminator="\n")
    pd.concat(receivers, ignore_index=True).to_csv(DERIVED / "receiver_candidates.csv", index=False, lineterminator="\n")
    pd.concat(shots, ignore_index=True).to_csv(DERIVED / "shots.csv", index=False, lineterminator="\n")
    if tracking:
        pd.concat(tracking, ignore_index=True).to_csv(DERIVED / "tracking_1hz.csv", index=False, lineterminator="\n")
    (DERIVED / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "provider": "skillcorner", "matches": manifests,
        "source": "SkillCorner Open Data", "tracking_sample_rate_hz": 1,
        "tracking_import_complete": bool(tracking) and len(tracking) == len(match_ids),
    }, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

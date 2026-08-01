#!/usr/bin/env python3
"""Static and artifact integrity checks used by the acceptance document."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_engine.dataset_registry import file_sha256, load_manifest, write_json_atomic


def main() -> int:
    manifest_path = ROOT / "data/world_model/dataset_manifest.json"
    manifest = load_manifest(manifest_path)
    trace_dir = Path(manifest["source_root"])
    hashes_ok = all(file_sha256(trace_dir / item["path"]) == item["sha256"] for item in manifest["files"])
    groups = {name: {item["group"] for item in manifest["files"] if item["split"] == name} for name in ("train", "dev", "sealed_test")}
    disjoint = not (groups["train"] & groups["dev"] or groups["train"] & groups["sealed_test"] or groups["dev"] & groups["sealed_test"])
    baseline = json.loads((ROOT / "data/calibration/statsbomb_match_baselines.json").read_text(encoding="utf-8"))
    env_hits = []
    pattern = re.compile(r"os\.(?:environ|getenv)")
    for path in (ROOT / "src").rglob("*.py"):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                env_hits.append(f"{path.relative_to(ROOT).as_posix()}:{line_no}")
    allowed_env_prefixes = (
        "src/cli.py:",
        "src/simulation/runtime.py:",
        "src/match_engine/calibration/",
    )
    core_env_hits = [hit for hit in env_hits if not hit.startswith(allowed_env_prefixes)]
    tracking = []
    for game in (1, 2):
        derived = ROOT / f"data/external/metrica/pass_candidates_game{game}.csv"
        derived_manifest = json.loads(derived.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        tracking.append({
            "game": game,
            "events": derived_manifest["events"],
            "candidate_rows": derived_manifest["candidate_rows"],
            "hash_ok": file_sha256(derived) == derived_manifest["derived_sha256"],
        })
    ranker = json.loads((ROOT / "data/world_model/receiver_ranker.json").read_text(encoding="utf-8"))
    try:
        import torch
        payload = torch.load(ROOT / "data/world_model/latent_wm.pt", map_location="cpu", weights_only=False)
        checkpoint_version = int(payload.get("version", 0))
    except Exception:
        checkpoint_version = 0
    report = {
        "dataset": {"hashes_ok": hashes_ok, "groups_disjoint": disjoint, "summary": manifest["summary"]},
        "real_reference": {"source": baseline.get("source"), "team_matches": baseline.get("n_team_matches", 0), "ok": baseline.get("n_team_matches", 0) >= 100},
        "real_tracking": {
            "source": "Metrica Sports sample-data",
            "games": tracking,
            "sealed_evaluation": ranker.get("sealed_evaluation", {}),
            "ok": all(item["hash_ok"] for item in tracking) and bool(ranker.get("sealed_evaluation", {}).get("candidate_ready")),
        },
        "checkpoint": {"version": checkpoint_version, "ok": checkpoint_version >= 7},
        "architecture": {"standings_service": (ROOT / "src/simulation/standings.py").is_file(), "stable_app_api": (ROOT / "src/app.py").is_file()},
        "compatibility_env_reads": {
            "count": len(env_hits), "core_count": len(core_env_hits),
            "locations": env_hits,
            "policy": "only CLI, calibration sandbox, and runtime boundary may access process environment",
        },
    }
    report["integrity_ready"] = all((hashes_ok, disjoint, report["real_reference"]["ok"], report["real_tracking"]["ok"], report["checkpoint"]["ok"], report["architecture"]["standings_service"], not core_env_hits))
    out = ROOT / "reports/acceptance/integrity.json"
    write_json_atomic(out, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["integrity_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

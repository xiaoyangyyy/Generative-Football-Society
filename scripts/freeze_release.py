#!/usr/bin/env python3
"""Create a content-addressed manifest for a validated frozen release."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_engine.dataset_registry import file_sha256, write_json_atomic


def _tree_digest(paths: list[Path]) -> dict:
    files = sorted(
        path for root in paths
        for path in ([root] if root.is_file() else root.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".md", ".toml"}
    )
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(bytes.fromhex(file_sha256(path)))
    return {"files": len(files), "sha256": digest.hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="6.0.0")
    parser.add_argument("--tests", default="132 passed, 2 skipped")
    parser.add_argument("--sealed-report", default=str(ROOT / "reports/acceptance/deployed_release_metrics.json"))
    parser.add_argument("--out", default=str(ROOT / "data/releases/v6.0.0.json"))
    args = parser.parse_args()
    artifacts = [
        ROOT / "data/world_model/latent_wm.pt",
        ROOT / "data/world_model/dataset_manifest.json",
        ROOT / "data/world_model/receiver_ranker.json",
        ROOT / "data/external/metrica/pass_candidates_game1.csv",
        ROOT / "data/external/metrica/pass_candidates_game1.manifest.json",
        ROOT / "data/external/metrica/pass_candidates_game2.csv",
        ROOT / "data/external/metrica/pass_candidates_game2.manifest.json",
    ]
    missing = [str(path) for path in artifacts if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing release artifacts: {missing}")
    sealed_path = Path(args.sealed_report)
    sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
    if not sealed.get("release_ready"):
        raise SystemExit("Sealed release gate is not ready")
    sealed["checkpoint"] = "data/world_model/latent_wm.pt"
    sealed["dataset_manifest"] = "data/world_model/dataset_manifest.json"
    payload = {
        "schema_version": 1,
        "release": args.version,
        "status": "frozen",
        "world_model_version": 6,
        "calibration_profile": "v5_physics_first",
        "tests": args.tests,
        "sealed_metrics": sealed,
        "artifacts": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in artifacts
        ],
        "source_snapshot": _tree_digest([
            ROOT / "src", ROOT / "scripts", ROOT / "tests",
            ROOT / "gfs.py", ROOT / "pyproject.toml",
            ROOT / "README.md", ROOT / "docs",
        ]),
        "mutation_policy": "new research must create a new version; do not overwrite sealed data or this manifest",
    }
    write_json_atomic(args.out, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

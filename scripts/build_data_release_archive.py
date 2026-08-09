#!/usr/bin/env python3
"""Build, check, or materialize the licensed IDSSE data supplement."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infrastructure import file_sha256  # noqa: E402


REGISTRY = ROOT / "data/evaluation/data_license_registry_v1.json"
MANIFEST = ROOT / "data/evaluation/data_release_manifest_v1.json"
ATTRIBUTION = ROOT / "docs/IDSSE_ATTRIBUTION.md"
SPORTEC_PATTERNS = (
    "data/external/sportec/derived/manifest.json",
    "data/frame_world/v8/sportec_*.npz",
    "data/frame_world/v82_actions/sportec_*.jsonl",
    "data/frame_world/v82_actions/sportec_*.npz",
    "data/frame_world/v84_pressure/sportec_*.npz",
    "data/frame_world/v85_pass_triplets/sportec_*.npz",
)
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def _relative(path: Path) -> str:
    resolved = path.resolve()
    resolved.relative_to(ROOT)
    return resolved.relative_to(ROOT).as_posix()


def approved_paths() -> list[Path]:
    paths: set[Path] = {ATTRIBUTION.resolve()}
    for pattern in SPORTEC_PATTERNS:
        matches = list(ROOT.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"approved pattern has no files: {pattern}")
        paths.update(path.resolve() for path in matches)
    ordered = sorted(paths, key=_relative)
    if any(path.is_symlink() or not path.is_file() for path in ordered):
        raise ValueError("approved archive paths must be regular files")
    return ordered


def archive_bytes(paths: list[Path] | None = None) -> bytes:
    selected = approved_paths() if paths is None else paths
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in selected:
            info = zipfile.ZipInfo(_relative(path), date_time=FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    return buffer.getvalue()


def expected_manifest() -> dict:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    sources = {row["source_id"]: row for row in registry.get("sources") or []}
    sportec = sources.get("sportec_idsse") or {}
    paths = approved_paths()
    entries = [
        {
            "path": _relative(path),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "role": (
                "required_attribution"
                if path == ATTRIBUTION.resolve()
                else "licensed_derived_data"
            ),
        }
        for path in paths
    ]
    payload = archive_bytes(paths)
    excluded = sorted(
        row["source_id"]
        for row in registry.get("sources") or []
        if row.get("archive_eligible") is not True
    )
    return {
        "schema_version": 1,
        "archive_id": "gfs-idsse-licensed-data-supplement-v1",
        "status": "manifest_addressed_subset_ready",
        "format": "deterministic_zip_stored",
        "source_id": "sportec_idsse",
        "license": "CC-BY-4.0",
        "official_dataset_doi": sportec.get("official_dataset_doi"),
        "official_publication_doi": sportec.get("official_publication_doi"),
        "attribution": "docs/IDSSE_ATTRIBUTION.md",
        "change_notice_included": True,
        "file_count": len(entries),
        "total_uncompressed_bytes": sum(row["bytes"] for row in entries),
        "entries": entries,
        "deterministic_archive": {
            "fixed_timestamp": "1980-01-01T00:00:00Z",
            "compression": "stored",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
        "excluded_source_ids": excluded,
        "external_calls_made": False,
        "matches_executed": 0,
        "training_executed": False,
        "formal_experiment_executed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write-manifest", action="store_true")
    action.add_argument("--check", action="store_true")
    action.add_argument("--materialize", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    expected = expected_manifest()
    if args.write_manifest:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(
            json.dumps(expected, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        result = {"status": "written", "path": str(MANIFEST)}
    elif args.check:
        current = (
            json.loads(MANIFEST.read_text(encoding="utf-8"))
            if MANIFEST.is_file()
            else None
        )
        matches = current == expected
        result = {"status": "current" if matches else "stale", "path": str(MANIFEST)}
        print(json.dumps(result, ensure_ascii=False))
        return 0 if matches else 1
    else:
        output = args.materialize.resolve()
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite: {output}")
        current = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if current != expected:
            raise RuntimeError("data release manifest is stale")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(archive_bytes())
        result = {
            "status": "materialized",
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": file_sha256(output),
        }
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

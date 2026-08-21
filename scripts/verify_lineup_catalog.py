"""Audit player-level season-manager lineup coverage without running matches."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.lineup import build_squad_catalog


ROTATIONS = ("strongest", "balanced", "rotate")
SAFE_DEGRADATIONS = {"roster_missing_goalkeeper"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_lineup_catalog(root: str | Path) -> dict[str, Any]:
    resolved = Path(root).resolve()
    roster_paths = sorted((resolved / "data/rosters").glob("*.json"))
    teams = []
    manifest_rows = []
    for path in roster_paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        team = str(raw.get("team_id") or path.stem)
        catalog = build_squad_catalog(resolved, team)
        issues = []
        if catalog.get("available"):
            for rotation in ROTATIONS:
                lineup = (catalog.get("automatic") or {}).get(rotation) or {}
                if (
                    len(lineup.get("starters") or []) != 11
                    or len(lineup.get("bench") or []) > 12
                    or lineup.get("roster_fingerprint")
                    != catalog.get("roster_fingerprint")
                ):
                    issues.append(f"invalid_{rotation}_lineup")
            state = "player_level_ready" if not issues else "invalid"
        else:
            reason = str(catalog.get("reason") or "unknown")
            state = "safe_degraded" if reason in SAFE_DEGRADATIONS else "invalid"
            if state == "invalid":
                issues.append(f"unexpected_degradation:{reason}")
        teams.append({
            "team": team,
            "state": state,
            "reason": catalog.get("reason"),
            "selectable_players": sum(
                1 for player in catalog.get("players") or []
                if player.get("selectable")
            ),
            "roster_fingerprint": catalog.get("roster_fingerprint"),
            "issues": issues,
        })
        manifest_rows.append(f"{path.name}:{_sha256(path)}")
    invalid = [team for team in teams if team["state"] == "invalid"]
    ready = [team for team in teams if team["state"] == "player_level_ready"]
    degraded = [team for team in teams if team["state"] == "safe_degraded"]
    manifest_sha = hashlib.sha256("\n".join(manifest_rows).encode("utf-8")).hexdigest()
    return {
        "schema_version": 1,
        "verification": "gfs_season_manager_lineup_catalog",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_bounded_lineup_coverage" if not invalid and teams else "failed",
        "passed": bool(teams) and not invalid,
        "summary": {
            "roster_teams": len(teams),
            "player_level_ready": len(ready),
            "safe_degraded": len(degraded),
            "invalid": len(invalid),
        },
        "safe_degradation_contract": sorted(SAFE_DEGRADATIONS),
        "teams": teams,
        "artifact_sha256": {
            "roster_manifest": manifest_sha,
            "src/simulation/lineup.py": _sha256(resolved / "src/simulation/lineup.py"),
            "src/data_engine/roster_loader.py": _sha256(
                resolved / "src/data_engine/roster_loader.py"
            ),
        },
        "external_calls_made": False,
        "matches_executed": 0,
        "training_executed": False,
        "limitations": [
            "this audit validates roster completeness and frozen lineup construction, not match outcomes",
            "safe degradation preserves team-level gameplay but does not claim a player-level lineup",
            "source roster role accuracy still depends on the frozen roster data pipeline",
        ],
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--out", default="data/evaluation/lineup_catalog_verification_v1.json",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    report = verify_lineup_catalog(root)
    output = Path(args.out)
    if not output.is_absolute():
        output = root / output
    _atomic_json(output, report)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

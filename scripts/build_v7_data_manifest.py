#!/usr/bin/env python3
"""Build the reproducible v7 cross-provider data inventory."""
import hashlib, json
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()
def main():
    base = ROOT / "data/external"
    sportec = json.loads((base / "sportec/derived/manifest.json").read_text())
    statsbomb = json.loads((base / "statsbomb360/derived/manifest.json").read_text())
    skill = json.loads((base / "skillcorner/derived/manifest.json").read_text())
    metrica = list((base / "metrica").glob("pass_candidates_game*.csv"))
    compact = list((base / "metrica").glob("*.csv"))
    for provider in ("sportec", "statsbomb360", "skillcorner"):
        compact += list((base / provider / "derived").glob("*.csv"))
    providers = {
      "metrica": {"matches": len(metrica), "tracking_complete": True, "candidate_rows": sum(len(pd.read_csv(p)) for p in metrica)},
      "sportec": {"matches": len(sportec["matches"]), "tracking_complete": True, "tracking_frames": sum(m["tracking_frames"] for m in sportec["matches"]), "passes": sum(m["passes"] for m in sportec["matches"]), "shots": sum(m["shots"] for m in sportec["matches"])},
      "statsbomb360": {"matches": len(statsbomb["matches"]), "tracking_complete": False, "spatial_context": "event_freeze_frame_360", "passes": sum(m["passes"] for m in statsbomb["matches"]), "shots": sum(m["shots"] for m in statsbomb["matches"]), "excluded_invalid_upstream": statsbomb.get("excluded_invalid_upstream", [])},
      "skillcorner": {"matches": len(skill["matches"]), "tracking_complete": bool(skill["tracking_import_complete"]), "tracking_frames": sum(m["frames"] for m in skill["matches"]), "passes": sum(m["passes"] for m in skill["matches"]), "receiver_candidates": sum(m["receiver_candidates"] for m in skill["matches"]), "event_rows_imported": True}}
    report = {"schema_version": 1, "candidate": "7.0.0", "providers": providers,
              "continuous_tracking_providers": [name for name, value in providers.items() if value.get("tracking_complete")],
              "derived_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(compact)},
              "tracking_evidence_ready": providers["metrica"]["matches"] + providers["sportec"]["matches"] >= 8}
    target = ROOT / "data/releases/v7-data-candidate.json"
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())

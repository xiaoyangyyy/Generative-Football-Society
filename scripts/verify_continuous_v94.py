#!/usr/bin/env python3
"""Verify v9.4 artifacts, official Game 3 sources, and deployment boundary."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    failures=[]; release=json.loads((ROOT/"data/releases/v9.4.0-candidate.json").read_text()); manifest=json.loads((ROOT/"data/frame_world/v94_temporal_providers/manifest.json").read_text()); registry=json.loads((ROOT/"data/releases/model_registry.json").read_text()); current=json.loads((ROOT/"data/releases/current.json").read_text())
    for item in release["artifacts"]:
        path=ROOT/item["path"]
        if not path.is_file() or path.stat().st_size!=item["bytes"] or sha(path)!=item["sha256"]: failures.append(f"artifact:{item['path']}")
    game=next((x for x in manifest["matches"] if x["provider"]=="metrica" and x["match_id"]=="game3"),None)
    if game is None: failures.append("game3_missing")
    else:
        base=ROOT/"data/external/metrica/raw/Sample_Game_3"
        for name,value in game["source_sha256"].items():
            path=base/name
            if not path.is_file() or sha(path)!=value: failures.append(f"source:{name}")
    for provider in ("metrica","skillcorner","statsbomb360"):
        entry=registry["models"].get(f"continuous_time_lopo_{provider}:9.4.0-candidate",{}); path=Path(entry.get("artifact",""))
        if entry.get("status")!="evaluated" or not path.is_file() or sha(path)!=entry.get("sha256"): failures.append(f"registry:{provider}")
    if current.get("active")!="7.0.0" or current.get("rollback",{}).get("release")!="6.0.0": failures.append("deployment_boundary")
    result={"release":"9.4.0-candidate","metrica_matches":manifest["providers"]["metrica"]["matches"],"failures":failures,"ok":not failures}; print(json.dumps(result,indent=2)); return 0 if not failures else 1
if __name__=="__main__": raise SystemExit(main())
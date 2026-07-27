#!/usr/bin/env python3
"""Fast integrity, isolation, and lifecycle verification for frozen v8.9 LODO."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    failures=[]; release_path=ROOT/"data/releases/v8.9.0-candidate.json"; release=json.loads(release_path.read_text()); report=json.loads((ROOT/"reports/acceptance/temporal_mark_v89_lopo.json").read_text()); data=json.loads((ROOT/"data/frame_world/v89_temporal_providers/manifest.json").read_text()); registry=json.loads((ROOT/"data/releases/model_registry.json").read_text()); current=json.loads((ROOT/"data/releases/current.json").read_text())
    for item in release["artifacts"]:
        path=ROOT/item["path"]
        if not path.is_file() or path.stat().st_size!=item["bytes"] or digest(path)!=item["sha256"]: failures.append(f"release:{item['path']}")
    for item in data["matches"]:
        path=ROOT/item["file"]
        if not path.is_file() or digest(path)!=item["sha256"]: failures.append(f"data:{item['file']}")
    if not report.get("ready") or len(report.get("folds",[]))!=2: failures.append("report_ready")
    for fold in report.get("folds",[]):
        if fold["held_provider"] in fold["train_providers"] or not fold.get("ready") or not all(fold["gates"].values()): failures.append(f"isolation:{fold['held_provider']}")
    expected={"skillcorner":(10,5451),"statsbomb360":(417,306584)}
    for provider,(matches,events) in expected.items():
        value=data["providers"].get(provider,{})
        if (value.get("matches"),value.get("events"))!=(matches,events): failures.append(f"counts:{provider}")
    for name in ("temporal_mark_lopo_skillcorner","temporal_mark_lopo_statsbomb360"):
        entry=registry["models"].get(f"{name}:8.9.0-candidate",{})
        if entry.get("status")!="evaluated" or not Path(entry.get("artifact","")).is_file() or digest(Path(entry["artifact"]))!=entry.get("sha256"): failures.append(f"registry:{name}")
    if current.get("active")!="7.0.0" or current.get("rollback",{}).get("release")!="6.0.0": failures.append("deployment_boundary")
    result={"release":"8.9.0-candidate","data_files":len(data["matches"]),"folds":len(report.get("folds",[])),"failures":failures,"ok":not failures}; print(json.dumps(result,indent=2)); return 0 if not failures else 1
if __name__=="__main__": raise SystemExit(main())

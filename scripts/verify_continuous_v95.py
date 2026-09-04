#!/usr/bin/env python3
"""Verify v9.5 shadow artifacts and the unchanged deployment boundary."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    failures=[]; release=json.loads((ROOT/"data/releases/v9.5.0-candidate.json").read_text()); report=json.loads((ROOT/"reports/acceptance/continuous_clock_shadow_v95.json").read_text()); current=json.loads((ROOT/"data/releases/current.json").read_text())
    for item in release.get("artifacts",[]):
        path=ROOT/item["path"]
        if not path.is_file() or path.stat().st_size!=item["bytes"] or sha(path)!=item["sha256"]: failures.append(f"artifact:{item['path']}")
    if release.get("status")!="frozen_shadow_candidate" or not report.get("promotion_ready") or not all(report.get("gates",{}).values()): failures.append("shadow_gates")
    if current.get("active")!="7.0.0" or current.get("rollback",{}).get("release")!="6.0.0": failures.append("deployment_boundary")
    result={"release":"9.5.0-candidate","pairs":len(report.get("pairs",[])),"gates":len(report.get("gates",{})),"active":current.get("active"),"failures":failures,"ok":not failures}; print(json.dumps(result,indent=2)); return 0 if not failures else 1
if __name__=="__main__": raise SystemExit(main())

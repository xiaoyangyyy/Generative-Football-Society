#!/usr/bin/env python3
"""Verify v9.7 dual-clock candidate and deployment boundary."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 failures=[];release=json.loads((ROOT/"data/releases/v9.7.0-candidate.json").read_text());report=json.loads((ROOT/"reports/acceptance/subtick_reception_shadow_v97.json").read_text());current=json.loads((ROOT/"data/releases/current.json").read_text())
 for item in release.get("artifacts",[]):
  p=ROOT/item["path"]
  if not p.is_file() or p.stat().st_size!=item["bytes"] or sha(p)!=item["sha256"]:failures.append(f"artifact:{item['path']}")
 if release.get("status")!="frozen_dual_clock_candidate" or not report.get("ready") or not all(report.get("gates",{}).values()):failures.append("candidate_gates")
 agg=report.get("aggregate",{});protocol=report.get("protocol",{})
 if protocol.get("seconds")!=5400 or protocol.get("dt")!=5.5:failures.append("full_match_protocol")
 if agg.get("applied")!=agg.get("scheduled") or agg.get("cancelled")!=0:failures.append("reconciliation")
 if current.get("active")!="7.0.0" or current.get("rollback",{}).get("release")!="6.0.0":failures.append("deployment_boundary")
 result={"release":"9.7.0-candidate","status":"evaluated_not_deployed","events":agg.get("scheduled"),"active":current.get("active"),"failures":failures,"ok":not failures};print(json.dumps(result,indent=2));return 0 if not failures else 1
if __name__=="__main__":raise SystemExit(main())

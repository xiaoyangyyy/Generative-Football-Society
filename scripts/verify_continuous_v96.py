#!/usr/bin/env python3
"""Verify v9.6 rejection evidence and unchanged production pointers."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 failures=[];release=json.loads((ROOT/"data/releases/v9.6.0-candidate.json").read_text());report=json.loads((ROOT/"reports/acceptance/continuous_clock_long_shadow_v96.json").read_text());current=json.loads((ROOT/"data/releases/current.json").read_text())
 for item in release.get("artifacts",[]):
  path=ROOT/item["path"]
  if not path.is_file() or path.stat().st_size!=item["bytes"] or sha(path)!=item["sha256"]:failures.append(f"artifact:{item['path']}")
 if release.get("status")!="frozen_negative_evidence" or report.get("long_shadow_ready") is not False:failures.append("rejection_contract")
 if report.get("dt_seconds")!=5.5 or report.get("seconds_per_match")!=5400:failures.append("production_protocol")
 if report.get("aggregate",{}).get("gated_ticks")!=0:failures.append("cadence_diagnosis")
 if current.get("active")!="7.0.0" or current.get("rollback",{}).get("release")!="6.0.0":failures.append("deployment_boundary")
 result={"release":"9.6.0-candidate","status":"rejected_for_deployment","active":current.get("active"),"failures":failures,"ok":not failures};print(json.dumps(result,indent=2));return 0 if not failures else 1
if __name__=="__main__":raise SystemExit(main())

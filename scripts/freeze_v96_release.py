#!/usr/bin/env python3
"""Freeze v9.6 long-shadow rejection evidence without deployment."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data_engine.dataset_registry import write_json_atomic
VERSION="9.6.0-candidate"
def descriptor(path):return {"path":path.relative_to(ROOT).as_posix(),"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
def main():
 report=ROOT/"reports/acceptance/continuous_clock_long_shadow_v96.json";data=json.loads(report.read_text());g=data["gates"]
 required_pass=(g["verified_v93_artifact"] and g["full_match_horizon"] and g["mirrored_protocol"] and g["deterministic_treatment"] and g["completion_stable"] and g["mirror_possession_bias"]);required_fail=(not g["clock_exercised"] and not g["real_shot_rate_improved"] and not g["real_xg_rate_improved"] and not data["long_shadow_ready"])
 if not(required_pass and required_fail):raise RuntimeError("v9.6 does not match the accepted rejection contract")
 paths=[report,ROOT/"scripts/evaluate_continuous_clock_long_shadow_v96.py",ROOT/"data/releases/v9.5.0-candidate.json",ROOT/"reports/acceptance/continuous_clock_shadow_v95.json"]
 release={"schema_version":2,"release":VERSION,"status":"frozen_negative_evidence","parent_research":"9.5.0-candidate","parent_deployed":"7.0.0","rollback":"6.0.0","decision":"reject v9.5 production promotion; require independent sub-tick reception queue","artifacts":[descriptor(p) for p in paths],"gates":{"full_90_minute_shadow":True,"production_cadence_tested":True,"deterministic":True,"cadence_confounding_detected":True,"shot_xg_regression_detected":True,"promotion_rejected":True,"deployment_unchanged":True}}
 write_json_atomic(ROOT/f"data/releases/v{VERSION}.json",release);print(json.dumps({"release":VERSION,"status":release["status"],"artifacts":len(paths)},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())

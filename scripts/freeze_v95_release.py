#!/usr/bin/env python3
"""Freeze the v9.5 continuous-clock shadow candidate without deployment."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.data_engine.dataset_registry import write_json_atomic
VERSION="9.5.0-candidate"
def descriptor(path): return {"path":path.relative_to(ROOT).as_posix(),"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
def main():
    report=ROOT/"reports/acceptance/continuous_clock_shadow_v95.json"; evidence=json.loads(report.read_text())
    if not evidence.get("promotion_ready") or not all(evidence.get("gates",{}).values()): raise RuntimeError("v9.5 shadow evidence not ready")
    paths=[report,ROOT/"data/frame_world/continuous_time_v93_operational.pt",ROOT/"data/frame_world/continuous_intervals_v93.json",ROOT/"src/match_engine/continuous_micro_clock.py",ROOT/"src/match_engine/action_engine.py",ROOT/"src/match_engine/match_micro_runner.py",ROOT/"src/match_engine/micro_config.py",ROOT/"src/match_engine/state.py",ROOT/"scripts/evaluate_continuous_clock_shadow_v95.py",ROOT/"tests/test_continuous_micro_clock.py"]
    release={"schema_version":2,"release":VERSION,"status":"frozen_shadow_candidate","parent_research":"9.4.0-candidate","temporal_parent":"9.3.0-candidate","parent_deployed":"7.0.0","rollback":"6.0.0","deployment_policy":"shadow accepted; explicit production release decision still required","artifacts":[descriptor(p) for p in paths],"gates":evidence["gates"]|{"deployment_unchanged":True}}
    write_json_atomic(ROOT/f"data/releases/v{VERSION}.json",release); print(json.dumps({"release":VERSION,"artifacts":len(paths),"gates":len(release["gates"])},indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())

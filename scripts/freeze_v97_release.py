#!/usr/bin/env python3
"""Freeze v9.7 dual-clock evidence into immutable source snapshots."""
from __future__ import annotations
import hashlib
import json
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data_engine.dataset_registry import write_json_atomic
VERSION="9.7.0-candidate"
def descriptor(p):return {"path":p.relative_to(ROOT).as_posix(),"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
def main():
 report=ROOT/"reports/acceptance/subtick_reception_shadow_v97.json";data=json.loads(report.read_text())
 if not data.get("ready") or data.get("protocol",{}).get("seconds")!=5400 or not all(data.get("gates",{}).values()):raise RuntimeError("v9.7 full-match evidence not ready")
 sources=["src/match_engine/subtick_reception_queue.py","src/match_engine/action_engine.py","src/match_engine/match_micro_runner.py","src/match_engine/micro_config.py","src/match_engine/state.py","scripts/evaluate_subtick_reception_shadow_v97.py","tests/test_subtick_reception_queue.py"]
 snap=ROOT/"data/releases/snapshots/v9.7";snap.mkdir(parents=True,exist_ok=True);snapshots=[]
 for rel in sources:
  dst=snap/rel.replace("/","__");shutil.copyfile(ROOT/rel,dst);snapshots.append(dst)
 paths=[report,ROOT/"data/frame_world/continuous_time_v93_operational.pt",ROOT/"data/frame_world/continuous_intervals_v93.json",ROOT/"data/frame_world/frame_world_v85_pass_triplet_lodo_skillcorner.pt",*snapshots]
 release={"schema_version":2,"release":VERSION,"status":"frozen_dual_clock_candidate","parent_research":"9.6.0-candidate","parent_deployed":"7.0.0","rollback":"6.0.0","deployment_policy":"evaluated full-match shadow; default-off and not deployed","architecture":{"macro_tick_s":5.5,"subtick_queue":True,"transition_contract":"damped learned residual only","rng_draws":0},"artifacts":[descriptor(p) for p in paths],"gates":data["gates"]|{"immutable_source_snapshot":True,"deployment_unchanged":True}}
 write_json_atomic(ROOT/f"data/releases/v{VERSION}.json",release);print(json.dumps({"release":VERSION,"artifacts":len(paths),"gates":len(release["gates"])},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())

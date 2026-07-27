#!/usr/bin/env python3
"""Matched-seed production-cadence shadow for the dual-clock reception queue."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.app import run_micro_match
from src.data_engine.dataset_registry import write_json_atomic
from src.match_engine.micro_config import MicroMatchConfig
FIXTURES=[("Brazil","Argentina"),("Argentina","Brazil"),("France","Spain"),("Spain","France")];SEEDS=[7,42,101,313]
def cfg(enabled,seconds):
 c=MicroMatchConfig.fast_demo();c.dt_default=5.5;c.match_seconds=float(seconds);c.enable_subtick_reception_queue=enabled;return c
def scalar(s):return {"passes":s.passes_home+s.passes_away,"shots":s.shots_home+s.shots_away,"completion":(s.pass_completion_home+s.pass_completion_away)/2,"possession_home":s.possession_home,"xg":s.micro_xg_home+s.micro_xg_away,"queue":s.subtick_reception_queue}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--seconds",type=float,default=600.);args=ap.parse_args();rows=[];start=time.perf_counter()
 for i,(home,away) in enumerate(FIXTURES):
  b=scalar(run_micro_match(home,away,seed=SEEDS[i],config=cfg(False,args.seconds)));t=scalar(run_micro_match(home,away,seed=SEEDS[i],config=cfg(True,args.seconds)));rows.append({"home":home,"away":away,"seed":SEEDS[i],"baseline":b,"treatment":t,"delta":{k:t[k]-b[k] for k in ("passes","shots","completion","possession_home","xg")}})
 repeat=scalar(run_micro_match(FIXTURES[0][0],FIXTURES[0][1],seed=SEEDS[0],config=cfg(True,args.seconds)));scale=args.seconds/5400.;targets={"passes":1070.54*scale,"shots":23.34*scale,"xg":2.6*scale};errors={name:{arm:float(np.mean([abs(r[arm][name]-target) for r in rows])) for arm in ("baseline","treatment")} for name,target in targets.items()};scheduled=sum(r["treatment"]["queue"]["scheduled"] for r in rows);applied=sum(r["treatment"]["queue"]["applied"] for r in rows);cancelled=sum(r["treatment"]["queue"]["cancelled"] for r in rows);mean_disp=float(np.mean([r["treatment"]["queue"]["mean_applied_displacement_m"] for r in rows]));pos=float(np.mean([r["delta"]["possession_home"] for r in rows]))
 gates={"production_cadence":True,"deterministic":repeat==rows[0]["treatment"],"queue_exercised":scheduled>=100 and applied>0,"reconciliation_rate":applied/max(1,scheduled)>=.95 and cancelled/max(1,scheduled)<=.05,"zero_rng_draws":all(r["treatment"]["queue"]["rng_draws"]==0 for r in rows),"nontrivial_local_effect":mean_disp>.01,"pass_rate_noninferior":errors["passes"]["treatment"]<=errors["passes"]["baseline"]+5,"shot_rate_noninferior":errors["shots"]["treatment"]<=errors["shots"]["baseline"]+1,"xg_rate_noninferior":errors["xg"]["treatment"]<=errors["xg"]["baseline"]+.08,"completion_stable":float(np.mean([abs(r["delta"]["completion"]) for r in rows]))<=.05,"mirror_possession_bias":abs(pos)<=.05};gates={k:bool(v) for k,v in gates.items()};report={"version":"9.7.0-candidate","protocol":{"seconds":args.seconds,"dt":5.5,"fixtures":"two mirrored pairs","macro_actions":"unchanged; max one per tick","subtick_rng_draws":0},"pairs":rows,"aggregate":{"targets":targets,"mean_absolute_target_errors":errors,"scheduled":scheduled,"applied":applied,"cancelled":cancelled,"application_rate":applied/max(1,scheduled),"mean_applied_displacement_m":mean_disp,"mean_signed_mirror_possession_drift":pos,"elapsed_s":time.perf_counter()-start},"gates":gates,"ready":all(gates.values()),"deployment_unchanged":True};write_json_atomic(ROOT/"reports/acceptance/subtick_reception_shadow_v97.json",report);print(json.dumps(report["aggregate"]|{"gates":gates,"ready":report["ready"]},indent=2));return 0 if report["ready"] else 1
if __name__=="__main__":raise SystemExit(main())

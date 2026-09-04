#!/usr/bin/env python3
"""Matched-seed v7 vs v9.3 continuous-clock shadow tournament audit."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.app import run_micro_match
from src.match_engine.frame_world.calibration import load_calibrated_temporal_router
from src.match_engine.micro_config import MicroMatchConfig
from src.data_engine.dataset_registry import write_json_atomic

FIXTURES=[("Brazil","Argentina"),("Argentina","Brazil"),("France","Spain"),("Spain","France"),("Germany","England"),("England","Germany"),("Japan","Morocco"),("Morocco","Japan")]

def config(enabled,seconds):
    cfg=MicroMatchConfig.fast_demo(); cfg.dt_default=.5; cfg.match_seconds=float(seconds); cfg.enable_continuous_event_clock=enabled; return cfg

def scalar(s):
    return {"passes":s.passes_home+s.passes_away,"shots":s.shots_home+s.shots_away,"completion":(s.pass_completion_home+s.pass_completion_away)/2,"possession_home":s.possession_home,"xg":s.micro_xg_home+s.micro_xg_away,"goals":s.goals_micro_home+s.goals_micro_away,"clock":s.continuous_clock}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seconds',type=float,default=120.); ap.add_argument('--pairs',type=int,default=8); args=ap.parse_args()
    artifact=ROOT/'data/frame_world/continuous_intervals_v93.json'; router=load_calibrated_temporal_router(artifact); rows=[]; started=time.perf_counter()
    for i,(home,away) in enumerate(FIXTURES[:args.pairs]):
        seed=[7,42,101,313,911,2026,4096,8191][i]
        baseline=run_micro_match(home,away,seed=seed,config=config(False,args.seconds))
        treatment=run_micro_match(home,away,seed=seed,config=config(True,args.seconds))
        b,t=scalar(baseline),scalar(treatment); rows.append({"home":home,"away":away,"seed":seed,"baseline":b,"treatment":t,"delta":{k:t[k]-b[k] for k in ('passes','shots','completion','possession_home','xg','goals')}})
    repeat=scalar(run_micro_match(FIXTURES[0][0],FIXTURES[0][1],seed=7,config=config(True,args.seconds)))
    reductions=np.array([1-r['treatment']['passes']/max(1,r['baseline']['passes']) for r in rows]); completion=np.array([r['delta']['completion'] for r in rows]); possession=np.array([r['delta']['possession_home'] for r in rows]); xg=np.array([r['delta']['xg'] for r in rows]); shot_ratio=sum(r['treatment']['shots'] for r in rows)/max(1,sum(r['baseline']['shots'] for r in rows)); plans=sum(r['treatment']['clock']['plans'] for r in rows); gated=sum(r['treatment']['clock']['gated_ticks'] for r in rows)
    scale=args.seconds/5400.; targets={"passes":2*535.27*scale,"shots":2*11.67*scale,"xg":2.6*scale}
    errors={name:{"baseline":float(np.mean([abs(r["baseline"][name]-target) for r in rows])),"treatment":float(np.mean([abs(r["treatment"][name]-target) for r in rows]))} for name,target in targets.items()}
    gates={"verified_v93_artifact":set(router.temporal_models)=={"metrica","skillcorner","statsbomb360"},"deterministic_treatment":repeat==rows[0]["treatment"],"clock_exercised":plans>0 and gated>0,"action_density_reduction":.2<=float(np.median(reductions))<=.9,"real_pass_rate_improved":errors["passes"]["treatment"]<errors["passes"]["baseline"] and np.mean([r["treatment"]["passes"] for r in rows])<=2*targets["passes"],"real_shot_rate_improved":errors["shots"]["treatment"]<errors["shots"]["baseline"],"real_xg_rate_improved":errors["xg"]["treatment"]<errors["xg"]["baseline"],"completion_stable":float(np.mean(np.abs(completion)))<=.15,"mirror_possession_bias":float(abs(np.mean(possession)))<=.08}
    gates={key:bool(value) for key,value in gates.items()}
    report={"version":"9.5.0-shadow","comparison":"deployed v7 default clock vs v9.3 calibrated continuous clock","policy":"matched fixture and seed; default production path unchanged","seconds_per_match":args.seconds,"pairs":rows,"aggregate":{"median_pass_density_reduction":float(np.median(reductions)),"mean_absolute_completion_drift":float(np.mean(np.abs(completion))),"mean_absolute_possession_drift":float(np.mean(np.abs(possession))),"mean_xg_drift":float(np.mean(xg)),"shot_volume_ratio":float(shot_ratio),"real_rate_targets":targets,"mean_absolute_target_errors":errors,"plans":plans,"gated_ticks":gated,"elapsed_s":time.perf_counter()-started},"gates":gates,"promotion_ready":all(gates.values()),"deployment_unchanged":True}
    write_json_atomic(ROOT/'reports/acceptance/continuous_clock_shadow_v95.json',report); print(json.dumps(report['aggregate']|{"gates":gates,"promotion_ready":report['promotion_ready']},indent=2)); return 0 if report['promotion_ready'] else 1
if __name__=='__main__': raise SystemExit(main())

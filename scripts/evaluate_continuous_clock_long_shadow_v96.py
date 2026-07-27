#!/usr/bin/env python3
"""Checkpointed full-match v7 vs continuous-clock mirrored shadow audit."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.app import run_micro_match
from src.data_engine.dataset_registry import write_json_atomic
from src.match_engine.frame_world.calibration import load_calibrated_temporal_router
from src.match_engine.micro_config import MicroMatchConfig

FIXTURES=[("Brazil","Argentina"),("Argentina","Brazil"),("France","Spain"),("Spain","France"),("Germany","England"),("England","Germany"),("Japan","Morocco"),("Morocco","Japan")]
SEEDS=[7,42,101,313,911,2026,4096,8191]

def config(enabled,seconds,dt):
    cfg=MicroMatchConfig.fast_demo();cfg.dt_default=float(dt);cfg.match_seconds=float(seconds);cfg.enable_continuous_event_clock=enabled;return cfg

def scalar(s):
    return {"passes":s.passes_home+s.passes_away,"shots":s.shots_home+s.shots_away,"completion":(s.pass_completion_home+s.pass_completion_away)/2,"possession_home":s.possession_home,"xg":s.micro_xg_home+s.micro_xg_away,"goals":s.goals_micro_home+s.goals_micro_away,"clock":s.continuous_clock}

def aggregate(rows,seconds):
    scale=seconds/5400.;targets={"passes":2*535.27*scale,"shots":2*11.67*scale,"xg":2.6*scale}
    errors={name:{arm:float(np.mean([abs(r[arm][name]-target) for r in rows])) for arm in ("baseline","treatment")} for name,target in targets.items()}
    reductions=np.array([1-r["treatment"]["passes"]/max(1,r["baseline"]["passes"]) for r in rows]);completion=np.array([r["delta"]["completion"] for r in rows]);possession=np.array([r["delta"]["possession_home"] for r in rows])
    return {"real_rate_targets":targets,"mean_absolute_target_errors":errors,"median_pass_density_reduction":float(np.median(reductions)),"mean_absolute_completion_drift":float(np.mean(np.abs(completion))),"mean_signed_mirror_possession_drift":float(np.mean(possession)),"mean_absolute_possession_drift":float(np.mean(np.abs(possession))),"plans":sum(r["treatment"]["clock"]["plans"] for r in rows),"gated_ticks":sum(r["treatment"]["clock"]["gated_ticks"] for r in rows)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--seconds",type=float,default=5400.);ap.add_argument("--dt",type=float,default=2.);ap.add_argument("--pairs",type=int,default=4);ap.add_argument("--resume",action="store_true");args=ap.parse_args()
    if args.pairs%2: raise SystemExit("pairs must be even to preserve mirrored fixtures")
    artifact=ROOT/"data/frame_world/continuous_intervals_v93.json";router=load_calibrated_temporal_router(artifact);checkpoint=ROOT/"reports/acceptance/continuous_clock_long_shadow_v96.checkpoint.json";target=ROOT/"reports/acceptance/continuous_clock_long_shadow_v96.json";rows=[]
    if args.resume and checkpoint.is_file():
        saved=json.loads(checkpoint.read_text());
        if saved.get("seconds")!=args.seconds or saved.get("dt")!=args.dt or saved.get("pairs_requested")!=args.pairs: raise RuntimeError("checkpoint protocol mismatch")
        rows=saved.get("pairs",[])
    started=time.perf_counter()
    for i,(home,away) in enumerate(FIXTURES[:args.pairs]):
        if i<len(rows): continue
        baseline=run_micro_match(home,away,seed=SEEDS[i],config=config(False,args.seconds,args.dt));treatment=run_micro_match(home,away,seed=SEEDS[i],config=config(True,args.seconds,args.dt));b,t=scalar(baseline),scalar(treatment)
        rows.append({"home":home,"away":away,"seed":SEEDS[i],"baseline":b,"treatment":t,"delta":{k:t[k]-b[k] for k in ("passes","shots","completion","possession_home","xg","goals")}})
        write_json_atomic(checkpoint,{"version":"9.6.0-long-shadow","seconds":args.seconds,"dt":args.dt,"pairs_requested":args.pairs,"pairs":rows})
    repeat=scalar(run_micro_match(FIXTURES[0][0],FIXTURES[0][1],seed=SEEDS[0],config=config(True,args.seconds,args.dt)));agg=aggregate(rows,args.seconds);errors=agg["mean_absolute_target_errors"]
    gates={"verified_v93_artifact":set(router.temporal_models)=={"metrica","skillcorner","statsbomb360"},"full_match_horizon":args.seconds>=5400,"mirrored_protocol":args.pairs%2==0,"deterministic_treatment":repeat==rows[0]["treatment"],"clock_exercised":agg["plans"]>0 and agg["gated_ticks"]>0,"real_pass_rate_improved":errors["passes"]["treatment"]<errors["passes"]["baseline"] and errors["passes"]["treatment"]<=.35*agg["real_rate_targets"]["passes"],"real_shot_rate_improved":errors["shots"]["treatment"]<errors["shots"]["baseline"],"real_xg_rate_improved":errors["xg"]["treatment"]<errors["xg"]["baseline"],"completion_stable":agg["mean_absolute_completion_drift"]<=.12,"mirror_possession_bias":abs(agg["mean_signed_mirror_possession_drift"])<=.05}
    gates={k:bool(v) for k,v in gates.items()};report={"version":"9.6.0-long-shadow","comparison":"v7 default tick clock vs v9.3 calibrated continuous clock","seconds_per_match":args.seconds,"dt_seconds":args.dt,"pairs":rows,"aggregate":agg|{"elapsed_this_run_s":time.perf_counter()-started},"gates":gates,"long_shadow_ready":all(gates.values()),"deployment_unchanged":True};write_json_atomic(target,report);print(json.dumps(report["aggregate"]|{"gates":gates,"long_shadow_ready":report["long_shadow_ready"]},indent=2));return 0 if report["long_shadow_ready"] else 1
if __name__=="__main__":raise SystemExit(main())

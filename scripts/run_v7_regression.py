#!/usr/bin/env python3
"""Fixed-seed distribution, determinism, and latency regression for v7."""
from __future__ import annotations
from dataclasses import asdict
import json, math, sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.app import run_micro_match, run_monte_carlo

def timed(call):
    start=time.perf_counter(); value=call(); return value,(time.perf_counter()-start)*1000
def scalar_micro(summary):
    data=asdict(summary)
    keys=("goals_micro_home","goals_micro_away","shots_home","shots_away","passes_home","passes_away","pass_completion_home","pass_completion_away","micro_xg_home","micro_xg_away")
    return {key:data[key] for key in keys}
def main():
    monte,seeds,times=[],[7,42,101,2026],[]
    for seed in seeds:
        rows,elapsed=timed(lambda seed=seed:run_monte_carlo(20,seed=seed)); monte.append(rows); times.append(elapsed)
    repeated,repeat_ms=timed(lambda:run_monte_carlo(20,seed=42))
    aggregate={row["team"]:0.0 for row in monte[0]}
    for rows in monte:
        for row in rows: aggregate[row["team"]]+=row["probability"]/len(monte)
    values=np.array(list(aggregate.values())); positive=values[values>0]
    pairs=[("Brazil","Argentina",42),("France","Spain",101),("Germany","England",2026),("Japan","Morocco",7)]
    micro=[]; micro_times=[]
    for home,away,seed in pairs:
        summary,elapsed=timed(lambda home=home,away=away,seed=seed:run_micro_match(home,away,seed=seed,fast=True))
        micro.append({"home":home,"away":away,"seed":seed,**scalar_micro(summary)}); micro_times.append(elapsed)
    repeated_micro,repeat_micro_ms=timed(lambda:run_micro_match("Brazil","Argentina",seed=42,fast=True))
    repeat_scalar=scalar_micro(repeated_micro)
    numeric=[v for row in micro for v in row.values() if isinstance(v,(int,float))]
    gates={
      "monte_probability_mass":all(abs(sum(r["probability"] for r in rows)-1)<1e-9 for rows in monte),
      "monte_deterministic":monte[1]==repeated,
      "champion_diversity":int((values>0).sum())>=8,
      "champion_concentration":float(values.max())<=.35,
      "micro_finite":bool(np.isfinite(numeric).all()),
      "micro_deterministic":micro[0] | {"home":"Brazil","away":"Argentina","seed":42} == {"home":"Brazil","away":"Argentina","seed":42,**repeat_scalar},
      "micro_score_sane":float(np.mean([r["goals_micro_home"]+r["goals_micro_away"] for r in micro]))<=8,
      "micro_completion_sane":all(0<=r[key]<=1 for r in micro for key in ("pass_completion_home","pass_completion_away")),
      "monte_p95_latency":float(np.quantile(times+[repeat_ms],.95))<=20000,
      "micro_p95_latency":float(np.quantile(micro_times+[repeat_micro_ms],.95))<=30000}
    report={"version":"7.0.0","seeds":seeds,"monte_carlo":{"simulations":80,"champion_teams":int((values>0).sum()),"max_share":float(values.max()),"entropy":float(-(positive*np.log(positive)).sum()),"p95_ms":float(np.quantile(times+[repeat_ms],.95))},"micro":{"matches":micro,"mean_total_goals":float(np.mean([r["goals_micro_home"]+r["goals_micro_away"] for r in micro])),"p95_ms":float(np.quantile(micro_times+[repeat_micro_ms],.95))},"gates":gates,"release_ready":all(gates.values())}
    target=ROOT/"reports/acceptance/v7_regression.json"; target.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8"); print(json.dumps(report,indent=2)); return 0 if report["release_ready"] else 1
if __name__=="__main__": raise SystemExit(main())

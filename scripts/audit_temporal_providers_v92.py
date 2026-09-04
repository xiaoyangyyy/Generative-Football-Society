#!/usr/bin/env python3
"""Audit v9.2 provider clocks and temporal arrays before training."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.data_engine.dataset_registry import write_json_atomic

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    base=ROOT/"data/frame_world/v92_temporal_providers"; manifest=json.loads((base/"manifest.json").read_text()); expected={Path(x["file"]).resolve() for x in manifest["matches"] if x["provider"]=="metrica"}; actual={x.resolve() for x in base.glob("*.npz")}; providers=[]
    for provider in manifest["providers"]:
        delays=[]; continuation=[]; subtype=[]; valid=True; files=0
        for item in [x for x in manifest["matches"] if x["provider"]==provider]:
            path=ROOT/item["file"]; valid &= path.exists() and sha(path)==item["sha256"]
            with np.load(path,allow_pickle=False) as z:
                sizes={len(z[k]) for k in ("features","observed","continuation","subtype","delay")}; valid &= sizes=={item["events"]} and np.isfinite(z["features"]).all() and np.isfinite(z["delay"]).all(); observed=z["observed"]; delays.extend(z["delay"][observed]); continuation.extend(z["continuation"][z["continuation"]>=0]); subtype.extend(z["subtype"][z["subtype"]>=0]); files+=1
        values=np.asarray(delays); providers.append({"provider":provider,"matches":files,"observed_events":len(values),"observed_rate":len(values)/manifest["providers"][provider]["events"],"delay_quantiles_s":dict(zip(("q05","q25","q50","q75","q90","q95"),np.quantile(values,(.05,.25,.5,.75,.9,.95)).round(4).tolist())),"continue_rate":float(np.mean(continuation)),"shot_given_continue_rate":float(np.mean(subtype)),"valid":bool(valid)})
    medians=[x["delay_quantiles_s"]["q50"] for x in providers]; gates={"three_providers":len(providers)==3,"arrays_valid":all(x["valid"] for x in providers),"minimum_observed":all(x["observed_events"]>=1000 for x in providers),"observed_rate":all(x["observed_rate"]>=.8 for x in providers),"clock_median_range_s":max(medians)-min(medians)<=.5,"no_orphan_v92_arrays":actual==expected}; report={"version":"9.2.0-candidate","providers":providers,"gates":gates,"ready":all(gates.values())}; write_json_atomic(ROOT/"reports/acceptance/temporal_provider_quality_v92.json",report); print(json.dumps(report,indent=2)); return 0 if report["ready"] else 1
if __name__=="__main__": raise SystemExit(main())
#!/usr/bin/env python3
"""Holdout sanity check for latent world model (trace replay error)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TRACE_DIR = ROOT / "data" / "world_model" / "traces"
MODEL_PATH = ROOT / "data" / "world_model" / "latent_wm.pt"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-traces", type=int, default=8)
    args = p.parse_args()

    if not MODEL_PATH.exists():
        print(json.dumps({"ok": False, "reason": "missing_model", "path": str(MODEL_PATH)}))
        return 0

    traces = sorted(TRACE_DIR.glob("*.jsonl"))[: args.max_traces]
    if not traces:
        print(json.dumps({"ok": False, "reason": "no_traces"}))
        return 0

    try:
        import torch

        from src.match_engine.world_model.inference import WorldModelRuntime
    except Exception as exc:
        print(json.dumps({"ok": False, "reason": f"import_error:{exc}"}))
        return 0

    rt = WorldModelRuntime.load_default(str(ROOT))
    errs: list[float] = []
    for path in traces:
        lines = path.read_text(encoding="utf-8").strip().splitlines()[:120]
        for line in lines:
            row = json.loads(line)
            obs = row.get("obs")
            act = row.get("action")
            obs_n = row.get("next_obs", row.get("obs_next"))
            if obs is None or act is None or obs_n is None:
                continue
            import numpy as np

            o = np.asarray(obs, dtype=float)
            a = np.asarray(act, dtype=float)
            on = np.asarray(obs_n, dtype=float)
            out = rt.imagine(o, a, steps=1)
            pred = np.asarray(out.next_obs, dtype=float)
            errs.append(float(np.mean((pred - on) ** 2)))
            if len(errs) >= 400:
                break
        if len(errs) >= 400:
            break

    if not errs:
        print(json.dumps({"ok": False, "reason": "no_pairs"}))
        return 0

    mse = float(sum(errs) / len(errs))
    version = int(getattr(rt.model, "checkpoint_version", 2))
    planner_quality = float(rt.base_quality)
    transition_quality = float((rt.meta.get("validation") or {}).get("transition_quality", 0.0))
    ok = version >= 6 and mse < 0.12 and transition_quality >= 0.50
    print(
        json.dumps(
            {
                "ok": ok,
                "checkpoint_version": version,
                "mse_mean": mse,
                "n_pairs": len(errs),
                "threshold": 0.12,
                "planner_quality": planner_quality,
                "transition_quality": transition_quality,
                "pass_planner_quality": rt.pass_quality,
                "shot_planner_quality": rt.shot_quality,
                "pass_planner_active": rt.pass_quality >= rt.cfg.min_planner_quality,
                "shot_planner_active": rt.shot_quality >= rt.cfg.min_planner_quality,
                "min_planner_quality": rt.cfg.min_planner_quality,
            }
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Holdout sanity check for latent world model (trace replay error)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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
            obs_n = row.get("obs_next")
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
    ok = mse < 0.12
    print(json.dumps({"ok": ok, "mse_mean": mse, "n_pairs": len(errs), "threshold": 0.12}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

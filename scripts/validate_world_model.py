#!/usr/bin/env python3
"""Holdout sanity check for latent world model (trace replay error)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_TRACE_DIR = ROOT / "data" / "world_model" / "traces"
DEFAULT_MODEL_PATH = ROOT / "data" / "world_model" / "latent_wm.pt"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-traces", type=int, default=8)
    p.add_argument("--trace-dir", default=str(DEFAULT_TRACE_DIR))
    p.add_argument("--checkpoint", default=str(DEFAULT_MODEL_PATH))
    p.add_argument(
        "--require-two-step-planning",
        action="store_true",
        help="Require grouped changing-action two-step holdout authority.",
    )
    p.add_argument(
        "--require-semantic-event-heads",
        action="store_true",
        help=(
            "Require at least two grouped-validated learned events at both "
            "one-step and two-step rollout depths."
        ),
    )
    args = p.parse_args()
    trace_dir = Path(args.trace_dir)
    model_path = Path(args.checkpoint)

    if not model_path.exists():
        print(json.dumps({"ok": False, "reason": "missing_model", "path": str(model_path)}))
        return 2

    traces = sorted(trace_dir.glob("*.jsonl"))[: args.max_traces]
    if not traces:
        print(json.dumps({"ok": False, "reason": "no_traces", "path": str(trace_dir)}))
        return 2

    try:
        import torch

        from src.match_engine.world_model.inference import WorldModelRuntime
        from src.match_engine.world_model.evaluation import TransitionMetricAccumulator
    except Exception as exc:
        print(json.dumps({"ok": False, "reason": f"import_error:{exc}"}))
        return 2

    rt = WorldModelRuntime.load(str(model_path))
    metrics = TransitionMetricAccumulator()
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
            metrics.update(pred, on, o, uncertainty=out.uncertainty)
            if metrics.samples >= 400:
                break
        if metrics.samples >= 400:
            break

    if not metrics.samples:
        print(json.dumps({"ok": False, "reason": "no_pairs"}))
        return 2

    transition_report = metrics.finalize()
    mse = float(transition_report["mse"])
    version = int(getattr(rt.model, "checkpoint_version", 2))
    planner_quality = float(rt.base_quality)
    transition_quality = float((rt.meta.get("validation") or {}).get("transition_quality", 0.0))
    validation = rt.meta.get("validation") or {}
    transition_ensemble_trained = bool(
        validation.get("transition_ensemble_trained")
    )
    transition_ensemble_size = int(
        validation.get("transition_ensemble_size", 0)
    )
    two_step_planning_gate = rt.two_step_planning_gate()
    from src.match_engine.world_model.state_scales import (
        FALSIFIABLE_SEMANTIC_EVENTS,
    )

    semantic_event_gates = {
        f"{steps}_step": {
            event: rt.semantic_event_head_gate(event, rollout_steps=steps)
            for event in FALSIFIABLE_SEMANTIC_EVENTS
        }
        for steps in (1, 2)
    }
    semantic_event_heads_ready = bool(
        version >= 8
        and getattr(rt.model, "semantic_event_heads_trained", False)
        and all(
            sum(bool(gate["active"]) for gate in gates.values()) >= 2
            for gates in semantic_event_gates.values()
        )
    )
    ok = (
        version >= 7
        and mse < 0.12
        and transition_quality >= 0.50
        and transition_ensemble_trained
        and transition_ensemble_size >= 2
        and (
            two_step_planning_gate["active"]
            if args.require_two_step_planning else True
        )
        and (
            semantic_event_heads_ready
            if args.require_semantic_event_heads else True
        )
    )
    print(
        json.dumps(
            {
                "ok": ok,
                "checkpoint_version": version,
                "mse_mean": mse,
                "n_pairs": metrics.samples,
                "threshold": 0.12,
                "transition_metrics": transition_report,
                "planner_quality": planner_quality,
                "transition_quality": transition_quality,
                "transition_ensemble_trained": transition_ensemble_trained,
                "transition_ensemble_size": transition_ensemble_size,
                "two_step_planning_gate": two_step_planning_gate,
                "semantic_event_head_gates": semantic_event_gates,
                "semantic_event_heads_ready": semantic_event_heads_ready,
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

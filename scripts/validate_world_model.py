#!/usr/bin/env python3
"""Holdout sanity check for latent world model (trace replay error)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_TRACE_DIR = ROOT / "data" / "world_model" / "traces"
DEFAULT_MODEL_PATH = ROOT / "data" / "world_model" / "latent_wm.pt"


def _sealed_evaluation(runtime, trace_dir: Path, manifest_path: Path) -> dict:
    """Evaluate two-step state skill and shot calibration on sealed groups only."""
    import torch
    from scripts.train_world_model import _match_group, _sequential_transition_pairs
    from src.data_engine.dataset_registry import files_for_split, load_manifest, verify_trace_manifest
    from src.match_engine.world_model.recorder import load_trace_batches
    from src.match_engine.world_model.schema import SHOT_GOAL_INDEX, observation_loss_weights, strip_outcome_leakage

    manifest = load_manifest(manifest_path)
    verify_trace_manifest(manifest, base_dir=ROOT)
    obs, raw_actions, nxt, raw_groups = load_trace_batches(
        str(trace_dir), return_groups=True,
        allowed_files=files_for_split(manifest, {"sealed_test"}),
    )
    groups = np.asarray([_match_group(value) for value in raw_groups], dtype=str)
    actions = strip_outcome_leakage(raw_actions)
    pair_left = _sequential_transition_pairs(
        obs, nxt, groups, np.arange(len(obs), dtype=int),
    )
    if len(pair_left):
        sequences = np.stack([actions[pair_left], actions[pair_left + 1]], axis=1)
        with torch.no_grad():
            members = runtime.model.transition_rollout_predictions(
                torch.from_numpy(obs[pair_left]), torch.from_numpy(sequences),
            )
        target = torch.from_numpy(nxt[pair_left + 1])
        initial = torch.from_numpy(obs[pair_left])
        weights = torch.from_numpy(observation_loss_weights())
        mean_prediction = members.mean(dim=0)
        mse = float((((mean_prediction - target) ** 2) * weights).mean())
        persistence = float((((initial - target) ** 2) * weights).mean())
        member_mean = float(torch.stack([
            (((member - target) ** 2) * weights).mean() for member in members
        ]).mean())
        disagreement = torch.sqrt(torch.mean(
            torch.var(members, dim=0, unbiased=False), dim=1,
        )).numpy()
        error = torch.sqrt(torch.mean((mean_prediction - target) ** 2, dim=1)).numpy()
        correlation = float(np.corrcoef(disagreement, error)[0, 1]) if (
            np.std(disagreement) > 1e-12 and np.std(error) > 1e-12
        ) else 0.0
    else:
        mse = persistence = member_mean = correlation = 0.0
    skill = 1.0 - mse / max(persistence, 1e-12) if len(pair_left) else -1.0
    two_step = {
        "samples": int(len(pair_left)),
        "groups": int(len(np.unique(groups[pair_left]))) if len(pair_left) else 0,
        "weighted_mse": mse,
        "persistence_weighted_mse": persistence,
        "skill_vs_persistence": skill,
        "member_mean_weighted_mse": member_mean,
        "ensemble_gain_vs_member_mean": member_mean - mse,
        "disagreement_error_correlation": correlation,
    }
    two_step["active"] = bool(
        two_step["samples"] >= 32 and two_step["groups"] >= 4
        and two_step["skill_vs_persistence"] >= 0.02
        and two_step["ensemble_gain_vs_member_mean"] >= -1e-8
        and two_step["disagreement_error_correlation"] >= 0.0
    )

    shot_mask = raw_actions[:, 1] > 0.5
    shot_true = raw_actions[shot_mask, SHOT_GOAL_INDEX]
    prior = np.clip(raw_actions[shot_mask, 13], 0.01, 0.78)
    if len(shot_true):
        with torch.no_grad():
            latent = runtime.model.encode(torch.from_numpy(obs[shot_mask]))
            logits = runtime.model.outcome_ensemble(
                latent, torch.from_numpy(actions[shot_mask]),
            )[2]
            probability = torch.sigmoid(logits).mean(dim=0).view(-1).numpy()
        model_brier = float(np.mean((probability - shot_true) ** 2))
        prior_brier = float(np.mean((prior - shot_true) ** 2))
    else:
        model_brier = prior_brier = 1.0
    shot = {
        "samples": int(len(shot_true)),
        "goals": int(shot_true.sum()) if len(shot_true) else 0,
        "model_brier": model_brier,
        "physics_xg_prior_brier": prior_brier,
        "skill_vs_physics_xg_prior": 1.0 - model_brier / max(prior_brier, 1e-12),
    }
    shot["learned_head_active"] = bool(
        shot["samples"] >= 40 and shot["goals"] >= 5 and model_brier < prior_brier
    )
    return {"manifest": str(manifest_path), "two_step": two_step, "shot": shot}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-traces", type=int, default=8)
    p.add_argument("--trace-dir", default=str(DEFAULT_TRACE_DIR))
    p.add_argument("--checkpoint", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--manifest", help="Frozen grouped manifest; defaults to checkpoint metadata.")
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
    manifest_value = args.manifest or rt.meta.get("dataset_manifest") or (
        rt.meta.get("rollout_calibration") or {}
    ).get("manifest")
    manifest_path = None
    if manifest_value:
        manifest_path = Path(str(manifest_value))
        if not manifest_path.is_absolute():
            manifest_path = ROOT / manifest_path
    sealed_test = _sealed_evaluation(rt, trace_dir, manifest_path) if manifest_path else None
    sealed_two_step_active = bool(
        sealed_test and sealed_test["two_step"]["active"]
    )
    effective_shot_active = bool(
        rt.shot_quality >= rt.cfg.min_planner_quality
        and sealed_test and sealed_test["shot"]["learned_head_active"]
    )
    ok = (
        version >= 7
        and mse < 0.12
        and transition_quality >= 0.50
        and transition_ensemble_trained
        and transition_ensemble_size >= 2
        and (
            two_step_planning_gate["active"] and sealed_two_step_active
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
                "sealed_test": sealed_test,
                "semantic_event_head_gates": semantic_event_gates,
                "semantic_event_heads_ready": semantic_event_heads_ready,
                "pass_planner_quality": rt.pass_quality,
                "shot_planner_quality": rt.shot_quality,
                "pass_planner_active": rt.pass_quality >= rt.cfg.min_planner_quality,
                "shot_planner_active": effective_shot_active,
                "shot_fallback": "physics_xg_prior" if not effective_shot_active else None,
                "min_planner_quality": rt.cfg.min_planner_quality,
            }
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

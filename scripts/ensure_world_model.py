#!/usr/bin/env python3
"""
Ensure data/world_model/latent_wm.pt exists and is a validated v8 checkpoint.
Used by restart_full_run.ps1 before LLM full tournament.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Retrain even if checkpoint exists")
    parser.add_argument("--collect-pairs", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=45)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt = os.path.join(base_dir, "data", "world_model", "latent_wm.pt")

    os.environ.setdefault("MATCH_MICRO", "1")
    os.environ.setdefault("MATCH_MICRO_SCORE", "1")
    os.environ.setdefault("MATCH_SCHEDULED_SHOTS", "0")
    os.environ.setdefault("MATCH_WM_RECORD", "1")
    os.environ.setdefault("MATCH_BALL_LOG", "1")
    os.environ.setdefault("MATCH_WM_TRANSITION", "gru")

    need_train = args.force or not os.path.isfile(ckpt)
    if not need_train:
        try:
            from src.match_engine.world_model.model import load_checkpoint

            model, cfg, meta = load_checkpoint(ckpt)
            version = int(getattr(model, "checkpoint_version", 2))
            validation = meta.get("validation") or {}
            quality = float(validation.get("transition_quality", 0.0))
            if version < 8:
                raise RuntimeError(f"legacy checkpoint v{version}; v8 required")
            if quality < 0.50:
                raise RuntimeError(
                    f"transition quality {quality:.3f} below checkpoint gate 0.500"
                )
            if not bool(validation.get("transition_ensemble_trained")):
                raise RuntimeError("transition ensemble is not independently trained")
            members = int(validation.get("transition_ensemble_size", 0))
            if members < 2:
                raise RuntimeError(f"only {members} transition ensemble members")
            primary_mse = float(validation.get("transition_primary_mse", 1.0))
            ensemble_mse = float(validation.get("weighted_obs_mse", 1.0))
            if ensemble_mse > primary_mse * 1.05:
                raise RuntimeError(
                    "transition ensemble mean degrades primary MSE by more than 5%"
                )
            semantic = validation.get("semantic_event_heads") or {}
            if not bool(getattr(model, "semantic_event_heads_trained", False)):
                raise RuntimeError("semantic event heads are not trained")
            if not bool(semantic.get("trained")):
                raise RuntimeError("semantic event training contract is missing")
            if int(semantic.get("one_step_optimization_steps", 0)) <= 0:
                raise RuntimeError("one-step semantic event objective was not optimized")
            if int(semantic.get("two_step_optimization_steps", 0)) <= 0:
                raise RuntimeError("two-step semantic event objective was not optimized")
            print(f"[ensure_world_model] OK: {ckpt}")
            return
        except Exception as exc:
            print(f"[ensure_world_model] Checkpoint invalid ({exc}); retraining.")
            need_train = True

    if need_train:
        import subprocess

        print("[ensure_world_model] Collecting micro traces...")
        subprocess.check_call(
            [
                sys.executable,
                os.path.join(base_dir, "scripts", "collect_world_model_traces.py"),
                "--pairs",
                str(args.collect_pairs),
            ],
            cwd=base_dir,
        )
        print("[ensure_world_model] Collecting shot-rich traces...")
        subprocess.check_call(
            [
                sys.executable,
                os.path.join(base_dir, "scripts", "collect_world_model_traces.py"),
                "--pairs",
                str(max(16, args.collect_pairs // 2)),
                "--seed",
                "71000",
                "--run-tag",
                "shot_rich",
                "--shot-boost",
                "1.2",
            ],
            cwd=base_dir,
        )
        ball_dir = os.path.join(base_dir, "outputs", "ball_log")
        if os.path.isdir(ball_dir) and any(os.scandir(ball_dir)):
            print("[ensure_world_model] Backfilling from ball_log...")
            subprocess.check_call(
                [sys.executable, os.path.join(base_dir, "scripts", "backfill_wm_from_ball_log.py")],
                cwd=base_dir,
            )
        print("[ensure_world_model] Training GRU world model...")
        subprocess.check_call(
            [
                sys.executable,
                os.path.join(base_dir, "scripts", "train_world_model.py"),
                "--epochs",
                str(args.epochs),
                "--use-ball-log",
            ],
            cwd=base_dir,
        )
        print(f"[ensure_world_model] Ready: {ckpt}")


if __name__ == "__main__":
    main()

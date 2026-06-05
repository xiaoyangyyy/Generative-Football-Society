#!/usr/bin/env python3
"""
Ensure data/world_model/latent_wm.pt exists and is compatible (GRU v2).
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
    parser.add_argument("--collect-pairs", type=int, default=32)
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

            load_checkpoint(ckpt)
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

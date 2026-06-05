#!/usr/bin/env python3
"""Export ball_log jsonl → world_model training jsonl (dense pass/shot/intercept labels)."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    from pathlib import Path

    from src.match_engine.world_model.ball_log_dataset import transitions_from_ball_log_file
    from src.match_engine.world_model.config import default_trace_dir

    ball_dir = Path(base_dir) / "outputs" / "ball_log"
    out_dir = Path(default_trace_dir(base_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "_ball_log_backfill.jsonl"

    total = 0
    with open(out_path, "w", encoding="utf-8") as out_fp:
        for fp in sorted(ball_dir.glob("*.jsonl")):
            rows = transitions_from_ball_log_file(fp)
            for row in rows:
                out_fp.write(
                    json.dumps(
                        {
                            "obs": row["obs"].tolist(),
                            "action": row["action"].tolist(),
                            "next_obs": row["next_obs"].tolist(),
                            "meta": {**row.get("meta", {}), "backfill_from": fp.name},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                total += 1

    print(f"Wrote {total} transitions → {out_path}")


if __name__ == "__main__":
    main()
